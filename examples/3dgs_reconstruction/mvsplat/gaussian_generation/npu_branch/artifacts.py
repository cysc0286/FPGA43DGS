"""Validate deployment assets and translate logical NCHW to audited host layouts."""
import json
import re
from pathlib import Path, PurePosixPath, PureWindowsPath
import numpy as np
from common import FULL_WEIGHT_SHA256, sha
from gaussian_generation.npu_branch.catalog import validate_selection


def relative_file(root, name):
    root = Path(root).resolve()
    # Legacy compiler manifests were written on Windows. Interpret separators
    # as a portable relative path, never as Linux filename characters.
    portable = PurePosixPath(str(name).replace("\\", "/"))
    if portable.is_absolute() or PureWindowsPath(name).drive or ".." in portable.parts:
        raise ValueError("Artifact outside bundle or missing: " + str(name))
    path = root.joinpath(*portable.parts).resolve()
    if root not in path.parents or not path.is_file():
        raise ValueError("Artifact outside bundle or missing: " + str(name))
    return path


def validate_bundle(root, names, backend):
    names = validate_selection(names)
    root = Path(root).resolve()
    data = json.loads((root / "manifest.json").read_text())
    schema = "mvsplat_compiled_v1" if backend == "npu" else "mvsplat_partitions_v1"
    if data["schema"] != schema or data["weight_sha256"] != FULL_WEIGHT_SHA256:
        raise ValueError("Bundle schema/checkpoint mismatch")
    result = {}
    for name in names:
        item = data["partitions"][name]
        if not item["complete"]:
            raise ValueError("Incomplete partition: " + name)
        if backend == "npu":
            from gaussian_generation.npu_branch.compile_graphs import audit
            graph, raw = relative_file(root, item["graph"]), relative_file(root, item["raw"])
            actual = audit(graph, raw, item["export"])
            if actual != item["audit"]:
                raise ValueError("Compiled placement/ABI/hash changed")
            meta = item["export"]
            result[name] = dict(meta=meta, input=actual["input"], output=actual["output"],
                                graph=str(graph), raw=str(raw), audit=actual)
        else:
            graph = relative_file(root, name + "/model.onnx")
            if sha(graph) != item["graph_sha256"] or not item["host_onnx_passed"]:
                raise ValueError("Unvalidated ONNX graph")
            output_layout = item.get('onnx_output_layout', 'NCHW')
            if output_layout not in ('NCHW', 'NHWC'):
                raise ValueError('Unreviewed ONNX output layout')
            shape = item['output_shape']
            output_shape = shape if output_layout == 'NCHW' else [shape[0], shape[2], shape[3], shape[1]]
            if item.get('onnx_output_shape', output_shape) != output_shape:
                raise ValueError('ONNX output axes differ from logical shape')
            result[name] = dict(meta=item, graph=str(graph),
                input=dict(layout="NCHW", shape=item["input_shape"]),
                output=dict(layout=output_layout, shape=output_shape))
        from gaussian_generation.npu_branch.spatial import validate_tiling
        tiling = validate_tiling(result[name]['meta'])
        if tiling and name not in ('to_gaussians','to_disparity','fused_heads'):
            raise ValueError('Spatial execution is restricted to local prediction heads')
        if name.startswith('transformer_ffn_'):
            from gaussian_generation.npu_branch.ffn_contract import validate_adapter
            spec = validate_adapter(result[name]['meta'])
            if spec['partition'] != name:
                raise ValueError('FFN adapter names a different layer')
        if name == 'corr_depth_pipeline':
            meta = result[name]['meta']
            if (meta.get('full_module') is not True or meta.get('returns_list') is not False
                    or meta.get('dtype') != 'float32' or meta.get('logical_layout') != 'NCHW'
                    or meta.get('input_shape') != [2, 256, 32, 32]
                    or meta.get('output_shape') != [2, 128, 32, 32]
                    or meta.get('composed_modules') !=
                       ['corr_refine_net', 'regressor_residual', 'depth_head_lowres']
                    or set(meta.get('source_provenance', {})) !=
                       {'corr_refine_net', 'regressor_residual', 'depth_head'}
                    or not meta.get('oracle_provenance')):
                raise ValueError('Continuous depth graph composition/ABI/provenance is not audited')
        if name == 'refine_heads_pipeline':
            meta = result[name]['meta']
            components = ['proj_feature', 'refine_unet', 'to_gaussians', 'to_disparity']
            if (meta.get('full_module') is not True or meta.get('returns_list') is not False
                    or meta.get('dtype') != 'float32' or meta.get('logical_layout') != 'NCHW'
                    or meta.get('input_shape') != [2, 133, 128, 128]
                    or meta.get('output_shape') != [2, 86, 128, 128]
                    or meta.get('composed_modules') != components
                    or set(meta.get('source_provenance', {})) != set(components)
                    or meta.get('upsampler_included') is not False
                    or meta.get('upsampled_feature_reused') is not True
                    or meta.get('full_view_batch_preserved') is not True
                    or meta.get('spatial_tiling_used') is not False
                    or meta.get('packed_input') != [dict(name=n, channels=c) for n, c in
                        (('upsampled_features',128),('rgb',3),('coarse_disparity',1),('pdf_max',1))]
                    or meta.get('packed_output') != [dict(name='raw_gaussians',channels=84),
                        dict(name='delta_disparity_raw_density',channels=2)]
                    or not meta.get('oracle_provenance')):
                raise ValueError('Continuous refinement/head composition/ABI/provenance is not audited')
            if meta.get('oracle_provenance') != 'source_onnx_derived_chain_on_explicit_packed_diagnostic_inputs':
                raise ValueError('Continuous refinement oracle provenance is not audited')
            for source in meta['source_provenance'].values():
                if (not isinstance(source, dict) or any(
                        not isinstance(source.get(key), str) or
                        re.fullmatch(r'[0-9a-f]{64}', source[key]) is None
                        for key in ('manifest_sha256', 'graph_sha256', 'oracle_sha256', 'context_sha256'))
                        or not isinstance(source.get('source_oracle_check'), dict)
                        or source['source_oracle_check'].get('passed') is not True):
                    raise ValueError('Continuous refinement source provenance is not audited')
    return data, result


def _blocked_copy(out, source, rows=8, channels=16):
    """Copy a layout view in cache-sized tiles, without a full float temporary.

    Both arguments have the same logical shape. Tiling the height and channel
    dimensions bounds the active source/destination cache lines on ARM cores.
    """
    for n in range(source.shape[0]):
        for h in range(0, source.shape[1], rows):
            for c in range(0, source.shape[3], channels):
                np.copyto(out[n, h:h+rows, :, c:c+channels],
                          source[n, h:h+rows, :, c:c+channels], casting='no')


def to_wire(array, spec, out=None, *, blocked=False):
    if spec["layout"] == "NHWC":
        array = array.transpose(0, 2, 3, 1)
    elif spec["layout"] != "NCHW":
        raise ValueError("Unsupported layout")
    if list(array.shape) != spec["shape"] or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError("Input dtype/shape/finite contract failed")
    if out is None:
        return np.ascontiguousarray(array)
    if (out.shape != array.shape or out.dtype != np.float32 or
            not out.flags.c_contiguous or not out.flags.writeable):
        raise ValueError("Destination buffer ABI mismatch")
    # A transposed view can be copied directly into IPC. Do not materialize a
    # second, full-size contiguous NHWC float buffer first.
    if blocked and spec['layout'] == 'NHWC':
        _blocked_copy(out, array)
    else:
        np.copyto(out, array, casting="no")
    return out


def from_wire(array, spec):
    if list(array.shape) != spec["shape"] or not np.isfinite(array).all():
        raise ValueError("Output shape/finite contract failed")
    if spec["layout"] == "NHWC":
        array = array.transpose(0, 3, 1, 2)
    elif spec["layout"] != "NCHW":
        raise ValueError("Unsupported output layout")
    # Copy before the reusable output buffer is assigned to the next job.
    return np.array(array, dtype=np.float32, order="C", copy=True)
