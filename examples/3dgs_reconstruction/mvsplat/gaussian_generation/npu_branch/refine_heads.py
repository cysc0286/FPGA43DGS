"""Explicit, opt-in projection/refinement/two-head seam; no default patching.

The CPU upsampler remains outside this boundary. A single full two-view NCHW
input packs [upsampled features, RGB, disparity, confidence] without changing
their values. The executor returns both raw heads in the same transaction.
This module neither registers a partition nor qualifies an NPU implementation.
"""
import ast
import hashlib
import inspect
import types
import weakref

import numpy as np


SCHEMA = 'mvsplat_refine_heads_seam_v1'
METHOD = '_refine_to_heads'
MARKER = '_mgs_refine_heads_seam'
OWNER = '_mgs_refine_heads_adapter'
SEAM_DOC = 'Unchanged MVSplat feature projection, refinement and both raw prediction heads.'
COMPONENTS = ('proj_feature', 'refine_unet', 'to_gaussians', 'to_disparity')
INPUT_CHANNELS = (128, 3, 1, 1)
OUTPUT_CHANNELS = (84, 2)

ORIGINAL_BLOCK = '''        proj_feature = self.proj_feature(proj_feat_in_fullres)
        refine_out = self.refine_unet(torch.cat(
            (extra_info["images"], proj_feature, fullres_disps, pdf_max), dim=1
        ))

        # gaussians head
        raw_gaussians_in = [refine_out,
                            extra_info["images"], proj_feat_in_fullres]
        raw_gaussians_in = torch.cat(raw_gaussians_in, dim=1)
        raw_gaussians = self.to_gaussians(raw_gaussians_in)
'''

ORIGINAL_DISPARITY = '            delta_disps_density = self.to_disparity(refine_out)\n'
REPLACEMENT_BLOCK = '''        if self.wo_depth_refine:
''' + ''.join('    ' + line if line.strip() else line
              for line in ORIGINAL_BLOCK.splitlines(keepends=True)) + '''        else:
            raw_gaussians, delta_disps_density = self._refine_to_heads(
                proj_feat_in_fullres, extra_info["images"], fullres_disps, pdf_max
            )
'''
REPLACEMENT_DISPARITY = '            # Both raw heads were returned by the explicit refinement seam.\n'

METHOD_SOURCE = '''    _mgs_refine_heads_seam = 'mvsplat_refine_heads_seam_v1'

    def _refine_to_heads(self, proj_feat_in_fullres, images, fullres_disps, pdf_max):
        """Unchanged MVSplat feature projection, refinement and both raw prediction heads."""
        proj_feature = self.proj_feature(proj_feat_in_fullres)
        refine_out = self.refine_unet(torch.cat(
            (images, proj_feature, fullres_disps, pdf_max), dim=1
        ))
        raw_gaussians_in = torch.cat((refine_out, images, proj_feat_in_fullres), dim=1)
        raw_gaussians = self.to_gaussians(raw_gaussians_in)
        delta_disps_density = self.to_disparity(refine_out)
        return raw_gaussians, delta_disps_density

'''


def _sha(source):
    return hashlib.sha256(source.encode('utf-8')).hexdigest()


def _dump(nodes):
    return [ast.dump(node, include_attributes=False) for node in nodes]


def patch_source(source):
    """Return a strictly matched vendor-source copy; write or mutate nothing.

    The ablation retains the original projection/refinement/Gaussian sequence
    and never calls a nonexistent disparity head. Only the ordinary branch
    uses the new method. Correlation-depth source patching may precede or follow
    this transformation; neither operation owns the other's statements.
    """
    if not isinstance(source, str):
        raise TypeError('Expected original DepthPredictorMultiView source text')
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError('Invalid original predictor source') from error
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)
               and node.name == 'DepthPredictorMultiView']
    if len(classes) != 1:
        raise ValueError('Expected exactly one top-level DepthPredictorMultiView class')
    cls = classes[0]
    forwards = [node for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == 'forward']
    if (len(forwards) != 1 or isinstance(forwards[0], ast.AsyncFunctionDef)
            or forwards[0].decorator_list):
        raise ValueError('Expected one undecorated predictor forward method')
    if any(token in source for token in (METHOD, MARKER, OWNER)):
        raise ValueError('Predictor source already contains a refinement-head seam')
    if '\r' in source:
        raise ValueError('Normalize predictor source to LF before exact source matching')
    if source.count(ORIGINAL_BLOCK) != 1 or source.count(ORIGINAL_DISPARITY) != 1:
        raise ValueError('Original projection/refinement/head blocks changed; review patch')
    forward = forwards[0]
    lines = source.splitlines(keepends=True)
    begin = sum(map(len, lines[:forward.lineno - 1]))
    end = sum(map(len, lines[:forward.end_lineno]))
    position = source.index(ORIGINAL_BLOCK)
    disparity_position = source.index(ORIGINAL_DISPARITY)
    if not begin < position < position + len(ORIGINAL_BLOCK) < disparity_position < end:
        raise ValueError('Original refinement/head blocks must be ordered inside predictor forward')
    first_line = source.count('\n', 0, position) + 1
    last_line = first_line + ORIGINAL_BLOCK.count('\n') - 1
    statements = [node for node in forward.body
                  if first_line <= node.lineno and node.end_lineno <= last_line]
    reference = ast.parse('def original(self):\n' + ORIGINAL_BLOCK).body[0].body
    if _dump(statements) != _dump(reference):
        raise ValueError('Original refinement block must be executable top-level forward statements')
    disparity_line = source.count('\n', 0, disparity_position) + 1
    disparity_reference = ast.parse('def original(self):\n' + ORIGINAL_DISPARITY).body[0].body[0]
    ablations = [node for node in forward.body if isinstance(node, ast.If)
                 and isinstance(node.test, ast.Attribute) and node.test.attr == 'wo_depth_refine'
                 and isinstance(node.test.value, ast.Name) and node.test.value.id == 'self']
    if (len(ablations) != 1 or not ablations[0].orelse
            or ablations[0].orelse[0].lineno != disparity_line
            or _dump([ablations[0].orelse[0]]) != _dump([disparity_reference])):
        raise ValueError('Original disparity call must begin the depth-ablation else branch')
    moved = {'proj_feature', 'refine_out', 'raw_gaussians_in'}
    for node in ast.walk(forward):
        if isinstance(node, ast.Name) and node.id in moved:
            if not (first_line <= node.lineno <= last_line or node.lineno == disparity_line):
                raise ValueError('Moved refinement temporary escapes the reviewed blocks')
        if isinstance(node, ast.Name) and node.id == 'delta_disps_density':
            if not ablations[0].orelse[0].lineno <= node.lineno <= ablations[0].end_lineno:
                raise ValueError('Raw disparity temporary escapes its original else branch')
    # Replace from later offsets first, then insert before forward. Other
    # generated methods and all untouched downstream statements stay intact.
    patched = source[:disparity_position] + REPLACEMENT_DISPARITY + source[
        disparity_position + len(ORIGINAL_DISPARITY):]
    patched = patched[:position] + REPLACEMENT_BLOCK + patched[position + len(ORIGINAL_BLOCK):]
    patched = patched[:begin] + METHOD_SOURCE + patched[begin:]
    try:
        ast.parse(patched)
    except SyntaxError as error:
        raise ValueError('Generated refinement-head source is invalid') from error
    return patched, dict(schema=SCHEMA, original_sha256=_sha(source), modified_sha256=_sha(patched),
        target='DepthPredictorMultiView.forward', method=METHOD, matched_blocks=2,
        source_written=False, operations=list(COMPONENTS), cpu_upsampler_preserved=True,
        ablation_cpu_path_preserved=True, ablation_npu_supported=False,
        parameter_names_changed=False, downstream_operations_changed=False,
        default_runtime_changed=False, npu_executed=False,
        input_channels=list(INPUT_CHANNELS), packed_input_channels=sum(INPUT_CHANNELS),
        output_channels=list(OUTPUT_CHANNELS), packed_output_channels=sum(OUTPUT_CHANNELS))


class RefineHeadsAdapter:
    """Own only the generated synchronous method, without hooks or caches."""
    def __init__(self, predictor, execute):
        import torch
        if not isinstance(predictor, torch.nn.Module) or not callable(execute):
            raise ValueError('Expected predictor module and callable executor')
        if hasattr(predictor, OWNER):
            raise ValueError('Refinement-head adapter is already attached or its owner slot is occupied')
        if hasattr(predictor, '_mgs_fused_heads_adapter'):
            raise ValueError('Refinement-head pipeline conflicts with fused prediction heads')
        original = getattr(predictor, METHOD, None)
        if (getattr(type(predictor), MARKER, None) != SCHEMA or not inspect.ismethod(original)
                or original.__self__ is not predictor or original.__func__.__doc__ != SEAM_DOC
                or len(inspect.signature(original).parameters) != 4):
            raise ValueError('Predictor lacks the generated refinement-head seam')
        if getattr(predictor, 'wo_depth_refine', None) is not False:
            raise ValueError('Refinement-head NPU ablation candidate is unsupported')
        self.modules = {name: getattr(predictor, name, None) for name in COMPONENTS}
        proj, refine, gaussian, disparity = (self.modules[name] for name in COMPONENTS)

        def head(module, incoming, hidden, outgoing):
            return (type(module) is torch.nn.Sequential and len(module) == 3
                and type(module[0]) is torch.nn.Conv2d and type(module[1]) is torch.nn.GELU
                and type(module[2]) is torch.nn.Conv2d
                and (module[0].in_channels, module[0].out_channels,
                     module[2].in_channels, module[2].out_channels)
                    == (incoming, hidden, hidden, outgoing))

        if (type(proj) is not torch.nn.Conv2d or (proj.in_channels, proj.out_channels) != (128, 32)
                or type(refine) is not torch.nn.Sequential or len(refine) != 4
                or type(refine[0]) is not torch.nn.Conv2d
                or (refine[0].in_channels, refine[0].out_channels) != (37, 32)
                or type(refine[1]) is not torch.nn.GroupNorm
                or (refine[1].num_groups, refine[1].num_channels) != (4, 32)
                or type(refine[2]) is not torch.nn.GELU
                or not isinstance(refine[3], torch.nn.Module)
                or not head(gaussian, 163, 168, 84) or not head(disparity, 32, 64, 2)):
            raise ValueError('Original refinement-head module/channel contract differs or a component is replaced')
        if any(parameter.device.type != 'cpu' or parameter.dtype != torch.float32
               for module in self.modules.values() for parameter in module.parameters()):
            raise ValueError('Refinement-head reference parameters must remain CPU FP32')
        self.predictor, self.execute, self.original = predictor, execute, original
        self.had_instance_method = METHOD in predictor.__dict__
        self.instance_method = predictor.__dict__.get(METHOD)
        self.closed = self.failed = self.active = False
        self._check_eval()
        owner = self

        def proxy(bound_predictor, up, images, disparity, confidence):
            if bound_predictor is not owner.predictor:
                raise ValueError('Refinement-head seam was rebound to another predictor')
            return owner._forward(up, images, disparity, confidence)

        self.proxy = types.MethodType(proxy, predictor)
        try:
            setattr(predictor, OWNER, weakref.ref(self))
            setattr(predictor, METHOD, self.proxy)
        except BaseException:
            self._restore()
            self.closed = True
            raise

    def _check_eval(self):
        if self.predictor.training or any(module.training
                for original in self.modules.values() for module in original.modules()):
            raise ValueError('Refinement-head adapter is inference/eval-only')

    def _forward(self, *values):
        import torch
        if self.closed or self.failed or self.active:
            raise RuntimeError('Refinement-head adapter is closed, failed or already active')
        self._check_eval()
        if (getattr(self.predictor, 'wo_depth_refine', None) is not False
                or hasattr(self.predictor, '_mgs_fused_heads_adapter')
                or any(getattr(self.predictor, name, None) is not module
                       for name, module in self.modules.items())):
            raise ValueError('Attached refinement-head model contract changed or conflicts')
        spatial = None
        for value, channels in zip(values, INPUT_CHANNELS):
            if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                    or value.dtype != torch.float32 or value.requires_grad or value.ndim != 4
                    or value.shape[:2] != (2, channels) or any(size < 1 for size in value.shape[2:])
                    or not torch.isfinite(value).all()):
                raise ValueError('Expected finite two-view CPU FP32 refinement inputs without gradients')
            if spatial is not None and value.shape[2:] != spatial:
                raise ValueError('Refinement inputs must share the same spatial shape')
            spatial = value.shape[2:]
        packed = np.empty((2, sum(INPUT_CHANNELS), *spatial), dtype=np.float32)
        offset = 0
        for value, channels in zip(values, INPUT_CHANNELS):
            packed[:, offset:offset + channels] = value.detach().numpy()
            offset += channels
        self.active = True
        try:
            result = self.execute(packed)
            if (not isinstance(result, np.ndarray) or result.dtype != np.float32
                    or result.shape != (2, sum(OUTPUT_CHANNELS), *spatial)
                    or not np.isfinite(result).all()):
                raise ValueError('Invalid refinement-head FP32 NCHW output shape or finite values')
            # Independent owned arrays: no head retains the whole shared result
            # and the next worker transaction cannot alter either returned value.
            return tuple(torch.from_numpy(np.array(part, dtype=np.float32, order='C', copy=True))
                         for part in (result[:, :OUTPUT_CHANNELS[0]], result[:, OUTPUT_CHANNELS[0]:]))
        except BaseException:
            self.failed = True
            raise
        finally:
            self.active = False

    def _restore(self):
        if self.had_instance_method:
            setattr(self.predictor, METHOD, self.instance_method)
        elif METHOD in self.predictor.__dict__:
            delattr(self.predictor, METHOD)
        reference = getattr(self.predictor, OWNER, None)
        if isinstance(reference, weakref.ReferenceType) and reference() is self:
            delattr(self.predictor, OWNER)

    def close(self):
        if self.closed:
            return
        if self.active:
            raise RuntimeError('Cannot detach an active refinement-head invocation')
        self._restore()
        self.closed = True

    def __enter__(self):
        if self.closed:
            raise RuntimeError('Refinement-head adapter is closed')
        return self

    def __exit__(self, *args):
        self.close()


def attach_refine_heads(predictor, execute):
    """Attach explicitly; the caller owns close and separate graph qualification.

    ``execute`` receives [up128, RGB3, disparity1, confidence1] NCHW float32 and
    returns [raw_gaussians84, raw_disparity_density2]. It must not recurse into
    this seam or use the untouched modules as a production CPU fallback.
    """
    return RefineHeadsAdapter(predictor, execute)
