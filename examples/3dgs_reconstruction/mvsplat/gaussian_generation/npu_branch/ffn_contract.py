"""Torch-free FFN boundary metadata and independent-token execution.

Imported by the SDK worker's asset validation. Keep this module free of Torch
so loading candidate metadata does not create a second full inference runtime.
"""
import numpy as np


FFN_PARTITIONS = {
    'transformer_ffn_' + str(i): 'backbone.transformer.layers.' + str(i) + '.cross_attn_ffn.mlp'
    for i in range(6)
}
ADAPTER_SCHEMA = 'mvsplat_token_ffn_1x1_v1'



def adapter_spec(name, full_input, full_output, execution_input, execution_output):
    spec = dict(schema=ADAPTER_SCHEMA, partition=name, path=FFN_PARTITIONS[name],
                full_input_shape=list(full_input), full_output_shape=list(full_output),
                token_layout='NLC', image_layout='NCHW', halo=0,
                semantics='Original Linear/GELU/Linear only; no LayerNorm or residual moved')
    validate_adapter(dict(input_shape=list(execution_input), output_shape=list(execution_output),
                          ffn_adapter=spec))
    return spec


def validate_adapter(meta):
    spec = meta.get('ffn_adapter')
    if (not isinstance(spec, dict) or spec.get('schema') != ADAPTER_SCHEMA or
            spec.get('partition') not in FFN_PARTITIONS or
            spec.get('path') != FFN_PARTITIONS[spec['partition']] or
            spec.get('halo') != 0 or spec.get('token_layout') != 'NLC' or
            spec.get('image_layout') != 'NCHW'):
        raise ValueError('Unreviewed FFN token adapter')
    full_in, full_out = spec['full_input_shape'], spec['full_output_shape']
    actual_in, actual_out = meta['input_shape'], meta['output_shape']
    if (any(len(shape) != 4 or any(type(n) is not int or n < 1 for n in shape)
            for shape in (full_in, full_out, actual_in, actual_out)) or
            full_in[0] != full_out[0] or full_in[2:] != full_out[2:] or
            actual_in[0] != actual_out[0] or actual_in[2:] != actual_out[2:] or
            actual_in[1] != full_in[1] or actual_out[1] != full_out[1] or
            actual_in[0] not in (1, full_in[0]) or full_in[0] % actual_in[0] or
            any(a > b for a, b in zip(actual_in[2:], full_in[2:]))):
        raise ValueError('FFN token/execution shapes disagree')
    return spec


def execute_pointwise(value, meta, forward):
    """Exact independent token tiles; no halo, resizing or token removal."""
    spec = validate_adapter(meta)
    if list(value.shape) != spec['full_input_shape'] or value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError('FFN full input shape/dtype/finite contract failed')
    tile_in, tile_out = meta['input_shape'], meta['output_shape']
    if list(value.shape) == tile_in:
        result = forward(value)
        if list(result.shape) != tile_out or result.dtype != np.float32 or not np.isfinite(result).all():
            raise ValueError('FFN output shape/dtype/finite contract failed')
        return result
    output = np.empty(spec['full_output_shape'], dtype=np.float32)
    batch, _, height, width = value.shape
    nb, _, th, tw = tile_in
    for n in range(0, batch, nb):
        for y in range(0, height, th):
            for x in range(0, width, tw):
                h, w = min(th, height-y), min(tw, width-x)
                tile = value[n:n+nb, :, y:y+h, x:x+w]
                if h != th or w != tw:
                    padded = np.zeros(tile_in, dtype=np.float32)
                    padded[:, :, :h, :w] = tile
                    tile = padded
                result = forward(tile)
                if (list(result.shape) != tile_out or result.dtype != np.float32 or
                        not np.isfinite(result).all()):
                    raise ValueError('FFN tile output shape/dtype/finite contract failed')
                output[n:n+nb, :, y:y+h, x:x+w] = result[:, :, :h, :w]
    return output
