"""Bounded spatial execution for the two local Conv/GELU/Conv prediction heads.

Image boundaries retain each layer's original zero padding. Interior execution
windows overlap by the two-convolution radius; only fully supported pixels are
copied back. This is not Gaussian pruning, image resizing, or CPU head fallback.
"""
import numpy as np


def validate_tiling(meta):
    spec = meta.get('spatial_tiling')
    if spec is None:
        return None
    if spec.get('schema') != 'mvsplat_local_head_tiles_v1' or spec.get('halo') != 2:
        raise ValueError('Unreviewed spatial head contract')
    full_in, full_out = spec['input_shape'], spec['output_shape']
    tile_in, tile_out = meta['input_shape'], meta['output_shape']
    if (len(full_in) != 4 or len(full_out) != 4 or len(tile_in) != 4 or len(tile_out) != 4 or
            any(type(v) is not int or v < 1 for shape in (full_in,full_out,tile_in,tile_out) for v in shape) or
            full_in[0] != full_out[0] or full_in[2:] != full_out[2:] or
            tile_in[0] != 1 or tile_out[0] != 1 or tile_in[2:] != tile_out[2:] or
            tile_in[1] != full_in[1] or tile_out[1] != full_out[1]):
        raise ValueError('Spatial head shapes disagree')
    core = spec['core_size']
    if type(core) is not int or core < 1 or any(core > v-4 for v in tile_in[2:]):
        raise ValueError('Insufficient two-pixel head halo')
    if any(t > f for t,f in zip(tile_in[2:],full_in[2:])):
        raise ValueError('Execution tile exceeds full image')
    return spec


def axis_windows(length, tile, core, halo=2):
    for start in range(0,length,core):
        end=min(start+core,length)
        origin=min(max(start-halo,0),length-tile)
        yield slice(origin,origin+tile),slice(start,end),slice(start-origin,end-origin)


def execute_spatial(value, meta, forward):
    spec=validate_tiling(meta)
    if spec is None or list(value.shape) != spec['input_shape'] or value.dtype != np.float32:
        raise ValueError('Full spatial input contract differs')
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite full spatial input')
    output=np.empty(spec['output_shape'],np.float32)
    tile=meta['input_shape']
    for batch in range(value.shape[0]):
        for source_y,target_y,keep_y in axis_windows(value.shape[2],tile[2],spec['core_size']):
            for source_x,target_x,keep_x in axis_windows(value.shape[3],tile[3],spec['core_size']):
                result=forward(value[batch:batch+1,:,source_y,source_x])
                if list(result.shape) != meta['output_shape'] or not np.isfinite(result).all():
                    raise ValueError('Bad spatial head tile output')
                output[batch:batch+1,:,target_y,target_x]=result[:,:,keep_y,keep_x]
    return output
