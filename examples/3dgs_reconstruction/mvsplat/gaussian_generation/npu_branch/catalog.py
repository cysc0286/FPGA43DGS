"""Explicit network boundaries, including opt-in continuous depth execution."""
CORR_DEPTH_COMPONENTS = ('corr_refine_net', 'regressor_residual', 'depth_head')
FUSED_HEAD_COMPONENTS = ('to_gaussians', 'to_disparity')
REFINE_HEADS_COMPONENTS = ('proj_feature', 'refine_unet', 'to_gaussians', 'to_disparity')

PARTITIONS = {
    "backbone_cnn": "backbone.backbone",
    "regressor_residual": "depth_predictor.regressor_residual",
    "depth_head": "depth_predictor.depth_head_lowres",
    "upsampler": "depth_predictor.upsampler",
    "proj_feature": "depth_predictor.proj_feature",
    "to_gaussians": "depth_predictor.to_gaussians",
    "to_disparity": "depth_predictor.to_disparity",
    "corr_refine_net": "depth_predictor.corr_refine_net",
    "refine_unet": "depth_predictor.refine_unet",
    "fused_heads": "depth_predictor.to_gaussians",
    "corr_depth_pipeline": "depth_predictor._corr_to_depth_logits",
    "refine_heads_pipeline": "depth_predictor._refine_to_heads",
    **{f"transformer_ffn_{i}": f"backbone.transformer.layers.{i}.cross_attn_ffn.mlp"
       for i in range(6)},
}


def validate_selection(names):
    """Reject duplicate work and incompatible adapters before allocating sessions."""
    if isinstance(names, (str, bytes)):
        raise ValueError('Select unique, known partitions explicitly')
    names = list(names)
    if (not names or any(not isinstance(name, str) or name not in PARTITIONS for name in names)
            or len(set(names)) != len(names)):
        raise ValueError('Select unique, known partitions explicitly')
    for composite, components in (('fused_heads', FUSED_HEAD_COMPONENTS),
                                  ('corr_depth_pipeline', CORR_DEPTH_COMPONENTS),
                                  ('refine_heads_pipeline', REFINE_HEADS_COMPONENTS)):
        if composite in names and any(name in names for name in components):
            label = 'Fused heads' if composite == 'fused_heads' else composite
            raise ValueError(label + ' and its separate components are mutually exclusive')
    if 'refine_heads_pipeline' in names and 'fused_heads' in names:
        raise ValueError('Refinement pipeline and fused heads are mutually exclusive')
    return names


def semantic_coverage(names):
    """Original operations covered, independently of graph boundary count."""
    covered = set(validate_selection(names))
    for composite, components in (('fused_heads', FUSED_HEAD_COMPONENTS),
                                  ('corr_depth_pipeline', CORR_DEPTH_COMPONENTS),
                                  ('refine_heads_pipeline', REFINE_HEADS_COMPONENTS)):
        if composite in covered:
            covered.remove(composite)
            covered.update(components)
    return covered


def locate(model, name):
    module = model
    for component in PARTITIONS[name].split("."):
        module = getattr(module, component)
    return module


def replace(model, name, module):
    path = PARTITIONS[name].split(".")
    parent = model
    for component in path[:-1]:
        parent = getattr(parent, component)
    setattr(parent, path[-1], module)
