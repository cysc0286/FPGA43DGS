"""Static, unary network partitions. Geometry/attention outside these stay CPU."""
PARTITIONS = {
    "backbone_cnn": "backbone.backbone",
    "regressor_residual": "depth_predictor.regressor_residual",
    "depth_head": "depth_predictor.depth_head_lowres",
    "upsampler": "depth_predictor.upsampler",
    "proj_feature": "depth_predictor.proj_feature",
    "to_gaussians": "depth_predictor.to_gaussians",
    "to_disparity": "depth_predictor.to_disparity",
}


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
