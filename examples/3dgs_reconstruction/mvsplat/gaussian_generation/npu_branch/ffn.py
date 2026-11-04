"""Checkpoint-preserving 4D boundaries for six multiview Transformer FFNs.

Only Linear/GELU/Linear executes behind this boundary. LayerNorm, attention,
residuals and per-layer target updates remain owned by the original model.
No production graph is selected or qualified merely by importing this module.
"""
import torch
from torch import nn


from gaussian_generation.npu_branch.ffn_contract import (
    ADAPTER_SCHEMA, FFN_PARTITIONS, adapter_spec, execute_pointwise, validate_adapter,
)

def locate_ffn(model, name):
    if name not in FFN_PARTITIONS:
        raise ValueError('Unknown FFN partition: ' + name)
    module = model
    for part in FFN_PARTITIONS[name].split('.'):
        module = getattr(module, part)
    return module


def validate_mlp(module):
    if (not isinstance(module, nn.Sequential) or len(module) != 3 or
            not isinstance(module[0], nn.Linear) or not isinstance(module[1], nn.GELU) or
            not isinstance(module[2], nn.Linear) or
            module[0].out_features != module[2].in_features):
        raise ValueError('Expected exactly Linear -> GELU -> Linear')
    if any(p.dtype != torch.float32 or p.device.type != 'cpu' for p in module.parameters()):
        raise ValueError('FFN export requires unchanged CPU FP32 parameters')
    return module[0].in_features, module[0].out_features, module[2].out_features


class PointwiseFFN(nn.Module):
    """Use the original per-token matrices as 1x1 convolution kernels."""
    def __init__(self, original):
        super().__init__()
        incoming, hidden, outgoing = validate_mlp(original)
        self.first = nn.Conv2d(incoming, hidden, 1, bias=original[0].bias is not None)
        self.activation = nn.GELU(approximate=original[1].approximate)
        self.second = nn.Conv2d(hidden, outgoing, 1, bias=original[2].bias is not None)
        # Bind the exact trained storage, not fresh/rounded/retrained weights.
        # Each wrapper receives its own layer's matrices; never share across FFNs.
        for conv, linear in ((self.first, original[0]), (self.second, original[2])):
            conv.weight = nn.Parameter(linear.weight.detach().reshape(*linear.weight.shape, 1, 1),
                                       requires_grad=False)
            if linear.bias is not None:
                conv.bias = nn.Parameter(linear.bias.detach(), requires_grad=False)
        self.eval()

    def forward(self, value):
        return self.second(self.activation(self.first(value)))


def token_to_image(value, height, width, channels):
    if (value.ndim != 3 or value.shape[1] != height*width or value.shape[2] != channels or
            value.dtype != torch.float32 or value.device.type != 'cpu'):
        raise ValueError('Static CPU FP32 FFN token shape differs')
    return value.transpose(1, 2).reshape(value.shape[0], channels, height, width)


def image_to_token(value):
    if value.ndim != 4:
        raise ValueError('Expected NCHW FFN result')
    return value.flatten(2).transpose(1, 2)




class TokenFFNProxy(nn.Module):
    """Restore the original token ABI around an explicitly selected worker."""
    def __init__(self, runtime, name):
        super().__init__()
        self.runtime, self.name = runtime, name
        self.meta = runtime.parts[name]['meta']
        self.spec = validate_adapter(self.meta)
        if self.spec['partition'] != name:
            raise ValueError('FFN adapter binds a different layer')

    def forward(self, value):
        batch, channels, height, width = self.spec['full_input_shape']
        image = token_to_image(value, height, width, channels)
        if image.shape[0] != batch:
            raise ValueError('FFN token batch differs')
        result = execute_pointwise(image.detach().numpy(), self.meta,
                                  lambda tile: self.runtime.execute(self.name, tile))
        return image_to_token(torch.from_numpy(result))


def attach_ffn_partitions(model, runtime):
    """Explicit optional attach; defaults and CPU paths stay unchanged."""
    selected = [name for name in runtime.parts if name in FFN_PARTITIONS]
    proxies = {}
    for name in selected:
        original = locate_ffn(model, name)
        incoming, _, outgoing = validate_mlp(original)
        proxy = TokenFFNProxy(runtime, name)
        if (proxy.spec['full_input_shape'][1] != incoming or
                proxy.spec['full_output_shape'][1] != outgoing):
            raise ValueError('FFN compiled channels differ from original layer')
        proxies[name] = proxy
    # Validate every selected boundary before mutating any original module.
    for name, proxy in proxies.items():
        path = FFN_PARTITIONS[name].split('.')
        parent = model
        for part in path[:-1]:
            parent = getattr(parent, part)
        setattr(parent, path[-1], proxy)
    return selected
