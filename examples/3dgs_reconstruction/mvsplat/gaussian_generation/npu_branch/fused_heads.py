"""Two local prediction heads behind one explicitly scoped NPU transaction.

The original depth predictor calls Gaussian prediction before disparity. The
adapter preserves that call order, but caches only the two disparity channels
from a single fused result. A cache can be consumed only in the same frame by
the very refinement tensor that produced the Gaussian-head input.
"""
import hashlib
import weakref

import numpy as np


def split_output(value, gaussian_channels=84, disparity_channels=2):
    """Validate and split a NCHW result, detaching the small pending output."""
    if (not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.ndim != 4 or
            value.shape[1] != gaussian_channels+disparity_channels or
            not np.isfinite(value).all()):
        raise ValueError('Invalid fused prediction-head output')
    return value[:, :gaussian_channels], value[:, gaussian_channels:].copy()


def torch_fused_heads(gaussian, disparity, refine_channels=32):
    """Wrap unchanged Torch heads for reference/export, not NPU fallback."""
    import torch

    class Fused(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gaussian = gaussian
            self.disparity = disparity

        def forward(self, value):
            return torch.cat((self.gaussian(value),
                              self.disparity(value[:, :refine_channels])), dim=1)

    return Fused().eval()


def _tensor_signature(value):
    # Inference tensors deliberately have no version counter. Hash a view of
    # their storage so an in-place edit cannot consume a stale prediction.
    try:
        return ('version', value._version)
    except RuntimeError:
        array = value.detach().numpy()
        contiguous = np.ascontiguousarray(array)
        return ('sha256', hashlib.sha256(memoryview(contiguous).cast('B')).digest())


class FusedHeadAdapter:
    def __init__(self, predictor, execute, *, input_channels=163,
                 refine_channels=32, gaussian_channels=84, disparity_channels=2):
        import torch
        if hasattr(predictor, '_mgs_refine_heads_adapter'):
            raise ValueError('Fused prediction heads conflict with the refinement-head pipeline')
        previous = getattr(predictor, '_mgs_fused_heads_adapter', None)
        if previous is not None and previous() is not None:
            raise ValueError('Fused heads are already attached')
        if (not callable(execute) or any(type(v) is not int or v < 1 for v in
                (input_channels, refine_channels, gaussian_channels, disparity_channels)) or
                refine_channels > input_channels):
            raise ValueError('Invalid fused-head channel/executor contract')
        self.predictor, self.execute = predictor, execute
        self.input_channels, self.refine_channels = input_channels, refine_channels
        self.gaussian_channels, self.disparity_channels = gaussian_channels, disparity_channels
        self.original = (predictor.to_gaussians, predictor.to_disparity)
        self.closed = False
        self.active = False
        self.sequence = 0
        self.refine = None
        self.pending = None
        self.signature = None
        self.generated = False
        self.handles = []
        owner = self
        predictor._mgs_fused_heads_adapter = weakref.ref(self)

        class GaussianProxy(torch.nn.Module):
            def forward(self, value):
                return owner.gaussians(value)

        class DisparityProxy(torch.nn.Module):
            def forward(self, value):
                return owner.disparity(value)

        try:
            self.handles.append(predictor.register_forward_pre_hook(self._begin))
            self.handles.append(predictor.refine_unet.register_forward_hook(self._bind))
            self.handles.append(predictor.register_forward_hook(self._finish))
            predictor.to_gaussians = GaussianProxy()
            predictor.to_disparity = DisparityProxy()
        except BaseException:
            self.close()
            raise

    def _begin(self, module, args):
        if self.closed or self.active or self.pending is not None:
            raise RuntimeError('Overlapping or unfinished fused-head frame')
        self.sequence += 1
        self.active = True
        self.generated = False
        self.refine = self.signature = None

    def _bind(self, module, args, value):
        if (not self.active or self.refine is not None or not hasattr(value, 'ndim') or
                value.ndim != 4 or value.shape[1] != self.refine_channels):
            raise ValueError('Unexpected refinement tensor for fused heads')
        self.refine = weakref.ref(value)

    def _finish(self, module, args, output):
        if not self.generated or self.pending is not None:
            raise RuntimeError('Fused result was not produced and consumed exactly once')
        self.active = False
        self.refine = self.signature = None

    def gaussians(self, value):
        import torch
        refine = self.refine() if self.refine is not None else None
        if (self.closed or not self.active or self.generated or self.pending is not None or
                refine is None or value.device.type != 'cpu' or value.dtype != torch.float32 or
                value.ndim != 4 or value.shape[1] != self.input_channels or
                value.shape[0] != refine.shape[0] or value.shape[2:] != refine.shape[2:] or
                value.requires_grad or refine.requires_grad):
            raise ValueError('Fused Gaussian input is outside its bound inference frame')
        if not torch.equal(value[:, :self.refine_channels], refine):
            raise ValueError('Fused Gaussian prefix differs from current refinement')
        self.signature = _tensor_signature(refine)
        result = self.execute(value.detach().numpy())
        if tuple(result.shape[:1]+result.shape[2:]) != tuple(value.shape[:1]+value.shape[2:]):
            raise ValueError('Fused head changed batch or spatial shape')
        gaussian, disparity = split_output(result, self.gaussian_channels, self.disparity_channels)
        self.pending = (self.sequence, disparity)
        self.generated = True
        return torch.from_numpy(gaussian)

    def disparity(self, value):
        import torch
        refine = self.refine() if self.refine is not None else None
        if (self.closed or not self.active or self.pending is None or
                self.pending[0] != self.sequence or value is not refine or
                _tensor_signature(value) != self.signature):
            raise ValueError('Stale, missing or mismatched fused disparity result')
        result = self.pending[1]
        self.pending = None
        self.signature = None
        return torch.from_numpy(result)

    def close(self):
        if self.closed:
            return
        self.closed = True
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.predictor.to_gaussians, self.predictor.to_disparity = self.original
        if getattr(self.predictor, '_mgs_fused_heads_adapter', lambda: None)() is self:
            delattr(self.predictor, '_mgs_fused_heads_adapter')
        self.refine = self.pending = self.signature = None
        self.active = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def attach_fused_heads(predictor, execute, **channels):
    """Install once; caller owns the returned closeable adapter.

    ``execute`` receives a full NCHW float32 NumPy input. Typically it is
    ``lambda x: partition_runtime.execute('fused_heads', x)``. It is not allowed
    to call the original CPU heads as a production fallback.
    """
    return FusedHeadAdapter(predictor, execute, **channels)
