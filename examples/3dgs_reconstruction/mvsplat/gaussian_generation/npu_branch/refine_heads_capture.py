"""Explicit one-forward CPU fixture capture for the refinement/head candidate.

This diagnostic never attaches automatically, changes weights or runs inference.
Context/checkpoint/source identities are supplied by the caller; hooks establish
that four original CPU target modules supplied the recorded arrays during one
ordered predictor invocation. This does not certify the upstream encoder or the
input's origin. No qualification is issued.
"""
import hashlib
import json
from pathlib import Path
import threading

import numpy as np


SCHEMA = 'mvsplat_refine_heads_inputs_v1'
COMPONENTS = ('proj_feature', 'refine_unet', 'to_gaussians', 'to_disparity')
DEFAULT_CHANNELS = (128, 3, 32, 84, 2)  # up, RGB, refinement, raw Gaussian, raw disparity
ADAPTER_OWNERS = ('_mgs_refine_heads_adapter', '_mgs_fused_heads_adapter',
                  '_mgs_corr_depth_adapter')
# Inspect class identities only. Importing runtime/worker here would unnecessarily
# load deployment code and could introduce SDK dependencies into a CPU capture.
PROXY_MODULES = frozenset(('gaussian_generation.npu_branch.runtime',
                           'gaussian_generation.npu_branch.ffn',
                           'gaussian_generation.npu_branch.fused_heads'))


def _identity(value, label):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError(label + ' must be a lowercase SHA256')
    return value


def _array_sha(value):
    """Hash logical C-order bytes without copying a whole channel slice."""
    digest = hashlib.sha256()
    if value.flags.c_contiguous:
        digest.update(memoryview(value).cast('B'))
    else:
        for sample in value:
            if not sample.flags.c_contiguous:
                raise ValueError('Unexpected noncontiguous fixture sample')
            digest.update(memoryview(sample).cast('B'))
    return digest.hexdigest()


class RefineHeadsCapture:
    """Capture a single eval/no-grad call, owning bounded FP32 array copies.

    The default channel contract is the real model's 133-input/86-output pair.
    A smaller explicit ``channels`` tuple exists for local contract tests only;
    its metadata records the different shape and cannot pass a fixed-model ABI.
    """
    def __init__(self, predictor, *, context_sha256, weight_sha256, source_hashes,
                 reference_device='host CPU', channels=DEFAULT_CHANNELS,
                 max_owned_bytes=32 * 1024 * 1024):
        import torch
        if not isinstance(predictor, torch.nn.Module):
            raise ValueError('Expected the original predictor module')
        if (len(channels) != 5 or any(type(n) is not int or n < 1 for n in channels)):
            raise ValueError('Expected five positive integer channel counts')
        self.channels = tuple(channels)
        if type(max_owned_bytes) is not int or max_owned_bytes < 1:
            raise ValueError('A positive capture-owned array byte budget is required')
        self.max_owned_bytes = max_owned_bytes
        self.context_sha256 = _identity(context_sha256, 'context_sha256')
        self.weight_sha256 = _identity(weight_sha256, 'weight_sha256')
        if not isinstance(source_hashes, dict) or not source_hashes:
            raise ValueError('Nonempty original source identities are required')
        self.source_hashes = {}
        for key, value in source_hashes.items():
            if not isinstance(key, str) or not key:
                raise ValueError('Original source identity name is invalid')
            self.source_hashes[key] = _identity(value, 'source hash')
        if reference_device not in ('host CPU', 'board ARM CPU'):
            raise ValueError('Only an explicit CPU reference can be captured')
        self.reference_device = reference_device
        self.predictor = predictor
        self.modules = {name: getattr(predictor, name, None) for name in COMPONENTS}
        if (any(not isinstance(m, torch.nn.Module) for m in self.modules.values())
                or len({id(m) for m in self.modules.values()}) != 4
                or any(m is predictor for m in self.modules.values())):
            raise ValueError('Four distinct original predictor modules are required')
        if self.channels == DEFAULT_CHANNELS:
            proj, refine, gaussian, disparity = (self.modules[name] for name in COMPONENTS)
            def original_head(module, incoming, hidden, outgoing):
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
                    or not original_head(gaussian, 163, 168, 84)
                    or not original_head(disparity, 32, 64, 2)):
                raise ValueError('Default capture requires the original full refinement/head structure')
        if getattr(predictor, 'wo_depth_refine', False) is not False:
            raise ValueError('Depth-refinement ablation cannot supply both raw heads')
        self._check_model()
        self.handles = []
        self.entered = self.closed = self.failed = self.complete = False
        self.active = False
        self.phase = 0
        self.root_calls = 0
        self.thread = None
        self.spatial = None
        self._input = self._expected = self._projected = self._refined = None
        self.peak_logical_owned_bytes = 0
        self._record = None

    def _check_model(self):
        import torch
        if self.predictor.training or any(m.training for m in self.predictor.modules()):
            raise ValueError('Fixture capture requires every module in eval mode')
        if any(p.device.type != 'cpu' or p.dtype != torch.float32
               for p in self.predictor.parameters()):
            raise ValueError('All predictor parameters must remain CPU FP32')
        if any(getattr(self.predictor, name, None) is not m for name, m in self.modules.items()):
            raise ValueError('A captured module was replaced')
        for module in self.predictor.modules():
            if any(hasattr(module, name) for name in ADAPTER_OWNERS):
                raise ValueError('Capture cannot run through an attached prediction adapter')
            if any(base.__module__ in PROXY_MODULES for base in type(module).__mro__):
                raise ValueError('Capture cannot include a deployed NPU Partition or proxy')

    def _bytes(self):
        arrays = (self._input, self._expected, self._projected, self._refined)
        return sum(x.nbytes for x in arrays if x is not None)

    def _allocate(self, name, shape):
        if self._bytes() + int(np.prod(shape)) * 4 > self.max_owned_bytes:
            raise MemoryError('Capture-owned logical array byte budget exceeded')
        setattr(self, name, np.empty(shape, dtype=np.float32))
        self.peak_logical_owned_bytes = max(self.peak_logical_owned_bytes, self._bytes())

    def _tensor(self, value, channels):
        import torch
        if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                or value.dtype != torch.float32 or value.requires_grad or value.ndim != 4
                or tuple(value.shape[:2]) != (2, channels)
                or any(size < 1 for size in value.shape[2:])
                or not torch.isfinite(value).all()):
            raise ValueError('Expected finite two-view CPU FP32 tensors without gradients')
        shape = tuple(value.shape[2:])
        if self.spatial is None:
            self.spatial = shape
        elif shape != self.spatial:
            raise ValueError('Captured spatial shapes changed within one invocation')
        return value.detach().numpy()

    def _step(self, phase):
        if (not self.active or self.phase != phase or self.failed or self.complete
                or threading.get_ident() != self.thread):
            raise ValueError('Refinement capture order, duplication or invocation boundary changed')
        self._check_model()
        self.phase += 1

    def _equal(self, actual, expected, label):
        if not np.array_equal(actual, expected):
            raise ValueError('Same-forward tensor mismatch: ' + label)

    def _root_pre(self, module, args):
        import torch
        if self.root_calls or self.active or self.complete or torch.is_grad_enabled():
            raise ValueError('Exactly one inference/no-grad predictor invocation is allowed')
        self._check_model()
        self.root_calls += 1
        self.active = True
        self.thread = threading.get_ident()

    def _root_post(self, module, args, result):
        if not self.active or self.phase != 8:
            raise ValueError('Predictor skipped a required refinement/head stage')
        self.active = False
        self.complete = True
        self._input.flags.writeable = self._expected.flags.writeable = False
        up, images, refine, gaussian, disparity = self.channels
        fields = dict(up=self._input[:, :up], images=self._input[:, up:up + images],
                      fullres_disps=self._input[:, up + images:up + images + 1],
                      pdf_max=self._input[:, up + images + 1:],
                      raw_gaussians=self._expected[:, :gaussian],
                      raw_disparity_density=self._expected[:, gaussian:])
        self._record = dict(schema=SCHEMA, complete=True, diagnostic_only=True,
            npu_executed=False, production_qualified=False,
            provenance='direct_original_cpu_refinement_head_hooks_one_predictor_invocation',
            reference_scope='four target modules only; upstream encoder execution and input origin are not certified',
            npu_execution_scope='captured original projection/refinement/Gaussian/disparity modules only',
            context_binding='caller-supplied context SHA256; same-forward arrays checked by hooks',
            context_sha256=self.context_sha256, weight_sha256=self.weight_sha256,
            source_hashes=dict(self.source_hashes), reference_device=self.reference_device,
            channels=list(self.channels), fixed_model_channels=self.channels == DEFAULT_CHANNELS,
            logical_layout='NCHW', dtype='float32', context_views=2,
            input_order=['up', 'images', 'fullres_disps', 'pdf_max'],
            output_order=['raw_gaussians', 'raw_disparity_density'],
            input_shape=list(self._input.shape), output_shape=list(self._expected.shape),
            input_sha256=_array_sha(self._input), expected_sha256=_array_sha(self._expected),
            fields={k:dict(shape=list(v.shape), sha256=_array_sha(v), bytes=v.nbytes)
                    for k, v in fields.items()},
            observed_stage_order=list(COMPONENTS), predictor_calls=self.root_calls,
            input_arrays_own_copies=True, all_cross_stage_comparisons_exact=True,
            peak_logical_owned_bytes=self.peak_logical_owned_bytes,
            max_logical_owned_bytes=self.max_owned_bytes,
            retained_logical_bytes=self._bytes(),
            memory_scope='capture-owned numpy arrays only; excludes model, Torch, SDK, allocator and PSS')

    def _pre(self, name, module, args):
        if len(args) != 1:
            raise ValueError('Expected one positional tensor per original component')
        up, images, refine, gaussian, disparity = self.channels
        if name == 'proj_feature':
            self._step(0)
            value = self._tensor(args[0], up)
            self._allocate('_input', (2, up + images + 2, *self.spatial))
            self._input[:, :up] = value
        elif name == 'refine_unet':
            self._step(2)
            value = self._tensor(args[0], images + refine + 2)
            self._equal(value[:, images:images + refine], self._projected, 'projected features')
            self._input[:, up:up + images] = value[:, :images]
            self._input[:, up + images:] = value[:, images + refine:]
            self._projected = None
        elif name == 'to_gaussians':
            self._step(4)
            value = self._tensor(args[0], refine + images + up)
            self._equal(value[:, :refine], self._refined, 'Gaussian refinement prefix')
            self._equal(value[:, refine:refine + images], self._input[:, up:up + images], 'RGB')
            self._equal(value[:, refine + images:], self._input[:, :up], 'full-resolution features')
        else:
            self._step(6)
            value = self._tensor(args[0], refine)
            self._equal(value, self._refined, 'disparity refinement input')
            self._refined = None

    def _post(self, name, module, args, result):
        up, images, refine, gaussian, disparity = self.channels
        if name == 'proj_feature':
            self._step(1)
            value = self._tensor(result, refine)
            self._allocate('_projected', value.shape)
            self._projected[...] = value
        elif name == 'refine_unet':
            self._step(3)
            value = self._tensor(result, refine)
            self._allocate('_refined', value.shape)
            self._refined[...] = value
        elif name == 'to_gaussians':
            self._step(5)
            value = self._tensor(result, gaussian)
            self._allocate('_expected', (2, gaussian + disparity, *self.spatial))
            self._expected[:, :gaussian] = value
        else:
            self._step(7)
            value = self._tensor(result, disparity)
            self._expected[:, gaussian:] = value

    def _guarded(self, callback):
        def guard(*args):
            try:
                return callback(*args)
            except BaseException:
                self.failed = True
                self._remove_hooks()
                self._release()
                raise
        return guard

    def _remove_hooks(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _release(self):
        self._input = self._expected = self._projected = self._refined = None
        self._record = None

    def __enter__(self):
        if self.entered or self.closed:
            raise RuntimeError('A capture is one-shot and cannot be re-entered')
        self.entered = True
        try:
            self._check_model()
            self.handles.append(self.predictor.register_forward_pre_hook(self._guarded(self._root_pre)))
            self.handles.append(self.predictor.register_forward_hook(self._guarded(self._root_post)))
            for name, module in self.modules.items():
                self.handles.append(module.register_forward_pre_hook(self._guarded(
                    lambda m, args, name=name: self._pre(name, m, args))))
                self.handles.append(module.register_forward_hook(self._guarded(
                    lambda m, args, result, name=name: self._post(name, m, args, result))))
        except BaseException:
            self.failed = True
            self._remove_hooks()
            self._release()
            self.closed = True
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._remove_hooks()
        self.closed = True
        if exc_type is not None or self.failed or not self.complete:
            self.failed = True
            self._release()
            if exc_type is None:
                raise ValueError('One complete same-forward capture was not obtained')

    @property
    def record(self):
        if not self.closed or self.failed or self._record is None:
            raise RuntimeError('Capture must finish successfully before reading its record')
        return json.loads(json.dumps(self._record))

    @property
    def arrays(self):
        self.record
        return dict(input=self._input, expected=self._expected)

    def save(self, directory):
        """Write a new diagnostic fixture directory without replacing any file."""
        record = self.record
        directory = Path(directory)
        directory.mkdir(exist_ok=False)
        fixture = directory / 'oracle.npz'
        np.savez(fixture, **self.arrays)
        digest = hashlib.sha256()
        with fixture.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        record.update(fixture='oracle.npz', fixture_sha256=digest.hexdigest())
        (directory / 'capture.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        return record


def capture_refine_heads(predictor, **kwargs):
    """Create an explicit context manager; the caller performs exactly one forward."""
    return RefineHeadsCapture(predictor, **kwargs)
