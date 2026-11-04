"""Opt-in seam for one unchanged correlation/residual/depth-logit graph.

Importing this module patches no source, model or runtime. ``patch_source``
returns a reviewed vendor copy for a future caller to save and hash. The
adapter owns only that explicit method; it never stubs the three CPU modules
or caches a result for a later call. This is preparation, not qualification.
"""
import ast
import hashlib
import inspect
import types
import weakref

import numpy as np


SCHEMA = 'mvsplat_corr_depth_seam_v1'
METHOD = '_corr_to_depth_logits'
MARKER = '_mgs_corr_depth_seam'
OWNER = '_mgs_corr_depth_adapter'
SEAM_DOC = 'Unchanged MVSplat correlation refinement, residual addition and depth logits.'

# Match the original full statement sequence, including the ablation branch.
# A changed upstream order or operation must be reviewed rather than guessed.
ORIGINAL_BLOCK = '''        # refine cost volume via 2D u-net
        if self.wo_cost_volume_refine:
            raw_correlation = self.corr_project(raw_correlation_in)
        else:
            raw_correlation = self.corr_refine_net(raw_correlation_in)  # (vb d h w)
            # apply skip connection
            raw_correlation = raw_correlation + self.regressor_residual(
                raw_correlation_in
            )

        # softmax to get coarse depth and density
        pdf = F.softmax(
            self.depth_head_lowres(raw_correlation), dim=1
        )  # [2xB, D, H, W]
'''

REPLACEMENT_BLOCK = '''        # refine cost volume, residual and depth logits through one explicit seam
        depth_logits = self._corr_to_depth_logits(raw_correlation_in)

        # softmax to get coarse depth and density
        pdf = F.softmax(
            depth_logits, dim=1
        )  # [2xB, D, H, W]
'''

METHOD_SOURCE = '''    _mgs_corr_depth_seam = 'mvsplat_corr_depth_seam_v1'

    def _corr_to_depth_logits(self, raw_correlation_in):
        """Unchanged MVSplat correlation refinement, residual addition and depth logits."""
        if self.wo_cost_volume_refine:
            raw_correlation = self.corr_project(raw_correlation_in)
        else:
            raw_correlation = self.corr_refine_net(raw_correlation_in)
            raw_correlation = raw_correlation + self.regressor_residual(
                raw_correlation_in
            )
        return self.depth_head_lowres(raw_correlation)

'''


def _sha(source):
    return hashlib.sha256(source.encode('utf-8')).hexdigest()


def patch_source(source):
    """Return a strictly matched source copy and its identity record.

    Both the regular and ``wo_cost_volume_refine`` CPU paths retain their
    exact operations. The NPU adapter deliberately rejects that ablation.
    No file is written and the original module parameters are not renamed.
    """
    if not isinstance(source, str):
        raise TypeError('Expected original DepthPredictorMultiView source text')
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ValueError('Invalid original predictor source') from error
    classes = [node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == 'DepthPredictorMultiView']
    if len(classes) != 1:
        raise ValueError('Expected exactly one top-level DepthPredictorMultiView class')
    cls = classes[0]
    forwards = [node for node in cls.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'forward']
    if (len(forwards) != 1 or isinstance(forwards[0], ast.AsyncFunctionDef) or
            forwards[0].decorator_list):
        raise ValueError('Expected one undecorated predictor forward method')
    if any(token in source for token in (METHOD, MARKER, OWNER)):
        raise ValueError('Predictor source already contains a correlation-depth seam')
    if '\r' in source:
        raise ValueError('Normalize predictor source to LF before exact source matching')
    if source.count(ORIGINAL_BLOCK) != 1:
        raise ValueError('Original correlation/residual/depth block changed; review patch')
    forward = forwards[0]
    lines = source.splitlines(keepends=True)
    begin = sum(map(len, lines[:forward.lineno - 1]))
    end = sum(map(len, lines[:forward.end_lineno]))
    position = source.index(ORIGINAL_BLOCK)
    if not begin < position < position + len(ORIGINAL_BLOCK) <= end:
        raise ValueError('Original correlation-depth block is outside predictor forward')
    first_line = source.count('\n', 0, position) + 1
    last_line = first_line + ORIGINAL_BLOCK.count('\n') - 1
    statements = [node for node in forward.body
                  if first_line <= node.lineno and node.end_lineno <= last_line]
    reference = ast.parse('def original(self):\n' + ORIGINAL_BLOCK).body[0].body
    if ([ast.dump(node, include_attributes=False) for node in statements] !=
            [ast.dump(node, include_attributes=False) for node in reference]):
        raise ValueError('Original block must be executable top-level forward statements')
    for node in ast.walk(forward):
        if isinstance(node, ast.Name) and (node.id == 'depth_logits' or
                (node.id == 'raw_correlation' and not first_line <= node.lineno <= last_line)):
            raise ValueError('Moved correlation temporary or new depth-logit name escapes the reviewed block')
    # Offsets derive from the untouched source: add the method at the beginning
    # of forward only after replacing the later, fully matched block.
    patched = source[:position] + REPLACEMENT_BLOCK + source[position + len(ORIGINAL_BLOCK):]
    patched = patched[:begin] + METHOD_SOURCE + patched[begin:]
    try:
        ast.parse(patched)
    except SyntaxError as error:
        raise ValueError('Generated correlation-depth source is invalid') from error
    metadata = dict(schema=SCHEMA, original_sha256=_sha(source),
        modified_sha256=_sha(patched), target='DepthPredictorMultiView.forward',
        method=METHOD, matched_blocks=1, source_written=False,
        operations=['corr_refine_net', 'regressor_residual', 'Add', 'depth_head_lowres'],
        ablation_cpu_path_preserved=True, ablation_npu_supported=False,
        softmax_remains_in_forward=True, parameter_names_changed=False,
        default_runtime_changed=False, npu_executed=False)
    return patched, metadata


class CorrDepthAdapter:
    """Own one explicit, synchronous CPU-FP32-to-NPU boundary until close.

    ``execute`` accepts a full two-view NCHW float32 NumPy array and returns
    complete depth logits in the same logical layout. The worker, not this
    adapter, must separately validate its static compiled shape and identity.
    No tensor or output from one invocation is reused by another invocation.
    """
    def __init__(self, predictor, execute):
        import torch
        if not isinstance(predictor, torch.nn.Module) or not callable(execute):
            raise ValueError('Expected predictor module and callable executor')
        if hasattr(predictor, OWNER):
            raise ValueError('Correlation-depth adapter is already attached or its owner slot is occupied')
        original = getattr(predictor, METHOD, None)
        if (getattr(type(predictor), MARKER, None) != SCHEMA or
                not inspect.ismethod(original) or original.__self__ is not predictor or
                original.__func__.__doc__ != SEAM_DOC or
                len(inspect.signature(original).parameters) != 1):
            raise ValueError('Predictor lacks the generated correlation-depth seam')
        if getattr(predictor, 'wo_cost_volume_refine', None) is not False:
            raise ValueError('Correlation-depth NPU ablation candidate is unsupported')
        self.modules = {name: getattr(predictor, name, None) for name in
                        ('corr_refine_net', 'regressor_residual', 'depth_head_lowres')}
        residual, head = self.modules['regressor_residual'], self.modules['depth_head_lowres']
        if (not all(isinstance(module, torch.nn.Module) for module in self.modules.values()) or
                not isinstance(residual, torch.nn.Conv2d) or
                not isinstance(head, torch.nn.Sequential) or len(head) != 3 or
                not isinstance(head[0], torch.nn.Conv2d) or
                not isinstance(head[1], torch.nn.GELU) or
                not isinstance(head[2], torch.nn.Conv2d) or
                residual.out_channels != head[0].in_channels or
                head[0].out_channels != head[2].in_channels or
                head[2].out_channels != residual.out_channels):
            raise ValueError('Original correlation-depth module/channel contract differs')
        if any(parameter.device.type != 'cpu' or parameter.dtype != torch.float32
               for module in self.modules.values() for parameter in module.parameters()):
            raise ValueError('Correlation-depth reference parameters must remain CPU FP32')
        self.predictor, self.execute = predictor, execute
        self.input_channels = residual.in_channels
        self.output_channels = head[2].out_channels
        self.original = original
        self.had_instance_method = METHOD in predictor.__dict__
        self.instance_method = predictor.__dict__.get(METHOD)
        self.closed = self.failed = self.active = False
        self._check_eval()
        owner = self

        def proxy(bound_predictor, value):
            if bound_predictor is not owner.predictor:
                raise ValueError('Correlation-depth seam was rebound to another predictor')
            return owner._forward(value)

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
            raise ValueError('Correlation-depth adapter is inference/eval-only')

    def _forward(self, value):
        import torch
        if self.closed or self.failed or self.active:
            raise RuntimeError('Correlation-depth adapter is closed, failed or already active')
        self._check_eval()
        if (getattr(self.predictor, 'wo_cost_volume_refine', None) is not False or
                any(getattr(self.predictor, name, None) is not module
                    for name, module in self.modules.items())):
            raise ValueError('Attached correlation-depth model contract changed')
        if (not isinstance(value, torch.Tensor) or value.device.type != 'cpu' or
                value.dtype != torch.float32 or value.requires_grad or value.ndim != 4 or
                value.shape[0] != 2 or value.shape[1] != self.input_channels or
                any(size < 1 for size in value.shape[2:]) or not torch.isfinite(value).all()):
            raise ValueError('Expected finite two-view CPU FP32 correlation input without gradients')
        array = np.ascontiguousarray(value.detach().numpy())
        expected_shape = (value.shape[0], self.output_channels, *value.shape[2:])
        self.active = True
        try:
            result = self.execute(array)
            if (not isinstance(result, np.ndarray) or result.dtype != np.float32 or
                    result.shape != expected_shape or not np.isfinite(result).all()):
                raise ValueError('Invalid correlation-depth FP32 NCHW output shape or finite values')
            # The returned Torch value owns this invocation's completed result,
            # even if a future worker call reuses its shared-memory output slot.
            return torch.from_numpy(np.array(result, dtype=np.float32, order='C', copy=True))
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
            raise RuntimeError('Cannot detach an active correlation-depth invocation')
        self._restore()
        self.closed = True

    def __enter__(self):
        if self.closed:
            raise RuntimeError('Correlation-depth adapter is closed')
        return self

    def __exit__(self, *args):
        self.close()


def attach_corr_depth(predictor, execute):
    """Attach explicitly; caller must close or use the returned context manager.

    The executor must not call the replaced method recursively or treat the
    unchanged CPU submodules as a production fallback. Graph correctness,
    input switching, SDK execution costs and image qualification remain separate.
    """
    return CorrDepthAdapter(predictor, execute)
