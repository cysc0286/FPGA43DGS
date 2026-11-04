"""One backend contract for file, button-camera and continuous-video jobs."""
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from gaussian_generation.npu_branch.catalog import PARTITIONS, validate_selection


def add_arguments(parser, *, default_backend='cpu'):
    parser.add_argument('--backend', choices=('cpu', 'onnx_reference', 'npu'),
                        default=default_backend)
    parser.add_argument('--partition-bundle', type=Path)
    parser.add_argument('--partitions', nargs='+', choices=list(PARTITIONS))
    parser.add_argument('--npu-library', type=Path)
    parser.add_argument('--oracle-graphs', type=Path)
    parser.add_argument('--worker-python', type=Path)
    parser.add_argument('--buffer-policy', choices=('shared', 'per_partition'), default='shared')
    parser.add_argument('--buffer-limit-mib', type=int, default=128)
    parser.add_argument('--npu-acceptance', type=Path,
                        help='Native FPGA image qualification receipt; otherwise strict tensor gate')


@dataclass(frozen=True)
class BackendConfig:
    backend: str = 'cpu'
    partition_bundle: Path | None = None
    partitions: tuple = ()
    npu_library: Path | None = None
    oracle_graphs: Path | None = None
    worker_python: Path | None = None
    buffer_policy: str = 'shared'
    buffer_limit_mib: int = 128
    npu_acceptance: Path | None = None

    @classmethod
    def from_args(cls, args):
        values = {name: getattr(args, name, field.default)
                  for name, field in cls.__dataclass_fields__.items()}
        values['partitions'] = tuple(values['partitions'] or ())
        for name in ('partition_bundle', 'npu_library', 'oracle_graphs', 'worker_python', 'npu_acceptance'):
            if values[name] is not None:
                values[name] = Path(values[name]).resolve()
        return cls(**values)

    def __post_init__(self):
        if self.backend not in ('cpu', 'onnx_reference', 'npu'):
            raise ValueError('Unknown generation backend')
        if self.buffer_policy not in ('shared', 'per_partition') or self.buffer_limit_mib <= 0:
            raise ValueError('Invalid NPU buffer policy/budget')
        if self.backend == 'cpu':
            if self.partitions or self.partition_bundle or self.npu_library or self.npu_acceptance or self.oracle_graphs:
                raise ValueError('CPU backend cannot silently ignore NPU configuration')
            return
        if not self.partition_bundle:
            raise ValueError('Non-CPU backend needs a bundle and unique explicit partitions')
        validate_selection(self.partitions)
        if self.backend == 'npu':
            if not self.npu_library:
                raise ValueError('NPU requires --npu-library')
            # Session count is not a correctness policy. Every selected session
            # must pass interleaved oracle checks, or an artifact-bound native
            # image receipt for this exact ordered set, before public READY.
        elif self.npu_acceptance:
            raise ValueError('Image qualification is only valid for actual NPU execution')

    def cli(self):
        """Forward every option unchanged to the reconstruction child, never viewer restore."""
        result = ['--backend', self.backend]
        if self.backend == 'cpu':
            return result
        for name in ('partition_bundle', 'npu_library', 'oracle_graphs', 'worker_python', 'npu_acceptance'):
            value = getattr(self, name)
            if value is not None:
                result += ['--' + name.replace('_', '-'), str(value)]
        return result + ['--buffer-policy', self.buffer_policy, '--buffer-limit-mib',
                         str(self.buffer_limit_mib), '--partitions', *self.partitions]


class GenerationSession:
    """Own the worker through readiness, model load, inference and orderly shutdown.

    Failure is explicit: never label CPU fallback as NPU, and never publish READY
    if an attached NPU fails its selected validation policy.
    """
    def __init__(self, args, out, *, threads, depth_chunk=16):
        from common import save
        self.resources = ExitStack()
        self.config = BackendConfig.from_args(args)
        self.partitions = None
        self.readiness = None
        out = Path(out)
        try:
            continuous_depth = 'corr_depth_pipeline' in self.config.partitions
            continuous_refinement = 'refine_heads_pipeline' in self.config.partitions
            if (continuous_depth or continuous_refinement) and (type(depth_chunk) is not int or not 1 <= depth_chunk <= 128):
                raise ValueError('Continuous depth requires depth_chunk in 1..128 before worker startup')
            if continuous_refinement:
                from gaussian_generation.runtime import validate_refine_heads_config
                validate_refine_heads_config(args.vendor)
            if self.config.npu_acceptance:
                if getattr(args, 'render_profile', 'mainline') != 'mainline' or not getattr(args, 'live_renderer', None):
                    raise ValueError('Image-qualified NPU requires the accepted native mainline renderer')
                from gaussian_generation.npu_branch.acceptance import validate_renderer
                validate_renderer(self.config.npu_acceptance, args.renderer, args.live_renderer)
            if self.config.backend != 'cpu':
                from gaussian_generation.npu_branch.runtime import PartitionRuntime
                from initialize.session import worker_environment
                c = self.config
                self.partitions = self.resources.enter_context(PartitionRuntime(
                    c.backend, c.partition_bundle, c.partitions, out / 'partition_runtime',
                    library=c.npu_library, worker_python=c.worker_python,
                    environment=worker_environment() if c.backend == 'npu' else None,
                    buffer_policy=c.buffer_policy, buffer_limit_mib=c.buffer_limit_mib))
                if c.backend == 'npu':
                    from gaussian_generation.npu_branch.readiness import validate_oracles
                    graphs = c.oracle_graphs or c.partition_bundle.parent / 'graphs'
                    self.readiness = validate_oracles(self.partitions, graphs, out / 'npu_readiness.json',
                                                     acceptance=c.npu_acceptance)
                    self.partitions.validation = self.readiness
            from gaussian_generation.runtime import ModelRuntime
            model_options = {}
            if continuous_depth:
                model_options['corr_depth_pipeline'] = True
            if continuous_refinement:
                model_options['refine_heads_pipeline'] = True
            self.model = ModelRuntime(args.weights, args.vendor, threads, depth_chunk,
                                      out / 'initialize', **model_options)
            if self.config.npu_acceptance:
                from gaussian_generation.npu_branch.acceptance import validate_model
                validate_model(self.config.npu_acceptance, self.model)
            if self.partitions is not None:
                self.model.attach_partitions(self.partitions)
        except BaseException as exc:
            save(out / 'generation_failure.json', dict(backend=self.config.backend, public_ready=False,
                 error=type(exc).__name__ + ': ' + str(exc)))
            self.close()
            raise

    def close(self):
        self.resources.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
