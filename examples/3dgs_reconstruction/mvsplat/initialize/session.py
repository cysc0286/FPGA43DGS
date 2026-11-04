"""Pre-video resources only; construction must complete before publishing READY."""
from contextlib import ExitStack
import gc
import os
from pathlib import Path
import time
from common import save


def render_environment():
    env = dict(os.environ)
    env.pop("LD_PRELOAD", None)
    env.pop("PYTHONPATH", None)
    env["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    env["LD_LIBRARY_PATH"] = "/root/heterogs_npu/sdk_3.36.1/usr/lib/aarch64-linux-gnu"
    return env


def worker_environment():
    env = dict(os.environ)
    env.pop("LD_PRELOAD", None)
    env["LD_LIBRARY_PATH"] = ("/root/heterogs_npu/sdk_3.36.1/usr/lib/aarch64-linux-gnu:"
                              "/root/fpga43dgs_reconstruction/arm_env/lib")
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    # The SDK allocator is process-local, while PL DDR is shared with the
    # renderer. Reserve its low workspace before loading NPU weights.
    env.setdefault("MGS_NPU_DDR_GUARD_MIB", "64")
    # The measured backbone SDK path can return the previous input's output.
    # Keep the qualified two-execution workaround in the production entry;
    # readiness still verifies input switches and binds this runtime identity.
    env.setdefault("MGS_NPU_SETTLE_BACKBONE", "1")
    return env


def protect_renderer_workspace(renderer, backend):
    """Bound both renderer DMA slots below the NPU's reserved DDR boundary."""
    if backend == 'npu':
        guard = int(worker_environment()['MGS_NPU_DDR_GUARD_MIB'])
        if not 1 <= guard <= 512:
            raise ValueError('Concurrent NPU/FPGA requires a positive DDR reservation')
        renderer.runtime.pl_ddr_limit_bytes = guard * 1024 * 1024


class WarmSession:
    def __init__(self, args, out):
        self.resources = ExitStack()
        started = time.monotonic()
        try:
            self.partitions = None
            # Device::Open resets shared device state. Open the renderer before
            # the retained NPU session, and close it last (ExitStack is LIFO).
            self.renderer = self.resources.enter_context(self._open_renderer(args, out))
            protect_renderer_workspace(self.renderer, args.backend)
            # NPU checks are in a separate SDK process. Import/load Torch before
            # OpenCV for the board's OpenMP environment; reject bad hardware
            # outputs before spending time and memory loading the full model.
            from gaussian_generation.npu_branch.config import GenerationSession
            generation = self.resources.enter_context(GenerationSession(args, out, threads=args.threads))
            self.model, self.partitions = generation.model, generation.partitions
            numerical_readiness = generation.readiness
            # Imports and package/device initialization are outside the video clock.
            from initialize import scene_preparation
            from gaussian_generation import adapter as export
            kernel_warmup = None
            if getattr(args, 'warm_kernels', False):
                kernel_warmup = self.model.warm_kernels(getattr(args, 'size', 128))
                save(out/'kernel_warmup.json', kernel_warmup)
                if self.partitions and args.backend == 'npu':
                    from gaussian_generation.npu_branch.readiness import validate_oracles
                    c = generation.config
                    numerical_readiness = validate_oracles(self.partitions,
                        c.oracle_graphs or c.partition_bundle.parent/'graphs',
                        out/'npu_readiness_after_warmup.json', acceptance=c.npu_acceptance)
                    self.partitions.validation = numerical_readiness
            gc.collect()
            self.record = dict(preheat_seconds_record_only=time.monotonic()-started,
                model_load_seconds_record_only=self.model.load_seconds, backend=args.backend,
                threads=dict(inference=args.threads, preparation=args.prepare_threads),
                warm_scope="weights, Python dependencies, selected compiled sessions, FPGA device/runtime",
                kernel_first_use_warmed=kernel_warmup is not None, kernel_warmup=kernel_warmup,
                scene_cache="Gaussian rows loaded once; projection remains per view",
                rss_mib_at_ready=self._rss())
            if getattr(args, "live_renderer", None):
                self.record["renderer_configuration"] = dict(self.renderer.runtime.configuration)
                self.record["scene_cache"] = "Gaussian rows and world covariance/opacity cached; per-view projection"
            if self.partitions:
                self.record["partitions"] = self.partitions.report()
                self.record["numerical_readiness"] = numerical_readiness
        except BaseException as exc:
            save(out/"initialize_failure.json", dict(
                error=type(exc).__name__+": "+str(exc), public_ready=False,
                preheat_seconds_record_only=time.monotonic()-started,
                partitions=self.partitions.report() if self.partitions is not None else None))
            self.resources.close()
            raise

    @staticmethod
    def _open_renderer(args, out):
        if getattr(args, "live_renderer", None):
            from rendering.pipeline_adapter import PipelineRenderer
            configuration = {key: getattr(args, "render_" + key, None)
                             for key in ("threads", "max_gaussians", "batch", "uniform_preview")}
            return PipelineRenderer(args.renderer, args.live_renderer, render_environment(),
                out / "renderer.log", profile=getattr(args, "render_profile", "mainline"),
                **{key: value for key, value in configuration.items() if value is not None})
        from rendering.resident_runtime import RendererRuntime
        return RendererRuntime(args.renderer, args.resident_binary, render_environment(), out / "renderer.log")

    @staticmethod
    def _rss():
        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
        except OSError:
            pass
        return None

    def close(self):
        self.resources.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
