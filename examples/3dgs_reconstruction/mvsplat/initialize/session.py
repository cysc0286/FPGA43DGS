"""Pre-video resources only; construction must complete before publishing READY."""
from contextlib import ExitStack
import gc
import os
from pathlib import Path
import time


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
    return env


class WarmSession:
    def __init__(self, args, out):
        self.resources = ExitStack()
        started = time.monotonic()
        try:
            # Import/load Torch before OpenCV for the board's OpenMP environment.
            from initialize.model_runtime import ModelRuntime
            self.model = ModelRuntime(args.weights, args.vendor, args.threads, 16, out / "initialize")
            self.partitions = None
            if args.backend != "cpu":
                from npu.runtime import PartitionRuntime
                self.partitions = self.resources.enter_context(PartitionRuntime(
                    args.backend, args.partition_bundle, args.partitions, out / "partition_runtime",
                    library=args.npu_library, worker_python=args.worker_python,
                    environment=worker_environment() if args.backend == "npu" else None))
                self.model.attach_partitions(self.partitions)
            # Imports and package/device initialization are outside the video clock.
            import video_input.prepare
            import export
            from initialize.renderer_runtime import RendererRuntime
            self.renderer = self.resources.enter_context(RendererRuntime(args.renderer,
                args.resident_binary, render_environment(), out / "renderer.log"))
            gc.collect()
            self.record = dict(preheat_seconds_record_only=time.monotonic()-started,
                model_load_seconds_record_only=self.model.load_seconds, backend=args.backend,
                threads=dict(inference=args.threads, preparation=args.prepare_threads),
                warm_scope="weights, Python dependencies, selected compiled sessions, FPGA device/runtime",
                kernel_first_use_warmed=False, scene_cache="Gaussian rows loaded once; projection remains per view",
                rss_mib_at_ready=self._rss())
            if self.partitions:
                self.record["partitions"] = self.partitions.report()
        except BaseException:
            self.resources.close()
            raise

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
