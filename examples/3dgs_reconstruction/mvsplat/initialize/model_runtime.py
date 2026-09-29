"""Keep the locked MVSplat encoder loaded across video jobs."""
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

from common import FULL_WEIGHT_SHA256, SOURCE_COMMIT, load_gaussians, save, sha


class ModelRuntime:
    def __init__(self, weights, vendor, threads, depth_chunk, runtime_dir):
        weights, vendor, runtime_dir = Path(weights), Path(vendor), Path(runtime_dir)
        if threads < 1 or sha(weights) != FULL_WEIGHT_SHA256:
            raise ValueError("Invalid thread count or MVSplat checkpoint")
        started = time.perf_counter()
        os.environ["WANDB_MODE"] = "disabled"
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        torch.set_num_threads(threads)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        from runtime import load_encoder
        active_vendor = vendor
        if depth_chunk:
            from memory_runtime import prepare_chunked_source
            active_vendor = prepare_chunked_source(vendor, runtime_dir / "runtime_source", depth_chunk)
        EncoderCostVolume, set_cfg = load_encoder(active_vendor)
        cfg = OmegaConf.merge(OmegaConf.load(vendor / "config/model/encoder/costvolume.yaml"),
                              OmegaConf.load(vendor / "config/experiment/re10k.yaml").model.encoder)
        cfg.unimatch_weights_path = None
        set_cfg(OmegaConf.create({"mode": "test", "dataset": {"view_sampler": {"num_context_views": 2}}}))
        model = EncoderCostVolume(cfg).eval()
        checkpoint = torch.load(weights, map_location="cpu", weights_only=True, mmap=True)
        state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        encoder_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
        model.load_state_dict(encoder_state, strict=True)
        self.metadata = dict(parameter_count=sum(p.numel() for p in model.parameters()),
                             checkpoint_encoder_tensor_count=len(encoder_state),
                             checkpoint_non_encoder_keys=[k for k in state if not k.startswith("encoder.")],
                             checkpoint_load="memory mapped; temporary state released before inference",
                             versions={name: importlib.metadata.version(name)
                                       for name in ("torch", "einops", "e3nn", "numpy")})
        del encoder_state, state, checkpoint
        gc.collect()
        self.model, self.torch, self.np = model, torch, np
        self.weights, self.vendor = weights, vendor
        self.weight_hash = sha(weights)
        self.source_hashes = {str(f.relative_to(vendor)): sha(f)
                              for f in sorted((vendor / "src").rglob("*.py"))}
        self.threads, self.depth_chunk = threads, depth_chunk
        self.config = cfg
        self.load_seconds = time.perf_counter() - started
        self.partition_runtime = None

    def attach_partitions(self, partitions):
        if self.partition_runtime is not None:
            raise ValueError("Partitions are already attached")
        partitions.attach(self.model)
        self.partition_runtime = partitions
        gc.collect()

    def infer(self, input_dir, out, context_sha256):
        input_dir, out = Path(input_dir), Path(out)
        if sha(input_dir / "context.npz") != context_sha256:
            raise ValueError("Context hash mismatch")
        out.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        backend = "cpu" if self.partition_runtime is None else self.partition_runtime.backend
        call_start = 0 if self.partition_runtime is None else len(self.partition_runtime.calls)
        record = dict(complete=False, device="cpu" if backend != "npu" else "cpu+npu", backend=backend,
                      machine=platform.machine(), platform=platform.platform(),
                      source_commit=SOURCE_COMMIT, weight_sha256=self.weight_hash,
                      input_sha256=context_sha256, threads=self.threads,
                      depth_chunk=self.depth_chunk, **self.metadata)
        try:
            from omegaconf import OmegaConf
            save(out / "source_files.json", self.source_hashes)
            save(out / "encoder_config.json", OmegaConf.to_container(self.config))
            with self.np.load(input_dir / "context.npz", allow_pickle=False) as data:
                context = {k: self.torch.from_numpy(data[k].copy()).unsqueeze(0) for k in data.files}
            if context["image"].shape[:3] != (1, 2, 3):
                raise ValueError("Expected two RGB context views")
            if self.partition_runtime and list(context["image"].shape) != self.partition_runtime.input_shape:
                raise ValueError("Input resolution differs from compiled partition bundle")
            record["input_shape"] = list(context["image"].shape)
            record["load_and_setup_seconds"] = time.perf_counter() - started
            with (out / "phases.jsonl").open("w", encoding="utf-8") as phase_log:
                hooks = []
                def phase(name):
                    def hook(module, args, output=None):
                        phase_log.write(json.dumps(dict(phase=name, seconds=time.perf_counter()-started)) + "\n")
                        phase_log.flush()
                    return hook
                for name, module in (("backbone", self.model.backbone),
                                     ("depth", self.model.depth_predictor),
                                     ("gaussians", self.model.gaussian_adapter)):
                    hooks.append(module.register_forward_pre_hook(phase(name + "_begin")))
                    hooks.append(module.register_forward_hook(phase(name + "_end")))
                t = time.perf_counter()
                try:
                    with self.torch.inference_mode():
                        gaussians = self.model(context, global_step=300000, deterministic=True)
                finally:
                    for hook in hooks:
                        hook.remove()
                record["inference_seconds"] = time.perf_counter() - t
            result = {key: getattr(gaussians, key)[0].detach().cpu().numpy() for key in
                      ("means", "covariances", "harmonics", "opacities")}
            self.np.savez(out / "gaussians.npz", **result)
            load_gaussians(out / "gaussians.npz")
            record.update(complete=True, gaussians=len(result["means"]), sh_degree=4,
                          gaussian_sha256=sha(out / "gaussians.npz"), training_steps=0)
            del gaussians, context, result
            gc.collect()
        except Exception as exc:
            record["error"] = type(exc).__name__ + ": " + str(exc)
            raise
        finally:
            record["process_seconds"] = time.perf_counter() - started
            record["scope"] = "Warm inference and output save; fixed weights loaded before video"
            record["npu_executed"] = False
            if self.partition_runtime is not None:
                record["partitions"] = self.partition_runtime.report(call_start)
                record["npu_executed"] = record["partitions"]["npu_executed"]
            save(out / "inference.json", record)
        return record
