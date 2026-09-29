"""Run the locked, full pretrained MVSplat encoder without the CUDA rasterizer."""
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time

from common import FULL_WEIGHT_SHA256, SOURCE_COMMIT, load_gaussians, new_directory, save, sha


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--vendor", type=Path, default=Path(__file__).resolve().parents[1] / "vendor/MVSplat_reference")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--depth-chunk", type=int, default=0)
    a = p.parse_args(argv)
    if a.threads < 1:
        p.error("threads must be positive")
    if sha(a.weights) != FULL_WEIGHT_SHA256:
        raise ValueError("Expected the locked official re10k full checkpoint")
    input_meta = json.loads((a.input / "input.json").read_text(encoding="utf-8"))
    if sha(a.input / "context.npz") != input_meta["context_sha256"]:
        raise ValueError("Input hash mismatch")
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(a.threads)
    out = new_directory(a.out)
    record = dict(complete=False, device="cpu", machine=platform.machine(), platform=platform.platform(),
                  source_commit=SOURCE_COMMIT, weight_sha256=sha(a.weights),
                  input_sha256=sha(a.input / "context.npz"), threads=a.threads)
    peak = [0]
    stop = threading.Event()
    import psutil
    process = psutil.Process()
    def monitor():
        while not stop.is_set():
            peak[0] = max(peak[0], process.memory_info().rss)
            stop.wait(.02)
    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    started = time.perf_counter()
    try:
        import numpy as np
        import torch
        from omegaconf import OmegaConf
        torch.set_num_threads(a.threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(0)
        from runtime import load_encoder
        active_vendor = a.vendor
        if a.depth_chunk:
            from memory_runtime import prepare_chunked_source
            active_vendor = prepare_chunked_source(a.vendor, out/"runtime_source", a.depth_chunk)
        record["depth_chunk"] = a.depth_chunk
        EncoderCostVolume, set_cfg = load_encoder(active_vendor)
        save(out / "source_files.json", {str(f.relative_to(a.vendor)): sha(f)
             for f in sorted((a.vendor / "src").rglob("*.py"))})
        cfg = OmegaConf.merge(OmegaConf.load(a.vendor / "config/model/encoder/costvolume.yaml"),
                              OmegaConf.load(a.vendor / "config/experiment/re10k.yaml").model.encoder)
        cfg.unimatch_weights_path = None
        set_cfg(OmegaConf.create({"mode": "test", "dataset": {"view_sampler": {"num_context_views": 2}}}))
        model = EncoderCostVolume(cfg).eval()
        checkpoint = torch.load(a.weights, map_location="cpu", weights_only=True, mmap=True)
        state = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        encoder_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
        model.load_state_dict(encoder_state, strict=True)
        record["parameter_count"] = sum(p.numel() for p in model.parameters())
        record["checkpoint_encoder_tensor_count"] = len(encoder_state)
        record["checkpoint_non_encoder_keys"] = [k for k in state if not k.startswith("encoder.")]
        del encoder_state, state, checkpoint
        gc.collect()
        record["checkpoint_load"] = "memory mapped; temporary state released before inference"
        record["versions"] = {name: importlib.metadata.version(name) for name in ("torch", "einops", "e3nn", "numpy")}
        save(out / "encoder_config.json", OmegaConf.to_container(cfg))
        with np.load(a.input / "context.npz", allow_pickle=False) as data:
            context = {k: torch.from_numpy(data[k].copy()).unsqueeze(0) for k in data.files}
        if context["image"].shape[:3] != (1, 2, 3):
            raise ValueError("Expected two RGB context views")
        record["input_shape"] = list(context["image"].shape)
        record["load_and_setup_seconds"] = time.perf_counter()-started
        print("Loaded checkpoint strictly; beginning CPU inference", flush=True)
        phase_log = (out/"phases.jsonl").open("w", encoding="utf-8")
        def phase(name):
            def hook(module, args, output=None):
                phase_log.write(json.dumps(dict(phase=name, seconds=time.perf_counter()-started,
                                               rss_mib=process.memory_info().rss/1024**2))+"\n")
                phase_log.flush()
            return hook
        for name, module in (("backbone", model.backbone), ("depth", model.depth_predictor),
                             ("gaussians", model.gaussian_adapter)):
            module.register_forward_pre_hook(phase(name+"_begin"))
            module.register_forward_hook(phase(name+"_end"))
        t = time.perf_counter()
        cpu_begin = process.cpu_times()
        with torch.inference_mode():
            gaussians = model(context, global_step=300000, deterministic=True)
        phase_log.close()
        record["inference_seconds"] = time.perf_counter()-t
        cpu_end = process.cpu_times()
        record["inference_cpu_seconds"] = cpu_end.user+cpu_end.system-cpu_begin.user-cpu_begin.system
        result = {key: getattr(gaussians, key)[0].detach().cpu().numpy() for key in
                  ("means", "covariances", "harmonics", "opacities")}
        np.savez(out / "gaussians.npz", **result)
        load_gaussians(out / "gaussians.npz")
        record.update(complete=True, gaussians=len(result["means"]), sh_degree=4,
                      gaussian_sha256=sha(out / "gaussians.npz"), training_steps=0)
    except Exception as exc:
        record["error"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        stop.set()
        monitor_thread.join()
        record["peak_process_rss_mib_sampled"] = peak[0]/1024**2
        record["process_seconds"] = time.perf_counter()-started
        record["scope"] = "Single cold CPU inference including output save separately from load; no NPU, GPU or per-scene training"
        save(out / "inference.json", record)
        print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
