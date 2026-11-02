"""Keep the locked MVSplat encoder loaded across video jobs."""
import gc
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import sys
import time

from common import FULL_WEIGHT_SHA256, SOURCE_COMMIT, load_gaussians, save, sha, validate_gaussians


def execution_source_hashes(active_vendor):
    """Bind qualification to code actually used, including generated schedules.

    Relative, stable keys keep relocation valid while any upstream patch,
    runtime handoff change or memory-scheduling change invalidates old receipts.
    """
    active_vendor = Path(active_vendor)
    hashes = {f.relative_to(active_vendor).as_posix(): sha(f)
              for f in sorted((active_vendor/'src').rglob('*.py'))}
    root = Path(__file__).resolve().parents[1]
    for relative in ('gaussian_generation/runtime.py', 'memory_runtime.py'):
        hashes['runtime/'+relative] = sha(root/relative)
    return hashes


def persistent_archive_directory(directory):
    """Reject RAM-backed output when disk spooling is requested to save RAM."""
    if not sys.platform.startswith('linux'):
        return  # Local Windows checks use a regular filesystem directory.
    target = Path(directory).resolve()
    candidates = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        left, separator, right = line.partition(' - ')
        if not separator:
            continue
        fields = left.split()
        if len(fields) < 5 or not right.split():
            continue
        mount = Path(re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), fields[4]))
        if target == mount or mount in target.parents:
            candidates.append((len(mount.parts), right.split()[0]))
    if not candidates:
        raise ValueError('Cannot verify persistent Gaussian archive filesystem')
    if max(candidates)[1] in ('tmpfs', 'ramfs', 'hugetlbfs'):
        raise ValueError('Disk Gaussian archive policy requires persistent storage, not a RAM filesystem')


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
        self.source_hashes = execution_source_hashes(active_vendor)
        self.threads, self.depth_chunk = threads, depth_chunk
        self.config = cfg
        self.load_seconds = time.perf_counter() - started
        self.partition_runtime = None
        self._pending_gaussians = None
        self._pending_archive = None

    def take_gaussians(self, expected_sha256):
        pending = self._pending_gaussians
        if pending is None or pending[0] != expected_sha256:
            raise ValueError("No validated in-memory Gaussians for this inference")
        self._pending_gaussians = None
        return pending[1]

    def warm_kernels(self, size=128):
        """Warm operators before video with a scene-independent synthetic pair.

        No checkpoint updates, input video access, Gaussian publication or scene
        reuse. The ordinary inference path must still run for every real video.
        """
        if size < 64 or size % 64:
            raise ValueError('Kernel warmup requires a supported square input size')
        torch = self.torch
        start = time.perf_counter()
        context = dict(image=torch.full((1,2,3,size,size), .5),
            extrinsics=torch.eye(4).repeat(1,2,1,1),
            intrinsics=torch.tensor([[.9,0.,.5],[0.,.9,.5],[0.,0.,1.]]).repeat(1,2,1,1),
            near=torch.ones((1,2)), far=torch.full((1,2), 100.))
        context['extrinsics'][0,1,0,3] = .2
        first = self.partition_runtime.sequence if self.partition_runtime else 0
        with torch.inference_mode():
            g = self.model(context, global_step=300000, deterministic=True)
        result = {key:getattr(g,key)[0].detach().numpy() for key in
                  ('means','covariances','harmonics','opacities')}
        validate_gaussians(result)
        record = dict(seconds_record_only=time.perf_counter()-start, input_shape=[1,2,3,size,size],
            input_kind='constant synthetic pair with fixed canonical baseline; no video data',
            discarded_gaussians=len(result['means']), training_steps=0,
            npu_executed=False, scene_published=False)
        if self.partition_runtime:
            record['partitions'] = self.partition_runtime.report(first)
            record['npu_executed'] = record['partitions']['npu_executed']
            if self.partition_runtime.backend == 'npu' and not record['npu_executed']:
                raise ValueError('Requested NPU kernel warmup did not execute NPU')
        del context, g, result
        gc.collect()
        return record

    def prepare_gaussian_archive(self, out, gaussians):
        """Bind an exact NPZ digest without necessarily retaining its full bytes.

        Disk mode deliberately trades pre-frame filesystem I/O for lower
        resident memory. It is explicit and rejects Linux RAM filesystems;
        memory mode keeps the original first-frame scheduling and fallback.
        """
        if self._pending_archive is not None:
            raise ValueError('Previous Gaussian archive audit is pending')
        policy = os.environ.get('MGS_GAUSSIAN_ARCHIVE', 'memory')
        out = Path(out)
        if policy == 'memory':
            with io.BytesIO() as stream:
                self.np.savez(stream, **gaussians)
                payload = stream.getvalue()
            digest = hashlib.sha256(payload).hexdigest()
            archive_bytes = resident_bytes = len(payload)
        elif policy == 'disk':
            persistent_archive_directory(out)
            payload = out/'.gaussians.pending.npz'
            created = False
            try:
                with payload.open('xb') as stream:
                    created = True
                    self.np.savez(stream, **gaussians)
                digest = sha(payload)
                archive_bytes = payload.stat().st_size
            except BaseException:
                # Only remove the partial file this call created, never an
                # earlier pending artifact or a successfully published scene.
                if created:
                    payload.unlink(missing_ok=True)
                raise
            resident_bytes = 0
        else:
            raise ValueError('Unknown Gaussian archive policy: '+policy)
        self._pending_archive = (out, payload, digest)
        return dict(policy=policy, sha256=digest, archive_bytes=archive_bytes,
                    resident_payload_bytes=resident_bytes)

    def persist_gaussians(self, record, gaussians):
        """Finish the deferred artifact audit after the first frame is available."""
        pending = self._pending_archive
        if pending is None or pending[2] != record["gaussian_sha256"]:
            raise ValueError("No matching deferred Gaussian archive")
        out, payload, digest = pending
        started = time.perf_counter()
        disk_archive = isinstance(payload, Path)
        archive = payload if disk_archive else out/'gaussians.npz'
        if not disk_archive:
            archive.write_bytes(payload)
        if sha(archive) != digest:
            raise ValueError("Deferred Gaussian archive hash mismatch")
        restored = load_gaussians(archive)
        if any(not self.np.array_equal(restored[key], gaussians[key]) for key in restored):
            raise ValueError("Deferred Gaussian archive differs from rendered parameters")
        if disk_archive:
            archive.replace(out/'gaussians.npz')
        self._pending_archive = None
        record["archive_verified"] = True
        record["deferred_archive_seconds"] = time.perf_counter()-started
        save(out/"inference.json", record)

    def attach_partitions(self, partitions):
        if self.partition_runtime is not None:
            raise ValueError("Partitions are already attached")
        partitions.attach(self.model)
        self.partition_runtime = partitions
        gc.collect()

    def infer(self, input_dir, out, context_sha256, defer_archive=False):
        if getattr(self, "_pending_archive", None) is not None:
            raise ValueError("Previous Gaussian archive audit is pending")
        self._pending_gaussians = None
        input_dir, out = Path(input_dir), Path(out)
        if sha(input_dir / "context.npz") != context_sha256:
            raise ValueError("Context hash mismatch")
        out.mkdir(parents=True, exist_ok=False)
        started = time.perf_counter()
        backend = "cpu" if self.partition_runtime is None else self.partition_runtime.backend
        call_start = 0 if self.partition_runtime is None else self.partition_runtime.sequence
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
            validate_gaussians(result)
            if backend == 'npu' and not self.partition_runtime.report(call_start)['npu_executed']:
                raise ValueError('NPU requested but no actual NPU partition executed')
            if defer_archive:
                archive = self.prepare_gaussian_archive(out, result)
                digest = archive['sha256']
                record['gaussian_archive'] = archive
            else:
                self.np.savez(out / "gaussians.npz", **result)
                load_gaussians(out / "gaussians.npz")
                digest = sha(out / "gaussians.npz")
            record.update(complete=True, gaussians=len(result["means"]), sh_degree=4,
                          gaussian_sha256=digest, training_steps=0,
                          archive_verified=not defer_archive)
            self._pending_gaussians = (record["gaussian_sha256"], result)
            del gaussians, context, result
            gc.collect()
        except Exception as exc:
            record["error"] = type(exc).__name__ + ": " + str(exc)
            raise
        finally:
            record["process_seconds"] = time.perf_counter() - started
            record["scope"] = ("Warm inference and validated memory output; archive audit follows first frame"
                if defer_archive else "Warm inference and output save; fixed weights loaded before video")
            record["npu_executed"] = False
            if self.partition_runtime is not None:
                record["partitions"] = self.partition_runtime.report(call_start)
                record["npu_executed"] = record["partitions"]["npu_executed"]
            save(out / "inference.json", record)
        return record
