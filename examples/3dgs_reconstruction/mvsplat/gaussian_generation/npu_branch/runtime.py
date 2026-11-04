"""Resident hybrid inference; all unselected MVSplat modules explicitly stay CPU."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import numpy as np
from collections import deque

from common import save, sha
from gaussian_generation.npu_branch.artifacts import from_wire, to_wire, validate_bundle
from gaussian_generation.npu_branch.catalog import (
    PARTITIONS, CORR_DEPTH_COMPONENTS, REFINE_HEADS_COMPONENTS, replace, validate_selection,
)
from gaussian_generation.npu_branch.buffers import buffer_plan, MappedBuffers
from gaussian_generation.npu_branch.protocol import PROTOCOL_VERSION
from gaussian_generation.npu_branch.spatial import execute_spatial


class PartitionRuntime:
    def __init__(self, backend, bundle, names, work, library=None,
                 worker_python=None, environment=None, timeout=90,
                 buffer_policy="shared", buffer_limit_mib=128):
        if backend not in ("onnx_reference", "npu"):
            raise ValueError("Expected an explicit accelerated or host validation backend")
        names = validate_selection(names)
        data, self.parts = validate_bundle(bundle, names, backend)
        if backend == 'npu' and any(p['meta'].get('spatial_tiling') or p['meta'].get('ffn_adapter')
                                    or name in ('corr_refine_net', 'refine_unet', 'corr_depth_pipeline', 'refine_heads_pipeline')
                                    for name, p in self.parts.items()):
            # Probes exposed stale results in existing large graphs/tiles.
            # New continuous candidates inherit that conservative policy; it
            # does not establish that two passes are necessary or sufficient
            # for a graph that still needs its own board qualification.
            environment = dict(os.environ if environment is None else environment)
            environment.setdefault('MGS_NPU_FRESH_INPUT', '1')
            if environment['MGS_NPU_FRESH_INPUT'] != '1':
                raise ValueError('Selected NPU graph requires fresh SDK input tensors')
        plan = buffer_plan(self.parts, buffer_policy)
        limit_bytes = int(buffer_limit_mib*1024**2)
        if limit_bytes <= 0 or plan["allocated_bytes"] > limit_bytes:
            raise ValueError("IPC buffer allocation exceeds configured byte budget")
        if backend == "npu" and (library is None or not Path(library).is_file()):
            raise ValueError("NPU backend requires a built ARM libmgs_npu.so")
        self.input_shape = data["input_shape"]
        self.backend, self.timeout = backend, timeout
        self.work = Path(work).resolve()
        self.work.mkdir(parents=True, exist_ok=False)
        self.buffers, self.calls, self.sequence = {}, deque(maxlen=256), 0
        self.arena = None
        self.lock = threading.Lock()
        self.responses = queue.Queue()
        self.process = None
        self.reader = None
        self.closed = False
        self.model_adapters = []
        self.model_replacements = []
        self.attached_model = None
        self.log = None
        self.failed = False
        request = dict(backend=backend, bundle=str(Path(bundle).resolve()), names=list(names),
                       library=None if library is None else str(Path(library).resolve()),
                       protocol_version=PROTOCOL_VERSION, buffer_plan=plan, buffer_limit_bytes=limit_bytes)
        try:
            self.arena = MappedBuffers(self.work, self.parts, plan, create=True, limit_bytes=limit_bytes)
            self.buffers = self.arena.views
            save(self.work / "request.json", request)
            self.log = (self.work / "worker.log").open("wb")
            self.process = subprocess.Popen([str(worker_python or sys.executable), "-m", "gaussian_generation.npu_branch.worker",
                str(self.work / "request.json")], cwd=Path(__file__).resolve().parents[2], env=environment,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1)
            self.reader = threading.Thread(target=self._read, daemon=True)
            self.reader.start()
            response = self._response()
            if (response.get("event") != "READY" or response.get("backend") != backend or
                    response.get("protocol_version") != PROTOCOL_VERSION or
                    response.get("buffer_plan") != plan):
                raise ValueError("Worker readiness mismatch")
            if backend == "npu" and any(response["bindings"].get(name, 0) <= 0 for name in names):
                raise ValueError("Missing runtime NPU compute binding")
            self.readiness = response
            self.provenance = dict(bundle_sha256=sha(Path(bundle) / "manifest.json"),
                oracle_manifest_sha256=data.get("source_manifest_sha256", sha(Path(bundle)/"manifest.json")),
                bridge_sha256=sha(library) if library is not None else None,
                buffer_policy=buffer_policy, shared_buffers_bytes=plan["allocated_bytes"],
                legacy_buffers_bytes=plan["logical_io_bytes"], protocol_version=PROTOCOL_VERSION,
                input_packing="direct layout copy into IPC buffer, no contiguous float temporary",
                transfer_scope="shared-file IPC plus SDK copies; not proven zero-copy")
        except BaseException:
            self.close()
            raise

    def _read(self):
        try:
            for line in self.process.stdout:
                self.responses.put(json.loads(line))
        except Exception as exc:
            self.responses.put(dict(event="ERROR", error=str(exc)))
        finally:
            self.responses.put(dict(event="ERROR", error="Worker stream closed"))

    def _response(self):
        try:
            value = self.responses.get(timeout=self.timeout)
        except queue.Empty:
            self.failed = True
            raise TimeoutError("Partition worker timed out") from None
        if value.get("event") == "ERROR":
            self.failed = True
            raise RuntimeError(value["error"])
        return value

    def execute(self, name, value):
        part = self.parts[name]
        if part['meta'].get('spatial_tiling'):
            # Hold the same transaction lock across the complete head. Worker
            # request IDs remain one per actual tile, with every call measured.
            with self.lock:
                return execute_spatial(value, part['meta'],
                    lambda tile: self._execute_once(name, tile))
        with self.lock:
            return self._execute_once(name, value)

    def _execute_once(self, name, value):
        if self.closed or self.failed:
            raise RuntimeError("Partition worker is closed or failed")
        started = time.perf_counter()
        part = self.parts[name]
        if list(value.shape) != part["meta"]["input_shape"]:
            raise ValueError("Static network shape mismatch for " + name)
        input_buffer, output_buffer = self.buffers[name]
        # Real-board transport probe: tiling helps large channel planes,
        # while Python tile dispatch slows small boundaries. Exact copies
        # only; never reduce precision or skip finite/shape validation.
        blocked = value.nbytes >= 4*1024**2
        to_wire(value, part["input"], out=input_buffer, blocked=blocked)
        packed = time.perf_counter()
        command = dict(command="FORWARD", name=name, sequence=self.sequence)
        try:
            self.process.stdin.write(json.dumps(command)+"\n")
            self.process.stdin.flush()
            response = self._response()
            if (response.get("event"), response.get("name"), response.get("sequence"),
                response.get("backend"), response.get("npu_executed")) != (
                "COMPLETE", name, self.sequence, self.backend, self.backend == "npu"):
                raise ValueError("Wrong task/sequence/backend in worker response")
            received = time.perf_counter()
            result = from_wire(output_buffer, part["output"])
            if list(result.shape) != part["meta"]["output_shape"]:
                raise ValueError("Logical output shape mismatch")
            finished = time.perf_counter()
            response.update(pack_seconds=packed-started, wait_seconds=received-packed,
                unpack_seconds=finished-received, total_seconds=finished-started,
                input_bytes=input_buffer.nbytes, output_bytes=output_buffer.nbytes,
                input_layout_copy='cache_tiled' if blocked else 'numpy_direct')
            self.calls.append(response)
            self.sequence += 1
            return result
        except BaseException:
            self.failed = True
            raise

    def attach(self, model):
        import torch
        if self.closed or self.failed:
            raise RuntimeError('Cannot attach a closed or failed partition worker')
        if getattr(self, 'attached_model', None) is not None:
            raise ValueError('Partition worker is already attached to a model')
        validate_selection(list(self.parts))
        runtime = self
        class Partition(torch.nn.Module):
            def __init__(self, name):
                super().__init__()
                self.name = name
            def forward(self, value):
                if value.device.type != "cpu" or value.dtype != torch.float32:
                    raise ValueError("Expected CPU float32 boundary tensor")
                result = torch.from_numpy(runtime.execute(self.name, value.detach().numpy()))
                return [result] if runtime.parts[self.name]["meta"]["returns_list"] else result
        # Resolve every ordinary/FFN replacement before changing the model.
        # Whole-method adapters validate original modules before replacements;
        # fused heads own their two modules/hooks and therefore close first.
        replacements = []
        for name in self.parts:
            if name in ('fused_heads', 'corr_depth_pipeline', 'refine_heads_pipeline'):
                continue
            path = PARTITIONS[name].split('.')
            parent = model
            for component in path[:-1]:
                parent = getattr(parent, component)
            replacements.append((parent, path[-1], getattr(parent, path[-1])))
        self.model_replacements = replacements
        try:
            # This is a generated method seam, never a replaceable nn.Module.
            # Its adapter validates the untouched method and three CPU modules
            # before any FFN/head/ordinary replacement changes the model.
            if 'corr_depth_pipeline' in self.parts:
                from gaussian_generation.npu_branch.corr_depth import attach_corr_depth
                self.model_adapters.append(attach_corr_depth(model.depth_predictor,
                    lambda value: self.execute('corr_depth_pipeline', value)))
            if 'refine_heads_pipeline' in self.parts:
                from gaussian_generation.npu_branch.refine_heads import attach_refine_heads
                self.model_adapters.append(attach_refine_heads(model.depth_predictor,
                    lambda value: self.execute('refine_heads_pipeline', value)))
            if any(name.startswith('transformer_ffn_') for name in self.parts):
                from gaussian_generation.npu_branch.ffn import attach_ffn_partitions
                attach_ffn_partitions(model, self)
            for name in self.parts:
                if name.startswith('transformer_ffn_') or name in ('fused_heads', 'corr_depth_pipeline', 'refine_heads_pipeline'):
                    continue
                replace(model, name, Partition(name))
            # Hook the actual refine module after its possible replacement.
            if 'fused_heads' in self.parts:
                from gaussian_generation.npu_branch.fused_heads import attach_fused_heads
                self.model_adapters.append(attach_fused_heads(model.depth_predictor,
                    lambda value: self.execute('fused_heads', value)))
            self.attached_model = model
        except BaseException:
            self._restore_model()
            raise

    def _restore_model(self):
        for adapter in reversed(getattr(self, 'model_adapters', [])):
            adapter.close()
        self.model_adapters = []
        for parent, attribute, original in reversed(getattr(self, 'model_replacements', [])):
            setattr(parent, attribute, original)
        self.model_replacements = []
        self.attached_model = None

    def report(self, start=None):
        if self.calls and start is not None and start < self.calls[0]['sequence']:
            raise ValueError('Requested partition report was evicted; persist per-inference receipts')
        calls = [c for c in self.calls if start is None or c['sequence'] >= start]
        return dict(backend=self.backend, npu_executed=any(c["npu_executed"] for c in calls),
            selected_partitions=list(self.parts), cpu_partitions=[n for n in PARTITIONS
                if n not in self.parts and n not in ('fused_heads', 'corr_depth_pipeline', 'refine_heads_pipeline') and not
                ('fused_heads' in self.parts and n in ('to_gaussians','to_disparity')) and not
                ('corr_depth_pipeline' in self.parts and n in CORR_DEPTH_COMPONENTS) and not
                ('refine_heads_pipeline' in self.parts and n in REFINE_HEADS_COMPONENTS)],
            spatial_execution={n:p['meta']['spatial_tiling'] for n,p in self.parts.items()
                               if p['meta'].get('spatial_tiling')},
            cpu_remainder="camera geometry, cost-volume warping, backbone window attention/normalization, Gaussian adapter; other unselected modules listed separately",
            calls=calls, total_partition_seconds=sum(c["total_seconds"] for c in calls),
            history_scope='bounded recent calls; complete per-inference slices are persisted separately',
            total_call_count=self.sequence, history_first_sequence=calls[0]['sequence'] if calls else None,
            transfer_bytes=sum(c["input_bytes"]+c["output_bytes"] for c in calls),
            transfer_bytes_scope="logical partition I/O bytes; not measured DMA/DDR traffic or all memory copies",
            validation=getattr(self, 'validation', None),
            **self.provenance)

    def close(self):
        if self.closed:
            return
        self.closed = True
        self._restore_model()
        if self.process is not None:
            try:
                if self.process.poll() is None:
                    self.process.stdin.write('{"command":"QUIT"}\n')
                    self.process.stdin.flush()
                    self.process.wait(timeout=35)
            except (OSError, subprocess.TimeoutExpired):
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            if self.reader is not None:
                self.reader.join(timeout=2)
            self.process.stdin.close()
            self.process.stdout.close()
        if self.arena is not None:
            self.arena.close()
        if self.log:
            self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
