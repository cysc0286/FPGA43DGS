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

from common import save, sha
from gaussian_generation.npu_branch.artifacts import from_wire, to_wire, validate_bundle
from gaussian_generation.npu_branch.catalog import PARTITIONS, replace
from gaussian_generation.npu_branch.buffers import buffer_plan, MappedBuffers
from gaussian_generation.npu_branch.protocol import PROTOCOL_VERSION


class PartitionRuntime:
    def __init__(self, backend, bundle, names, work, library=None,
                 worker_python=None, environment=None, timeout=90,
                 buffer_policy="shared", buffer_limit_mib=128):
        if backend not in ("onnx_reference", "npu"):
            raise ValueError("Expected an explicit accelerated or host validation backend")
        data, self.parts = validate_bundle(bundle, names, backend)
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
        self.buffers, self.calls, self.sequence = {}, [], 0
        self.arena = None
        self.lock = threading.Lock()
        self.responses = queue.Queue()
        self.process = None
        self.reader = None
        self.closed = False
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
        with self.lock:
            if self.closed or self.failed:
                raise RuntimeError("Partition worker is closed or failed")
            started = time.perf_counter()
            part = self.parts[name]
            if list(value.shape) != part["meta"]["input_shape"]:
                raise ValueError("Static network shape mismatch for " + name)
            input_buffer, output_buffer = self.buffers[name]
            to_wire(value, part["input"], out=input_buffer)
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
                    input_bytes=input_buffer.nbytes, output_bytes=output_buffer.nbytes)
                self.calls.append(response)
                self.sequence += 1
                return result
            except BaseException:
                self.failed = True
                raise

    def attach(self, model):
        import torch
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
        for name in self.parts:
            replace(model, name, Partition(name))

    def report(self, start=0):
        calls = self.calls[start:]
        return dict(backend=self.backend, npu_executed=any(c["npu_executed"] for c in calls),
            selected_partitions=list(self.parts), cpu_partitions=[n for n in PARTITIONS if n not in self.parts],
            cpu_remainder="camera geometry, cost-volume warping, multi-view attention, refinement UNets, Gaussian adapter",
            calls=calls, total_partition_seconds=sum(c["total_seconds"] for c in calls),
            transfer_bytes=sum(c["input_bytes"]+c["output_bytes"] for c in calls),
            transfer_bytes_scope="logical partition I/O bytes; not measured DMA/DDR traffic or all memory copies",
            **self.provenance)

    def close(self):
        if self.closed:
            return
        self.closed = True
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
