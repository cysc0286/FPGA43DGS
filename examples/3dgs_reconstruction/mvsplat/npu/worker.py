"""Separate SDK process; persistent sessions and bounded shared-file buffers."""
import ctypes as ct
import json
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
from npu.artifacts import validate_bundle
from npu.buffers import MappedBuffers
from npu.protocol import PROTOCOL_VERSION, check_bridge, check_command


def main():
    # Preserve protocol FD before sending SDK stdout diagnostics to stderr.
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    request_path = Path(sys.argv[1]).resolve()
    request = json.loads(request_path.read_text())
    if request.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("Worker protocol version mismatch")
    backend = request["backend"]
    if backend not in ("npu", "onnx_reference"):
        raise ValueError("Unknown backend")
    _, parts = validate_bundle(request["bundle"], request["names"], backend)
    sessions, buffers, sequence = {}, {}, 0
    library, device_lock, arena = None, None, None
    def reply(value):
        protocol.write(json.dumps(value, allow_nan=False)+"\n")
    try:
        arena = MappedBuffers(request_path.parent, parts, request["buffer_plan"],
                              limit_bytes=request["buffer_limit_bytes"])
        buffers = arena.views
        if backend == "npu":
            if platform.machine().lower() not in ("aarch64", "arm64"):
                raise RuntimeError("Real NPU backend requires ARM board")
            import fcntl
            device_lock = open("/run/lock/fpga43dgs-npu.lock", "a")
            fcntl.flock(device_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            library = ct.CDLL(request["library"])
            check_bridge(library)
            library.mgs_error.restype = ct.c_char_p
            library.mgs_create.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_size_t, ct.c_size_t]
            library.mgs_create.restype = ct.c_void_p
            library.mgs_destroy.argtypes = [ct.c_void_p]
            library.mgs_destroy.restype = None
            library.mgs_bindings.argtypes = [ct.c_void_p]
            library.mgs_bindings.restype = ct.c_int
            library.mgs_forward.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t,
                                            ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_double)]
            library.mgs_forward.restype = ct.c_int
        else:
            import onnxruntime as ort
        bindings = {}
        for name, part in parts.items():
            pair = buffers[name]
            if library:
                ctx = library.mgs_create(part["graph"].encode(), part["raw"].encode(), pair[0].nbytes, pair[1].nbytes)
                if not ctx:
                    raise RuntimeError(library.mgs_error().decode())
                sessions[name] = ctx
                bindings[name] = library.mgs_bindings(ctx)
            else:
                options = ort.SessionOptions()
                options.intra_op_num_threads = 2
                sessions[name] = ort.InferenceSession(part["graph"], sess_options=options, providers=["CPUExecutionProvider"])
                bindings[name] = 0
        reply(dict(event="READY", backend=backend, npu_executed=False, bindings=bindings,
                   protocol_version=PROTOCOL_VERSION, buffer_plan=arena.plan))
        for line in sys.stdin:
            command = json.loads(line)
            if not check_command(command, sequence, sessions):
                break
            name = command["name"]
            input_buffer, output_buffer = buffers[name]
            started = time.perf_counter()
            if not np.isfinite(input_buffer).all():
                raise ValueError("Nonfinite input")
            stages = [0., 0., 0.]
            if library:
                measured = (ct.c_double*3)()
                rc = library.mgs_forward(sessions[name], input_buffer.ctypes.data, input_buffer.nbytes,
                                         output_buffer.ctypes.data, output_buffer.nbytes, measured)
                if rc:
                    raise RuntimeError(library.mgs_error().decode())
                stages = list(measured)
            else:
                output_buffer[:] = sessions[name].run(None, {"features": input_buffer})[0]
            if not np.isfinite(output_buffer).all():
                raise ValueError("Nonfinite output")
            reply(dict(event="COMPLETE", sequence=sequence, name=name,
                backend=backend, npu_executed=library is not None,
                worker_seconds=time.perf_counter()-started, sdk_seconds=stages))
            sequence += 1
    except Exception as exc:
        reply(dict(event="ERROR", error=type(exc).__name__+": "+str(exc)))
        raise
    finally:
        if library:
            for ctx in sessions.values():
                library.mgs_destroy(ctx)
        if arena is not None:
            arena.close()
        if device_lock is not None:
            device_lock.close()
        protocol.close()


if __name__ == "__main__":
    main()
