"""Separate SDK process; persistent sessions and bounded shared-file buffers."""
import ctypes as ct
import json
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
from common import sha
from gaussian_generation.npu_branch.artifacts import validate_bundle
from gaussian_generation.npu_branch.buffers import MappedBuffers
from gaussian_generation.npu_branch.protocol import PROTOCOL_VERSION, check_bridge, check_command, check_direct_sfb


def runtime_identity(spatial_heads=()):
    """Hash loaded SDK binaries once, after session creation, outside video timing."""
    libraries = {}
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/'):
            path = Path(fields[5])
            if 'icraft' in path.name.lower() and path.is_file():
                libraries[str(path.resolve())] = sha(path)
    return dict(machine=platform.machine().lower(), sdk_libraries=libraries,
                ddr_guard_mib=int(os.environ.get('MGS_NPU_DDR_GUARD_MIB', '0')),
                **({'spatial_head_settle_passes': 2, 'spatial_heads': list(spatial_heads)} if spatial_heads else {}),
                **({'session_reapply': True} if os.environ.get('MGS_NPU_REAPPLY') == '1' else {}),
                **({'fresh_input_tensor': True} if os.environ.get('MGS_NPU_FRESH_INPUT') == '1' else {}),
                **({'force_output_wait': True} if os.environ.get('MGS_NPU_FORCE_WAIT') == '1' else {}),
                **({'direct_sfb_output': True} if os.environ.get('MGS_NPU_DIRECT_SFB') == '1' else {}),
                **({'backbone_settle_passes': 2} if os.environ.get('MGS_NPU_SETTLE_BACKBONE') == '1' else {}))


def sdk_passes(name, meta):
    if meta.get('spatial_tiling') and name in ('to_gaussians', 'to_disparity'):
        return 2
    return 2 if name == 'backbone_cnn' and os.environ.get('MGS_NPU_SETTLE_BACKBONE') == '1' else 1


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
    detailed_forward = None
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
            if os.environ.get('MGS_NPU_DIRECT_SFB') == '1':
                check_direct_sfb(library)
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
            # New bridge builds expose five independently measured buckets,
            # while ABI-2 libraries retain the historical three-bucket call.
            try:
                detailed_forward = library.mgs_forward_detailed
            except AttributeError:
                detailed_forward = None
            if detailed_forward is not None:
                detailed_forward.argtypes = [ct.c_void_p, ct.c_void_p, ct.c_size_t,
                                             ct.c_void_p, ct.c_size_t, ct.POINTER(ct.c_double)]
                detailed_forward.restype = ct.c_int
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
                try:
                    memory_layout = library.mgs_memory_layout
                except AttributeError:
                    if int(os.environ.get('MGS_NPU_DDR_GUARD_MIB', '0')):
                        raise RuntimeError('Bridge lacks required PL DDR reservation support; rebuild bridge')
                else:
                    memory_layout.argtypes = [ct.c_void_p]
                    memory_layout.restype = ct.c_char_p
                    raw = memory_layout(ctx)
                    layout = json.loads(raw) if raw else []
                    guard_bytes = int(os.environ.get('MGS_NPU_DDR_GUARD_MIB', '0')) * 1024 * 1024
                    if guard_bytes and (not layout or min(layout, key=lambda c: c['begin'])['bytes'] != guard_bytes):
                        raise RuntimeError('NPU PL DDR reservation could not be verified')
                    print('NPU_MEMORY_LAYOUT', name, layout, file=sys.stderr, flush=True)
            else:
                options = ort.SessionOptions()
                options.intra_op_num_threads = 2
                sessions[name] = ort.InferenceSession(part["graph"], sess_options=options, providers=["CPUExecutionProvider"])
                bindings[name] = 0
        reply(dict(event="READY", backend=backend, npu_executed=False, bindings=bindings,
                   runtime_identity=runtime_identity([n for n,p in parts.items() if p['meta'].get('spatial_tiling')]) if library else None,
                   protocol_version=PROTOCOL_VERSION, buffer_plan=arena.plan))
        for line in sys.stdin:
            command = json.loads(line)
            if not check_command(command, sequence, sessions):
                break
            name = command["name"]
            input_buffer, output_buffer = buffers[name]
            started = time.perf_counter()
            prepare_started = time.perf_counter()
            if not np.isfinite(input_buffer).all():
                raise ValueError("Nonfinite input")
            input_prepare_seconds = time.perf_counter()-prepare_started
            stages = [0., 0., 0.]
            stage_names = ["input_write", "execute_wait", "output_convert"]
            execution_passes = 1
            if library:
                width = 5 if detailed_forward is not None else 3
                forward = detailed_forward or library.mgs_forward
                execution_passes = sdk_passes(name, parts[name]['meta'])
                stages = [0.] * width
                # Explicit workaround for a board-observed one-request stale
                # result. Both full SDK calls are charged; this is not a cache
                # hit, zero-copy, or a claim that the underlying SDK is fixed.
                for _ in range(execution_passes):
                    measured = (ct.c_double*width)()
                    rc = forward(sessions[name], input_buffer.ctypes.data, input_buffer.nbytes,
                                 output_buffer.ctypes.data, output_buffer.nbytes, measured)
                    if rc:
                        raise RuntimeError(library.mgs_error().decode())
                    stages = [a+b for a,b in zip(stages, measured)]
                if detailed_forward is not None:
                    stage_names = ["input_write", "forward_submit", "wait", "output_convert", "sdk_total"]
            else:
                output_buffer[:] = sessions[name].run(None, {"features": input_buffer})[0]
            if not np.isfinite(output_buffer).all():
                raise ValueError("Nonfinite output")
            reply(dict(event="COMPLETE", sequence=sequence, name=name,
                backend=backend, npu_executed=library is not None,
                worker_seconds=time.perf_counter()-started,
                input_prepare_seconds=input_prepare_seconds,
                sdk_execution_passes=execution_passes,
                sdk_stage_names=stage_names, sdk_seconds=stages))
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
