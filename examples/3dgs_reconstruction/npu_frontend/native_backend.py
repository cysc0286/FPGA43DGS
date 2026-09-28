"""ctypes backend with one retained SDK session. ARM board execution only."""
import ctypes as ct
import hashlib
import platform
import time
from pathlib import Path
import numpy as np
from audit_graph import audit


class NativeNPUBackend:
    name = "icraft_native_resident"
    npu_executed = True

    def __init__(self, library, model, raw, tile=256):
        if platform.system() != "Linux" or platform.machine().lower() not in ("aarch64", "arm64"):
            raise RuntimeError("Native NPU execution requires the ARM board")
        self.audit = audit(Path(model), Path(raw), tile)
        self.library_sha256 = hashlib.sha256(Path(library).read_bytes()).hexdigest()
        self.tile = tile
        self.lib = ct.CDLL(str(Path(library).resolve()))
        self.lib.hgs_create.argtypes = [ct.c_char_p, ct.c_char_p, ct.c_int]
        self.lib.hgs_create.restype = ct.c_void_p
        self.lib.hgs_error.restype = ct.c_char_p
        self.lib.hgs_binding.argtypes = [ct.c_void_p]
        self.lib.hgs_destroy.argtypes = [ct.c_void_p]
        self.lib.hgs_dot.argtypes = [ct.c_void_p]+[ct.POINTER(ct.c_float)]*3+[ct.POINTER(ct.c_double)]
        self.output = np.empty((1, 1, tile, tile), np.float32)
        self.times = np.zeros(3, np.float64)
        self.total_times = np.zeros(3, np.float64)
        self.calls = 0
        start = time.perf_counter()
        self.handle = self.lib.hgs_create(str(Path(model).resolve()).encode(), str(Path(raw).resolve()).encode(), tile)
        self.init_s = time.perf_counter()-start
        if not self.handle:
            raise RuntimeError(self.lib.hgs_error().decode(errors="replace"))
        if self.lib.hgs_binding(self.handle) != 1:
            self.close()
            raise RuntimeError("No unique Matmul binding")

    def dot(self, left, right):
        if not self.handle:
            raise ValueError("NPU session closed")
        for arr, shape in [(left, (1, 1, self.tile, 128)), (right, (1, 1, 128, self.tile))]:
            if arr.dtype != np.float32 or arr.shape != shape or not arr.flags.c_contiguous:
                raise ValueError("Native input ABI mismatch")
        ptr = lambda a: a.ctypes.data_as(ct.POINTER(ct.c_float))
        error = self.lib.hgs_dot(self.handle, ptr(left), ptr(right), ptr(self.output), self.times.ctypes.data_as(ct.POINTER(ct.c_double)))
        if error:
            raise RuntimeError(self.lib.hgs_error().decode(errors="replace"))
        self.total_times += self.times
        self.calls += 1
        # The consumer must finish before the next call overwrites this buffer.
        return self.output

    def close(self):
        if getattr(self, "handle", None):
            self.lib.hgs_destroy(self.handle)
            self.handle = None
