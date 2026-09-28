"""Explicit CPU backends. ONNX execution on CPU is never reported as NPU."""
import time
import numpy as np


class NumpyBackend:
    name = "numpy_fp32_cpu"
    npu_executed = False

    def dot(self, left, right):
        return np.matmul(left, right)


class OnnxCPUBackend:
    name = "onnxruntime_cpu"
    npu_executed = False

    def __init__(self, model, threads=1):
        import onnxruntime as ort
        start = time.perf_counter()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
        if self.session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("Unexpected execution provider")
        self.init_s = time.perf_counter()-start

    def dot(self, left, right):
        return self.session.run(["scores"], {"left": left, "right": right})[0]
