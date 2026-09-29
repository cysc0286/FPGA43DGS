"""Boundary checks that prevent wrong layouts or false NPU attribution."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from npu.artifacts import from_wire, to_wire, relative_file, validate_bundle
from npu.compile_graphs import host_abi
from npu.runtime import PartitionRuntime


class NpuContractTests(unittest.TestCase):
    def test_layout_roundtrip_and_result_owns_memory(self):
        x = np.arange(2*3*4*5, dtype=np.float32).reshape(2,3,4,5)
        spec = dict(layout="NHWC", shape=[2,4,5,3])
        wire = to_wire(x, spec)
        self.assertTrue(wire.flags.c_contiguous)
        result = from_wire(wire, spec)
        wire[:] = -1
        np.testing.assert_array_equal(result, x)

    def test_invalid_dtype_and_shape_fail(self):
        spec = dict(layout="NCHW", shape=[1,3,2,2])
        for x in (np.zeros((1,3,2,2), np.float64), np.zeros((1,2,2,3), np.float32),
                  np.full((1,3,2,2), np.nan, np.float32)):
            with self.assertRaises(ValueError):
                to_wire(x, spec)

    def test_compiled_layout_needs_semantics_not_just_same_bytes(self):
        value = dict(element_dtype="@fp(32)", shape=[2,4,5,3], layout="@layout(NHWC)")
        self.assertEqual(host_abi(value, [2,3,4,5])["layout"], "NHWC")
        with self.assertRaises(ValueError):
            host_abi(value, [2,4,5,3])
        with self.assertRaises(ValueError):
            host_abi(dict(value, layout="@layout(***C)"), [2,3,4,5])

    def test_artifact_cannot_escape_bundle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/"bundle"
            root.mkdir()
            (root.parent/"outside").write_text("data")
            with self.assertRaises(ValueError):
                relative_file(root, "../outside")

    def test_worker_rejects_cpu_claiming_npu(self):
        # A stale or mislabeled reply must never be recorded as NPU completion.
        import queue
        import threading
        import io
        from types import SimpleNamespace
        runtime = PartitionRuntime.__new__(PartitionRuntime)
        runtime.lock = threading.Lock()
        runtime.closed = runtime.failed = False
        runtime.sequence = 0
        runtime.backend = "npu"
        spec = dict(shape=[1,1,1,1], layout="NCHW")
        runtime.parts = {"x":dict(input=spec, output=spec, meta=dict(input_shape=[1,1,1,1]))}
        runtime.buffers = {"x":[np.zeros((1,1,1,1),np.float32),np.zeros((1,1,1,1),np.float32)]}
        runtime.process = SimpleNamespace(stdin=io.StringIO())
        runtime.timeout = 1
        runtime.responses = queue.Queue()
        runtime.responses.put(dict(event="COMPLETE", name="x", sequence=0,
                                   backend="onnx_reference", npu_executed=False))
        with self.assertRaisesRegex(ValueError, "Wrong task"):
            runtime.execute("x", np.zeros((1,1,1,1),np.float32))
        self.assertTrue(runtime.failed)


if __name__ == "__main__":
    unittest.main()
