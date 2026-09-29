"""Boundary checks that prevent wrong layouts or false NPU attribution."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from npu.artifacts import from_wire, to_wire, relative_file, validate_bundle
from npu.compile_graphs import host_abi
from npu.runtime import PartitionRuntime
from npu.buffers import buffer_plan, MappedBuffers
from npu.protocol import check_command, check_bridge


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

    def test_direct_layout_copy_handles_noncontiguous_input(self):
        x = np.arange(120, dtype=np.float32).reshape(2,3,4,5)[:, :, ::-1, :]
        spec = dict(layout="NHWC", shape=[2,4,5,3])
        out = np.empty(spec["shape"], np.float32)
        self.assertIs(to_wire(x, spec, out=out), out)
        np.testing.assert_array_equal(from_wire(out, spec), x)
        with self.assertRaises(ValueError):
            to_wire(x, spec, out=np.empty(spec["shape"], np.float64))

    @staticmethod
    def small_parts():
        return {name: {k:dict(layout="NCHW", shape=shape) for k in ("input", "output")}
                for name, shape in (("a", [1,3,4,5]), ("b", [1,1,2,3]))}

    def test_shared_buffers_cross_mapping_reuse_preserves_returned_outputs(self):
        parts = self.small_parts()
        plan = buffer_plan(parts)
        self.assertLess(plan["allocated_bytes"], buffer_plan(parts, "per_partition")["allocated_bytes"])
        with tempfile.TemporaryDirectory() as directory:
            writer = MappedBuffers(directory, parts, plan, create=True)
            reader = MappedBuffers(directory, parts, plan)
            try:
                held = []
                for step, name in enumerate(("a", "b", "a", "b")):
                    value = np.full(parts[name]["input"]["shape"], step, np.float32)
                    to_wire(value, parts[name]["input"], out=writer.views[name][0])
                    np.testing.assert_array_equal(reader.views[name][0], value)
                    reader.views[name][1][:] = value + 1
                    held.append((from_wire(writer.views[name][1], parts[name]["output"]), step+1))
                for output, expected in held:
                    np.testing.assert_array_equal(output, np.full(output.shape, expected, np.float32))
                self.assertEqual(len(writer.maps), 2)
            finally:
                reader.close()
                writer.close()

    def test_buffer_budget_descriptor_and_truncated_file_fail_before_use(self):
        parts = self.small_parts()
        plan = buffer_plan(parts)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "budget"):
                MappedBuffers(directory, parts, plan, create=True, limit_bytes=1)
            self.assertEqual(list(Path(directory).iterdir()), [])
            with self.assertRaisesRegex(ValueError, "descriptor"):
                MappedBuffers(directory, parts, dict(plan, files={"../escape":240}), create=True)
            arena = MappedBuffers(directory, parts, plan, create=True)
            arena.close()
            (Path(directory)/"output.bin").write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "size mismatch"):
                MappedBuffers(directory, parts, plan)

    def test_invalid_commands_do_not_consume_sequence(self):
        good = dict(command="FORWARD", name="a", sequence=2)
        self.assertTrue(check_command(good, 2, ["a"]))
        self.assertFalse(check_command(dict(command="QUIT"), 2, ["a"]))
        for bad in (dict(good, command="RESET"), dict(good, sequence=1),
                    dict(good, sequence=2.0), dict(good, name="b"), dict(good, extra=1)):
            with self.assertRaises(ValueError):
                check_command(bad, 2, ["a"])

    def test_old_bridge_is_rejected_before_device_creation(self):
        from types import SimpleNamespace
        bridge = SimpleNamespace(mgs_abi_version=lambda: 1)
        with self.assertRaisesRegex(ValueError, "ABI mismatch"):
            check_bridge(bridge)

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
