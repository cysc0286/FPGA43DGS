import contextlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import numpy as np
from block_io import pack, verify, HEADER, INDEX
from matcher import blocks


class BlockIO(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        rng = np.random.default_rng(913)
        self.a = rng.integers(0, 65, (33, 128), dtype=np.uint8)
        self.b = rng.integers(0, 65, (35, 128), dtype=np.uint8)
        self.b[:33] = self.a
        np.savez(self.root/"case.npz", a=self.a, b=self.b)
        self.bundle = self.root/"bundle"
        with contextlib.redirect_stdout(io.StringIO()):
            pack(self.root/"case.npz", self.bundle, 32)
        self.output = self.root/"scores.bin"
        with self.output.open("wb") as f:
            f.write(HEADER.pack(b"HGSOUT01", 32, 4, 33, 35))
            for i, j, ni, nj, left, right in blocks(self.a, self.b, 32):
                f.write(INDEX.pack(i, j, ni, nj))
                f.write((left@right).astype("<f4").tobytes())

    def test_round_trip_tail_and_no_hardware_attribution(self):
        report = verify(self.bundle, self.output, self.root/"report.json")
        self.assertTrue(report["passed"])
        self.assertEqual(report["score_max_abs_error"], 0)
        self.assertFalse(report["npu_executed"])

    def test_reject_bad_headers_order_truncation_and_extra_data(self):
        original = self.output.read_bytes()
        bad_order = bytearray(original)
        struct.pack_into("<I", bad_order, HEADER.size, 32)
        for i, bad in enumerate([b"BADMAGIC"+original[8:], original[:-1], original+b"x", bad_order]):
            with self.subTest(i=i):
                self.output.write_bytes(bad)
                with self.assertRaises(ValueError):
                    verify(self.bundle, self.output, self.root/f"bad{i}.json")

    def test_reject_numerical_corruption_and_tampered_inputs(self):
        raw = bytearray(self.output.read_bytes())
        struct.pack_into("<f", raw, HEADER.size+INDEX.size, 2.0)
        self.output.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, "score or match"):
            verify(self.bundle, self.output, self.root/"bad_score.json")
        (self.bundle/"jobs.bin").write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "hash"):
            verify(self.bundle, self.output, self.root/"bad_input.json")

    @unittest.skipUnless(os.environ.get("HGS_BLOCK_RUNNER"), "Set HGS_BLOCK_RUNNER to the built CPU executable")
    def test_cpp_abi_and_invalid_job(self):
        runner = os.environ["HGS_BLOCK_RUNNER"]
        out = self.root/"cpp"
        cmd = [runner, "unused", "unused", str(self.bundle/"jobs.bin"), str(out)]
        subprocess.run(cmd, check=True, capture_output=True)
        self.assertEqual((out/"scores.bin").read_bytes(), self.output.read_bytes())
        self.assertFalse(json.loads((out/"status.json").read_text())["npu_executed"])
        # Prevent accidentally overwriting any earlier run.
        self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
        raw = bytearray((self.bundle/"jobs.bin").read_bytes())
        struct.pack_into("<I", raw, HEADER.size, 32)
        (self.bundle/"jobs.bin").write_bytes(raw)
        cmd[-1] = str(self.root/"invalid")
        self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
        self.assertFalse((self.root/"invalid/status.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
