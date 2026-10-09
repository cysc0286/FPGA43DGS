"""A loaded NPU session must not be advertised as numerically ready."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from common import sha
from gaussian_generation.npu_branch.readiness import validate_oracles


class ReadinessTests(unittest.TestCase):
    def run_gate(self, corrupt):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"backbone_cnn").mkdir()
            expected = np.ones((1, 1, 2, 2), dtype=np.float32)
            oracle = root/"backbone_cnn/oracle.npz"
            np.savez(oracle, input=expected, expected=expected)
            (root/"manifest.json").write_text("{}")
            meta = {"partitions": {"backbone_cnn": {"oracle_sha256": sha(oracle)}}}
            calls = []
            def execute(name, value):
                calls.append(name)
                # Corruption on reuse catches stale device buffers after READY.
                return value + (0.04 if corrupt and len(calls) == 2 else 0)
            worker = SimpleNamespace(parts={"backbone_cnn": {}}, execute=execute,
                provenance={"oracle_manifest_sha256": sha(root/"manifest.json")})
            with patch("gaussian_generation.npu_branch.readiness.validate_bundle", return_value=(meta, {})):
                if corrupt:
                    with self.assertRaisesRegex(ValueError, "numerical readiness"):
                        validate_oracles(worker, root, root/"readiness.json")
                else:
                    validate_oracles(worker, root, root/"readiness.json")
            result = json.loads((root/"readiness.json").read_text())
            self.assertEqual(result["complete"], not corrupt)
            self.assertEqual(len(calls), 2)

    def test_correct_repeated_execution_passes(self):
        self.run_gate(False)

    def test_corrupted_repeat_cannot_publish_readiness(self):
        self.run_gate(True)


if __name__ == "__main__":
    unittest.main()
