"""The public reconstruction command must route to the board MVSplat chain."""
import sys
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pipeline


class ReconstructEntry(unittest.TestCase):
    def test_legacy_bundle_does_not_require_mvsplat(self):
        source = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shutil.copy2(source / "pipeline.py", root / "pipeline.py")
            shutil.copytree(source / "modules", root / "modules",
                            ignore=shutil.ignore_patterns("__pycache__"))
            result = subprocess.run([sys.executable, str(root / "pipeline.py"),
                                     "gaussian", "--help"],
                                    cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--opensplat", result.stdout)

    def test_forwards_board_options(self):
        args = ["--video", "input.mp4", "--out", "new_run", "--pose-mode", "fast_pair", "--fused"]
        with patch("mvsplat.board_pipeline.main") as board:
            pipeline.main(["reconstruct", *args])
        board.assert_called_once_with(args)

    def test_defaults_to_quality_validated_pose_path(self):
        from mvsplat.board_pipeline import build_parser
        parsed = build_parser().parse_args(["--video", "input.mp4", "--out", "new_run"])
        self.assertEqual(parsed.pose_mode, "full_sfm")
        self.assertFalse(parsed.fused)


if __name__ == "__main__":
    unittest.main()
