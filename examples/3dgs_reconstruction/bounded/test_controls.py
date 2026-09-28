"""Host-only regression: actual child processes, content corruption, exact CLI wiring."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import psutil
from common import ROOT, cpu_env, save
from monitor import run
from run import commands, inventory, load_profile, verify_receipt


class Controls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_success_and_evidence_no_overwrite(self):
        folder = self.root/"ok"
        result = run([sys.executable, "-c", "print('measured')"], folder, cpu_env(1), timeout_s=3)
        self.assertEqual(result["status"], "passed")
        self.assertIn("measured", (folder/"output.log").read_text())
        before = (folder/"measurement.json").read_bytes()
        with self.assertRaises(FileExistsError):
            run([sys.executable, "-c", "pass"], folder, cpu_env(1))
        self.assertEqual(before, (folder/"measurement.json").read_bytes())

    def test_timeout_terminates_child(self):
        pidfile = self.root/"child.pid"
        code = "import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(30)"
        folder = self.root/"timeout"
        with self.assertRaisesRegex(RuntimeError, "timeout"):
            run([sys.executable, "-c", code, str(pidfile)], folder, cpu_env(1), timeout_s=1)
        self.assertFalse(psutil.pid_exists(int(pidfile.read_text())))
        self.assertEqual(json.loads((folder/"measurement.json").read_text())["status"], "timeout")

    def test_rss_limit_stops_allocating_process(self):
        folder = self.root/"rss"
        with self.assertRaisesRegex(RuntimeError, "rss_limit"):
            run([sys.executable, "-c", "import time; x=bytearray(160*1024*1024); time.sleep(20)"],
                folder, cpu_env(1), rss_mib=80, timeout_s=5)
        m = json.loads((folder/"measurement.json").read_text())
        self.assertGreater(m["peak_process_tree_rss_mib_sampled"], 80)
        self.assertIsNotNone(m["returncode"])

    def test_process_failure_is_not_success(self):
        folder = self.root/"failed"
        with self.assertRaisesRegex(RuntimeError, "process_failed"):
            run([sys.executable, "-c", "raise SystemExit(7)"], folder, cpu_env(1))
        self.assertEqual(json.loads((folder/"measurement.json").read_text())["returncode"], 7)

    def test_failed_parent_does_not_leave_observed_worker(self):
        pidfile = self.root/"orphan.pid"
        code = "import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(.4); raise SystemExit(9)"
        with self.assertRaisesRegex(RuntimeError, "process_failed"):
            run([sys.executable, "-c", code, str(pidfile)], self.root/"orphan", cpu_env(1), timeout_s=3)
        self.assertFalse(psutil.pid_exists(int(pidfile.read_text())))

    def test_resume_detects_output_corruption_and_command_change(self):
        (self.root/"frames").mkdir()
        image = self.root/"frames/frame.png"
        image.write_bytes(b"test receipt content")
        save(self.root/"frames.json", {})
        stage = self.root/"stages/frames"
        stage.mkdir(parents=True)
        save(stage/"receipt.json", {"outputs": inventory(self.root, "frames")})
        save(stage/"measurement.json", {"status": "passed", "command": ["original"]})
        verify_receipt(self.root, "frames", ["original"])
        with self.assertRaisesRegex(ValueError, "command"):
            verify_receipt(self.root, "frames", ["different"])
        image.write_bytes(b"corrupted")
        with self.assertRaisesRegex(ValueError, "content changed"):
            verify_receipt(self.root, "frames", ["original"])

    def test_plan_is_side_effect_free_and_forces_cpu(self):
        folder = self.root/"nonexistent"
        result = subprocess.run([sys.executable, str(ROOT/"run.py"), "--plan", "--video", "absent.mp4",
                                 "--opensplat", "absent.exe", "--run", str(folder)], capture_output=True, text=True, check=True)
        plan = json.loads(result.stdout)
        self.assertFalse(folder.exists())
        self.assertIn("--cpu", plan["commands"]["train"])
        self.assertFalse(plan["board_access"])
        c = load_profile("arm_candidate")
        cmd = plan["commands"]["sfm"]
        self.assertEqual(cmd[cmd.index("--max-features")+1], str(c["max_features"]))
        self.assertEqual(cmd[cmd.index("--render-width")+1], str(c["render_width"]))

    def test_archive_preserves_existing_version_rows(self):
        import csv
        import importlib.util
        import shutil
        spec = importlib.util.spec_from_file_location("archive_under_test", ROOT.parent/"collect_evidence.py")
        archive = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(archive)
        evidence = self.root/"evidence"
        evidence.mkdir()
        registry = evidence/"versions.csv"
        shutil.copy2(ROOT.parent/"evidence/versions.csv", registry)
        with registry.open(newline="") as f:
            original = list(csv.DictReader(f))
        candidate = self.root/"synthetic_archive_unit_fixture"
        (candidate/"renderer_input").mkdir(parents=True)
        save(candidate/"sfm.json", {"registered": 5, "points3D": 10, "mean_reprojection_error_px": 1, "has_cuda": False})
        save(candidate/"quality.json", {"native_cpu_vs_heldout": {"psnr_db": 25, "ssim_rgb_gaussian_11_sigma1p5_valid": .9}})
        save(candidate/"renderer_input/manifest.json", {"gaussians": 20, "model_sha256": "unit_fixture"})
        for name in ["frames", "sfm", "train", "export"]:
            save(candidate/(name+"_measurement.json"), {"wall_s": 1, "peak_process_tree_rss_mib_sampled": 1})
        (candidate/"train.log").write_text("Using CPU\nvalidation PSNR: 25.0\n")
        (candidate/"sfm.log").write_text("unit fixture")
        (candidate/"comparison.png").write_bytes(b"unit fixture copied, not decoded")
        with patch.object(archive, "ROOT", self.root), patch.object(sys, "argv", ["collect", "--runs", str(candidate)]):
            with patch("builtins.print"):
                archive.main()
            with self.assertRaisesRegex(ValueError, "Duplicate run"):
                archive.main()
        with registry.open(newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows[:-1], original)
        self.assertEqual(rows[-1]["run"], candidate.name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
