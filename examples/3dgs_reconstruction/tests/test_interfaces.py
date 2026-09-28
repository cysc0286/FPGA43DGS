"""Module-interface regression, including corrupted/cross-version artifacts."""
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.common import file_sha256
from modules.contracts import FrameSet, PoseSet, GaussianScene, RenderInput, RenderResult
from modules.gaussian_generation.training import command as train_command
from modules.rendering import render
from interface_fixture import make_reference


class Interfaces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        source = os.environ.get("HGS_TEST_REFERENCE_RUN")
        cls.reference_run = Path(source).resolve() if source else make_reference(Path(cls.fixture.name)/"reference")

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name)
        self.reference = self.reference_run

    def tearDown(self):
        self.tmp.cleanup()

    def copy(self, *names):
        for name in names:
            src, dst = self.reference/name, self.run/name
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)

    def test_registered_reference_interfaces(self):
        self.assertEqual(len(FrameSet.load(self.reference).names), 30)
        self.assertEqual(PoseSet.load(self.reference).heldout_image, "frame_000015.png")
        self.assertTrue(GaussianScene.load(self.reference).model.is_file())
        scene = RenderInput.load(self.reference/"renderer_input")
        self.assertEqual((scene.width, scene.height), (160, 120))

    def test_reject_missing_image(self):
        self.copy("frames", "frames.json")
        next((self.run/"frames").glob("*.png")).unlink()
        with self.assertRaisesRegex(ValueError, "Missing"):
            FrameSet.load(self.run)

    def test_reject_duplicate_frame_indices(self):
        self.copy("frames", "frames.json")
        path = self.run/"frames.json"
        data = json.loads(path.read_text())
        data["frames"][1]["video_frame"] = data["frames"][0]["video_frame"]
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "indices"):
            FrameSet.load(self.run)

    def test_pose_artifact_is_independent_of_video_and_database(self):
        self.copy("project", "project.json", "sfm.json")
        self.assertTrue(PoseSet.load(self.run).project.is_dir())

    def run_validate(self, *args):
        return subprocess.run([sys.executable, str(ROOT/"pipeline.py"), "validate",
                               "--run", str(self.run), *args], cwd=self.run,
                              capture_output=True, text=True)

    def test_cli_pose_handoff_without_video(self):
        self.copy("project", "project.json", "sfm.json")
        result = self.run_validate("--only", "pose")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([v["stage"] for v in json.loads(result.stdout)["checks"]], ["pose"])
        self.assertFalse((self.run/"frames.json").exists())

    def test_cli_renderer_handoff_without_reconstruction(self):
        self.copy("renderer_input")
        result = self.run_validate("--only", "export")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([v["stage"] for v in json.loads(result.stdout)["checks"]], ["export"])
        self.assertFalse((self.run/"project").exists())

    def test_cli_validate_scope_and_result_requirements(self):
        conflict = self.run_validate("--only", "pose", "--through", "export")
        self.assertEqual(conflict.returncode, 2)
        self.assertIn("not allowed with argument", conflict.stderr)
        missing = self.run_validate("--only", "render")
        self.assertEqual(missing.returncode, 2)
        self.assertIn("--out is required", missing.stderr)

    def test_cli_default_still_checks_upstream_artifacts(self):
        self.copy("renderer_input")
        result = self.run_validate()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("frames.json", result.stderr)

    def test_reject_noncentered_intrinsics(self):
        self.copy("project", "project.json", "sfm.json")
        path = self.run/"project.json"
        data = json.loads(path.read_text())
        next(iter(data["cameras"].values()))["params"][2] += 1
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "Intrinsics"):
            PoseSet.load(self.run)

    def test_reject_model_corruption(self):
        self.copy("renderer_input")
        with (self.run/"renderer_input/model.ply").open("ab") as f:
            f.write(b"x")
        with self.assertRaisesRegex(ValueError, "hash"):
            RenderInput.load(self.run/"renderer_input")

    def test_reject_wrong_ply_layout_even_with_updated_hash(self):
        self.copy("renderer_input")
        model = self.run/"renderer_input/model.ply"
        model.write_bytes(model.read_bytes().replace(b"property float opacity", b"property float invalid", 1))
        path = self.run/"renderer_input/manifest.json"
        data = json.loads(path.read_text())
        data["model_sha256"] = file_sha256(model)
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "layout"):
            RenderInput.load(model.parent)

    def test_reject_reflected_camera(self):
        self.copy("renderer_input")
        path = self.run/"renderer_input/novel_midpoint.bin"
        raw = path.read_bytes()
        values = list(struct.unpack("<4I14d", raw[8:]))
        values[4:7] = [-v for v in values[4:7]]
        path.write_bytes(raw[:8]+struct.pack("<4I14d", *values))
        with self.assertRaisesRegex(ValueError, "handedness"):
            RenderInput.load(path.parent)

    def test_reject_camera_path_escape(self):
        with self.assertRaises(ValueError):
            RenderInput.load(self.reference/"renderer_input", "../camera.bin")

    def test_render_plan_uses_frozen_package(self):
        # Exercise the actual release entry point without distributing board binaries.
        package = self.run/"renderer_package"
        package.mkdir()
        shutil.copy2(ROOT.parent/"3dgs_flicker_hw/release/render.py", package/"render.py")
        (package/"manifest.json").write_text(json.dumps({"scope": "plan-only test fixture"}))
        result = render(self.reference/"renderer_input", package,
                        self.run/"output", backend="cpu_dense", plan=True)
        self.assertEqual(result["status"], "plan_only")
        self.assertIn("cpu_dense", result["command"])
        self.assertFalse((self.run/"output").exists())

    def test_reject_incomplete_framebuffer(self):
        frame = self.run/"frame.bin"
        frame.write_bytes(b"GSSOUT01"+struct.pack("<II", 2, 2))
        (self.run/"frame.ppm").write_bytes(b"P6\n2 2\n255\n")
        (self.run/"result.json").write_text(json.dumps({"complete": True, "frame_sha256": file_sha256(frame)}))
        with self.assertRaisesRegex(ValueError, "length"):
            RenderResult.load(self.run)

    def test_training_command_preserves_cpu_contract(self):
        command = train_command(self.run, "opensplat", 1000, 6000, "frame_000015.png")
        self.assertIn("--cpu", command)
        self.assertEqual(command[command.index("--densify-until")+1], "750")
        self.assertEqual(command[-2:], ["-o", str(self.run/"splat.ply")])
        with self.assertRaises(ValueError):
            train_command(self.run, "opensplat", 20, 6000, "frame.png")


if __name__ == "__main__":
    unittest.main()
