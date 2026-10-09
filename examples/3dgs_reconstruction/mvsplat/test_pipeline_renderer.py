"""Verify the warm entry reaches the same renderer configuration as interaction.

The native process is mocked: these are integration contracts, not board timings.
"""
from contextlib import ExitStack
import io
import json
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from initialize.run import main
from initialize.session import WarmSession
from rendering.pipeline_adapter import PipelineRenderer
from rendering.runtime import LiveRenderer, load_mainline_profile


class PipelineRendererTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.verify = self.stack.enter_context(patch(
            "rendering.pipeline_adapter.frozen_renderer_module"))
        self.verify.return_value.verify.return_value = {"name": "frozen-test-package"}
        self.process = MagicMock()
        self.process.poll.return_value = 0
        self.popen = self.stack.enter_context(patch(
            "rendering.runtime.subprocess.Popen", return_value=self.process))
        self.line = self.stack.enter_context(patch.object(LiveRenderer, "_line", return_value=b"LIVE_READY 1"))

    def renderer(self, **options):
        return self.stack.enter_context(PipelineRenderer(
            self.root / "frozen", "native-renderer", {}, self.root / "renderer.log", **options))

    def test_default_calls_mainline_factory_and_passes_native_flags(self):
        with patch.object(LiveRenderer, "mainline", wraps=LiveRenderer.mainline) as factory:
            renderer = self.renderer()
        factory.assert_called_once()
        profile = load_mainline_profile()
        configuration = renderer.runtime.configuration
        for key, value in profile["runtime"].items():
            if key != "cpu":
                self.assertEqual(configuration[key], value, key)
        self.assertEqual(configuration["backend"], "fpga")
        self.assertEqual(configuration["profile"], profile["id"])
        self.assertEqual(configuration["expected_boot_sha256"],
                         profile["required_hardware"]["accepted_boot_sha256"])
        command = self.popen.call_args.args[0]
        for flag in ("--direct-collect", "--parallel-collect", "--fused-collect", "--neon-pack"):
            self.assertIn(flag, command)
        self.assertNotIn("--uniform-preview", command)
        self.assertEqual(command[command.index("--max-gaussians") + 1], "0")

    def test_identical_explicit_settings_remain_compatible(self):
        renderer = self.renderer(threads=4, max_gaussians=0, batch=2, uniform_preview=False)
        self.assertTrue(renderer.runtime.configuration["direct_collect"])
        self.assertTrue(renderer.runtime.configuration["neon_pack"])

    def test_mainline_rejects_changed_or_unknown_settings_before_device_access(self):
        for options in ({"threads": 1}, {"max_gaussians": 16384}, {"neon_pack": False},
                        {"batch": True}, {"misspelled_option": 0}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "custom"):
                self.renderer(**options)
        self.verify.assert_not_called()
        self.popen.assert_not_called()

    def test_custom_preserves_explicit_preview_and_labels_evidence(self):
        with patch.object(LiveRenderer, "mainline") as factory:
            renderer = self.renderer(profile="custom", max_gaussians=16384, uniform_preview=True)
        factory.assert_not_called()
        self.assertEqual(renderer.runtime.configuration["profile"], "custom")
        self.assertEqual(renderer.runtime.configuration["max_gaussians"], 16384)
        self.assertIn("--uniform-preview", self.popen.call_args.args[0])
        self.assertNotIn("expected_boot_sha256", renderer.runtime.configuration)

    def test_invalid_profile_is_rejected_before_device_access(self):
        with self.assertRaisesRegex(ValueError, "profile"):
            self.renderer(profile="typo")
        self.verify.assert_not_called()
        self.popen.assert_not_called()

    def test_frame_archive_keeps_rgb_raw_and_mainline_configuration(self):
        renderer = self.renderer()
        raw = b"GSSOUT01" + struct.pack("<II", 2, 1) + bytes(40)
        rgb = bytes(range(6))
        header = dict(width=2, height=1, rgb_bytes=len(rgb), raw_bytes=len(raw))
        camera = b"FLCAM001" + struct.pack("<4I", 2, 1, 2, 1) + struct.pack("<14d", *([1.] * 14))
        self.line.side_effect = [b"SCENE 2 2 0.1", b"FRAME " + json.dumps(header).encode()]
        with patch.object(renderer.runtime, "_send") as send, \
                patch.object(renderer.runtime, "_exact", side_effect=[rgb, raw]):
            renderer.load_rows(np.zeros((2, 62), dtype=np.float32))
            record = renderer.render_camera(camera, self.root / "frame")
        self.assertEqual(send.call_args.args[0], b"RENDER_RAW\n" + camera)
        self.assertEqual(renderer.last_rgb.tobytes(), rgb)
        self.assertEqual((self.root / "frame/frame.bin").read_bytes(), raw)
        saved = json.loads((self.root / "frame/result.json").read_text())
        self.assertTrue(saved["configuration"]["direct_collect"])
        self.assertTrue(saved["configuration"]["neon_pack"])
        self.assertEqual(saved["configuration"]["profile"], load_mainline_profile()["id"])
        self.assertEqual(record["resident_scene"]["gaussians"], 2)
        self.assertIn("archived", record["endpoint"])

    def warm_args(self, **options):
        defaults = dict(backend="cpu", weights="unused", vendor="unused", threads=4,
                        prepare_threads=1, renderer=self.root / "frozen", live_renderer="native-renderer",
                        resident_binary="legacy-renderer", render_profile="mainline", render_threads=None,
                        render_batch=None, render_max_gaussians=None, render_uniform_preview=None)
        defaults.update(options)
        return SimpleNamespace(**defaults)

    def test_warm_session_records_mainline_and_closes_native_process(self):
        with patch("initialize.model_runtime.ModelRuntime") as model:
            model.return_value.load_seconds = 0.
            with WarmSession(self.warm_args(), self.root) as session:
                configuration = session.record["renderer_configuration"]
                self.assertEqual(configuration["profile"], load_mainline_profile()["id"])
                self.assertTrue(configuration["direct_collect"])
                self.assertTrue(configuration["neon_pack"])
        self.process.stdin.close.assert_called_once()

    def test_warm_session_inherits_profile_changes_without_duplicate_defaults(self):
        profile = load_mainline_profile()
        profile["runtime"]["batch"] = 4
        with patch("initialize.model_runtime.ModelRuntime") as model, \
                patch("rendering.runtime.load_mainline_profile", return_value=profile):
            model.return_value.load_seconds = 0.
            with WarmSession(self.warm_args(), self.root) as session:
                self.assertEqual(session.record["renderer_configuration"]["batch"], 4)

    def test_legacy_renderer_remains_available_without_native_path(self):
        with patch("initialize.model_runtime.ModelRuntime") as model, \
                patch("initialize.renderer_runtime.RendererRuntime") as legacy:
            model.return_value.load_seconds = 0.
            with WarmSession(self.warm_args(live_renderer=None), self.root) as session:
                self.assertNotIn("renderer_configuration", session.record)
            legacy.assert_called_once()
        self.popen.assert_not_called()

    def test_cli_rejects_lossy_mainline_before_model_or_platform_initialization(self):
        args = ["--out", str(self.root / "run"), "--weights", "unused", "--vendor", "unused",
                "--renderer", "unused", "--live-renderer", "native-renderer",
                "--render-max-gaussians", "16384"]
        errors = io.StringIO()
        with patch("initialize.run.platform.machine") as machine, \
                patch("sys.stderr", errors), self.assertRaises(SystemExit) as error:
            main(args)
        self.assertEqual(error.exception.code, 2)
        self.assertIn("--render-profile custom", errors.getvalue())
        self.assertFalse((self.root / "run").exists())
        machine.assert_not_called()

    def test_board_sync_transfers_profile_and_changed_entrypoints_together(self):
        from initialize import board_sync
        destination = "/root/fpga43dgs_reconstruction/test_profile_contract"
        with patch.object(board_sync.remote, "connect") as connect, \
                patch.object(board_sync.remote, "run", return_value=(0, "")), \
                patch("sys.argv", ["board_sync", "--dest", destination]), \
                patch("sys.stdout", io.StringIO()):
            board_sync.main()
        uploads = connect.return_value.open_sftp.return_value.put.call_args_list
        targets = {call.args[1] for call in uploads}
        for name in ("initialize/run.py", "initialize/session.py", "rendering/runtime.py",
                     "rendering/pipeline_adapter.py", "rendering/mainline.json"):
            self.assertIn(destination + "/mvsplat/" + name, targets)
        for call in uploads:
            self.assertTrue(Path(call.args[0]).is_file(), call.args[0])
        connect.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
