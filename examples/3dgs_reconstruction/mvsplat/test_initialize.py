"""Lifecycle accounting for warm board execution."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import threading
import struct
import hashlib
import io
import numpy as np

from common import load_gaussians, validate_gaussians
from initialize.events import EventLog
from initialize.model_runtime import ModelRuntime
from initialize.renderer_runtime import framebuffer_rgb, frozen_renderer_module
from warm_pipeline import overlap_prepare_infer
from video_input.receipt import receive_file


class FramebufferTests(unittest.TestCase):
    def test_bulk_display_matches_frozen_scalar_rounding(self):
        reference = frozen_renderer_module(Path(__file__).resolve().parents[3] /
            "releases/3dgs_renderer_v1_20260928")
        samples = np.random.default_rng(103).uniform(-0.2, 1.2, (1024, 4)).astype(np.float32)
        samples[:3, :3] = [[0, 0.5, 1], [0.5/255, 1.5/255, 254.5/255], [-1, 2, -0.0]]
        payload = b"GSSOUT01"+struct.pack("<II", 32, 32)+b"".join(
            struct.pack("<4fI", *sample, 0xffffffff) for sample in samples)
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory)/"frame.bin", Path(directory)/"frame.ppm"
            source.write_bytes(payload)
            reference.write_ppm(source, output)
            width, height, rgb = framebuffer_rgb(payload)
            self.assertEqual(output.read_bytes(), f"P6\n{width} {height}\n255\n".encode()+rgb.tobytes())

    def test_bad_frame_is_rejected(self):
        good = b"GSSOUT01"+struct.pack("<II4fI", 1, 1, 0, 0, 0, 1, 0)
        for raw in (good[:-1], good+b"x", b"wrong", good[:16]+struct.pack("<4fI", 0, 0, 0, float("nan"), 0)):
            with self.assertRaises(ValueError):
                framebuffer_rgb(raw)


class EventLogTests(unittest.TestCase):
    def test_clock_uses_monotonic_and_cannot_be_overridden(self):
        with tempfile.TemporaryDirectory() as directory:
            log = EventLog(Path(directory) / "events.json")
            with self.assertRaises(ValueError):
                log.emit("READY", monotonic=0)
            with patch("initialize.events.time.monotonic", side_effect=[10., 100., 125., 130.]), \
                    patch("initialize.events.time.time", side_effect=[1000., 900., 800., 700.]):
                for name in ("READY", "VIDEO_COMPLETE", "SCENE_READY", "FRAME_COMPLETE"):
                    log.emit(name)
            self.assertEqual(log.summary()["scene_preparation_seconds"], 30.)
            self.assertNotIn("video_input_seconds_record_only", log.summary())

    def test_preexisting_file_receipt_is_not_video_capture_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = EventLog(root/"events.json")
            log.emit("READY")
            (root/"video.mp4").write_bytes(b"completed input")
            record = receive_file(root/"video.mp4", log)
            self.assertEqual(len(record["sha256"]), 64)
            self.assertIsNone(record["video_input_seconds_record_only"])
            self.assertFalse(log.summary()["complete"])

    def test_main_clock_excludes_preheat_and_video_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.json"
            log = EventLog(path)
            for name in ("READY", "VIDEO_COMPLETE", "SCENE_READY", "FRAME_COMPLETE"):
                log.emit(name)
            result = log.summary()
            self.assertTrue(result["complete"])
            self.assertAlmostEqual(result["scene_preparation_seconds"],
                                   result["gaussian_scene_seconds"] + result["first_render_seconds"])
            self.assertEqual([item["event"] for item in json.loads(path.read_text())["events"]],
                             ["READY", "VIDEO_COMPLETE", "SCENE_READY", "FRAME_COMPLETE"])

    def test_missing_or_repeated_event_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            log = EventLog(Path(directory) / "events.json")
            with self.assertRaises(ValueError):
                log.emit("VIDEO_COMPLETE")
            log.emit("READY")
            with self.assertRaises(ValueError):
                log.emit("READY")
            with self.assertRaises(ValueError):
                log.emit("FRAME_COMPLETE")


class OverlapTests(unittest.TestCase):
    def test_inference_stays_main_thread_and_overlaps_pose_work(self):
        entered = threading.Event()
        main_thread = threading.get_ident()
        def prepare(args, on_context_ready):
            self.assertNotEqual(threading.get_ident(), main_thread)
            on_context_ready("input", "digest")
            self.assertTrue(entered.wait(2))
        def infer(directory, out, digest):
            self.assertEqual(threading.get_ident(), main_thread)
            self.assertEqual((directory, digest), ("input", "digest"))
            entered.set()
            return dict(complete=True)
        result, timings = overlap_prepare_infer(prepare, [], infer, "out")
        self.assertTrue(result["complete"])
        self.assertGreaterEqual(timings["combined_seconds"], timings["inference_end_seconds"])

    def test_first_target_callback_is_published_before_remaining_pose_work(self):
        entered = threading.Event()
        seen = []
        def prepare(args, on_context_ready, on_first_target_ready):
            on_context_ready("input", "digest")
            on_first_target_ready("input", "digest", 7)
            self.assertTrue(entered.wait(2))
        def infer(directory, out, digest):
            entered.set()
            return dict(complete=True, gaussian_sha256="g")
        def first(directory, digest, target, result, timing):
            seen.append((directory, digest, target, result["complete"]))
        result, timings = overlap_prepare_infer(prepare, [], infer, "out", first)
        self.assertTrue(result["complete"])
        self.assertEqual(seen, [("input", "digest", 7, True)])
        self.assertIn("first_target_ready_seconds", timings)

    def test_failed_pose_never_publishes_successful_scene(self):
        def prepare(args, on_context_ready):
            on_context_ready("input", "digest")
            raise ValueError("PnP rejected")
        with self.assertRaisesRegex(ValueError, "PnP rejected"):
            overlap_prepare_infer(prepare, [], lambda *args: dict(complete=True), "out")

    def test_no_context_failure_does_not_deadlock(self):
        with self.assertRaisesRegex(ValueError, "did not publish"):
            overlap_prepare_infer(lambda *a, **k: None, [], lambda *a: None, "out")

    def test_duplicate_context_is_rejected(self):
        def prepare(args, on_context_ready):
            on_context_ready("input", "one")
            on_context_ready("input", "two")
        with self.assertRaisesRegex(ValueError, "more than once"):
            overlap_prepare_infer(prepare, [], lambda *args: dict(complete=True), "out")


class GaussianHandoffTests(unittest.TestCase):
    def test_deferred_archive_is_exact_and_rejects_changed_parameters(self):
        gaussian = dict(means=np.zeros((1, 3), np.float32),
                        covariances=np.eye(3, dtype=np.float32)[None],
                        harmonics=np.zeros((1, 3, 25), np.float32),
                        opacities=np.array([0.5], np.float32))
        stream = io.BytesIO()
        np.savez(stream, **gaussian)
        payload = stream.getvalue()
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.savez(root/"reference.npz", **gaussian)
            self.assertEqual((root/"reference.npz").read_bytes(), payload)
            runtime = ModelRuntime.__new__(ModelRuntime)
            runtime.np = np
            runtime._pending_archive = (root, payload, digest)
            record = dict(gaussian_sha256=digest, archive_verified=False)
            gaussian["means"][0, 0] = 1
            with self.assertRaisesRegex(ValueError, "differs from rendered"):
                runtime.persist_gaussians(record, gaussian)
            self.assertFalse(record["archive_verified"])
            gaussian["means"][0, 0] = 0
            runtime.persist_gaussians(record, gaussian)
            self.assertTrue(record["archive_verified"])
            self.assertIsNone(runtime._pending_archive)
            with self.assertRaisesRegex(ValueError, "No matching"):
                runtime.persist_gaussians(record, gaussian)

    def test_memory_and_npz_share_the_validation_contract(self):
        gaussian = dict(means=np.zeros((1, 3), np.float32),
                        covariances=np.eye(3, dtype=np.float32)[None],
                        harmonics=np.zeros((1, 3, 25), np.float32),
                        opacities=np.array([0.5], np.float32))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gaussians.npz"
            np.savez(path, **gaussian)
            self.assertTrue(all(np.array_equal(validate_gaussians(gaussian)[key],
                                               load_gaussians(path)[key]) for key in gaussian))
            gaussian["opacities"][0] = np.nan
            with self.assertRaisesRegex(ValueError, "opacities"):
                validate_gaussians(gaussian)

    def test_pending_gaussians_require_matching_hash_and_are_consumed_once(self):
        runtime = ModelRuntime.__new__(ModelRuntime)
        runtime._pending_gaussians = ("digest", {"means": "sample"})
        with self.assertRaisesRegex(ValueError, "No validated"):
            runtime.take_gaussians("different")
        self.assertEqual(runtime.take_gaussians("digest"), {"means": "sample"})
        with self.assertRaisesRegex(ValueError, "No validated"):
            runtime.take_gaussians("digest")


if __name__ == "__main__":
    unittest.main()
