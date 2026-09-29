"""Lifecycle accounting for warm board execution."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import threading

from initialize.events import EventLog
from warm_pipeline import overlap_prepare_infer
from video_input.receipt import receive_file


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


if __name__ == "__main__":
    unittest.main()
