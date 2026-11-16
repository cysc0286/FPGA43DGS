import copy
import json
import tempfile
import unittest
from pathlib import Path

from .camera_scan import generate_plan, write_plan


class CameraScanTests(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads(Path(__file__).with_name("camera_scan_profile.json").read_text(encoding="utf-8"))

    def test_default_plan_limits_slew_and_home(self):
        points, summary = generate_plan(self.profile)
        self.assertEqual(summary["capture_markers"], 5)
        self.assertEqual((points[0]["pan_us"], points[0]["tilt_us"]), (1500, 1500))
        self.assertEqual((points[-1]["pan_us"], points[-1]["tilt_us"]), (1500, 1500))
        self.assertEqual(points[-1]["event"], "end")
        for i, point in enumerate(points):
            self.assertEqual(point["time_ms"], i*20)
            self.assertTrue(500 <= point["pan_us"] <= 2500)
            self.assertTrue(1000 <= point["tilt_us"] <= 2000)
            if i:
                self.assertLessEqual(abs(point["pan_us"]-points[i-1]["pan_us"]), 10)
                self.assertLessEqual(abs(point["tilt_us"]-points[i-1]["tilt_us"]), 10)

    def test_multirow_serpentine_and_nominal_pitch_bounds(self):
        self.profile["scan"]["tilt_rows_deg"] = [-45, 0, 45]
        points, summary = generate_plan(self.profile)
        captures = [p for p in points if p["event"] == "capture"]
        self.assertEqual(summary["capture_markers"], 15)
        self.assertLess(captures[0]["pan_us"], captures[4]["pan_us"])
        self.assertGreater(captures[5]["pan_us"], captures[9]["pan_us"])
        self.assertEqual(summary["tilt_us_range"], [1000, 2000])

    def test_sweep_does_not_stop_at_intermediate_columns(self):
        self.profile["scan"]["mode"] = "sweep"
        _, summary = generate_plan(self.profile)
        self.assertEqual(summary["capture_markers"], 2)

    def test_reverse_direction_and_exact_unaligned_endpoint(self):
        self.profile["calibration"]["pan"]["direction"] = -1
        self.profile["scan"].update(pan_start_deg=25, pan_end_deg=-12, pan_step_deg=10)
        points, _ = generate_plan(self.profile)
        captures = [p for p in points if p["event"] == "capture"]
        self.assertEqual(len(captures), 5)
        self.assertEqual(captures[-1]["pan_us"], 1633)

    def test_invalid_profiles_rejected(self):
        cases = [
            ("scan", "tilt_rows_deg", [46]), ("scan", "pan_start_deg", -61),
            ("scan", "pan_speed_deg_s", 90), ("scan", "cycles", True),
            ("scan", "pan_step_deg", 0), ("scan", "settle_s", float("nan")),
            ("scan", "tilt_rows_deg", []), ("scan", "mode", "unknown"),
        ]
        for section, key, value in cases:
            with self.subTest(key=key):
                profile = copy.deepcopy(self.profile)
                profile[section][key] = value
                with self.assertRaises(ValueError):
                    generate_plan(profile)
        for key, value in (("kind", "continuous"), ("direction", True), ("us_per_degree", 100)):
            profile = copy.deepcopy(self.profile)
            profile["calibration"]["pan"][key] = value
            with self.assertRaises(ValueError):
                generate_plan(profile)

    def test_csv_hash_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/"scan.csv"
            profile = Path(__file__).with_name("camera_scan_profile.json")
            summary = write_plan(profile, out)
            self.assertEqual(out.read_text().splitlines()[0], "time_ms,pan_us,tilt_us,event")
            self.assertEqual(len(out.read_text().splitlines())-1, summary["samples"])
            self.assertTrue(out.with_suffix(".csv.json").is_file())
            before = out.read_bytes()
            with self.assertRaises(FileExistsError):
                write_plan(profile, out)
            self.assertEqual(before, out.read_bytes())

    def test_duration_budget(self):
        self.profile["scan"].update(pan_speed_deg_s=0.1, tilt_rows_deg=[-45, 0, 45], cycles=10)
        with self.assertRaisesRegex(ValueError, "one-hour"):
            generate_plan(self.profile)


class BowScanTests(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads(Path(__file__).with_name("camera_scan_bow.json").read_text(encoding="utf-8"))

    def test_three_rows_and_alternating_endpoints(self):
        points, summary = generate_plan(self.profile)
        markers = [p for p in points if p["event"] == "capture"]
        self.assertEqual(summary["rows"], [30, 0, -30])
        self.assertEqual(summary["capture_markers"], 6)
        self.assertEqual([p["pan_us"] for p in markers], [1167, 1833, 1833, 1167, 1167, 1833])
        self.assertEqual([p["tilt_us"] for p in markers], [1833, 1833, 1500, 1500, 1167, 1167])
        self.assertEqual(summary["tilt_us_range"], [1167, 1833])
        self.assertEqual((points[-1]["pan_us"], points[-1]["tilt_us"]), (1500, 1500))

    def test_pitch_changes_only_at_horizontal_ends(self):
        points, _ = generate_plan(self.profile)
        markers = [i for i, p in enumerate(points) if p["event"] == "capture"]
        for a, b in zip(points[markers[0]:markers[-1]], points[markers[0]+1:markers[-1]+1]):
            dp = b["pan_us"]-a["pan_us"]
            dt = b["tilt_us"]-a["tilt_us"]
            self.assertFalse(dp and dt, "Both axes changed inside a bow sweep")
            if dt:
                self.assertIn(a["pan_us"], (1167, 1833))
                self.assertLess(dt, 0)

    def test_horizontal_rows_never_pause_in_the_middle(self):
        points, _ = generate_plan(self.profile)
        markers = [i for i, p in enumerate(points) if p["event"] == "capture"]
        for row in range(3):
            begin, end = markers[2*row:2*row+2]
            moving = [p for p in points[begin+1:end] if p["event"] == "move"]
            self.assertEqual(len(moving), 150)
            self.assertEqual(len({p["tilt_us"] for p in moving}), 1)
            sign = 1 if row % 2 == 0 else -1
            for a, b in zip(moving, moving[1:]):
                self.assertGreater(sign*(b["pan_us"]-a["pan_us"]), 0)

    def test_fast15_preserves_coverage_slew_and_slow_fallback(self):
        slow = json.loads(Path(__file__).with_name("camera_scan_bow_conservative.json").read_text(encoding="utf-8"))
        fast = json.loads(Path(__file__).with_name("camera_scan_bow_fast15.json").read_text(encoding="utf-8"))
        points, summary = generate_plan(fast)
        _, previous = generate_plan(slow)
        self.assertEqual(summary['samples'], 583)
        self.assertAlmostEqual(summary['nominal_duration_seconds'], 11.66)
        self.assertAlmostEqual(previous['nominal_duration_seconds'], 46.06)
        for field in ('pan_start_deg', 'pan_end_deg', 'tilt_rows_deg'):
            self.assertEqual(fast['scan'][field], slow['scan'][field])
        for a, b in zip(points, points[1:]):
            self.assertLessEqual(abs(a['pan_us']-b['pan_us']), 7)
            self.assertLessEqual(abs(a['tilt_us']-b['tilt_us']), 7)

    def test_balanced20_retains_three_rows_and_fits_recording_window(self):
        points, summary = generate_plan(self.profile)
        self.assertEqual(summary['samples'], 833)
        self.assertAlmostEqual(summary['nominal_duration_seconds'], 16.66)
        self.assertTrue(15 <= summary['nominal_duration_seconds'] + 1 <= 20)
        self.assertLessEqual(summary['nominal_duration_seconds'] + 3, 20)
        for a, b in zip(points, points[1:]):
            self.assertLessEqual(abs(a['pan_us']-b['pan_us']), 5)
            self.assertLessEqual(abs(a['tilt_us']-b['tilt_us']), 5)


class SpiralScanTests(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads(Path(__file__).with_name("camera_scan_spiral.json").read_text(encoding="utf-8"))

    def test_spiral_changes_both_axes_continuously(self):
        points, summary = generate_plan(self.profile)
        markers = [i for i, p in enumerate(points) if p["event"] == "capture"]
        self.assertEqual(len(markers), 2)
        segment = [p for p in points[markers[0]+1:markers[1]] if p["event"] == "move"]
        self.assertEqual(len(segment), 600)
        for a, b in zip(segment, segment[1:]):
            self.assertGreaterEqual(b["pan_us"], a["pan_us"])
            self.assertLessEqual(b["tilt_us"], a["tilt_us"])
            self.assertLessEqual(b["pan_us"]-a["pan_us"], 3)
            self.assertLessEqual(a["tilt_us"]-b["tilt_us"], 2)
        self.assertEqual(summary["spiral"]["nominal_motion_seconds"], 12)
        self.assertEqual(summary["tilt_us_range"], [1167, 1833])
        self.assertEqual(summary["spiral"]["nominal_pan_speed_deg_s"], 5)
        self.assertEqual(summary["spiral"]["nominal_tilt_speed_deg_s"], -5)
        self.assertAlmostEqual(summary["spiral"]["nominal_pan_revolutions"], 1/6)
        self.assertFalse(summary["spiral"]["physical_360_verified"])
        self.assertEqual((points[-1]["pan_us"], points[-1]["tilt_us"]), (1500, 1500))

    def test_reverse_spiral_and_ascent(self):
        self.profile["scan"].update(pan_start_deg=30, pan_end_deg=-30, tilt_start_deg=-30, tilt_end_deg=30)
        _, summary = generate_plan(self.profile)
        self.assertLess(summary["spiral"]["nominal_pan_speed_deg_s"], 0)
        self.assertGreater(summary["spiral"]["nominal_tilt_speed_deg_s"], 0)

    def test_slow_axis_limits_both_to_keep_path_synchronized(self):
        self.profile["scan"]["tilt_speed_deg_s"] = 2
        _, summary = generate_plan(self.profile)
        self.assertEqual(summary["spiral"]["nominal_motion_seconds"], 30)
        self.assertEqual(summary["spiral"]["nominal_pan_speed_deg_s"], 2)

    def test_nominal_full_turn_requires_explicit_position_calibration(self):
        self.profile["calibration"]["pan"].update(us_per_degree=2000/360, min_deg=-180, max_deg=180)
        self.profile["scan"].update(pan_start_deg=-180, pan_end_deg=180)
        points, summary = generate_plan(self.profile)
        markers = [p for p in points if p["event"] == "capture"]
        self.assertEqual([p["pan_us"] for p in markers], [500, 2500])
        self.assertEqual(summary["spiral"]["nominal_pan_revolutions"], 1)
        self.assertFalse(summary["spiral"]["physical_360_verified"])

    def test_no_hidden_wraparound_or_flat_spiral(self):
        for changes in (dict(pan_end_deg=390), dict(cycles=2), dict(tilt_end_deg=30), dict(tilt_start_deg=46)):
            with self.subTest(changes=changes):
                profile = copy.deepcopy(self.profile)
                profile["scan"].update(changes)
                with self.assertRaises(ValueError):
                    generate_plan(profile)

    def test_original_raster_plan_size_unchanged(self):
        profile = json.loads(Path(__file__).with_name("camera_scan_profile.json").read_text(encoding="utf-8"))
        points, summary = generate_plan(profile)
        self.assertEqual(len(points), 777)
        self.assertEqual(summary["nominal_duration_seconds"], 15.54)


if __name__ == "__main__":
    unittest.main()
