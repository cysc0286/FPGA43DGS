"""Generate a bounded position-servo scan; never opens hardware or a camera."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

PERIOD_MS = 20
MAX_STEP_US = 10
MAX_ROWS = 180000
LIMITS = {"pan": (500, 2500), "tilt": (1000, 2000)}


def number(value, label, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    if not low <= value <= high:
        raise ValueError(f"{label} must be within {low}..{high}")
    return float(value)


def keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{label} fields must be {', '.join(expected)}")


def generate_plan(profile):
    keys(profile, ("name", "calibration", "scan"), "profile")
    if not isinstance(profile["name"], str) or not profile["name"].strip():
        raise ValueError("profile name is required")
    calibration = profile["calibration"]
    keys(calibration, ("pan", "tilt"), "calibration")
    for axis, limits in LIMITS.items():
        c = calibration[axis]
        keys(c, ("kind", "us_per_degree", "direction", "min_deg", "max_deg"), axis)
        if c["kind"] != "position":
            raise ValueError("Continuous-rotation servos need speed/encoder control, not this position runner")
        number(c["us_per_degree"], axis+" us_per_degree", 0.1, 100)
        if isinstance(c["direction"], bool) or c["direction"] not in (-1, 1):
            raise ValueError("direction must be -1 or 1")
        angle_limit = 45 if axis == "tilt" else 180
        lo = number(c["min_deg"], axis+" min_deg", -angle_limit, 0)
        hi = number(c["max_deg"], axis+" max_deg", 0, angle_limit)
        if lo >= hi:
            raise ValueError("calibrated minimum must be below maximum")
        for angle in (lo, hi):
            pulse = 1500 + c["direction"]*c["us_per_degree"]*angle
            if not limits[0]-1e-8 <= pulse <= limits[1]+1e-8:
                raise ValueError(axis+" calibrated range exceeds FPGA pulse limits")

    s = profile["scan"]
    mode = s.get("mode") if isinstance(s, dict) else None
    common = ("mode", "pan_start_deg", "pan_end_deg", "pan_speed_deg_s",
              "tilt_speed_deg_s", "settle_s", "dwell_s", "start_hold_s", "end_hold_s", "cycles")
    extra = ("tilt_start_deg", "tilt_end_deg") if mode == "spiral" else ("pan_step_deg", "tilt_rows_deg")
    keys(s, common+extra, "scan")
    if mode not in ("raster", "sweep", "spiral"):
        raise ValueError("scan mode must be raster, sweep or spiral")
    pc, tc = calibration["pan"], calibration["tilt"]
    start = number(s["pan_start_deg"], "pan_start_deg", pc["min_deg"], pc["max_deg"])
    end = number(s["pan_end_deg"], "pan_end_deg", pc["min_deg"], pc["max_deg"])
    if start == end:
        raise ValueError("pan scan needs distinct endpoints")
    pan_speed = number(s["pan_speed_deg_s"], "pan_speed_deg_s", 0.1, 90)
    tilt_speed = number(s["tilt_speed_deg_s"], "tilt_speed_deg_s", 0.1, 45)
    for axis, speed in (("pan", pan_speed), ("tilt", tilt_speed)):
        if speed*calibration[axis]["us_per_degree"]*PERIOD_MS/1000 > MAX_STEP_US + 1e-8:
            raise ValueError(axis+" speed exceeds 10us per frame")
    for field in ("settle_s", "dwell_s", "start_hold_s", "end_hold_s"):
        number(s[field], field, 0, 60)
    if not isinstance(s["cycles"], int) or isinstance(s["cycles"], bool) or not 1 <= s["cycles"] <= 10:
        raise ValueError("cycles must be an integer within 1..10")
    if mode == "spiral":
        tilt_start = number(s["tilt_start_deg"], "tilt_start_deg", tc["min_deg"], tc["max_deg"])
        tilt_end = number(s["tilt_end_deg"], "tilt_end_deg", tc["min_deg"], tc["max_deg"])
        if tilt_start == tilt_end:
            raise ValueError("spiral requires a changing tilt angle; use sweep for fixed tilt")
        if s["cycles"] != 1:
            raise ValueError("spiral supports one unwrapped pass; repeated turns require verified hardware")
        rows = []
        columns = [start, end]
    else:
        step = number(s["pan_step_deg"], "pan_step_deg", 0.1, 360)
        if not isinstance(s["tilt_rows_deg"], list) or not 1 <= len(s["tilt_rows_deg"]) <= 25:
            raise ValueError("tilt_rows_deg requires 1..25 rows")
        rows = [number(x, "tilt row", tc["min_deg"], tc["max_deg"]) for x in s["tilt_rows_deg"]]
        if mode == "sweep":
            columns = [start, end]
        else:
            count = math.ceil(abs(end-start)/step - 1e-10)
            columns = [start + (1 if end > start else -1)*step*i for i in range(count)] + [end]
    points = []
    current = (0.0, 0.0)

    def pulse(axis, angle):
        c = calibration[axis]
        return int(math.floor(1500 + c["direction"]*c["us_per_degree"]*angle + 0.5))

    def append(pan, tilt, event):
        if len(points) >= MAX_ROWS:
            raise ValueError("scan exceeds the one-hour plan budget")
        p, t = pulse("pan", pan), pulse("tilt", tilt)
        if not 500 <= p <= 2500 or not 1000 <= t <= 2000:
            raise ValueError("generated pulse exceeds hardware limits")
        if points and max(abs(p-points[-1]["pan_us"]), abs(t-points[-1]["tilt_us"])) > MAX_STEP_US:
            raise ValueError("rounded pulse exceeds per-frame slew")
        points.append(dict(time_ms=(len(points)*PERIOD_MS), pan_us=p, tilt_us=t, event=event))

    def hold(seconds, event):
        for _ in range(math.ceil(seconds*1000/PERIOD_MS - 1e-10)):
            append(*current, event)

    def move(target, event="move"):
        nonlocal current
        old = current
        n = max(1, math.ceil(max(abs(target[0]-old[0])/pan_speed,
                                 abs(target[1]-old[1])/tilt_speed)*1000/PERIOD_MS - 1e-10))
        for i in range(1, n+1):
            append(old[0]+(target[0]-old[0])*i/n, old[1]+(target[1]-old[1])*i/n, event)
        current = target
        return n*PERIOD_MS/1000

    append(0, 0, "start")
    hold(s["start_hold_s"], "settle")
    captures = 0
    spiral_seconds = None
    if mode == "spiral":
        # Angles stay unwrapped. No modulo reset or hidden rapid return is
        # inserted in the recorded spiral segment, even for a nominal full turn.
        move((start, tilt_start))
        hold(s["settle_s"], "settle")
        append(*current, "capture")
        hold(s["dwell_s"], "settle")
        spiral_seconds = move((end, tilt_end))
        hold(s["settle_s"], "settle")
        append(*current, "capture")
        hold(s["dwell_s"], "settle")
        captures = 2
    else:
        for cycle in range(s["cycles"]):
            for row_id, tilt in enumerate(rows):
                order = columns if row_id % 2 == 0 else list(reversed(columns))
                for pan in order:
                    move((pan, tilt))
                    # A sweep pauses only at its ends; raster pauses at each column.
                    hold(s["settle_s"], "settle")
                    append(*current, "capture")
                    hold(s["dwell_s"], "settle")
                    captures += 1
    move((0, 0), "home")
    hold(s["end_hold_s"], "home")
    append(*current, "end")
    summary = {"name": profile["name"], "samples": len(points), "period_ms": PERIOD_MS,
               "nominal_duration_seconds": len(points)*PERIOD_MS/1000,
               "mode": s["mode"], "rows": rows, "capture_markers": captures,
               "pan_us_range": [min(p["pan_us"] for p in points), max(p["pan_us"] for p in points)],
               "tilt_us_range": [min(p["tilt_us"] for p in points), max(p["tilt_us"] for p in points)],
               "hardware_tested": False,
               "calibration_note": "Angles are requested nominal positions, not encoder measurements. Verify servo type and calibration.",
               "capture_note": "capture is a log marker; this program does not start the USB camera."}
    if mode == "spiral":
        summary["spiral"] = {
            "pan_start_deg": start, "pan_end_deg": end,
            "tilt_start_deg": tilt_start, "tilt_end_deg": tilt_end,
            "nominal_pan_revolutions": abs(end-start)/360,
            "nominal_motion_seconds": spiral_seconds,
            "nominal_pan_speed_deg_s": (end-start)/spiral_seconds,
            "nominal_tilt_speed_deg_s": (tilt_end-tilt_start)/spiral_seconds,
            "angles_unwrapped": True, "physical_360_verified": False,
            "note": "Angular spiral of viewing directions; no vertical translation or extra parallax is implied."}
    return points, summary


def write_plan(profile_path, output):
    profile_path, output = Path(profile_path), Path(output)
    raw = profile_path.read_bytes()
    profile = json.loads(raw.decode("utf-8-sig"))
    points, summary = generate_plan(profile)
    metadata = output.with_suffix(output.suffix+".json")
    if output.exists() or metadata.exists():
        raise FileExistsError("Choose new output paths; existing scan evidence is not overwritten")
    created = []
    try:
        with output.open("x", encoding="ascii", newline="") as handle:
            created.append(output)
            writer = csv.DictWriter(handle, fieldnames=("time_ms", "pan_us", "tilt_us", "event"))
            writer.writeheader()
            writer.writerows(points)
        summary.update(profile_sha256=hashlib.sha256(raw).hexdigest(),
                       csv_sha256=hashlib.sha256(output.read_bytes()).hexdigest(), profile=profile)
        with metadata.open("x", encoding="utf-8") as handle:
            created.append(metadata)
            json.dump(summary, handle, ensure_ascii=False, indent=2)
    except BaseException:
        for path in created:
            path.unlink()
        raise
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=Path(__file__).with_name("camera_scan_bow.json"))
    parser.add_argument("--output", type=Path, required=True, help="New CSV file; does not execute motion")
    args = parser.parse_args()
    try:
        summary = write_plan(args.profile, args.output)
    except (ValueError, OSError) as error:
        parser.exit(1, str(error)+"\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
