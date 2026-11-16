"""Camera-rig servo control. Default: preview only; never guess Linux GPIO numbers.

Hardware execution requires a verified UIO mapping of camera_servo_pwm registers.
The current frozen rendering BOOT does not expose this block.
"""
from __future__ import annotations

import argparse
import json
import mmap
import os
import re
import struct
import time

MAGIC = 0x53525631
ABI = 2
CONTROL, PAN, TILT, HEARTBEAT = 0x10, 0x14, 0x18, 0x1C
PULSE_LIMITS = {PAN: (500, 2500), TILT: (1000, 2000)}


def validate_write(offset: int, value: int) -> None:
    if not isinstance(value, int) or not 0 <= value <= 0xFFFFFFFF:
        raise ValueError("register value must be an unsigned 32-bit integer")
    if offset in PULSE_LIMITS:
        low, high = PULSE_LIMITS[offset]
        if not low <= value <= high:
            raise ValueError(f"pulse must be within {low}..{high} us")
    elif offset == CONTROL:
        if value & ~3:
            raise ValueError("only the pan/tilt enable bits may be set")
    elif offset != HEARTBEAT:
        raise ValueError("register is not writable")


def smoke_plan(axis: str) -> dict:
    if axis not in ("pan", "tilt"):
        raise ValueError("test one axis at a time")
    return {"axis": axis, "header_physical_pin": 23 if axis == "pan" else 21,
            "ground_physical_pin": 2, "period_us": 20000,
            "tilt_limit_degrees": [-45, 45], "tilt_limit_us": [1000, 2000],
            "pulse_sequence_us": [1500, 1450, 1500, 1550, 1500],
            "slew_us_per_frame": 2, "hold_seconds": 0.3,
            "note": "Position-servo smoke test, not a continuous 360-degree scan."}


class UioServo:
    def __init__(self, path: str):
        if re.fullmatch(r"/dev/uio[0-9]+", path) is None:
            raise ValueError("specify the verified /dev/uioN device; /dev/mem is not supported")
        self.fd = os.open(path, os.O_RDWR | os.O_SYNC)
        self.mem = None
        try:
            self.mem = mmap.mmap(self.fd, 4096)
            if self.read(0) != MAGIC or self.read(4) != ABI:
                raise RuntimeError("UIO identity/ABI mismatch; no register was written")
            if self.read(0x0C) != 20000 or self.read(0x30) < 100000:
                raise RuntimeError("unexpected frame/watchdog configuration; no register was written")
            for offset, register in ((PAN, 0x34), (TILT, 0x38)):
                low, high = PULSE_LIMITS[offset]
                if self.read(register) != low | high << 16:
                    raise RuntimeError("hardware pulse limits differ from the calibrated profile; no register was written")
        except BaseException:
            self.close()
            raise

    def read(self, offset: int) -> int:
        return struct.unpack_from("<I", self.mem, offset)[0]

    def write(self, offset: int, value: int) -> None:
        validate_write(offset, value)
        struct.pack_into("<I", self.mem, offset, value)

    def close(self) -> None:
        if self.mem is not None:
            self.mem.close()
            self.mem = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


def run_smoke(device, axis: str, sleep=time.sleep) -> None:
    plan = smoke_plan(axis)
    address, mask = (PAN, 1) if axis == "pan" else (TILT, 2)
    try:
        device.write(CONTROL, 0)
        device.write(PAN, 1500)
        device.write(TILT, 1500)
        device.write(HEARTBEAT, 1)
        device.write(CONTROL, mask)
        current = 1500
        for target in plan["pulse_sequence_us"]:
            while current != target:
                current += max(-2, min(2, target - current))
                device.write(address, current)
                device.write(HEARTBEAT, 1)
                sleep(0.02)
            for _ in range(15):
                device.write(HEARTBEAT, 1)
                sleep(0.02)
    finally:
        device.write(CONTROL, 0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axis", choices=("pan", "tilt"), default="pan")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--uio", help="verified UIO device of the newly integrated PWM block")
    args = parser.parse_args()
    plan = smoke_plan(args.axis)
    if not args.execute:
        print(json.dumps({"mode": "dry_run", **plan}, ensure_ascii=False, indent=2))
        return
    if not args.uio:
        parser.error("--execute requires a verified --uio device; the frozen BOOT is unsupported")
    device = UioServo(args.uio)
    try:
        run_smoke(device, args.axis)
    finally:
        device.close()
    print("Completed the command sequence; physical motion must be observed separately.")


if __name__ == "__main__":
    main()
