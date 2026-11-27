import unittest

from .camera_servo import (CONTROL, HEARTBEAT, PAN, TILT, STOP, UioServo,
                           finish_pulse, run_smoke, smoke_plan, validate_write)


class FakeDevice:
    def __init__(self):
        self.writes = []
        self.control = 0
        self.status = 0

    def write(self, offset, value):
        self.writes.append((offset, value))
        if offset == CONTROL:
            self.control = self.status = value
        if offset == STOP:
            self.control = self.status = 0

    def read(self, offset):
        return self.control if offset == CONTROL else self.status


class CameraServoTests(unittest.TestCase):
    def test_tilt_limit_rejects_out_of_range_before_mmio(self):
        for pulse in (1000, 1500, 2000):
            validate_write(TILT, pulse)
        for pulse in (-1, 999, 2001, 2500, 0xFFFFFFFF, 1.5):
            with self.assertRaises(ValueError):
                validate_write(TILT, pulse)
        with self.assertRaises(ValueError):
            validate_write(0, 1)
        with self.assertRaises(ValueError):
            validate_write(CONTROL, 4)

    def test_uio_path_validation_does_not_open_guessed_gpio_or_memory(self):
        for path in ('/dev/mem', '/dev/gpiochip0', '/dev/uio0/../mem', '/dev/uio'):
            with self.assertRaises(ValueError):
                UioServo(path)

    def test_physical_header_is_not_linux_gpio_number(self):
        self.assertEqual(smoke_plan("pan")["header_physical_pin"], 23)
        self.assertEqual(smoke_plan("tilt")["header_physical_pin"], 21)
        with self.assertRaises(ValueError):
            smoke_plan("both")

    def test_small_position_test_and_heartbeat(self):
        for axis, offset, mask in [("pan", PAN, 1), ("tilt", TILT, 2)]:
            device = FakeDevice()
            run_smoke(device, axis, sleep=lambda seconds: None)
            self.assertEqual(device.writes[0], (CONTROL, 0))
            self.assertIn((CONTROL, mask), device.writes)
            self.assertEqual(device.writes[-1], (CONTROL, 0))
            self.assertEqual(device.writes[-2], (STOP, 1))
            pulses = [value for address, value in device.writes if address == offset]
            self.assertEqual(min(pulses), 1450)
            self.assertEqual(max(pulses), 1550)
            self.assertTrue(all(1000 <= value <= 2000 for address, value in device.writes if address == TILT))
            self.assertTrue(all(abs(b - a) <= 2 for a, b in zip(pulses, pulses[1:])))
            self.assertGreater(sum(address == HEARTBEAT for address, _ in device.writes), 100)

    def test_interrupted_test_disables_output(self):
        device = FakeDevice()

        def interrupt(seconds):
            raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            run_smoke(device, "pan", sleep=interrupt)
        self.assertEqual(device.writes[-1], (CONTROL, 0))

    def test_stop_requires_current_and_requested_disable(self):
        device = FakeDevice()
        device.control = device.status = 2
        polls = []
        def request_stop(offset, value):
            self.assertEqual((offset, value), (STOP, 1))
            device.control = 0
        device.write = request_stop
        def frame_boundary(_seconds):
            polls.append(1)
            if len(polls) == 3:
                device.status = 0
        finish_pulse(device, frame_boundary)
        self.assertEqual(len(polls), 3)

    def test_stop_timeout_watchdog_and_write_validation(self):
        device = FakeDevice()
        device.control = device.status = 2
        device.write = lambda _offset, _value: None
        with self.assertRaisesRegex(RuntimeError, 'not acknowledged'):
            finish_pulse(device, lambda _seconds: None)
        device.status = 0x100
        with self.assertRaisesRegex(RuntimeError, 'watchdog'):
            finish_pulse(device, lambda _seconds: None)
        validate_write(STOP, 1)
        for value in (0, 2, 0xFFFFFFFF):
            with self.assertRaises(ValueError):
                validate_write(STOP, value)


if __name__ == "__main__":
    unittest.main()
