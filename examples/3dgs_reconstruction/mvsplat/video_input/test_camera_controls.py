import json
import re
import subprocess
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from video_input.camera_controls import CameraControls, parse_controls
from video_input.recording import CameraRecorder
from web.recording_server import make_server


class Driver:
    def __init__(self):
        self.values = dict(exposure_auto=3, exposure_absolute=157, gain=0, brightness=0)
        self.writes = []
        self.fail = None
        self.ignore = None

    def __call__(self, command, **kwargs):
        assert command[:3] == ['v4l2-ctl', '-d', '/dev/video0']
        if '--set-ctrl' in command:
            key, value = command[-1].split('='); value = int(value)
            self.writes.append((key, value))
            if self.fail == key:
                self.fail = None
                return SimpleNamespace(returncode=1, stdout='injected USB control error')
            if self.ignore != key:
                self.values[key] = value
            self.ignore = None
            return SimpleNamespace(returncode=0, stdout='')
        lines = [f" exposure_auto 0x009a0901 (menu) : min=0 max=3 value={self.values['exposure_auto']}",
                 '    1: Manual Mode', '    3: Aperture Priority Mode']
        for key, minimum, maximum in [('exposure_absolute', 1, 5000), ('gain', 0, 100), ('brightness', -64, 64)]:
            flag = ' flags=inactive' if key == 'exposure_absolute' and self.values['exposure_auto'] == 3 else ''
            lines.append(f' {key} 0x00980900 (int) : min={minimum} max={maximum} step=1 value={self.values[key]}{flag}')
        return SimpleNamespace(returncode=0, stdout='\n'.join(lines))


class ControlsTests(unittest.TestCase):
    def setUp(self):
        self.driver = Driver()
        self.device = CameraControls('/dev/video0', self.driver)

    def test_read_discovers_modes_and_preserves_signed_ranges(self):
        result = self.device.read()
        self.assertEqual(result['mode'], 'auto')
        self.assertEqual(result['exposure_unit_ms'], 0.1)
        self.assertEqual(result['controls']['brightness']['min'], -64)
        self.assertEqual(self.driver.writes, [])

    def test_non_whitelisted_controls_cannot_be_discovered_or_written(self):
        self.assertEqual(parse_controls('pan_absolute 0x009a0908 (int) : min=0 max=100 value=1'), {})
        for request in ({'pan_absolute': 1}, {}, [], {'mode': 'bad'}, {'mode': True}, {'gain': True}, {'gain': 0.5}, {'gain': -1}, {'gain': 101}, {'gain': '1;anything'}, {'exposure_absolute': 1}):
            with self.subTest(request=request), self.assertRaises(ValueError): self.device.apply(request)
        self.assertEqual(self.driver.writes, [])

    def test_manual_mode_precedes_exposure_and_signed_brightness(self):
        result = self.device.apply(dict(mode='manual', exposure_absolute=100, gain=25, brightness=-5))
        self.assertEqual(self.driver.writes, [('exposure_auto', 1), ('exposure_absolute', 100), ('gain', 25), ('brightness', -5)])
        self.assertEqual(result['mode'], 'manual')
        self.assertEqual(result['controls']['exposure_absolute']['value'], 100)

    def test_return_to_auto_does_not_write_exposure(self):
        self.device.apply(dict(mode='manual', exposure_absolute=80))
        self.driver.writes.clear()
        self.assertEqual(self.device.apply(dict(mode='auto'))['mode'], 'auto')
        self.assertEqual(self.driver.writes, [('exposure_auto', 3)])

    def test_validate_every_value_before_any_write(self):
        with self.assertRaises(ValueError): self.device.apply(dict(mode='manual', exposure_absolute=100, gain=101))
        self.assertEqual(self.driver.writes, [])

    def test_partial_failure_restores_old_mode_exposure_and_gain(self):
        initial = dict(self.driver.values); self.driver.fail = 'gain'
        with self.assertRaises(RuntimeError): self.device.apply(dict(mode='manual', exposure_absolute=100, gain=25))
        self.assertEqual(initial, self.driver.values)

    def test_readback_mismatch_is_not_reported_success(self):
        self.driver.ignore = 'exposure_auto'
        with self.assertRaises(RuntimeError): self.device.apply(dict(mode='manual'))
        self.assertEqual(self.driver.values['exposure_auto'], 3)

    def test_absent_menu_does_not_invent_a_manual_mode(self):
        runner = lambda *a, **k: SimpleNamespace(returncode=0, stdout='')
        with self.assertRaises(ValueError): CameraControls('/dev/video0', runner).apply({'mode': 'manual'})

    def test_recorder_locks_before_driver_writes(self):
        camera = object.__new__(CameraRecorder)
        camera.lock = threading.RLock(); camera.closed = False; camera.maintenance = False
        camera.state = 'idle'; camera.build = None; camera.camera_controls = self.device; camera.fps = 60
        for state in ('starting', 'recording', 'stopping', 'uploading'):
            camera.state = state
            with self.subTest(state=state), self.assertRaises(ValueError): camera.set_camera_controls({'gain': 10})
        camera.state = 'idle'
        for phase in ('initializing', 'reconstructing', 'streaming', 'stopping'):
            camera.build = SimpleNamespace(status=lambda: dict(phase=phase))
            with self.subTest(phase=phase), self.assertRaises(ValueError): camera.set_camera_controls({'gain': 10})
        self.assertEqual(self.driver.writes, [])
        camera.build = None
        self.assertTrue(camera.get_camera_controls()['editable'])
        self.assertEqual(camera.set_camera_controls({'gain': 10})['controls']['gain']['value'], 10)


class ControlsHttpTests(unittest.TestCase):
    def setUp(self):
        self.device = CameraControls('/dev/video0', Driver())
        camera = SimpleNamespace(build=None, get_camera_controls=self.device.read, set_camera_controls=self.device.apply)
        self.server = make_server(camera, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True); self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_port}'
        with urllib.request.urlopen(self.base+'/capture') as response:
            html = response.read().decode()
        self.token = re.search(r'const token=("[^"]+")', html).group(1)
        self.token = json.loads(self.token)

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join()

    def post(self, body, token=None):
        headers = {'Content-Type': 'application/json', 'X-Session-Token': token or ''}
        return urllib.request.urlopen(urllib.request.Request(self.base+'/api/camera-controls', data=body, headers=headers))

    def test_authentication_is_required_before_writes(self):
        with self.assertRaises(urllib.error.HTTPError) as caught: self.post(b'{"gain":25}')
        self.assertEqual(caught.exception.code, 403)
        self.assertEqual(self.device.read()['controls']['gain']['value'], 0)

    def test_get_js_and_authenticated_apply(self):
        with urllib.request.urlopen(self.base+'/camera_controls.js') as r:
            self.assertIn(b'/api/camera-controls', r.read())
        with urllib.request.urlopen(self.base+'/api/camera-controls') as r:
            self.assertEqual(json.load(r)['mode'], 'auto')
        with self.post(b'{"mode":"manual","exposure_absolute":100,"gain":25}', self.token) as r:
            result = json.load(r)
            self.assertEqual(result['mode'], 'manual')
            self.assertEqual(result['controls']['gain']['value'], 25)

    def test_malformed_and_oversized_json_is_rejected(self):
        for data in (b'bad json', b'"not an object"', b' '*4097):
            with self.subTest(size=len(data)), self.assertRaises(urllib.error.HTTPError) as caught:
                self.post(data, self.token)
            self.assertEqual(caught.exception.code, 409)
        self.assertEqual(self.device.read()['controls']['gain']['value'], 0)


if __name__ == '__main__': unittest.main()
