"""Loopback HTTP and camera ABI checks; fake renderer never accesses FPGA."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np

from rendering.runtime import LiveFrame
from rendering.viewer import make_server, orbit_view, verify_saved_scene


BASE = dict(c2w=np.eye(4).tolist(), intrinsics_normalized=[[1, 0, .5], [0, 1, .5], [0, 0, 1]],
            width=128, height=128)


class ViewerTests(unittest.TestCase):
    def test_saved_scene_requires_model_gaussian_and_context_hashes(self):
        from common import sha
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('renderer_input','inference','input'):(root/name).mkdir()
            model=root/'renderer_input/model.ply';model.write_bytes(b'model')
            gaussians=root/'inference/gaussians.npz';gaussians.write_bytes(b'gaussians')
            context=root/'input/context.npz';context.write_bytes(b'context')
            (root/'renderer_input/manifest.json').write_text(json.dumps(dict(model_sha256=sha(model))))
            (root/'inference/inference.json').write_text(json.dumps(dict(complete=True,gaussian_sha256=sha(gaussians),input_sha256=sha(context))))
            (root/'input/input.json').write_text(json.dumps(dict(context_sha256=sha(context))))
            self.assertEqual(verify_saved_scene(root),root.resolve())
            for path in (model,gaussians,context):
                original=path.read_bytes();path.write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError,'validation'):verify_saved_scene(root)
                path.write_bytes(original)

    def test_orbit_identity_and_proper_camera_axes(self):
        np.testing.assert_allclose(orbit_view(BASE, {}, 10)["c2w"], np.eye(4), atol=1e-12)
        for yaw, pitch in ((30, -20), (-30, 20)):
            view = np.array(orbit_view(BASE, dict(yaw=yaw, pitch=pitch), 10)["c2w"])
            rotation = view[:3, :3]
            np.testing.assert_allclose(rotation.T@rotation, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(rotation), 1)
            np.testing.assert_allclose(view[:3, 3]+10*rotation[:, 2], [0, 0, 10], atol=1e-12)
        for controls in ({"distance": -1}, {"yaw": float("nan")}, {"request_id": True}, {"bad": 1}):
            with self.assertRaises(ValueError):
                orbit_view(BASE, controls, 10)

    def test_live_http_pixels_authentication_errors_and_quit(self):
        class FakeRenderer:
            calls = []
            def render_camera(self, camera):
                self.calls.append(camera)
                image = np.zeros((128, 128, 3), np.uint8)
                image[:, :, 0] = 230
                return LiveFrame(image, dict(wall_ms=1.25))
        renderer = FakeRenderer()
        server = make_server(renderer, BASE, 10, port=0)
        def run():
            while not server.stopping:
                server.handle_request()
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{server.server_port}"
        def post(path, payload, token=None):
            request = Request(url+path, data=json.dumps(payload).encode(), headers={
                "Authorization": "Bearer "+(token or server.session_token), "Content-Type": "application/json"})
            return urlopen(request, timeout=5)
        try:
            with urlopen(url, timeout=5) as response:
                self.assertIn("createImageBitmap", response.read().decode())
            with self.assertRaises(HTTPError) as failed:
                post("/render", {}, "bad")
            self.assertEqual(failed.exception.code, 403)
            with self.assertRaises(HTTPError) as failed:
                post("/render", {"yaw": 99})
            self.assertEqual(failed.exception.code, 400)
            with post("/render", {"request_id": 7, "yaw": 10}) as response:
                pixels = cv2.imdecode(np.frombuffer(response.read(), np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(response.headers["X-Request-Id"], "7")
                self.assertEqual(pixels[0, 0].tolist(), [0, 0, 230])
            self.assertEqual(len(renderer.calls), 1)
            self.assertEqual((len(renderer.calls[0]), renderer.calls[0][:8]), (136, b"FLCAM001"))
            with post("/quit", {}) as response:
                self.assertEqual(response.status, 200)
            thread.join(5)
            self.assertFalse(thread.is_alive())
        finally:
            server.stopping = True
            thread.join(2)
            server.server_close()

    def test_renderer_timeout_stops_session_instead_of_reusing_a_late_reply(self):
        class FailingRenderer:
            def render_camera(self, camera):
                raise TimeoutError("test delayed native reply")
        server = make_server(FailingRenderer(), BASE, 10, port=0)
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        try:
            request = Request(f"http://127.0.0.1:{server.server_port}/render", data=b"{}",
                              headers={"Authorization": "Bearer "+server.session_token})
            with self.assertRaises(HTTPError) as error:
                urlopen(request, timeout=5)
            self.assertEqual(error.exception.code, 500)
            thread.join(5)
            self.assertTrue(server.stopping)
            self.assertIn("TimeoutError", server.stats["fatal_error"])
        finally:
            server.server_close()


if __name__ == "__main__":
    unittest.main()
