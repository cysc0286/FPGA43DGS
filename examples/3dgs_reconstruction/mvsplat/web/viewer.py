"""Small HTTP camera controller over a resident LiveRenderer; no scene regeneration."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import secrets
import time

import cv2
import numpy as np

from rendering.camera import camera_view, orbit_view


PAGE = Path(__file__).with_name('viewer.html').read_text(encoding='utf-8')






def viewer_metadata(metadata=None):
    """Whitelist and validate optional manifest data before exposing it to UI."""
    source = metadata or {}
    result = {}
    if 'capture_center' in source:
        center = np.asarray(source['capture_center'], dtype=np.float64)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError('Invalid capture_center')
        result['capture_center'] = center.tolist()
    default = source.get('default_view_mode', 'orbit')
    if default not in ('look', 'orbit') or default == 'look' and 'capture_center' not in result:
        raise ValueError('Invalid default_view_mode or missing capture_center')
    result['default_view_mode'] = default
    if 'depth_observable' in source:
        if type(source['depth_observable']) is not bool:
            raise ValueError('Invalid depth_observable')
        result['depth_observable'] = source['depth_observable']
    if 'coverage' in source:
        coverage = source['coverage']
        if not isinstance(coverage, dict):
            raise ValueError('Invalid coverage')
        directions = np.asarray(coverage.get('directions', []), dtype=np.float64)
        if not directions.size:
            directions = np.empty((0, 3))
        if (directions.ndim != 2 or directions.shape[1] != 3 or len(directions) > 4096
                or not np.isfinite(directions).all() or np.any(np.linalg.norm(directions, axis=1) < 1e-8)):
            raise ValueError('Invalid coverage directions')
        selected = dict(directions=directions.tolist())
        for key, limit in (('crop_horizontal_fov_degrees', 180), ('observed_yaw_span_degrees', 720)):
            if key in coverage:
                value = coverage[key]
                if type(value) not in (int, float) or not np.isfinite(value) or not 0 < value <= limit:
                    raise ValueError('Invalid coverage '+key)
                selected[key] = value
        if 'pair_groups' in coverage:
            groups = coverage['pair_groups']
            if type(groups) is not int or not 1 <= groups <= 32:
                raise ValueError('Invalid coverage pair_groups')
            selected['pair_groups'] = groups
        for key in ('sample_covered_fraction', 'coverage_target', 'achieved_fraction'):
            if key in coverage:
                value = coverage[key]
                if type(value) not in (int, float) or not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError('Invalid coverage '+key)
                selected[key] = value
        if 'max_continuous_gap_degrees' in coverage:
            gap = coverage['max_continuous_gap_degrees']
            if type(gap) not in (int, float) or not np.isfinite(gap) or not 0 <= gap <= 360:
                raise ValueError('Invalid coverage maximum gap')
            selected['max_continuous_gap_degrees'] = gap
        for key in ('target_met', 'coverage_target_met'):
            if key in coverage:
                if type(coverage[key]) is not bool:
                    raise ValueError('Invalid coverage '+key)
                selected[key] = coverage[key]
        if 'covered_yaw_sample_degrees' in coverage:
            samples = coverage['covered_yaw_sample_degrees']
            if (not isinstance(samples, list) or len(samples) > 360
                    or any(type(v) is not int or not 0 <= v < 360 for v in samples)
                    or len(set(samples)) != len(samples)):
                raise ValueError('Invalid coverage yaw samples')
            selected['covered_yaw_sample_degrees'] = samples
        result['coverage'] = selected
    # Rotation-only input has no measured translation/depth. Offering orbit here
    # would let the browser leave the one camera centre actually observed.
    modes = (['look'] if default == 'look' and result.get('depth_observable') is False
             else ['look', 'orbit'] if 'capture_center' in result else ['orbit'])
    result['camera_controls'] = dict(modes=modes,
                                    default_mode=default, yaw=[-180, 180], pitch=[-85, 85])
    return result


def make_server(renderer, base, focus_distance, host="127.0.0.1", port=8080, *, live=None,
                points=None, generation=None, view_metadata=None):
    """One renderer owner. Browser selects locally, submits only on confirmation."""
    if live is None and (not np.isfinite(focus_distance) or focus_distance <= 0):
        raise ValueError("Invalid scene focus distance")
    from gaussian_generation.adapter import camera_bytes
    # Validate the initial view before publishing a URL or accepting clients.
    if live is None:camera_bytes(base)
    metadata = viewer_metadata(view_metadata)
    token = secrets.token_urlsafe(24)
    stats = dict(frames=0, errors=0, render_ms_total=0.)
    def selected_points(values):
        data = np.asarray(values if values is not None else [], dtype=np.float64).reshape(-1, 3)
        data = data[np.isfinite(data).all(axis=1)]
        step = max(1, (len(data)+1499)//1500)
        return data[::step].tolist()
    preview_points = selected_points(points)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)

        def log_message(self, *args):
            pass

        def reply(self, code, data, content="text/plain; charset=utf-8", **headers):
            self.send_response(code)
            self.send_header("Content-Type", content)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            for key, value in headers.items():
                self.send_header(key.replace("_", "-"), str(value))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                self.reply(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path in ('/viewer.js', '/viewer.css'):
                content = 'application/javascript; charset=utf-8' if self.path.endswith('.js') else 'text/css; charset=utf-8'
                self.reply(200, Path(__file__).with_name(self.path[1:]).read_bytes(), content)
            elif self.path == '/scene':
                if not secrets.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):
                    self.reply(403,b'Invalid session token');return
                if live is not None:
                    record = live.preview()
                else:
                    record = dict(base=base, focus_distance=focus_distance, points=preview_points,
                                  version=0, generation=generation or {}, **metadata)
                record['renderer_backend'] = getattr(renderer, 'configuration', {}).get('backend', 'unknown')
                self.reply(200,json.dumps(record,allow_nan=False).encode(),'application/json; charset=utf-8')
            elif self.path=='/state' and live is not None:
                if not secrets.compare_digest(self.headers.get('Authorization',''),'Bearer '+token):
                    self.reply(403,b'Invalid session token');return
                self.reply(200,json.dumps(live.snapshot(),allow_nan=False).encode(),'application/json; charset=utf-8')
            else:
                self.reply(404, b"Not found")

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid request size")
                # Consume this bounded request before replying: closing a socket
                # with unread POST bytes can reset it before the client sees 403.
                body = self.rfile.read(length)
                if len(body) != length:
                    raise ValueError("Incomplete request")
                if not secrets.compare_digest(self.headers.get("Authorization", ""), "Bearer "+token):
                    self.reply(403, b"Invalid session token")
                    return
                payload = json.loads(body)
                if self.path == "/quit":
                    self.reply(200, b"Session closed")
                    self.server.stopping = True
                elif self.path == "/render":
                    if live is not None:live.validate_controls(payload)
                    camera = None if live is not None else camera_bytes(camera_view(
                        base, payload, focus_distance, capture_center=metadata.get('capture_center'),
                        default_mode=metadata['default_view_mode'],
                        allowed_modes=metadata['camera_controls']['modes']))
                    try:
                        if live is not None and not live.snapshot()['has_scene']:
                            self.reply(503,'等待首个有效场景'.encode('utf-8'));return
                        frame = live.render_view(payload) if live is not None else renderer.render_camera(camera)
                    except (RuntimeError, TimeoutError, OSError, ValueError) as exc:
                        # A timed-out framed pipe cannot safely be retried: the
                        # delayed reply could otherwise be labelled as the next view.
                        self.server.stopping = True
                        stats["fatal_error"] = type(exc).__name__+": "+str(exc)
                        raise RuntimeError(stats["fatal_error"]) from exc
                    ok, png = cv2.imencode(".png", frame.rgb[:, :, ::-1])
                    if not ok:
                        raise RuntimeError("Image encoding failed")
                    stats["frames"] += 1
                    stats["render_ms_total"] += frame.metadata["wall_ms"]
                    self.reply(200, png.tobytes(), "image/png", X_Render_Ms=f'{frame.metadata["wall_ms"]:.2f}',
                               X_Request_Id=payload.get("request_id", 0),X_Scene_Version=frame.metadata.get('live_version',0))
                else:
                    self.reply(404, b"Not found")
            except (ValueError, KeyError, TypeError) as exc:
                stats["errors"] += 1
                self.reply(400, str(exc).encode("utf-8"))
            except (RuntimeError, TimeoutError, OSError, cv2.error) as exc:
                stats["errors"] += 1
                self.reply(500, str(exc).encode("utf-8"))

    server = HTTPServer((host, port), Handler)
    server.timeout = .5
    server.stopping = False
    server.session_token = token
    server.stats = stats
    return server


def serve_scene(renderer, prepared, inference, *, host="127.0.0.1", port=8080):
    meta = json.loads((Path(prepared)/"input.json").read_text(encoding="utf-8"))
    # Ring capture uses the earliest supplied camera as the relative heading.
    # A normal scene retains its established target-view reference.
    base = (min(meta['views'], key=lambda view: view['index']) if meta.get('default_view_mode') == 'look'
            else next(view for view in meta['views'] if view['role'] == 'target'))
    with np.load(Path(inference)/"gaussians.npz", allow_pickle=False) as archive:
        means = archive["means"]
    c2w = np.asarray(base["c2w"])
    depth = (means-c2w[:3, 3]) @ c2w[:3, 2]
    positive = depth[np.isfinite(depth) & (depth > 0)]
    if not len(positive):
        raise ValueError("No positive scene depth for viewer")
    generation_record = json.loads((Path(inference)/'inference.json').read_text(encoding='utf-8'))
    generation = {k:generation_record.get(k) for k in ('backend','npu_executed','inference_seconds')}
    server = make_server(renderer, base, float(np.median(positive)), host, port, points=means,
                         generation=generation, view_metadata=meta)
    shown_host = "<board-IP>" if host == "0.0.0.0" else host
    print(json.dumps(dict(event="VIEWER_READY", url=f"http://{shown_host}:{server.server_port}/#token={server.session_token}",
                          width=base["width"], height=base["height"])), flush=True)
    started = time.monotonic()
    try:
        while not server.stopping:
            server.handle_request()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    if "fatal_error" in server.stats:
        raise RuntimeError("Interactive renderer stopped: "+server.stats["fatal_error"])
    return dict(server.stats, elapsed_seconds=time.monotonic()-started,
                scope="interactive requests; PNG encoding/network/display excluded from render_ms")


def verify_saved_scene(scene):
    """Reopen the existing model only after checking its saved artifact hashes."""
    from common import sha
    from gaussian_generation.adapter import validate_inference_archive
    scene=Path(scene).resolve()
    manifest=json.loads((scene/'renderer_input/manifest.json').read_text(encoding='utf-8'))
    inference=json.loads((scene/'inference/inference.json').read_text(encoding='utf-8'))
    prepared=json.loads((scene/'input/input.json').read_text(encoding='utf-8'))
    try:
        validate_inference_archive(scene/'inference', inference)
    except (ValueError, KeyError, OSError) as exc:
        raise ValueError('Saved scene artifact validation failed: '+str(exc)) from exc
    manifest_kind = manifest.get('gaussian_identity_kind', 'npz_file_v1')
    if (manifest_kind not in ('npz_file_v1', 'array_content_v1')
            or manifest_kind != inference.get('gaussian_identity_kind', 'npz_file_v1')
            or ('gaussian_sha256' in manifest
                and manifest['gaussian_sha256'] != inference['gaussian_sha256'])):
        raise ValueError('Saved scene artifact validation failed: Gaussian export identity mismatch')
    if inference.get('gaussian_identity_kind') == 'array_content_v1':
        if (manifest.get('gaussian_identity_kind') != 'array_content_v1'
                or manifest.get('gaussian_sha256') != inference['gaussian_sha256']
                or manifest.get('gaussian_archive_sha256') != inference['gaussian_archive_sha256']):
            raise ValueError('Saved scene artifact validation failed: Gaussian export identity mismatch')
    if (not inference.get('complete')
            or sha(scene/'renderer_input/model.ply')!=manifest.get('model_sha256')
            or sha(scene/'input/context.npz')!=inference.get('input_sha256')
            or prepared.get('context_sha256')!=inference.get('input_sha256')):
        raise ValueError('Saved scene artifact validation failed')
    return scene


def main(argv=None):
    import argparse
    import signal
    from initialize.session import render_environment
    from rendering.pipeline_adapter import PipelineRenderer
    p=argparse.ArgumentParser(description='Reopen a saved FPGA scene without Gaussian inference')
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--renderer',type=Path,required=True)
    p.add_argument('--live-renderer',type=Path,required=True)
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=8080)
    a=p.parse_args(argv)
    scene=verify_saved_scene(a.scene)
    def interrupted(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    with PipelineRenderer(a.renderer,a.live_renderer,render_environment(),
                          scene/('viewer_'+secrets.token_hex(3)+'.renderer.log')) as renderer:
        renderer.load_scene(scene/'renderer_input/model.ply')
        result=serve_scene(renderer.runtime,scene/'input',scene/'inference',host=a.host,port=a.port)
    print(json.dumps(dict(event='VIEWER_COMPLETE',**result)),flush=True)


if __name__=='__main__':main()
