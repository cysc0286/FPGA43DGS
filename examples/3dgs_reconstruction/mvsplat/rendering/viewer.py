"""Small HTTP camera controller over a resident LiveRenderer; no scene regeneration."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import secrets
import time

import cv2
import numpy as np


PAGE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>3DGS 场景</title>
<style>
:root{color-scheme:light dark;font-family:system-ui,sans-serif}body{max-width:800px;margin:24px auto;padding:0 16px}
canvas{display:block;width:100%;max-width:640px;background:#555;touch-action:none;image-rendering:auto}
.controls{display:flex;flex-wrap:wrap;gap:16px;margin:16px 0}label{display:flex;align-items:center;gap:8px}
button,input{font:inherit}button{padding:8px 14px}#status{min-height:2em}input{max-width:160px}
</style><h1>3DGS 场景</h1><p>拖动画面旋转；滚轮缩放；按住 Shift 拖动平移。</p>
<canvas id="view" width="128" height="128" aria-label="重建场景；也可使用下方滑块调整视角"></canvas>
<div class="controls"><label>水平角度<input id="yaw" type="range" min="-30" max="30" value="0"></label>
<label>俯仰角度<input id="pitch" type="range" min="-20" max="20" value="0"></label>
<label>远近<input id="distance" type="range" min="0.6" max="1.5" step="0.01" value="1"></label>
<button id="reset">初始视角</button><button id="quit">结束会话</button></div><p id="status" role="status"></p>
<script>
const token=new URLSearchParams(location.hash.slice(1)).get('token')||'';
const canvas=document.getElementById('view'), ctx=canvas.getContext('2d'), status=document.getElementById('status');
const fields=['yaw','pitch','distance'], state={yaw:0,pitch:0,distance:1,pan_x:0,pan_y:0};
let revision=0, busy=false, stopped=false, pointer=null;
const clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
function sync(){for(const key of fields)document.getElementById(key).value=state[key];}
async function request(path,body){const r=await fetch(path,{method:'POST',headers:{'Authorization':'Bearer '+token,
  'Content-Type':'application/json'},body:JSON.stringify(body)}); if(!r.ok)throw Error(await r.text());return r;}
async function draw(){if(busy||stopped)return;busy=true;const id=revision;
  try{const response=await request('/render',{...state,request_id:id});const blob=await response.blob();
    const image=await createImageBitmap(blob);canvas.width=image.width;canvas.height=image.height;
    ctx.drawImage(image,0,0);image.close();status.textContent='原生分辨率 '+canvas.width+'×'+canvas.height+
      ' · 本次渲染 '+response.headers.get('X-Render-Ms')+' ms';
  }catch(error){status.textContent=error.message;}finally{busy=false;if(id!==revision&&!stopped)draw();}}
function changed(){revision++;sync();draw();}
for(const key of fields)document.getElementById(key).addEventListener('input',e=>{state[key]=Number(e.target.value);changed();});
canvas.addEventListener('pointerdown',e=>{pointer={id:e.pointerId,x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointermove',e=>{if(!pointer||pointer.id!==e.pointerId)return;
  const dx=e.clientX-pointer.x,dy=e.clientY-pointer.y;pointer.x=e.clientX;pointer.y=e.clientY;
  if(e.shiftKey){state.pan_x=clamp(state.pan_x-dx/1000,-.25,.25);state.pan_y=clamp(state.pan_y-dy/1000,-.25,.25);}
  else{state.yaw=clamp(state.yaw+dx*.15,-30,30);state.pitch=clamp(state.pitch-dy*.15,-20,20);}changed();});
for(const event of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(event,()=>pointer=null);
canvas.addEventListener('wheel',e=>{e.preventDefault();state.distance=clamp(state.distance*Math.exp(e.deltaY*.001),.6,1.5);changed();},{passive:false});
document.getElementById('reset').onclick=()=>{Object.assign(state,{yaw:0,pitch:0,distance:1,pan_x:0,pan_y:0});changed();};
document.getElementById('quit').onclick=async()=>{stopped=true;try{await request('/quit',{});status.textContent='会话已结束';}
  catch(error){stopped=false;status.textContent=error.message;}};
draw();
</script></html>'''


def orbit_view(base, controls, focus_distance):
    allowed = {"yaw", "pitch", "distance", "pan_x", "pan_y", "request_id"}
    if not isinstance(controls, dict) or set(controls)-allowed:
        raise ValueError("Unknown camera controls")
    limits = dict(yaw=(-30, 30, 0), pitch=(-20, 20, 0), distance=(.6, 1.5, 1),
                  pan_x=(-.25, .25, 0), pan_y=(-.25, .25, 0))
    values = {}
    for key, (low, high, default) in limits.items():
        value = controls.get(key, default)
        if type(value) not in (int, float) or not np.isfinite(value) or not low <= value <= high:
            raise ValueError("Invalid camera control: " + key)
        values[key] = value
    request_id = controls.get("request_id", 0)
    if type(request_id) is not int or not 0 <= request_id <= 2**53-1:
        raise ValueError("Invalid request_id")
    original = np.asarray(base["c2w"], dtype=np.float64)
    right, down, forward = original[:3, :3].T
    pivot = original[:3, 3] + forward*focus_distance
    pivot += focus_distance*(right*values["pan_x"] + down*values["pan_y"])
    def rotation(axis, degrees):
        angle = np.radians(degrees)
        x, y, z = axis
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        return np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew@skew)
    offset = rotation(down, values["yaw"]) @ rotation(right, values["pitch"]) @ (-forward*focus_distance)
    position = pivot + offset*values["distance"]
    look = -offset/np.linalg.norm(offset)
    right = np.cross(down, look)
    right /= np.linalg.norm(right)
    down = np.cross(look, right)
    c2w = np.eye(4)
    c2w[:3, :3] = np.column_stack((right, down, look))
    c2w[:3, 3] = position
    return dict(base, c2w=c2w.tolist())


def make_server(renderer, base, focus_distance, host="127.0.0.1", port=8080):
    """One renderer owner and one active request. Browser coalesces pending views."""
    if not np.isfinite(focus_distance) or focus_distance <= 0:
        raise ValueError("Invalid scene focus distance")
    from gaussian_generation.adapter import camera_bytes
    # Validate the initial view before publishing a URL or accepting clients.
    camera_bytes(base)
    token = secrets.token_urlsafe(24)
    stats = dict(frames=0, errors=0, render_ms_total=0.)

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
            else:
                self.reply(404, b"Not found")

        def do_POST(self):
            if not secrets.compare_digest(self.headers.get("Authorization", ""), "Bearer "+token):
                self.reply(403, b"Invalid session token")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("Invalid request size")
                payload = json.loads(self.rfile.read(length))
                if self.path == "/quit":
                    self.reply(200, b"Session closed")
                    self.server.stopping = True
                elif self.path == "/render":
                    camera = camera_bytes(orbit_view(base, payload, focus_distance))
                    try:
                        frame = renderer.render_camera(camera)
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
                               X_Request_Id=payload.get("request_id", 0))
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
    base = next(view for view in meta["views"] if view["role"] == "target")
    with np.load(Path(inference)/"gaussians.npz", allow_pickle=False) as archive:
        means = archive["means"]
    c2w = np.asarray(base["c2w"])
    depth = (means-c2w[:3, 3]) @ c2w[:3, 2]
    positive = depth[np.isfinite(depth) & (depth > 0)]
    if not len(positive):
        raise ValueError("No positive scene depth for viewer")
    server = make_server(renderer, base, float(np.median(positive)), host, port)
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
    scene=Path(scene).resolve()
    manifest=json.loads((scene/'renderer_input/manifest.json').read_text(encoding='utf-8'))
    inference=json.loads((scene/'inference/inference.json').read_text(encoding='utf-8'))
    prepared=json.loads((scene/'input/input.json').read_text(encoding='utf-8'))
    if (not inference.get('complete')
            or sha(scene/'renderer_input/model.ply')!=manifest.get('model_sha256')
            or sha(scene/'inference/gaussians.npz')!=inference.get('gaussian_sha256')
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
