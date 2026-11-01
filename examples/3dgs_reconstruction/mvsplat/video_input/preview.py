"""Bounded MJPEG preview from Linux UVC, with no model, calibration or FPGA."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import subprocess
import threading
import time

from video_input.capture import validate_capture


PAGE = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>USB 摄像头预览</title>
<style>:root{font-family:system-ui;color-scheme:light dark}body{max-width:1000px;margin:20px auto;padding:0 12px}
img{display:block;width:100%;height:auto}button{font:inherit;padding:8px 16px}</style>
<h1>USB 摄像头预览</h1><p>这是摄像头原始画面。预览不需要标定。</p>
<img id="camera" alt="摄像头实时画面"><p id="status" role="status">正在获取画面…</p>
<button id="pause">暂停显示</button><script>
let active=true; const img=document.getElementById('camera'), status=document.getElementById('status');
async function update(){if(active){try{const r=await fetch('/frame.jpg?t='+Date.now(),{cache:'no-store'});
if(!r.ok)throw Error(await r.text());const data=await r.blob();const old=img.src;img.src=URL.createObjectURL(data);
img.onload=()=>{if(old.startsWith('blob:'))URL.revokeObjectURL(old)};
status.textContent='原生 JPEG · 帧号 '+r.headers.get('X-Frame-Id')+' · 采集 '+r.headers.get('X-Capture-Fps')+' fps';
}catch(e){status.textContent=e.message}}setTimeout(update,250)}
document.getElementById('pause').onclick=e=>{active=!active;e.target.textContent=active?'暂停显示':'继续显示'};update();
</script></html>'''


class JpegFrames:
    """Extract complete JPEGs from arbitrary pipe chunks without unbounded RAM."""
    def __init__(self, limit=8*1024**2):
        self.buffer = bytearray()
        self.limit = limit

    def feed(self, data):
        self.buffer.extend(data)
        frames = []
        while True:
            start = self.buffer.find(b'\xff\xd8')
            if start < 0:
                self.buffer[:] = self.buffer[-1:]
                break
            if start:
                del self.buffer[:start]
            end = self.buffer.find(b'\xff\xd9', 2)
            if end < 0:
                if len(self.buffer) > self.limit:
                    raise ValueError('JPEG exceeds preview buffer limit')
                break
            if end+2 > self.limit:
                raise ValueError('JPEG exceeds preview frame limit')
            frames.append(bytes(self.buffer[:end+2]))
            del self.buffer[:end+2]
        return frames


class Preview:
    def __init__(self, device, width, height, fps, log, *, recording=None,
                 storage_limit=None, max_seconds=20):
        validate_capture(width, height, fps, 1, 0)
        if not Path(device).exists():
            raise ValueError('Camera capture node does not exist')
        self.lock = threading.Lock()
        self.latest = None
        self.count = 0
        self.error = None
        self.first = None
        self.last = None
        self.log = Path(log).open('wb')
        command = ['ffmpeg', '-hide_banner', '-loglevel', 'warning',
                   '-f', 'v4l2', '-input_format', 'mjpeg', '-video_size', f'{width}x{height}',
                   '-framerate', str(fps), '-i', str(device)]
        if recording is not None:
            command += ['-n', '-map', '0:v:0', '-an', '-c:v', 'copy', '-t', str(max_seconds),
                        '-cluster_time_limit','250','-cluster_size_limit',str(1024**2),
                        '-fs', str(storage_limit), str(recording)]
        command += ['-map', '0:v:0', '-an', '-c:v', 'copy']
        if recording is not None:
            command += ['-t', str(max_seconds)]
        command += ['-f', 'image2pipe', 'pipe:1']
        self.command = command
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE if recording else subprocess.DEVNULL, stdout=subprocess.PIPE,
                                            stderr=self.log, bufsize=0)
            if recording is not None:
                # -fs is a muxer soft limit: old Matroska can flush a large cluster
                # beyond it. Keep a kernel ceiling as the final fail-closed guard.
                import resource
                resource.prlimit(self.process.pid,resource.RLIMIT_FSIZE,(int(storage_limit),int(storage_limit)))
        except BaseException:
            if getattr(self,'process',None) is not None:
                if self.process.poll() is None:
                    self.process.kill(); self.process.wait()
                if self.process.stdout:self.process.stdout.close()
                if self.process.stdin:self.process.stdin.close()
            self.log.close()
            raise
        self.thread = threading.Thread(target=self.read, daemon=True)
        self.thread.start()

    def read(self):
        parser = JpegFrames()
        try:
            while True:
                chunk = self.process.stdout.read(65536)
                if not chunk:
                    raise RuntimeError('Camera stream ended; check USB connection and ffmpeg log')
                for frame in parser.feed(chunk):
                    with self.lock:
                        now = time.monotonic()
                        self.latest = frame
                        self.count += 1
                        self.first = self.first or now
                        self.last = now
        except (ValueError, OSError, RuntimeError) as exc:
            with self.lock:
                self.error = str(exc)

    def snapshot(self):
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            if self.latest is None or time.monotonic()-self.last > 3:
                raise RuntimeError('Waiting for fresh camera frames')
            fps = (self.count-1)/(self.last-self.first) if self.count > 1 else 0
            return self.latest, self.count, fps

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.thread.join(5)
        self.process.stdout.close()
        if self.process.stdin:
            self.process.stdin.close()
        self.log.close()

    def finish_recording(self):
        """Ask FFmpeg to write its muxer trailer before releasing the camera."""
        if self.process.poll() is None:
            try:
                self.process.stdin.write(b'q\n')
                self.process.stdin.flush()
            except BrokenPipeError:
                pass
            self.process.wait(timeout=8)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='/dev/video0')
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=720)
    p.add_argument('--fps', type=float, default=60)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=8081)
    p.add_argument('--log', type=Path, required=True)
    a = p.parse_args(argv)
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    preview = Preview(a.device, a.width, a.height, a.fps, a.log)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            status, content, headers = 200, 'text/html; charset=utf-8', {}
            if self.path == '/':
                data = PAGE.encode()
            elif self.path.startswith('/frame.jpg'):
                try:
                    data, count, fps = preview.snapshot()
                    content = 'image/jpeg'
                    headers = {'X-Frame-Id': str(count), 'X-Capture-Fps': f'{fps:.2f}'}
                except RuntimeError as exc:
                    status, data = 503, str(exc).encode()
            else:
                status, data = 404, b'Not found'
            self.send_response(status)
            self.send_header('Content-Type', content)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            for key, value in headers.items():
                self.send_header(key, value)
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = None
    try:
        server = ThreadingHTTPServer((a.host, a.port), Handler)
        print(json.dumps(dict(event='PREVIEW_READY', host=a.host, port=server.server_port)), flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server:
            server.server_close()
        preview.close()


if __name__ == '__main__':
    main()
