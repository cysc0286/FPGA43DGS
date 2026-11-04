"""Authenticated capture/upload HTTP routes; video ownership stays in video_input."""
import json
from pathlib import Path
import secrets
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

PAGE = Path(__file__).with_name('recording_page.html').read_text(encoding='utf-8')

def make_server(camera, host='127.0.0.1', port=8081):
    token=secrets.token_urlsafe(24)
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(15)
        def log_message(self,*args): pass
        def reply(self,code,data,content='application/json; charset=utf-8',headers=None):
            self.send_response(code)
            self.send_header('Content-Type',content)
            self.send_header('Content-Length',str(len(data)))
            self.send_header('Cache-Control','no-store')
            for key,value in (headers or {}).items(): self.send_header(key,value)
            self.end_headers()
            try: self.wfile.write(data)
            except (BrokenPipeError,ConnectionResetError): pass
        def do_GET(self):
            route=urlsplit(self.path).path
            if route in ('/','/capture','/upload'):
                calibration=getattr(getattr(camera.build,'args',None),'calibration',None)
                label='实测标定内参' if calibration else '估算内参'
                page=(Path(__file__).with_name('upload_page.html').read_text(encoding='utf-8')
                      if route=='/upload' else PAGE.replace('估算内参',label))
                self.reply(200,page.replace('__TOKEN__',json.dumps(token)).encode(),'text/html; charset=utf-8')
            elif self.path=='/api/status':
                self.reply(200,json.dumps(camera.status(),ensure_ascii=False).encode())
            elif self.path.startswith('/frame.jpg'):
                try:
                    frame,count,fps,captured=camera.packet()
                    self.reply(200,frame,'image/jpeg',{'X-Frame-Id':str(count),'X-Capture-Fps':f'{fps:.2f}',
                        'X-Capture-Monotonic':str(captured)})
                except RuntimeError as exc: self.reply(503,str(exc).encode(),'text/plain; charset=utf-8')
            else: self.reply(404,b'{}')
        def do_POST(self):
            supplied=self.headers.get('X-Session-Token','')
            if self.headers.get('Authorization','').startswith('Bearer '):
                supplied=self.headers['Authorization'][7:]
            if not secrets.compare_digest(supplied.encode('utf-8'),token.encode('utf-8')):
                self.reply(403,b'{"error":"Invalid session"}'); return
            body_started=False
            try:
                route=urlsplit(self.path)
                if route.path=='/api/prepare/upload':
                    camera.prepare_upload();self.reply(202,b'{"accepted":true}');return
                if route.path=='/api/upload':
                    if self.headers.get('Transfer-Encoding') or self.headers.get('Content-Length') is None:
                        raise ValueError('上传需要明确的 Content-Length，不支持分块编码')
                    lengths=self.headers.get_all('Content-Length',[])
                    if len(lengths)!=1 or not lengths[0].isdecimal():raise ValueError('Invalid upload length')
                    name=parse_qs(route.query).get('filename',['video.mp4'])
                    if len(name)!=1:raise ValueError('Invalid filename')
                    owner=self
                    class UploadBody:
                        def read(self,size):
                            nonlocal body_started
                            body_started=True
                            return owner.rfile.read(size)
                    record=camera.receive_upload(UploadBody(),int(lengths[0]),name[0])
                    self.reply(202,json.dumps(dict(accepted=True,upload=record),ensure_ascii=False).encode());return
                if self.path=='/api/cleanup':
                    self.reply(200,json.dumps(camera.cleanup_history(),ensure_ascii=False).encode());return
                if self.path=='/api/cleanup/all':
                    if self.headers.get('X-Confirm-Delete')!='all-session-records':
                        raise ValueError('清空全部记录需要明确确认')
                    self.reply(200,json.dumps(camera.clear_all_history(),ensure_ascii=False).encode());return
                action={'/api/start':camera.start,'/api/stop':camera.stop,'/api/reconstruct':camera.reconstruct,
                        '/api/live/start':camera.start_stream,'/api/live/stop':camera.stop_stream}.get(self.path)
                if action is None: self.reply(404,b'{}'); return
                action()
                self.reply(202,b'{"accepted":true}')
            except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
                # An early rejection with a small unread body can otherwise
                # reset TCP before the client sees the response on Windows.
                # Never drain an unbounded/unknown body or one partly consumed.
                length=self.headers.get('Content-Length','')
                if not body_started and length.isdecimal() and 0<int(length)<=65536:
                    try:
                        self.connection.settimeout(.25)
                        self.rfile.read(int(length))
                    except OSError:pass
                self.reply(409,json.dumps(dict(error=str(exc)),ensure_ascii=False).encode())
    return ThreadingHTTPServer((host,port),Handler)
