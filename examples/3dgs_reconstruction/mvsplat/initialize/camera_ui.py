"""Coordinate button capture, the existing pipeline CLI and its live viewer."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

from video_input.recording import CameraRecorder, make_server


def reconstruction_error(detail):
    if 'timeout' in detail:
        return '会话达到 30 分钟保护上限，已保存的场景仍保留。可重新开始会话。'
    if 'No valid context pair' in detail:
        return '没有找到足够重叠且清晰的视角。请对准同一组静止物体，缓慢横向平移，录制 5～8 秒；避免绕物体大幅旋转或快速靠近。'
    if 'rss_limit' in detail or 'host_reserve_limit' in detail:
        return '本次重建触及内存保护，已停止。请结束其他任务后重试。'
    if 'No space left' in detail:
        return '板端存储空间不足，本次未完成。请点击“清理历史记录”后重试。'
    return '本次重建未完成，请展开诊断详情查看原因，原录像已保留。'


class ReconstructionJob:
    def __init__(self,camera,args):
        self.camera,self.args=camera,args
        self.lock=threading.RLock()
        self.phase,self.error,self.url='idle',None,None
        self.process,self.worker,self.stopping=None,None,False
        self.cancel=threading.Event()
        self.clip_path=None
        self.run_path=None
        self.error_details=None
        self.mode='clip'
        self.stream_stop_requested=False

    def status(self):
        with self.lock:
            live={}
            if self.mode=='streaming' and self.run_path:
                try:
                    path=Path(self.run_path)/'stream.json'
                    if path.stat().st_size<=1024**2:live=json.loads(path.read_text(encoding='utf-8'))
                except (OSError,ValueError):pass
            phase=self.phase
            if self.mode=='streaming' and phase in ('initializing','streaming'):
                if phase=='streaming' and live.get('phase')=='stopped':phase='ready'
                elif self.stream_stop_requested:phase='stopping'
            return dict(phase=phase,error=self.error,details=self.error_details,url=self.url,
                        clip_path=self.clip_path,run_path=self.run_path,mode=self.mode,
                        live={k:v for k,v in live.items() if k not in ('points','trajectory')})

    def start_stream(self):
        with self.lock:
            if self.stopping or self.status()['phase'] in ('initializing','reconstructing','streaming','stopping'):
                raise ValueError('已有采集或重建正在进行')
            self.phase,self.error,self.url='initializing',None,None
            self.mode='streaming';self.stream_stop_requested=False;self.error_details=None;self.clip_path=None
            self.run_path=str(self.args.out/('stream_'+time.strftime('%Y%m%d_%H%M%S')+'_'+os.urandom(3).hex()))
            previous=self.worker;self.cancel.set();self.cancel=threading.Event()
            self.worker=threading.Thread(target=self.run,args=(None,self.cancel,previous,None,True),daemon=True)
            self.worker.start()

    def stop_stream(self):
        with self.lock:
            if self.mode!='streaming' or self.status()['phase'] not in ('initializing','streaming','stopping'):
                raise ValueError('当前没有连续采集任务')
            self.stream_stop_requested=True
            if Path(self.run_path).is_dir():(Path(self.run_path)/'stop.requested').touch()

    def stop_viewer(self):
        if self.process and self.process.poll() is None:
            # measure forwards TERM to the isolated pipeline process group in finally.
            try: os.killpg(self.process.pid,signal.SIGTERM)
            except ProcessLookupError: return
            try: self.process.wait(10)
            except subprocess.TimeoutExpired:
                raise RuntimeError('上次重建进程尚未退出，请先检查状态')

    def reset(self):
        """Release an idle/ended viewer before its saved files are removed."""
        with self.lock:
            if self.status()['phase'] in ('initializing','reconstructing','streaming','stopping'):
                raise ValueError('请先结束当前采集或重建')
            self.cancel.set();worker=self.worker
        self.stop_viewer()
        if worker:
            worker.join(12)
            if worker.is_alive():raise RuntimeError('上次查看器尚未退出，未清空记录')
        with self.lock:
            self.phase,self.error,self.url='idle',None,None
            self.process,self.worker=None,None
            self.clip_path,self.run_path,self.error_details=None,None,None
            self.mode='clip';self.stream_stop_requested=False;self.cancel=threading.Event()

    def restore_stream(self,folder):
        from common import sha
        folder=Path(folder).resolve()
        if folder.parent!=self.camera.storage_root or not folder.name.startswith('stream_'):
            raise ValueError('恢复连续结果必须位于当前录制目录中')
        latest=json.loads((folder/'latest.json').read_text(encoding='utf-8'))
        if latest.get('slot') not in ('slot_a','slot_b'):raise ValueError('无效的连续结果指针')
        scene=folder/latest['slot'];frame=scene/'board/first_frame/frame.bin'
        if scene.is_symlink() or scene.resolve().parent!=folder or sha(frame)!=latest.get('frame_sha256'):
            raise ValueError('连续结果校验失败')
        with self.lock:
            if self.stopping or self.status()['phase'] in ('initializing','reconstructing','streaming','stopping'):
                raise ValueError('请先结束当前采集或重建')
            self.phase,self.error,self.url='initializing',None,None
            self.mode='streaming';self.stream_stop_requested=False;self.error_details=None;self.clip_path=None
            self.run_path=str(folder)
            previous=self.worker;self.cancel.set();self.cancel=threading.Event()
            self.worker=threading.Thread(target=self.run,args=(None,self.cancel,previous,scene),daemon=True)
            self.worker.start()

    def start(self,clip,scene=None):
        with self.lock:
            if self.stopping or self.status()['phase'] in ('initializing','reconstructing','streaming','stopping'):
                raise ValueError('已有重建正在进行')
            if scene is not None:
                scene=Path(scene).resolve()
                if scene.parent!=self.camera.storage_root or not scene.name.startswith('scene_') or not scene.is_dir():
                    raise ValueError('恢复场景必须位于当前录制目录中')
                events=json.loads((scene/'events.json').read_text(encoding='utf-8'))['events']
                if not any(e.get('event')=='VIDEO_COMPLETE' and e.get('path')==str(clip) for e in events):
                    raise ValueError('恢复场景与当前录像的来源不一致')
            self.phase,self.error,self.url='initializing',None,None
            self.error_details=None
            self.mode='clip';self.stream_stop_requested=False
            self.clip_path=str(clip)
            self.run_path=str(scene or self.args.out/('scene_'+time.strftime('%Y%m%d_%H%M%S')+'_'+os.urandom(3).hex()))
            previous=self.worker
            self.cancel.set()
            self.cancel=threading.Event()
            self.worker=threading.Thread(target=self.run,args=(clip,self.cancel,previous,scene),daemon=True)
            self.worker.start()

    def run(self,clip,cancel,previous=None,scene=None,stream=False):
        args=self.args
        run=Path(self.run_path)
        log_path=run/('viewer_resume_'+os.urandom(3).hex()+'.log') if scene else Path(str(run)+'.log')
        measurement_path=Path(str(log_path)+'.measurement.json') if scene else Path(str(run)+'.measurement.json')
        try:
            if previous:
                previous.join(12)
                if previous.is_alive(): raise RuntimeError('上次查看器尚未退出，请检查状态')
            with self.lock: self.phase,self.error,self.url='initializing',None,None
            self.stop_viewer()
            if self.stopping: return
            if not stream:self.camera.pause_preview()
            cli=[sys.executable,str(Path(__file__).resolve().parents[2]/'pipeline.py'),'initialize',
                '--video',str(clip),'--out',str(run),'--auto-select','--weights',str(args.weights),
                '--vendor',str(args.vendor),'--renderer',str(args.renderer),'--live-renderer',str(args.live_renderer),
                '--render-profile','mainline','--threads','4','--prepare-threads','1','--serial-prepare',
                '--interactive','--viewer-host','127.0.0.1','--viewer-port',str(args.viewer_port)]
            if args.calibration: cli+=['--calibration',str(args.calibration)]
            else: cli+=['--focal-ratio',str(args.focal_ratio)]
            if scene is not None:
                cli=[sys.executable,'-m','rendering.viewer','--scene',str(scene),
                     '--renderer',str(args.renderer),'--live-renderer',str(args.live_renderer),
                     '--host','127.0.0.1','--port',str(args.viewer_port)]
            if stream:
                cli=[sys.executable,'-m','initialize.streaming','--source-url',f'http://127.0.0.1:{args.port}/frame.jpg',
                     '--out',str(run),'--weights',str(args.weights),'--vendor',str(args.vendor),
                     '--renderer',str(args.renderer),'--live-renderer',str(args.live_renderer),
                     '--host','127.0.0.1','--port',str(args.viewer_port)]
                cli+=['--calibration',str(args.calibration)] if args.calibration else ['--focal-ratio',str(args.focal_ratio)]
            command=['/usr/bin/python3',str(Path(__file__).resolve().parents[1]/'measure.py'),
                '--out',str(measurement_path),'--rss-mib','650','--reserve-mib','128','--timeout','1800','--']+cli
            env=dict(os.environ,PYTHONUNBUFFERED='1')
            with log_path.open('xb') as log:
                self.process=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                              env=env,start_new_session=True)
            while self.process.poll() is None and not self.stopping and not cancel.is_set():
                if stream and self.stream_stop_requested and run.is_dir():(run/'stop.requested').touch()
                events=run/'events.json'
                if events.exists():
                    names=[e['event'] for e in json.loads(events.read_text())['events']]
                    if 'READY' in names:
                        with self.lock:
                            if self.phase=='initializing': self.phase='reconstructing'
                if self.url is None:
                    for line in log_path.read_text(errors='replace').splitlines():
                        if '"event": "VIEWER_READY"' in line:
                            event=json.loads(line)
                            token=urlsplit(event['url']).fragment
                            with self.lock:self.url=args.viewer_public_url.rstrip('/')+'/#'+token;self.phase='streaming' if stream else 'ready'
                            self.camera.resume_preview()
                            break
                time.sleep(.3)
            if self.stopping or cancel.is_set(): return
            result=(dict(complete=True) if scene or stream else (json.loads((run/'warm_result.json').read_text()) if (run/'warm_result.json').exists() else {}))
            measurement=json.loads(measurement_path.read_text()) if measurement_path.exists() else {}
            if stream and (run/'stream.json').exists():
                live=json.loads((run/'stream.json').read_text())
                if live.get('phase')=='error':raise RuntimeError(live.get('message','Continuous session failed'))
            if not result.get('complete') or self.process.returncode or measurement.get('status')!='passed':
                raise RuntimeError(result.get('error') or '重建进程退出（'+measurement.get('status','no receipt')+'），请查看 '+str(run)+'.log')
            with self.lock: self.phase,self.url='complete',None
        except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
            self.stop_viewer()
            with self.lock:
                self.phase,self.error,self.url='error',reconstruction_error(str(exc)),None
                self.error_details=str(exc)
        finally:
            try: self.camera.resume_preview()
            except OSError as exc:
                with self.lock: self.error='预览恢复失败：'+str(exc)

    def close(self):
        self.stopping=True
        self.cancel.set()
        self.stop_viewer()
        if self.worker: self.worker.join(12)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='/dev/video0')
    p.add_argument('--width',type=int,default=1280)
    p.add_argument('--height',type=int,default=720)
    p.add_argument('--fps',type=float,default=60.)
    p.add_argument('--max-seconds',type=float,default=10.)
    p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--port',type=int,default=8081)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume-clip',type=Path,help='Restore one verified completed capture.json after restarting UI')
    p.add_argument('--resume-scene',type=Path,help='Reopen a saved scene without running reconstruction again')
    p.add_argument('--resume-stream',type=Path,help='Reopen the verified latest slot of an ended continuous session')
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--vendor',type=Path,required=True)
    p.add_argument('--renderer',type=Path,required=True)
    p.add_argument('--live-renderer',type=Path,required=True)
    p.add_argument('--calibration',type=Path)
    p.add_argument('--focal-ratio',type=float,default=None)
    p.add_argument('--viewer-port',type=int,default=8080)
    p.add_argument('--viewer-public-url',required=True,help='URL reachable in the user browser, without token')
    a=p.parse_args(argv)
    if a.resume_scene and a.resume_stream:p.error('Choose one saved scene or stream')
    if a.resume_scene and not a.resume_clip:p.error('--resume-scene requires --resume-clip')
    if bool(a.calibration)==bool(a.focal_ratio): p.error('Choose measured --calibration or explicit approximate --focal-ratio')
    if a.focal_ratio is not None and (not math.isfinite(a.focal_ratio) or a.focal_ratio<=0): p.error('Invalid focal ratio')
    public=urlsplit(a.viewer_public_url)
    if public.scheme not in ('http','https') or not public.netloc or public.fragment or public.query:
        p.error('Invalid public viewer URL')
    for key in ('out','weights','vendor','renderer','live_renderer','calibration'):
        value=getattr(a,key)
        if value is not None: setattr(a,key,value.resolve())
    def interrupted(signum,frame): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    camera=CameraRecorder(a.device,a.width,a.height,a.fps,a.out,max_seconds=a.max_seconds)
    camera.build=ReconstructionJob(camera,a)
    server=None
    try:
        # A user may intentionally have cleared every record since the launch
        # command was saved. Missing restore paths then mean an empty UI.
        if a.resume_clip and a.resume_clip.is_file():camera.restore_clip(a.resume_clip)
        if a.resume_scene and a.resume_scene.is_dir() and camera.clip:camera.build.start(camera.clip['path'],a.resume_scene)
        if a.resume_stream and (a.resume_stream/'latest.json').is_file():camera.build.restore_stream(a.resume_stream)
        server=make_server(camera,a.host,a.port)
        a.port=server.server_port
        print(json.dumps(dict(event='CAMERA_UI_READY',host=a.host,port=server.server_port)),flush=True)
        server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        if server: server.server_close()
        camera.close()


if __name__=='__main__': main()
