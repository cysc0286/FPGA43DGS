"""Lifecycle for bounded sampling, one resident encoder and one FPGA owner."""
import argparse
import json
from pathlib import Path
import shutil
import signal
import threading
import time


def atomic_json(path, value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.new')
    temporary.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


class LivePipeline:
    def __init__(self, model, exchange, out, *, calibration=None, focal_ratio=None):
        self.model,self.exchange,self.out=model,exchange,Path(out).resolve()
        self.calibration,self.focal_ratio=calibration,focal_ratio
        self.anchor=None;self.active=None;self.failures=0;self.last_failure=None

    def progress(self, **values):
        self.exchange.update(**values)
        atomic_json(self.out/'stream.json',dict(schema='camera_live_v1',**self.exchange.snapshot()))

    def process(self, packets):
        import numpy as np
        from common import sha
        from video_input.streaming import decode_window,prepare_window
        from pose_estimation.streaming import LocalAnchor
        from gaussian_generation.adapter import main as export
        from gaussian_generation.adapter import gaussian_rows
        started=time.monotonic();source_time=packets[-1].monotonic
        self.progress(phase='geometry',message=None)
        frames=decode_window(packets,self.calibration)
        h,w=next(iter(frames.values())).shape[:2]
        intrinsic=np.array([[self.focal_ratio*w,0,(w-1)/2],[0,self.focal_ratio*w,(h-1)/2],[0,0,1]],np.float64) if self.calibration is None else self.calibration.intrinsic
        meta=dict(kind='approximate',focal_ratio=self.focal_ratio,distortion='assumed zero') if self.calibration is None else self.calibration.metadata()
        if self.anchor is None:self.anchor=LocalAnchor(intrinsic)
        if not np.allclose(self.anchor.intrinsic,intrinsic):raise ValueError('Camera intrinsics changed; start a new session')
        chosen=self.anchor.select(frames)
        selected=set(chosen[0]+chosen[1])
        source_time=max(p.monotonic for p in packets if p.sequence in selected)
        self.progress(**self.anchor.snapshot())
        # Exactly one latest scene plus one candidate, with fixed owned names.
        slot=self.out/('slot_b' if self.active=='slot_a' else 'slot_a')
        if slot.exists():
            if slot.is_symlink() or getattr(slot,'is_junction',lambda:False)() or slot.resolve().parent!=self.out or slot.name==self.active:
                raise ValueError('Invalid candidate directory')
            for path in slot.rglob('*'):
                if path.is_symlink() or getattr(path,'is_junction',lambda:False)():raise ValueError('Linked candidate directory')
            shutil.rmtree(slot)
        if shutil.disk_usage(self.out).free<96*1024**2:raise OSError('No space left for a bounded live candidate; clean history first')
        slot.mkdir()
        prepared=prepare_window(frames,chosen,intrinsic,meta,slot/'input',packets)
        geometry_seconds=time.monotonic()-started
        self.progress(phase='inference')
        inference=self.model.infer(slot/'input',slot/'inference',prepared['context_sha256'])
        if not inference.get('complete') or inference.get('input_sha256')!=prepared['context_sha256']:
            raise ValueError('Live inference/context contract failed')
        gaussians=self.model.take_gaussians(inference['gaussian_sha256'])
        export(['--input',str(slot/'input'),'--inference',str(slot/'inference'),'--out',str(slot/'renderer_input')],gaussians=gaussians)
        rows,_=gaussian_rows(gaussians)
        base=next(v for v in prepared['views'] if v['role']=='target')
        self.progress(phase='publishing')
        record={}
        def commit(frame,version):
            completed=frame.metadata['host_complete_monotonic']
            frame.archive(slot/'board/first_frame')
            record.update(complete=True,version=version,slot=slot.name,source_last_monotonic=source_time,
                    window_latest_monotonic=packets[-1].monotonic,
                    completed_monotonic=completed,window_to_frame_seconds=completed-source_time,
                    processing_seconds=completed-started,geometry_seconds=geometry_seconds,
                    inference_seconds=inference['inference_seconds'],frame_sha256=sha(slot/'board/first_frame/frame.bin'),
                    scope='latest selected frame host JPEG receive to complete RGB; not exposure time; continuous input has no VIDEO_COMPLETE',
                    coordinate_frame='fixed_initial_anchor',context_indices=prepared['context_indices'],target_indices=prepared['target_indices'])
            atomic_json(slot/'live_result.json',record)
            atomic_json(self.out/'latest.json',record)
        self.exchange.publish(rows,base,gaussians['means'],source_time,commit=commit)
        self.active=slot.name
        self.progress(last_result=record,message=None)
        return record

    def reject(self, exc):
        self.failures+=1;self.last_failure=str(exc)
        self.progress(failures=self.failures,phase='sampling',message='视角重叠不足或当前更新失败，保留上次场景；请缓慢平移并保持对准起始区域。',
                             last_diagnostic=self.last_failure[:2048])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-url',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--weights',type=Path,required=True);p.add_argument('--vendor',type=Path,required=True)
    p.add_argument('--renderer',type=Path,required=True);p.add_argument('--live-renderer',type=Path,required=True)
    p.add_argument('--calibration',type=Path);p.add_argument('--focal-ratio',type=float)
    p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8080)
    a=p.parse_args(argv)
    if bool(a.calibration)==bool(a.focal_ratio):p.error('Choose measured calibration or explicit approximate focal ratio')
    if a.focal_ratio is not None and (not 0<a.focal_ratio<10):p.error('Invalid focal ratio')
    from video_input.streaming import LiveSampler
    sampler=LiveSampler(a.source_url,stop_path=a.out/'stop.requested')
    a.out.mkdir(parents=True,exist_ok=False)
    atomic_json(a.out/'stream.json',dict(schema='camera_live_v1',phase='warming'))
    server=None;exchange=None
    def interrupted(signum,frame):raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    try:
        # Match the board's established Torch-before-OpenCV import order.
        from gaussian_generation.runtime import ModelRuntime
        model=ModelRuntime(a.weights,a.vendor,2,16,a.out/'initialize')
        import cv2
        cv2.setNumThreads(1)
        from initialize.session import render_environment
        from rendering.pipeline_adapter import PipelineRenderer
        from rendering.streaming import SceneExchange
        from rendering.viewer import make_server
        from video_input.calibration import Calibration
        calibration=Calibration(a.calibration) if a.calibration else None
        with PipelineRenderer(a.renderer,a.live_renderer,render_environment(),a.out/'renderer.log') as renderer:
            exchange=SceneExchange(renderer.runtime)
            pipeline=LivePipeline(model,exchange,a.out,calibration=calibration,focal_ratio=a.focal_ratio)
            server=make_server(renderer.runtime,None,None,a.host,a.port,live=exchange)
            def serve():
                while not server.stopping:server.handle_request()
            thread=threading.Thread(target=serve,daemon=True);thread.start()
            print(json.dumps(dict(event='VIEWER_READY',url=f'http://{a.host}:{server.server_port}/#token={server.session_token}',mode='streaming')),flush=True)
            try:
                sampler.start();last=-1;last_attempt=0.
                pipeline.progress(phase='sampling')
                while not server.stopping and not (a.out/'stop.requested').exists():
                    packets,source=sampler.snapshot(last)
                    exchange.update(source=source)
                    if source['source_resets']:raise RuntimeError('Camera source restarted; stop and start a new reference session')
                    if packets and time.monotonic()-last_attempt>=2:
                        last=packets[-1].sequence;last_attempt=time.monotonic()
                        try:pipeline.process(packets)
                        except (ValueError,cv2.error) as exc:
                            if exchange.failure:raise
                            pipeline.reject(exc)
                    atomic_json(a.out/'stream.json',dict(schema='camera_live_v1',**exchange.snapshot()))
                    time.sleep(.2)
                sampler.close()
                if server.stats.get('fatal_error'):raise RuntimeError(server.stats['fatal_error'])
                pipeline.progress(phase='stopped',message=('采集已结束；可继续查看最后一个有效场景。' if exchange.base is not None else '采集已结束；尚未生成有效场景，可重新开始。'))
                from video_input.recording import prune_stream_candidate
                storage=prune_stream_candidate(a.out,a.out.resolve().parent)
                pipeline.progress(storage=storage)
                # Keep the final scene interactive until closed/replaced by the UI.
                while not server.stopping:time.sleep(.2)
                if server.stats.get('fatal_error'):raise RuntimeError(server.stats['fatal_error'])
            finally:
                server.stopping=True
                # A request already in progress finishes before the owner closes.
                thread.join(125)
                server.server_close()
    except KeyboardInterrupt:pass
    except BaseException as exc:
        state=exchange.snapshot() if exchange else {}
        atomic_json(a.out/'stream.json',dict(state,schema='camera_live_v1',phase='error',message=str(exc)[:2048]))
        raise
    finally:
        sampler.close()
        if server:server.stopping=True;server.server_close()


if __name__=='__main__':main()
