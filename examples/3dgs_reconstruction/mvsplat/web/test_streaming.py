"""Framework contracts; synthetic geometry/fake runtime never attest hardware."""
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request,urlopen

import cv2
import numpy as np

from video_input.streaming import FramePacket,FrameWindow,LiveSampler,prepare_window
from video_input.recording import CameraRecorder
from web.recording_server import make_server as camera_server
from video_input.test_recording import FakePreview
from pose_estimation.streaming import LocalAnchor
from pose_estimation import test_camera_selection
from rendering.streaming import SceneExchange
from rendering.runtime import LiveFrame
from web.viewer import make_server
from web.test_viewer import BASE
from web.camera_ui import ReconstructionJob,main as camera_ui_main
from initialize.streaming import LivePipeline


class Renderer:
    def __init__(self):self.calls=[];self.fail=False
    def load_rows(self,rows):self.calls.append('load')
    def render_camera(self,camera,**kwargs):
        self.calls.append('render')
        if self.fail:raise TimeoutError('delayed framed reply')
        return LiveFrame(np.zeros((128,128,3),np.uint8),dict(complete=True,wall_ms=1.),b'raw')


class StreamingTests(unittest.TestCase):
    def test_bounded_latest_window_has_no_pending_job_queue(self):
        window=FrameWindow(byte_limit=1024)
        for i in range(200):window.push(FramePacket(i,i*.5,b'x'*80),i*.5)
        self.assertEqual(len(window.frames),12);self.assertEqual(window.bytes,960)
        selected=window.latest()
        self.assertEqual([p.sequence for p in selected],list(range(188,200)))
        self.assertIsNone(window.latest(after=199))
        self.assertFalse(window.push(selected[-1],99.5))
        with self.assertRaises(ValueError):window.push(FramePacket(200,80,b'x'),100)
        window.push(FramePacket(0,100,b'x'),100)
        self.assertEqual(window.resets,1);self.assertEqual(len(window.frames),1)

    def test_sampler_rejects_remote_sources_and_stale_snapshots(self):
        for url in ('https://127.0.0.1/x','http://example.com/x','http://user:pass@127.0.0.1/x'):
            with self.assertRaises(ValueError):LiveSampler(url)
        sampler=LiveSampler('http://127.0.0.1:1/frame.jpg')
        for i in range(8):sampler.window.push(FramePacket(i,10+i*.5,b'x'),10+i*.5)
        with patch('video_input.streaming.time.monotonic',return_value=20):
            self.assertIsNone(sampler.snapshot()[0])

    def test_live_sampler_uses_atomic_preview_timestamp_and_stops(self):
        with tempfile.TemporaryDirectory() as root:
            camera=CameraRecorder('/dev/video0',1280,720,60,root,factory=FakePreview)
            server=camera_server(camera,port=0)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            stop=Path(root)/'stop.requested'
            sampler=LiveSampler(f'http://127.0.0.1:{server.server_port}/frame.jpg',.1,stop)
            try:
                sampler.start();time.sleep(.3)
                self.assertIsNone(sampler.error);self.assertEqual(sampler.window.received,1)
                stop.touch();sampler.thread.join(1)
                self.assertFalse(sampler.thread.is_alive())
            finally:sampler.close();server.shutdown();server.server_close();camera.close()

    def test_anchor_retains_origin_and_scale_across_two_windows(self):
        helper=test_camera_selection.CameraGeometryTests();k,f0=helper.synthetic_features(0)
        _,f1=helper.synthetic_features(.2);_,f2=helper.synthetic_features(.4)
        image=np.random.default_rng(1).integers(0,256,(270,480,3),dtype=np.uint8)
        anchor=LocalAnchor(k)
        with patch('pose_estimation.geometry.image_features',side_effect=[f0,f1,f2]):
            initial=anchor.select({0:image,5:image,9:image})
        scale=initial[2][0][0,3]/.4
        _,f3=helper.synthetic_features(.6);_,f4=helper.synthetic_features(.8)
        with patch('pose_estimation.streaming.image_features',side_effect=[f4,f3]):
            chosen=anchor.select({15:image,20:image})
        self.assertEqual((chosen[0],chosen[1]),([0,20],[15]))
        np.testing.assert_allclose(chosen[2][0][:3,3],[.8*scale,0,0],atol=1e-3)
        np.testing.assert_allclose(chosen[3][15][0][:3,3],[.6*scale,0,0],atol=1e-3)
        before=anchor.snapshot()
        with self.assertRaises(ValueError):anchor.select({30:np.zeros_like(image),40:np.zeros_like(image)})
        self.assertEqual(before,anchor.snapshot())

    def test_prepared_input_keeps_target_held_out_and_centered(self):
        frames={i:np.full((270,480,3),i,np.uint8) for i in (0,5,9)}
        k=np.array([[252,0,239.5],[0,252,134.5],[0,0,1.]])
        pose=np.eye(4);pose[0,3]=1
        middle=pose.copy();middle[0,3]=.5
        chosen=([0,9],[5],(pose,None,None,None,None),{5:(middle,{})},{})
        packets=[FramePacket(i,float(i),b'jpeg') for i in frames]
        with tempfile.TemporaryDirectory() as root:
            meta=prepare_window(frames,chosen,k,dict(kind='approximate'),Path(root)/'input',packets)
            with np.load(Path(root)/'input/context.npz') as data:
                self.assertEqual(data['image'].shape,(2,3,128,128))
                self.assertAlmostEqual(float(data['image'][1].mean()),9/255,places=6)
                np.testing.assert_allclose(data['intrinsics'][:,:2,2],.5)
            self.assertEqual(meta['target_indices'],[5]);self.assertEqual(meta['context_indices'],[0,9])

    def test_candidate_disk_commit_precedes_public_version_and_failure_is_fatal(self):
        renderer=Renderer();exchange=SceneExchange(renderer)
        rows=np.zeros((2,62),np.float32);means=np.array([[0,0,10.],[1,0,12.]])
        seen=[]
        exchange.publish(rows,BASE,means,1.,commit=lambda frame,version:seen.append((exchange.version,version)))
        self.assertEqual(seen,[(0,1)]);self.assertEqual(exchange.version,1)
        def fail(frame,version):raise OSError('disk full')
        with self.assertRaises(RuntimeError):exchange.publish(rows,BASE,means,2.,commit=fail)
        self.assertEqual(exchange.version,1);self.assertIn('disk full',exchange.failure)
        before=list(renderer.calls)
        with self.assertRaises(RuntimeError):exchange.render_view({})
        self.assertEqual(before,renderer.calls)

    def test_model_refresh_keeps_the_interactive_camera_reference(self):
        exchange=SceneExchange(Renderer());rows=np.zeros((2,62));means=np.array([[0,0,10.]])
        exchange.publish(rows,BASE,means,1.)
        moved=dict(BASE,c2w=np.eye(4).tolist());moved['c2w'][0][3]=2
        exchange.publish(rows,moved,means,2.)
        self.assertEqual(exchange.base,BASE);self.assertEqual(exchange.version,2)
        with self.assertRaises(ValueError):exchange.publish(np.zeros((2,61)),BASE,means,3.)
        self.assertEqual(exchange.version,2);self.assertIsNone(exchange.failure)

    def test_model_load_and_camera_requests_never_overlap(self):
        entered=threading.Event();release=threading.Event()
        class BlockingRenderer(Renderer):
            def load_rows(self,rows):
                super().load_rows(rows);entered.set();release.wait(2)
        renderer=BlockingRenderer();exchange=SceneExchange(renderer)
        publish=threading.Thread(target=exchange.publish,args=(np.zeros((2,62)),BASE,np.array([[0,0,10.]]),1.))
        publish.start();self.assertTrue(entered.wait(1))
        view=threading.Thread(target=exchange.render_view,args=({},));view.start();time.sleep(.05)
        self.assertEqual(renderer.calls,['load']);release.set();publish.join(2);view.join(2)
        self.assertEqual(renderer.calls,['load','render','render'])

    def test_dynamic_bad_controls_do_not_break_fpga_session(self):
        renderer=Renderer();exchange=SceneExchange(renderer)
        exchange.publish(np.zeros((2,62)),BASE,np.array([[0,0,10.]]),time.monotonic())
        server=make_server(renderer,None,None,port=0,live=exchange)
        def serve():
            while not server.stopping:server.handle_request()
        thread=threading.Thread(target=serve,daemon=True);thread.start()
        def post(payload):
            return urlopen(Request(f'http://127.0.0.1:{server.server_port}/render',data=json.dumps(payload).encode(),
                           headers={'Authorization':'Bearer '+server.session_token}),timeout=3)
        try:
            with self.assertRaises(HTTPError) as error:post(dict(yaw=181))
            self.assertEqual(error.exception.code,400);self.assertFalse(server.stopping)
            with post({}) as response:self.assertEqual(response.headers['X-Scene-Version'],'1')
            renderer.fail=True
            with self.assertRaises(HTTPError):post({})
            self.assertTrue(server.stopping);self.assertIsNotNone(exchange.failure)
        finally:server.stopping=True;thread.join(2);server.server_close()

    def test_ui_stream_busy_and_stop_lifecycle_and_latest_cleanup(self):
        from common import sha
        with tempfile.TemporaryDirectory() as root:
            camera=CameraRecorder('/dev/video0',1280,720,60,root,factory=FakePreview)
            job=ReconstructionJob(camera,SimpleNamespace(out=Path(root)));camera.build=job
            try:
                with patch('web.camera_ui.threading.Thread'):
                    camera.start_stream()
                self.assertEqual(job.status()['mode'],'streaming')
                with self.assertRaises(ValueError):camera.start()
                with self.assertRaises(ValueError):camera.cleanup_history()
                run=Path(job.run_path);run.mkdir()
                camera.stop_stream();self.assertTrue((run/'stop.requested').exists())
                self.assertEqual(job.status()['phase'],'stopping')
                job.phase='streaming'
                (run/'stream.json').write_text('{"phase":"stopped"}')
                self.assertEqual(job.status()['phase'],'ready')
                job.worker=None
                old=Path(root)/'stream_20261101_010000_aaaaaa';old.mkdir();(old/'stream.json').write_text('{}')
                frame=run/'slot_a/board/first_frame/frame.bin';frame.parent.mkdir(parents=True);frame.write_bytes(b'frame')
                (run/'latest.json').write_text(json.dumps(dict(slot='slot_a',frame_sha256=sha(frame))))
                self.assertIn(old.name,camera.cleanup_history()['removed']);self.assertTrue(frame.exists())
            finally:camera.close()

    def test_repeated_start_stops_previous_owner_and_launches_another_worker(self):
        with tempfile.TemporaryDirectory() as root:
            camera=CameraRecorder('/dev/video0',1280,720,60,root,factory=FakePreview)
            args=SimpleNamespace(out=Path(root),weights='weights',vendor='vendor',renderer='renderer',
                                 live_renderer='live',viewer_port=8080,port=8081,calibration=None,
                                 focal_ratio=.52,viewer_public_url='http://127.0.0.1:18434')
            job=ReconstructionJob(camera,args);camera.build=job;children=[]
            def popen(command,**kwargs):
                child=SimpleNamespace(returncode=None)
                child.poll=lambda:child.returncode
                run=Path(command[command.index('--out',command.index('--source-url'))+1]);run.mkdir()
                kwargs['stdout'].write(b'{"event": "VIEWER_READY", "url": "http://127.0.0.1:8080/#token=test"}\n')
                kwargs['stdout'].flush();children.append(child)
                return child
            def stop():
                if job.process:job.process.returncode=0
            def wait_for(test):
                deadline=time.monotonic()+3
                while not test() and time.monotonic()<deadline:time.sleep(.01)
                self.assertTrue(test(),job.status())
            try:
                with patch('web.camera_ui.subprocess.Popen',side_effect=popen),patch.object(job,'stop_viewer',side_effect=stop):
                    camera.start_stream();wait_for(lambda:job.phase=='streaming')
                    first=job.run_path
                    (Path(first)/'stream.json').write_text('{"phase":"stopped"}')
                    self.assertEqual(job.status()['phase'],'ready')
                    camera.start_stream();wait_for(lambda:len(children)==2 and job.phase=='streaming')
                    self.assertIsNotNone(children[0].returncode)
                    self.assertNotEqual(first,job.run_path)
                    camera.stop_stream();(Path(job.run_path)/'stream.json').write_text('{"phase":"stopped"}')
                    camera.clear_all_history()
                    self.assertEqual(job.status()['phase'],'idle');self.assertIsNone(job.process)
                    self.assertIsNone(camera.clip)
            finally:camera.close()

    def test_restoring_stopped_metadata_is_not_ready_before_viewer_launch(self):
        with tempfile.TemporaryDirectory() as root:
            camera=CameraRecorder('/dev/video0',1280,720,60,root,factory=FakePreview)
            job=ReconstructionJob(camera,SimpleNamespace(out=Path(root)));camera.build=job
            try:
                job.mode='streaming';job.phase='initializing';job.run_path=root
                (Path(root)/'stream.json').write_text('{"phase":"stopped"}')
                self.assertEqual(job.status()['phase'],'initializing')
            finally:camera.close()

    def test_ui_restart_ignores_restore_paths_removed_by_clear_all(self):
        with tempfile.TemporaryDirectory() as root:
            for option in ('--resume-scene','--resume-stream'):
                args=['--out',root,'--weights',root+'/weights','--vendor',root+'/vendor',
                      '--renderer',root+'/renderer','--live-renderer',root+'/live',
                      '--focal-ratio','.52','--viewer-public-url','http://127.0.0.1:18434','--backend','cpu',
                      '--resume-clip',root+'/deleted/capture.json',option,root+'/deleted_scene']
                with patch('web.camera_ui.CameraRecorder') as camera,patch('web.camera_ui.make_server') as server,patch('builtins.print'):
                    server.return_value.server_port=8081
                    server.return_value.serve_forever.side_effect=KeyboardInterrupt
                    camera_ui_main(args)
                    camera.return_value.restore_clip.assert_not_called()
                    self.assertEqual(camera.return_value.build.status()['phase'],'idle')

    def test_failed_geometry_preserves_successful_slot_and_pointer(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root);(path/'latest.json').write_text('{"version":1}')
            slot=path/'slot_a';slot.mkdir();(slot/'keep').write_bytes(b'previous scene')
            pipeline=LivePipeline(None,SceneExchange(Renderer()),path,focal_ratio=.52);pipeline.active='slot_a'
            image=np.zeros((270,480,3),np.uint8)
            packets=[FramePacket(i,float(i),b'jpeg') for i in range(8)]
            with patch('video_input.streaming.decode_window',return_value={i:image for i in range(8)}):
                with self.assertRaises(ValueError):pipeline.process(packets)
            self.assertEqual((path/'latest.json').read_text(),'{"version":1}')
            self.assertEqual((slot/'keep').read_bytes(),b'previous scene')


if __name__=='__main__':unittest.main()
