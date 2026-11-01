"""Manual recording state, rejection and authenticated HTTP regressions."""
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from video_input.recording import CameraRecorder, make_server,prune_stream_candidate


class FakePreview:
    records=[]
    def __init__(self,device,width,height,fps,log,**kwargs):
        self.process=SimpleNamespace(poll=lambda:None,returncode=0)
        self.closed=False
        if kwargs:
            self.records.append(self)
            Path(kwargs['recording']).write_bytes(b'stub clip')
    def snapshot(self): return b'jpeg',60,60.
    def packet(self):return self.snapshot()+(time.monotonic(),)
    def finish_recording(self): self.finished=True
    def close(self): self.closed=True


class RecordingTests(unittest.TestCase):
    def clip(self,root,name):
        import hashlib
        path=Path(root)/name;path.mkdir()
        video=path/'capture.mkv';video.write_bytes(b'clip bytes')
        record=dict(complete=True,path=str(video.resolve()),sha256=hashlib.sha256(b'clip bytes').hexdigest(),
                    requested=dict(width=1280,height=720,fps=60),duration_seconds=3.,frames=180,measured_fps=60.,stop_label='手动结束')
        (path/'capture.json').write_text(json.dumps(record),encoding='utf-8')
        return path

    def scene(self,root,name,clip):
        from common import sha
        path=Path(root)/name;path.mkdir()
        (path/'renderer_input').mkdir();(path/'renderer_input/model.ply').write_bytes(b'model')
        (path/'inference').mkdir();(path/'inference/inference.json').write_text('{"complete":true}')
        frame=path/'board/target/frame.bin';frame.parent.mkdir(parents=True);frame.write_bytes(b'framebuffer')
        (path/'events.json').write_text(json.dumps(dict(events=[dict(event='VIDEO_COMPLETE',path=str(clip/'capture.mkv')),
                                                             dict(event='FRAME_COMPLETE',frame_sha256=sha(frame))])))
        return path

    def test_inference_without_verified_first_frame_does_not_replace_last_success(self):
        with tempfile.TemporaryDirectory() as root:
            camera=self.camera(root)
            try:
                source=self.clip(root,'clip_20261101_100000_aaaaaa')
                last=self.scene(root,'scene_20261101_100100_aaaaaa',source)
                current=self.clip(root,'clip_20261101_110000_bbbbbb')
                failed=self.scene(root,'scene_20261101_110100_bbbbbb',current)
                (failed/'board/target/frame.bin').write_bytes(b'corrupt')
                camera.restore_clip(current/'capture.json')
                result=camera.cleanup_history()
                self.assertIn(failed.name,result['removed'])
                self.assertTrue(last.exists());self.assertTrue(source.exists());self.assertTrue(current.exists())
            finally:camera.close()

    def test_cleanup_retains_current_result_and_leaves_unrelated_paths(self):
        with tempfile.TemporaryDirectory() as root:
            camera=self.camera(root)
            try:
                old=self.clip(root,'clip_20261101_100000_aaaaaa')
                stale=self.scene(root,'scene_20261101_100100_aaaaaa',old)
                (Path(str(stale)+'.log')).write_bytes(b'old log')
                current=self.clip(root,'clip_20261101_110000_bbbbbb')
                scene=self.scene(root,'scene_20261101_110100_bbbbbb',current)
                marker=Path(root)/'weights';marker.mkdir();(marker/'keep').write_bytes(b'weights')
                unknown=Path(root)/'clip_20261101_090000_cccccc';unknown.mkdir();(unknown/'notes').write_bytes(b'unknown')
                camera.restore_clip(current/'capture.json')
                camera.build=SimpleNamespace(status=lambda:dict(phase='ready',clip_path=str(current/'capture.mkv'),run_path=str(scene)),close=lambda:None)
                result=camera.cleanup_history()
                self.assertFalse(old.exists());self.assertFalse(stale.exists());self.assertFalse(Path(str(stale)+'.log').exists())
                self.assertTrue((current/'capture.mkv').exists());self.assertTrue((scene/'renderer_input/model.ply').exists())
                self.assertTrue((marker/'keep').exists());self.assertTrue((unknown/'notes').exists())
                self.assertGreater(result['freed_bytes'],0)
                self.assertEqual(camera.state,'recorded')
                self.assertEqual(camera.cleanup_history()['removed'],[])
            finally:camera.close()

    def test_cleanup_protects_last_scene_source_when_new_clip_not_rebuilt(self):
        with tempfile.TemporaryDirectory() as root:
            camera=self.camera(root)
            try:
                source=self.clip(root,'clip_20261101_100000_aaaaaa')
                scene=self.scene(root,'scene_20261101_100100_aaaaaa',source)
                current=self.clip(root,'clip_20261101_110000_bbbbbb')
                camera.restore_clip(current/'capture.json')
                self.assertEqual(camera.cleanup_history()['removed'],[])
                self.assertTrue(source.exists());self.assertTrue(scene.exists());self.assertTrue(current.exists())
            finally:camera.close()

    def test_cleanup_rejects_busy_and_never_follows_symlinks(self):
        with tempfile.TemporaryDirectory() as root,tempfile.TemporaryDirectory() as outside:
            camera=self.camera(root)
            try:
                old=self.clip(root,'clip_20261101_100000_aaaaaa')
                current=self.clip(root,'clip_20261101_110000_bbbbbb')
                camera.restore_clip(current/'capture.json')
                victim=Path(outside)/'keep';victim.write_bytes(b'outside')
                try:(old/'linked').symlink_to(victim)
                except OSError:link=False
                else:link=True
                camera.state='recording'
                with self.assertRaises(ValueError):camera.cleanup_history()
                camera.state='recorded'
                camera.build=SimpleNamespace(status=lambda:dict(phase='reconstructing'),close=lambda:None)
                with self.assertRaises(ValueError):camera.cleanup_history()
                camera.build=None
                if link:
                    self.assertEqual(camera.cleanup_history()['removed'],[])
                    self.assertTrue(old.exists())
                self.assertEqual(victim.read_bytes(),b'outside')
            finally:camera.close()

    def test_ui_restart_restores_only_hash_verified_completed_capture(self):
        import hashlib
        with tempfile.TemporaryDirectory() as folder:
            camera=self.camera(folder)
            try:
                clip=Path(folder)/'saved';clip.mkdir()
                path=clip/'capture.mkv';path.write_bytes(b'valid')
                data=dict(complete=True,path=str(path.resolve()),sha256=hashlib.sha256(b'valid').hexdigest(),
                    requested=dict(width=1280,height=720,fps=60),duration_seconds=3.,frames=180,measured_fps=60.,stop_label='手动结束')
                meta=clip/'capture.json';meta.write_text(json.dumps(data),encoding='utf-8')
                camera.restore_clip(meta)
                self.assertEqual(camera.state,'recorded')
                self.assertEqual(camera.clip['path'],str(path.resolve()))
                path.write_bytes(b'tampered')
                with self.assertRaisesRegex(ValueError,'校验'):camera.restore_clip(meta)
            finally:camera.close()

    def wait(self,camera,state):
        deadline=time.monotonic()+3
        while camera.status()['state']!=state and time.monotonic()<deadline: time.sleep(.01)
        self.assertEqual(camera.status()['state'],state,camera.status())

    def camera(self,folder): return CameraRecorder('/dev/video0',1280,720,60,folder,factory=FakePreview)

    def test_start_stop_publishes_only_validated_clip_and_can_record_again(self):
        with tempfile.TemporaryDirectory() as folder,patch('video_input.recording.inspect_clip',return_value=(
            dict(codec_name='mjpeg',width=1280,height=720,fps=60),[i/60 for i in range(121)])):
            camera=self.camera(folder)
            try:
                for _ in range(2):
                    camera.start();self.wait(camera,'recording')
                    with self.assertRaises(ValueError): camera.start()
                    camera.stop();self.wait(camera,'recorded')
                    self.assertTrue(Path(camera.clip['path']).is_file())
                    self.assertEqual(camera.clip['stop_reason'],'manual')
                self.assertEqual(len(list(Path(folder).glob('clip_*/capture.json'))),2)
                self.assertEqual(camera.snapshot()[2],60)
            finally: camera.close()

    def test_too_short_clip_is_not_available_to_reconstruction(self):
        with tempfile.TemporaryDirectory() as folder,patch('video_input.recording.inspect_clip',return_value=(
            dict(codec_name='mjpeg',width=1280,height=720,fps=60),[i/60 for i in range(10)])):
            camera=self.camera(folder)
            try:
                camera.start();self.wait(camera,'recording');camera.stop();self.wait(camera,'error')
                self.assertIsNone(camera.clip)
                self.assertIn('不足',camera.error)
                with self.assertRaises(ValueError): camera.reconstruct()
                self.assertFalse(list(Path(folder).glob('clip_*/capture.mkv')))
            finally: camera.close()

    def test_low_disk_and_stale_camera_reject_without_background_recording(self):
        with tempfile.TemporaryDirectory() as folder:
            camera=self.camera(folder)
            try:
                with patch('video_input.recording.shutil.disk_usage',return_value=SimpleNamespace(free=65*1024**2)):
                    with self.assertRaises(OSError): camera.start()
                with patch.object(camera.preview,'snapshot',side_effect=RuntimeError('disconnected')):
                    with self.assertRaises(RuntimeError): camera.start()
                self.assertEqual(camera.state,'idle')
                self.assertIsNone(camera.worker)
            finally: camera.close()

    def test_finalization_failure_keeps_partial_and_preview_recovers(self):
        with tempfile.TemporaryDirectory() as folder,patch('video_input.recording.inspect_clip',side_effect=ValueError('bad PTS')):
            camera=self.camera(folder)
            try:
                camera.start();self.wait(camera,'recording');camera.stop();self.wait(camera,'error')
                self.assertIn('bad PTS',camera.error)
                self.assertTrue(list(Path(folder).glob('clip_*/capture.partial.mkv')))
                self.assertEqual(camera.snapshot()[2],60)
            finally: camera.close()

    def test_http_mutations_need_token_and_double_start_is_conflict(self):
        with tempfile.TemporaryDirectory() as folder,patch('video_input.recording.inspect_clip',return_value=(
            dict(codec_name='mjpeg',width=1280,height=720,fps=60),[i/60 for i in range(121)])):
            camera=self.camera(folder);server=make_server(camera,port=0)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            url='http://127.0.0.1:'+str(server.server_port)
            try:
                page=urlopen(url+'/').read().decode()
                token=json.loads(page.split('const token=',1)[1].split(';',1)[0])
                def post(action,auth=True):
                    return urlopen(Request(url+'/api/'+action,method='POST',headers={'X-Session-Token':token} if auth else {}))
                with self.assertRaises(HTTPError) as exc: post('start',False)
                self.assertEqual(exc.exception.code,403)
                with self.assertRaises(HTTPError) as exc: post('cleanup',False)
                self.assertEqual(exc.exception.code,403)
                self.assertEqual(post('start').status,202);self.wait(camera,'recording')
                with self.assertRaises(HTTPError) as exc: post('cleanup')
                self.assertEqual(exc.exception.code,409)
                with self.assertRaises(HTTPError) as exc: post('start')
                self.assertEqual(exc.exception.code,409)
                state=json.load(urlopen(url+'/api/status'))
                self.assertEqual(state['state'],'recording')
                self.assertEqual(post('stop').status,202);self.wait(camera,'recorded')
                self.assertEqual(post('cleanup').status,200)
            finally: server.shutdown();server.server_close();thread.join();camera.close()

    def test_reconstruction_receives_only_completed_clip(self):
        with tempfile.TemporaryDirectory() as folder:
            camera=self.camera(folder)
            calls=[]
            camera.build=SimpleNamespace(status=lambda:dict(phase='idle',url=None),start=calls.append,close=lambda:None)
            try:
                with self.assertRaises(ValueError): camera.reconstruct()
                camera.state='recorded';camera.clip=dict(path='/validated/capture.mkv')
                camera.reconstruct()
                self.assertEqual(calls,['/validated/capture.mkv'])
            finally: camera.close()

    def test_automatic_stop_distinguishes_time_limit_from_camera_disconnect(self):
        class EndedPreview(FakePreview):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                if kwargs: self.process=SimpleNamespace(poll=lambda:0,returncode=0)
        for count,expected in ((1201,'recorded'),(121,'error')):
            with self.subTest(count=count),tempfile.TemporaryDirectory() as folder,patch(
                    'video_input.recording.inspect_clip',return_value=(
                    dict(codec_name='mjpeg',width=1280,height=720,fps=60),[i/60 for i in range(count)])):
                camera=CameraRecorder('/dev/video0',1280,720,60,folder,factory=EndedPreview)
                try:
                    camera.start();self.wait(camera,expected)
                    if expected=='recorded': self.assertEqual(camera.clip['stop_reason'],'time_limit')
                    else: self.assertIn('断流',camera.error)
                finally: camera.close()

    def stream(self,root,name):
        from common import sha
        path=Path(root)/name;path.mkdir()
        (path/'stream.json').write_text('{"phase":"stopped"}')
        frame=path/'slot_b/board/first_frame/frame.bin';frame.parent.mkdir(parents=True);frame.write_bytes(b'latest')
        (path/'latest.json').write_text(json.dumps(dict(complete=True,slot='slot_b',frame_sha256=sha(frame))))
        spare=path/'slot_a';spare.mkdir();(spare/'old').write_bytes(b'old candidate')
        return path

    def test_finished_stream_prunes_only_spare_after_latest_hash_check(self):
        with tempfile.TemporaryDirectory() as root:
            stream=self.stream(root,'stream_20261101_100000_aaaaaa')
            frame=stream/'slot_b/board/first_frame/frame.bin'
            frame.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):prune_stream_candidate(stream,root)
            self.assertTrue((stream/'slot_a/old').exists())
            frame.write_bytes(b'latest')
            result=prune_stream_candidate(stream,root)
            self.assertEqual(result['removed'],['slot_a']);self.assertGreater(result['freed_bytes'],0)
            self.assertEqual(frame.read_bytes(),b'latest')
            self.assertEqual(prune_stream_candidate(stream,root)['removed'],[])

    def test_restart_reclaims_old_streams_and_candidates_but_keeps_latest_and_clips(self):
        with tempfile.TemporaryDirectory() as root:
            camera=self.camera(root);calls=[]
            try:
                old=self.stream(root,'stream_20261101_100000_aaaaaa')
                current=self.stream(root,'stream_20261101_110000_bbbbbb')
                clip=self.clip(root,'clip_20261101_090000_cccccc')
                camera.build=SimpleNamespace(status=lambda:dict(phase='complete',mode='streaming',run_path=str(current)),
                    start_stream=lambda:calls.append('start'),close=lambda:None)
                free_before=123*1024**2
                def disk(path):
                    return SimpleNamespace(free=free_before+(21*1024**2 if not (current/'slot_a').exists() else 0))
                with patch('video_input.recording.shutil.disk_usage',side_effect=disk):camera.start_stream()
                self.assertEqual(calls,['start']);self.assertFalse(old.exists())
                self.assertFalse((current/'slot_a').exists());self.assertTrue((current/'slot_b').exists());self.assertTrue(clip.exists())
            finally:camera.close()

    def test_clear_all_deletes_current_latest_and_orphans_but_preserves_unknown_files(self):
        with tempfile.TemporaryDirectory() as root:
            camera=self.camera(root)
            try:
                clip=self.clip(root,'clip_20261101_100000_aaaaaa');scene=self.scene(root,'scene_20261101_100100_aaaaaa',clip)
                stream=self.stream(root,'stream_20261101_110000_bbbbbb')
                camera.restore_clip(clip/'capture.json');resets=[]
                camera.build=SimpleNamespace(status=lambda:dict(phase='ready'),reset=lambda:resets.append(True),close=lambda:None)
                orphan=Path(root)/'stream_20261101_120000_cccccc.log';orphan.write_bytes(b'failed launch')
                keep=Path(root)/'weights';keep.mkdir();(keep/'keep').write_bytes(b'weights')
                unknown=Path(root)/'clip_20261101_080000_dddddd';unknown.mkdir();(unknown/'notes').write_bytes(b'user notes')
                result=camera.clear_all_history()
                self.assertEqual(resets,[True]);self.assertFalse(clip.exists());self.assertFalse(scene.exists());self.assertFalse(stream.exists());self.assertFalse(orphan.exists())
                self.assertTrue((keep/'keep').exists());self.assertTrue((unknown/'notes').exists())
                self.assertIsNone(camera.clip);self.assertEqual(camera.state,'idle');self.assertEqual(camera.elapsed,0)
                self.assertFalse(camera.maintenance);self.assertEqual(result['kept'],[])
                self.assertEqual(camera.clear_all_history()['removed'],[])
            finally:camera.close()

    def test_clear_all_rejects_busy_and_releases_camera_lock_while_resetting_viewer(self):
        with tempfile.TemporaryDirectory() as root:
            camera=self.camera(root)
            try:
                camera.state='recording'
                with self.assertRaises(ValueError):camera.clear_all_history()
                camera.state='idle'
                camera.build=SimpleNamespace(status=lambda:dict(phase='streaming'),close=lambda:None)
                with self.assertRaises(ValueError):camera.clear_all_history()
                entered=threading.Event();release=threading.Event();errors=[]
                def reset():
                    camera.resume_preview();entered.set();release.wait(2)
                camera.build=SimpleNamespace(status=lambda:dict(phase='ready'),reset=reset,close=lambda:None)
                def clear():
                    try:camera.clear_all_history()
                    except Exception as exc:errors.append(str(exc))
                thread=threading.Thread(target=clear,daemon=True);thread.start()
                self.assertTrue(entered.wait(1));self.assertTrue(camera.status()['maintenance'])
                with self.assertRaises(ValueError):camera.start_stream()
                with self.assertRaises(ValueError):camera.cleanup_history()
                release.set();thread.join(2);self.assertFalse(thread.is_alive());self.assertEqual(errors,[])
            finally:camera.close()

    def test_clear_all_never_follows_internal_links(self):
        with tempfile.TemporaryDirectory() as root,tempfile.TemporaryDirectory() as outside:
            camera=self.camera(root)
            try:
                clip=self.clip(root,'clip_20261101_100000_aaaaaa');target=Path(outside)/'keep';target.write_bytes(b'outside')
                try:(clip/'linked').symlink_to(target)
                except OSError:self.skipTest('Symlink creation unavailable in this Windows session')
                result=camera.clear_all_history()
                self.assertIn(clip.name,result['skipped']);self.assertTrue(clip.exists());self.assertEqual(target.read_bytes(),b'outside')
            finally:camera.close()

    def test_clear_all_http_requires_token_and_explicit_confirmation(self):
        with tempfile.TemporaryDirectory() as root:
            camera=self.camera(root);server=make_server(camera,port=0)
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            url=f'http://127.0.0.1:{server.server_port}'
            try:
                clip=self.clip(root,'clip_20261101_100000_aaaaaa');camera.restore_clip(clip/'capture.json')
                token=json.loads(urlopen(url).read().decode().split('const token=',1)[1].split(';',1)[0])
                def post(headers):return urlopen(Request(url+'/api/cleanup/all',method='POST',headers=headers))
                with self.assertRaises(HTTPError) as error:post({})
                self.assertEqual(error.exception.code,403)
                with self.assertRaises(HTTPError) as error:post({'X-Session-Token':token})
                self.assertEqual(error.exception.code,409);self.assertTrue(clip.exists())
                with post({'X-Session-Token':token,'X-Confirm-Delete':'all-session-records'}) as response:
                    self.assertEqual(response.status,200);self.assertIn(clip.name,json.load(response)['removed'])
                self.assertFalse(clip.exists());self.assertIsNone(camera.status()['clip'])
            finally:server.shutdown();server.server_close();thread.join();camera.close()


if __name__=='__main__': unittest.main()
