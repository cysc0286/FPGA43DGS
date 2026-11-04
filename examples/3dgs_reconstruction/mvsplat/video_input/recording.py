"""Button-controlled UVC recording, sharing one input with the preview output."""
import json
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import threading
import time

from common import save, sha
from video_input.capture import inspect_clip, validate_capture
from video_input.preview import Preview


SESSION_PATTERN=r'(?:(?:clip|scene|stream)_\d{8}_\d{6}_[0-9a-f]{6}|upload_\d{8}_\d{6}_[0-9a-f]{8})'


def session_marker(path):
    if path.name.startswith('upload_'):return 'upload.json'
    if path.name.startswith('clip_'):return 'capture.json'
    return 'stream.json' if path.name.startswith('stream_') else 'events.json'


def checked_path(path,root):
    """Only regular files/directories within the verified root, without links."""
    if path.is_symlink() or getattr(path,'is_junction',lambda:False)():return False
    if root not in path.resolve().parents:return False
    if path.is_dir():return all(checked_path(p,root) for p in path.iterdir())
    return path.is_file()


def remove_path(path,root):
    if not checked_path(path,root):raise ValueError('历史记录路径已改变，未执行删除')
    if path.is_dir():
        total=sum(remove_path(p,root) for p in path.iterdir())
        path.rmdir();return total
    size=path.stat().st_size;path.unlink();return size


def prune_stream_candidate(folder,root):
    """An ended stream needs only its hash-verified latest slot, not its spare."""
    folder,root=Path(folder),Path(root).resolve()
    if (folder.is_symlink() or getattr(folder,'is_junction',lambda:False)()
            or folder.resolve().parent!=root or not re.fullmatch(r'stream_\d{8}_\d{6}_[0-9a-f]{6}',folder.name)):
        raise ValueError('连续会话目录边界不一致')
    try:
        state=json.loads((folder/'stream.json').read_text(encoding='utf-8'))
        latest=json.loads((folder/'latest.json').read_text(encoding='utf-8'))
    except FileNotFoundError:return dict(removed=[],freed_bytes=0)
    if state.get('phase')!='stopped':return dict(removed=[],freed_bytes=0)
    slot=latest.get('slot')
    if slot not in ('slot_a','slot_b') or not latest.get('complete'):raise ValueError('连续结果指针无效，未回收候选')
    frame=folder/slot/'board/first_frame/frame.bin'
    if (not checked_path(frame,root) or sha(frame)!=latest.get('frame_sha256')):
        raise ValueError('最新结果校验失败，未回收候选')
    spare=folder/('slot_b' if slot=='slot_a' else 'slot_a')
    if not spare.exists() and not spare.is_symlink():return dict(removed=[],freed_bytes=0)
    return dict(removed=[spare.name],freed_bytes=remove_path(spare,root))


class CameraRecorder:
    def __init__(self, device, width, height, fps, out, *, max_seconds=20, factory=Preview,
                 auto_process=False):
        validate_capture(width, height, fps, max_seconds, 0)
        self.device, self.width, self.height, self.fps = device, width, height, fps
        self.out, self.max_seconds, self.factory = Path(out), max_seconds, factory
        self.out.mkdir(parents=True, exist_ok=True)
        self.storage_root=self.out.resolve()
        self.lock = threading.RLock()
        self.state, self.error, self.clip = 'idle', None, None
        self.preview, self.worker, self.build = None, None, None
        self.stop_event, self.closed = threading.Event(), False
        self.maintenance=False
        self.started, self.elapsed = None, 0.
        self.auto_process = auto_process
        self.video_complete_monotonic = None
        self.video_complete_epoch = None
        self.input_kind='usb_camera'
        self.upload=None
        self.preview_error=None
        try:self.resume_preview()
        except (OSError,ValueError,subprocess.SubprocessError) as exc:
            # The PC-file entry remains usable when the USB camera is absent.
            self.preview_error=str(exc)

    def resume_preview(self):
        with self.lock:
            if self.preview is None and not self.closed:
                self.preview = self.factory(self.device, self.width, self.height, self.fps,
                                            self.out/'preview.log')
                self.preview_error=None

    def pause_preview(self):
        with self.lock:
            preview, self.preview = self.preview, None
        if preview:
            preview.close()

    def snapshot(self):
        with self.lock:
            if self.preview is None:
                try:self.resume_preview()
                except (OSError,ValueError,subprocess.SubprocessError) as exc:
                    self.preview_error=str(exc)
                    raise RuntimeError('USB 摄像头不可用：'+str(exc)) from exc
                if self.preview is None:raise RuntimeError('摄像头预览不可用')
            return self.preview.snapshot()

    def packet(self):
        with self.lock:
            if self.preview is None:raise RuntimeError(self.preview_error or '摄像头预览尚未就绪')
            return self.preview.packet()

    def start_stream(self):
        with self.lock:
            if self.closed or self.maintenance or self.state in ('starting','recording','stopping','uploading'):
                raise ValueError('请先结束当前录制并等待保存完成')
            if not self.build:raise ValueError('连续重建入口尚未配置')
            build=self.build.status()
            if build['phase'] in ('initializing','reconstructing','streaming','stopping'):
                raise ValueError('请先结束当前采集或重建')
            self.packet()
            # Finished sessions otherwise accumulate another pair of ~20 MiB
            # slots per restart. Keep the latest usable result and current run.
            retained=self.cleanup_history(stream_only=True)
            for name in retained['kept']:
                if name.startswith('stream_'):prune_stream_candidate(self.storage_root/name,self.storage_root)
            if shutil.disk_usage(self.out).free<128*1024**2:
                raise OSError('连续模式需至少 128 MiB 可用空间，请先清理历史记录')
            self.input_kind='usb_camera';self.upload=None;self.clip=None
            self.video_complete_monotonic=self.video_complete_epoch=None
            self.state,self.error,self.elapsed,self.started='idle',None,0.,None
            self.build.start_stream()

    def stop_stream(self):
        with self.lock:
            if not self.build:raise ValueError('连续重建入口尚未配置')
            self.build.stop_stream()

    def status(self):
        with self.lock:
            elapsed = time.monotonic()-self.started if self.started and self.state in ('recording','stopping') else self.elapsed
            clip = None if self.clip is None else {k:self.clip[k] for k in
                ('frames','duration_seconds','measured_fps','stop_label')}
            build=self.build.status() if self.build else dict(phase='idle',url=None)
            build['matches_clip']=bool(self.clip and build.get('clip_path')==self.clip['path'])
            # Include stop/finalization latency, before a validated clip can be
            # handed to the warmed inference process. Scene events remain a
            # separate VIDEO_COMPLETE->FRAME_COMPLETE measurement.
            post_elapsed=(max(0.,time.monotonic()-self.video_complete_monotonic)
                          if self.video_complete_monotonic is not None else 0.)
            if build.get('processing_complete') and build['matches_clip']:
                post_elapsed=build['processing_elapsed_seconds']
            if build.get('scene_restored'):post_elapsed=None
            return dict(state=self.state, elapsed_seconds=max(0.,elapsed), max_seconds=self.max_seconds,
                        free_mib=shutil.disk_usage(self.out).free/1024**2, clip=clip, error=self.error,
                        preview_error=self.preview_error,
                        build=build,maintenance=self.maintenance,input_kind=self.input_kind,
                        upload=({k:v for k,v in self.upload.items() if k!='frame_pts_seconds'} if self.upload else None),
                        processing_started_epoch=None if build.get('scene_restored') else self.video_complete_epoch,
                        postcapture_elapsed_seconds=post_elapsed,auto_process=self.auto_process)

    def start(self):
        with self.lock:
            if self.closed or self.maintenance or self.state in ('starting','recording','stopping','uploading'):
                raise ValueError('已有录制正在进行，请先结束')
            if self.build and self.build.status()['phase'] in ('initializing','reconstructing','streaming','stopping'):
                raise ValueError('请等待当前重建完成')
            self.snapshot()  # Refuse to start on a disconnected camera.
            available = shutil.disk_usage(self.out).free-64*1024**2
            if available < max(8*1024**2,self.width*self.height*2):
                raise OSError('空间不足，请点击“清理历史记录”；需保留 64 MiB 系统空间')
            folder = self.out/('clip_'+time.strftime('%Y%m%d_%H%M%S')+'_'+secrets.token_hex(3))
            folder.mkdir(exist_ok=False)
            self.state, self.error, self.clip, self.elapsed, self.started = 'starting', None, None, 0., None
            self.input_kind='usb_camera';self.upload=None
            self.video_complete_monotonic = self.video_complete_epoch = None
            self.stop_event.clear()
            self.worker=threading.Thread(target=self.record,args=(folder,min(256*1024**2,available)),daemon=True)
            self.worker.start()

    def prepare_upload(self):
        """Explicit optional warmup; the single job releases the other input's owner."""
        with self.lock:
            if self.closed or self.maintenance or self.state in ('starting','recording','stopping','uploading'):
                raise ValueError('当前输入尚未结束，不能切换预热模式')
            if not self.build:raise ValueError('板端重建服务未配置')
            self.build.prime(input_kind='uploaded_video')

    def receive_upload(self,stream,length,filename):
        """PC sends bytes only; validation, decode, inference and rendering stay on board."""
        from video_input.upload import stream_receive
        with self.lock:
            if self.closed or self.maintenance or self.state in ('starting','recording','stopping','uploading'):
                raise ValueError('已有输入正在进行，请等待完成')
            if not self.build:raise ValueError('板端重建服务未配置')
            if self.build.status()['phase'] in ('initializing','reconstructing','streaming','stopping'):
                raise ValueError('请等待当前预热或重建完成')
            self.state,self.error,self.clip='uploading',None,None
            self.input_kind='uploaded_video';self.upload=None
            self.video_complete_monotonic=self.video_complete_epoch=None
        preview_paused=False
        def received(monotonic,epoch):
            with self.lock:
                if self.closed:raise RuntimeError('服务已关闭，未提交上传视频')
                self.video_complete_monotonic=monotonic
                self.video_complete_epoch=epoch
        try:
            record=stream_receive(stream,length,self.out,filename,on_received=received)
            with self.lock:
                if self.closed:raise RuntimeError('服务已关闭，未提交上传视频')
                # A preheated job submits directly to its retained stdin and
                # does not enter ReconstructionJob.run again. Release USB
                # preview here for both the warm and cold dispatch paths.
                preview_paused=True
                self.pause_preview()
                self.clip=self.upload=record
                self.state='recorded';self.elapsed=record['duration_seconds']
                # The receiver already published these exact receipt timestamps
                # before final storage/probing. Never start a later success clock.
                if self.video_complete_monotonic is None:
                    received(record['upload_complete_monotonic'],record['upload_complete_epoch'])
                self.build.start(record['path'],input_kind='uploaded_video')
            return record
        except (OSError,ValueError,RuntimeError,subprocess.SubprocessError):
            with self.lock:
                self.state='error';self.error='上传或视频校验失败，本次视频未完成重建。'
            if preview_paused:
                try:self.resume_preview()
                except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
                    with self.lock:self.preview_error='USB 预览恢复失败：'+str(exc)
            raise

    def stop(self):
        with self.lock:
            if self.state not in ('starting','recording','stopping'):
                raise ValueError('当前没有正在录制的短片')
            self.state='stopping'
            if self.video_complete_monotonic is None:
                self.video_complete_monotonic = time.monotonic()
                self.video_complete_epoch = time.time()
            self.stop_event.set()

    def record(self, folder, limit):
        started=time.monotonic()
        partial, final=folder/'capture.partial.mkv',folder/'capture.mkv'
        record=dict(complete=False, requested=dict(width=self.width,height=self.height,fps=self.fps),
                    storage_limit_bytes=limit,timestamp_scope='demuxed PTS; not exposure timestamps')
        preview=None
        try:
            if self.auto_process and self.build and hasattr(self.build, 'prepare_capture'):
                self.build.prepare_capture()
                if self.stop_event.is_set():
                    raise ValueError('采集开始前已取消')
                started = time.monotonic()
            self.pause_preview()
            preview=self.factory(self.device,self.width,self.height,self.fps,folder/'ffmpeg.log',
                recording=partial,storage_limit=limit,max_seconds=self.max_seconds)
            with self.lock: self.preview=preview
            deadline=time.monotonic()+12
            while True:
                try: preview.snapshot(); break
                except RuntimeError:
                    if preview.process.poll() is not None or time.monotonic()>deadline:
                        raise RuntimeError('摄像头录制启动失败，请检查连接和日志')
                    if self.stop_event.wait(.05): break
            with self.lock:
                self.started=time.monotonic()
                if not self.stop_event.is_set(): self.state='recording'
            boundary=None
            while preview.process.poll() is None and not self.stop_event.wait(.1):
                if (partial.exists() and partial.stat().st_size>=max(0,limit-8*1024**2)
                        or shutil.disk_usage(folder).free<=72*1024**2):
                    boundary='storage_limit'
                    break
                if time.monotonic()-started>self.max_seconds+15:
                    raise TimeoutError('摄像头录制超时')
            manual=self.stop_event.is_set()
            with self.lock:
                if self.video_complete_monotonic is None:
                    self.video_complete_monotonic = time.monotonic()
                    self.video_complete_epoch = time.time()
            preview.finish_recording()
            if preview.process.returncode not in (0,255):
                raise RuntimeError('摄像头或录像写入异常，请检查日志')
            with self.lock: self.state='stopping'
            stream,times=inspect_clip(partial)
            duration=times[-1]-times[0]
            measured=(len(times)-1)/duration
            if duration<1.:
                raise ValueError('短片不足 1 秒，请重新录制并缓慢水平转满一圈')
            if (stream['codec_name']!='mjpeg' or (stream['width'],stream['height'])!=(self.width,self.height)
                    or abs(stream['fps']-self.fps)>max(.1,self.fps*.01) or measured<self.fps*.9):
                raise ValueError('摄像头模式或帧率未通过检查，短片未交付重建')
            if (not manual and boundary is None and duration<self.max_seconds-.3
                    and partial.stat().st_size<limit-max(1024**2,self.width*self.height*2)):
                raise RuntimeError('摄像头提前断流，短片未交付重建，请检查连接')
            partial.rename(final)
            reason='manual' if manual else (boundary or ('time_limit' if duration>=self.max_seconds-.3 else 'storage_limit'))
            record.update(complete=True,path=str(final.resolve()),sha256=sha(final),frames=len(times),actual=stream,
                frame_pts_seconds=times,duration_seconds=duration,measured_fps=measured,
                stop_reason=reason,stop_label={'manual':'手动结束','time_limit':'达到时长上限，自动结束','storage_limit':'达到空间上限，自动结束'}[reason])
            with self.lock: self.clip=record; self.elapsed=duration
        except (OSError,RuntimeError,ValueError,subprocess.SubprocessError,TimeoutError) as exc:
            record['error']=str(exc)
            with self.lock: self.error=str(exc)
        finally:
            if preview: preview.close()
            with self.lock:
                if self.preview is preview: self.preview=None
            record['acquisition_seconds']=time.monotonic()-started
            try: save(folder/'capture.json',record)
            except OSError as exc:
                record['complete']=False
                with self.lock: self.clip=None; self.error='短片记录保存失败：'+str(exc)
            try: self.resume_preview()
            except OSError as exc:
                with self.lock: self.error='预览恢复失败：'+str(exc)
            with self.lock: self.state='recorded' if record['complete'] else 'error'
        # Never submit a partial, disconnected, or unverified clip. Dispatch
        # outside the camera lock to avoid camera/job lock inversion.
        if record['complete'] and self.auto_process and not self.closed:
            try:
                self.pause_preview()
                self.build.start(self.clip['path'])
            except (OSError, RuntimeError, ValueError) as exc:
                with self.lock: self.error='自动处理未启动：'+str(exc)
                self.resume_preview()

    def reconstruct(self):
        with self.lock:
            if self.maintenance:raise ValueError('正在清理记录，请稍候')
            if not self.build: raise ValueError('重建入口尚未配置')
            if self.state!='recorded' or not self.clip: raise ValueError('请先结束录制并等待短片检查完成')
            self.build.start(self.clip['path'],input_kind=self.input_kind)

    def restore_clip(self,metadata):
        """Restore a completed owned clip across a controlled UI restart."""
        metadata=Path(metadata)
        linked=lambda p:p.is_symlink() or getattr(p,'is_junction',lambda:False)()
        if (linked(self.out) or self.out.resolve()!=self.storage_root or linked(metadata)
                or linked(metadata.parent)):
            raise ValueError('恢复录像的目录或文件边界不一致')
        metadata=metadata.resolve()
        if self.storage_root not in metadata.parents or not metadata.is_file():
            raise ValueError('恢复录像必须位于当前录制目录中')
        record=json.loads(metadata.read_text(encoding='utf-8'))
        if not isinstance(record,dict):raise ValueError('恢复录像的元数据格式无效')
        uploaded=(metadata.name=='upload.json' or metadata.parent.name.startswith('upload_')
                  or record.get('source_kind')=='uploaded_video'
                  or record.get('schema')=='mvsplat_uploaded_video_v1')
        if uploaded:
            from video_input.upload import VIDEO_SUFFIXES
            if (metadata.name!='upload.json' or metadata.parent.parent!=self.storage_root
                    or not re.fullmatch(r'upload_\d{8}_\d{6}_[0-9a-f]{8}',metadata.parent.name)
                    or record.get('schema')!='mvsplat_uploaded_video_v1'
                    or record.get('source_kind')!='uploaded_video' or record.get('complete') is not True
                    or not isinstance(record.get('path'),str)):
                raise ValueError('恢复上传视频的来源或元数据校验失败')
            path=Path(record['path'])
            if (not path.is_absolute() or path.parent!=metadata.parent
                    or path.name not in {'video'+suffix for suffix in VIDEO_SUFFIXES|{'.video'}}
                    or not checked_path(path,self.storage_root)
                    or str(path.resolve())!=record['path']
                    or sha(path)!=record.get('sha256')
                    or type(record.get('declared_bytes')) is not int
                    or record['declared_bytes']<=0
                    or type(record.get('received_bytes')) is not int
                    or record['received_bytes']!=record['declared_bytes']
                    or path.stat().st_size!=record['received_bytes']):
                raise ValueError('恢复上传视频的文件校验失败')
        else:
            path=metadata.parent/'capture.mkv'
            mode=record.get('requested',{})
            if (metadata.name!='capture.json' or not isinstance(mode,dict)
                    or not record.get('complete') or not checked_path(path,self.storage_root)
                    or str(path.resolve())!=record.get('path') or sha(path)!=record.get('sha256')
                    or (mode.get('width'),mode.get('height'),mode.get('fps'))!=(self.width,self.height,self.fps)):
                raise ValueError('恢复录像的文件校验或摄像头模式不一致')
        if any(key not in record for key in ('duration_seconds','frames','measured_fps','stop_label')):
            raise ValueError('恢复录像的文件校验或摄像头模式不一致')
        with self.lock:
            if (self.closed or self.maintenance or self.state in ('starting','recording','stopping','uploading')
                    or self.build and self.build.status()['phase'] in ('initializing','reconstructing','streaming','stopping')):
                raise ValueError('请先结束当前输入或重建')
            self.clip,self.state,self.error=record,'recorded',None
            self.elapsed=record['duration_seconds']
            self.input_kind='uploaded_video' if uploaded else 'usb_camera'
            self.upload=record if uploaded else None
            # These timestamps belong to a previous request/process lifetime.
            # Reopening a saved scene must not reuse them as a new upload clock.
            self.video_complete_monotonic=self.video_complete_epoch=self.started=None

    def cleanup_history(self, *, stream_only=False):
        """Delete only owned inactive sessions; retain current and last usable result."""
        with self.lock:
            if self.closed or self.maintenance or self.state in ('starting','recording','stopping','uploading'):
                raise ValueError('请先结束录制并等待保存完成，再清理历史记录')
            build=self.build.status() if self.build else {}
            if build.get('phase') in ('initializing','reconstructing','streaming','stopping'):
                raise ValueError('请等待当前重建完成，再清理历史记录')
            root=self.storage_root
            # Refuse a substituted root. Deletion must stay inside the configured
            # session directory, including on Windows junction/reparse surfaces.
            if self.out.is_symlink() or self.out.resolve()!=root:
                raise ValueError('录制目录边界不一致，未执行清理')
            pattern=SESSION_PATTERN
            entries={p.name:p for p in root.iterdir() if re.fullmatch(pattern,p.name)
                     and p.is_dir() and not p.is_symlink() and p.resolve().parent==root
                     and (p/session_marker(p)).is_file()}
            protected=set()
            def retain(path):
                if not path:return
                candidate=Path(path).resolve()
                if root not in candidate.parents:return
                name=candidate.relative_to(root).parts[0]
                if name in entries:protected.add(name)
            if self.clip:retain(self.clip['path'])
            retain(build.get('clip_path'));retain(build.get('run_path'))
            clips=[];scenes=[];streams=[]
            for name,path in entries.items():
                try:
                    if name.startswith(('clip_','upload_')):
                        data=json.loads((path/session_marker(path)).read_text(encoding='utf-8'))
                        video=Path(data.get('path',''))
                        if (data.get('complete') and video.parent==path and video.is_file()
                                and checked_path(video,root)):
                            clips.append(path)
                    elif name.startswith('stream_'):
                        data=json.loads((path/'latest.json').read_text(encoding='utf-8'))
                        if data.get('slot') not in ('slot_a','slot_b'):continue
                        frame=path/data['slot']/'board/first_frame/frame.bin'
                        if not frame.is_symlink() and path in frame.resolve().parents and sha(frame)==data.get('frame_sha256'):streams.append(path)
                    elif (path/'renderer_input/model.ply').is_file():
                        data=json.loads((path/'inference/inference.json').read_text(encoding='utf-8'))
                        events=json.loads((path/'events.json').read_text(encoding='utf-8'))['events']
                        hashes={e['frame_sha256'] for e in events if e.get('event')=='FRAME_COMPLETE' and e.get('frame_sha256')}
                        verified=any(not frame.is_symlink() and path in frame.resolve().parents and sha(frame) in hashes
                                     for frame in path.glob('board/*/frame.bin'))
                        if data.get('complete') and verified:scenes.append(path)
                except (OSError,ValueError,KeyError,TypeError,AttributeError):continue
            if clips:retain(max(clips,key=lambda p:(p.stat().st_mtime_ns,p.name)))
            if streams:retain(max(streams,key=lambda p:((p/'latest.json').stat().st_mtime_ns,p.name)))
            if scenes:
                last=max(scenes,key=lambda p:((p/'renderer_input/model.ply').stat().st_mtime_ns,p.name));retain(last)
                # The receipt records the original video path independently of
                # the current recording (which may not have been rebuilt yet).
                try:
                    events=json.loads((last/'events.json').read_text(encoding='utf-8'))['events']
                    sources=[event['path'] for event in events if event.get('event')=='VIDEO_COMPLETE' and event.get('path')]
                    if not sources:raise ValueError('Missing original video receipt')
                    for source in sources:retain(source)
                except (OSError,ValueError,KeyError,TypeError):
                    # Without provenance we cannot know which clip the last
                    # model depends on; leave clips intact rather than guess.
                    protected.update(p.name for p in clips)
            selected=[p for name,p in sorted(entries.items()) if name not in protected
                      and (not stream_only or name.startswith('stream_'))]
            # Inspect every target before deleting any, never follow links.
            # A symlink/junction inside a candidate causes that entry to be kept.
            def checked(path):
                if path.is_symlink() or getattr(path,'is_junction',lambda:False)():return False
                if root not in path.resolve().parents:return False
                if path.is_dir():return all(checked(p) for p in path.iterdir())
                return path.is_file()
            selected=[p for p in selected if checked(p)]
            sidecars=[]
            for path in selected:
                for suffix in ('.log','.measurement.json'):
                    sidecar=root/(path.name+suffix)
                    if sidecar.exists() and checked(sidecar):sidecars.append(sidecar)
            removed=[];total=0
            def remove(path):
                nonlocal total
                # Native pathlib operations throughout: no computed shell rm.
                if (path.is_symlink() or getattr(path,'is_junction',lambda:False)()
                        or root not in path.resolve().parents):
                    raise ValueError('历史记录路径已改变，已停止清理')
                if path.is_dir():
                    for child in path.iterdir():remove(child)
                    path.rmdir()
                else:
                    size=path.stat().st_size;path.unlink();total+=size
            for path in selected+sidecars:
                if not checked(path):raise ValueError('历史记录路径已改变，请刷新后重试')
                remove(path);removed.append(path.name)
            return dict(removed=removed,freed_bytes=total,kept=sorted(protected),
                        free_mib=shutil.disk_usage(root).free/1024**2)

    def clear_all_history(self):
        """Close the inactive viewer and delete all recognized owned sessions."""
        with self.lock:
            if self.closed or self.maintenance or self.state in ('starting','recording','stopping','uploading'):
                raise ValueError('请先结束录制并等待保存完成')
            if self.build and self.build.status()['phase'] in ('initializing','reconstructing','streaming','stopping'):
                raise ValueError('请先结束连续采集或等待重建完成，再清空全部记录')
            root=self.storage_root
            if self.out.is_symlink() or getattr(self.out,'is_junction',lambda:False)() or self.out.resolve()!=root:
                raise ValueError('录制目录边界不一致，未执行清空')
            self.maintenance=True
        try:
            # The worker's finally resumes preview under camera.lock. Do not
            # hold that lock while cancelling/joining it or closing its viewer.
            if self.build:self.build.reset()
            with self.lock:
                if self.out.resolve()!=root or self.out.is_symlink():raise ValueError('目录已改变，未执行清空')
                pattern=SESSION_PATTERN
                selected=[];skipped=[]
                for path in sorted(root.iterdir()):
                    if not re.fullmatch(pattern,path.name):continue
                    marker=session_marker(path)
                    if not path.is_dir() or not (path/marker).is_file():continue
                    if checked_path(path,root):selected.append(path)
                    else:skipped.append(path.name)
                # Pair receipts/logs also cover failed launches with no folder.
                for path in sorted(root.iterdir()):
                    if re.fullmatch(pattern+r'\.(log|measurement\.json)',path.name):
                        if checked_path(path,root):selected.append(path)
                        else:skipped.append(path.name)
                self.clip=None;self.state='idle';self.error=None;self.started=None;self.elapsed=0.
                self.upload=None;self.input_kind='usb_camera'
                self.video_complete_monotonic=self.video_complete_epoch=None
                total=0;removed=[]
                for path in selected:
                    total+=remove_path(path,root);removed.append(path.name)
                return dict(removed=removed,skipped=skipped,freed_bytes=total,kept=[],free_mib=shutil.disk_usage(root).free/1024**2)
        finally:
            with self.lock:self.maintenance=False

    def close(self):
        with self.lock: self.closed=True; self.stop_event.set()
        if self.worker: self.worker.join(45)
        if self.build: self.build.close()
        self.pause_preview()
