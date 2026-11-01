"""Button-controlled UVC recording, sharing one input with the preview output."""
import json
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from common import save, sha
from video_input.capture import inspect_clip, validate_capture
from video_input.preview import Preview


PAGE = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>摄像头采集与重建</title>
<style>:root{font-family:system-ui;color:#17212f;background:#eef2f6;color-scheme:light}
*{box-sizing:border-box}body{max-width:1320px;margin:24px auto;padding:0 16px}h1{font-size:26px;margin-bottom:8px}
.workspace{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px;align-items:start}.workspace>.card{margin:0;min-width:0}#capture-card{order:1}#continuous-card{order:2}
.hint{color:#526175;line-height:1.6}.card{background:white;border-radius:16px;padding:16px;margin:16px 0;box-shadow:0 4px 20px #17212f0a}
img{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#111827;border-radius:10px}
.controls{display:flex;flex-wrap:wrap;gap:10px;margin-top:16px}.toolbar{background:white;padding:8px 0 16px;margin:0}button,a.button{font:inherit;border:0;border-radius:9px;padding:12px 20px;background:#e6ebf2;color:#17212f;text-decoration:none;cursor:pointer}
#start,#rebuild,a.button{background:#1764cf;color:white}#stop{background:#d83344;color:white}button:disabled{opacity:.4;cursor:default}
#clear-all{background:#b21e2f;color:white}
#time{font-size:30px;font-variant-numeric:tabular-nums}#record-status{font-weight:600}#error{color:#b21e2f;white-space:pre-wrap;overflow-wrap:anywhere}
@media(max-width:800px){.workspace{grid-template-columns:minmax(0,1fr)}}
@media(max-width:480px){body{padding:0 10px;margin:16px auto}.card{padding:12px}button{flex:1;padding:12px 8px}.toolbar button{flex:1 1 calc(50% - 10px)}h1{font-size:23px}}</style>
<h1>摄像头采集与重建</h1><p class="hint">先调清焦点，保持场景静止。连续模式下缓慢横向平移，并保持对准起始区域。</p>
<div class="workspace"><section class="card" id="continuous-card"><h2>连续采集与渲染</h2><p class="hint">持续预览、更新局部点云，后台分批生成高斯场景；更新期间可继续操作上一个场景。使用估算内参、128×128 输出。只保留近期窗口和最新结果，不保存整段录像。单次会话含预热和查看最多 30 分钟；关闭页面不会停止采集。</p>
<div class="controls"><button id="live-start" disabled>开始连续重建</button><button id="live-stop" disabled>结束连续采集</button></div>
<p id="live-status" role="status">等待开始</p><iframe id="live-viewer" title="连续点云与高斯场景" hidden style="border:0;width:100%;height:850px"></iframe></section>
<section class="card" id="capture-card"><h2>原始预览与短片录制</h2><div class="controls toolbar"><button id="start" disabled>开始录制</button><button id="stop" disabled>结束录制</button><button id="rebuild" disabled>重建场景</button><button id="pause">暂停显示</button></div>
<img id="camera" alt="摄像头实时预览"><p id="preview-status" class="hint">正在连接摄像头…</p>
<div id="time">00:00.0</div><p id="record-status" role="status">正在读取状态…</p>
<p id="limits" class="hint"></p><p id="clip" class="hint"></p></section></div>
<section class="card"><h2>重建场景</h2><p class="hint">录制结束后可重建，也可以重新录制。当前使用估算内参，输出为 128×128。</p>
<div class="controls"><a id="viewer" class="button" hidden target="_blank">打开交互场景</a></div>
<p id="build-status" role="status">等待短片</p><p id="error" role="alert"></p>
<details id="diagnostics" hidden><summary>查看诊断详情</summary><pre id="error-detail" style="white-space:pre-wrap;overflow-wrap:anywhere"></pre></details></section>
<section class="card"><h2>板端存储</h2><p class="hint">清理本页面产生的旧录像、失败记录和旧重建文件，保留当前短片及最近成功的场景和来源录像。删除后不可恢复；模型权重和系统文件保留。</p>
<div class="controls"><button id="cleanup" disabled>清理历史记录（保留最新）</button><button id="clear-all" disabled>清空全部记录（含最新结果）</button></div>
<p class="hint">清空全部记录会关闭当前场景，并删除本页面的全部录像、重建结果和会话日志，无法恢复。</p><p id="cleanup-status" class="hint" role="status"></p></section>
<script>const token=__TOKEN__;let active=true,busy=false,last=null,actionError='';const $=id=>document.getElementById(id);
async function command(action){if(busy)return;busy=true;actionError='';buttons();$('error').textContent='';try{
const headers={'X-Session-Token':token};if(action==='cleanup/all')headers['X-Confirm-Delete']='all-session-records';
const r=await fetch('/api/'+action,{method:'POST',headers});const d=await r.json();if(!r.ok)throw Error(d.error);if(action==='cleanup'||action==='cleanup/all')$('cleanup-status').textContent=(action==='cleanup/all'?'已清空 ':'已清理 ')+d.removed.length+' 项，释放 '+(d.freed_bytes/1048576).toFixed(1)+' MiB；当前可用 '+d.free_mib.toFixed(0)+' MiB。'+(d.skipped&&d.skipped.length?' 有 '+d.skipped.length+' 项因路径或链接异常而保留。':'');
}catch(e){actionError=e.message;$('error').textContent=actionError}finally{busy=false;await status()}}
function buttons(){if(!last)return;const building=last.maintenance||['initializing','reconstructing','streaming','stopping'].includes(last.build.phase);
$('start').disabled=busy||building||['starting','recording','stopping'].includes(last.state);
$('stop').disabled=busy||!['starting','recording'].includes(last.state);$('rebuild').disabled=busy||building||!last.clip||['starting','recording','stopping'].includes(last.state);
$('cleanup').disabled=busy||building||['starting','recording','stopping'].includes(last.state);
$('clear-all').disabled=$('cleanup').disabled;
$('live-start').disabled=busy||building||['starting','recording','stopping'].includes(last.state);
$('live-stop').disabled=busy||last.build.mode!=='streaming'||!['initializing','streaming'].includes(last.build.phase)}
async function status(){try{const r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw Error('状态连接中断');last=await r.json();
const labels={idle:'准备就绪',starting:'正在启动录制，请稍候…',recording:'● 正在录制',stopping:'正在保存和检查短片…',recorded:'短片已保存',error:'本次录制未完成'};
$('record-status').textContent=labels[last.state]||last.state;const t=last.elapsed_seconds;
$('time').textContent=String(Math.floor(t/60)).padStart(2,'0')+':'+(t%60).toFixed(1).padStart(4,'0');
$('limits').textContent='短片模式：单段最长 '+last.max_seconds+' 秒；可用空间 '+last.free_mib.toFixed(0)+' MiB。达到时长或空间上限会自动结束。暂停显示和关闭页面均不会停止录制。';
$('clip').textContent=last.clip?last.clip.frames+' 帧 · '+last.clip.duration_seconds.toFixed(2)+' 秒 · '+last.clip.measured_fps.toFixed(2)+' fps · '+last.clip.stop_label:'';
const phases={idle:'等待短片',initializing:'正在加载模型和 FPGA…',reconstructing:'正在选帧、重建与渲染，请稍候…',streaming:'连续更新中',stopping:'已请求停止采样，正在完成当前更新…',ready:'场景已就绪，可以交互查看',complete:'重建会话已结束',error:'重建失败，可查看原因后重新录制'};
$('build-status').textContent=last.build.mode==='streaming'&&last.build.phase==='ready'&&!last.build.live.has_scene?'本次采集已结束，尚无有效场景':(last.build.mode!=='streaming'&&last.build.phase==='ready'&&!last.build.matches_clip?'上次场景已就绪；当前短片尚未重建':phases[last.build.phase]||last.build.phase);$('viewer').hidden=!last.build.url;$('viewer').href=last.build.url||'#';
const live=last.build.live||{},liveLabels={warming:'模型预热中',sampling:'等待有效视角',geometry:'估计位姿与点云',inference:'生成高斯模型',publishing:'校验并切换场景',stopped:'采集已结束',error:'更新已停止'};
$('live-status').textContent=last.build.mode==='streaming'?(last.build.phase==='stopping'?phases.stopping:(liveLabels[live.phase]||phases[last.build.phase]))+' · 已更新 '+(live.version||0)+' 次'+(live.last_result?' · 最近窗口到首帧 '+live.last_result.window_to_frame_seconds.toFixed(1)+' 秒':'')+(live.message?' · '+live.message:'')+(live.source&&live.source.error?' · 采集异常：'+live.source.error:''):'等待开始';
const liveUrl=last.build.mode==='streaming'&&last.build.url?last.build.url:'';$('live-viewer').hidden=!liveUrl;if(liveUrl&&$('live-viewer').getAttribute('src')!==liveUrl)$('live-viewer').src=liveUrl;else if(!liveUrl)$('live-viewer').removeAttribute('src');
$('error').textContent=last.error||last.build.error||actionError;
$('diagnostics').hidden=!last.build.details;$('error-detail').textContent=last.build.details||'';buttons();
}catch(e){$('record-status').textContent=e.message;for(const id of ['start','stop','rebuild','live-start','live-stop','cleanup','clear-all'])$(id).disabled=true}}
async function frame(){if(active){try{const r=await fetch('/frame.jpg?t='+Date.now(),{cache:'no-store'});if(!r.ok)throw Error(await r.text());
const old=$('camera').src;$('camera').src=URL.createObjectURL(await r.blob());if(old.startsWith('blob:'))URL.revokeObjectURL(old);
$('preview-status').textContent='实时预览 · '+r.headers.get('X-Capture-Fps')+' fps 采集';}catch(e){$('preview-status').textContent=e.message}}setTimeout(frame,250)}
$('start').onclick=()=>command('start');$('stop').onclick=()=>command('stop');$('rebuild').onclick=()=>command('reconstruct');
$('cleanup').onclick=()=>command('cleanup');$('live-start').onclick=()=>command('live/start');$('live-stop').onclick=()=>command('live/stop');$('pause').onclick=e=>{active=!active;e.target.textContent=active?'暂停显示':'继续显示'};
$('clear-all').onclick=()=>{if(confirm('确定清空全部记录（包括最新结果）？将关闭当前场景，删除本页面的全部录像、重建结果和会话日志。此操作无法恢复。'))command('cleanup/all')};
status();setInterval(status,500);frame();</script></html>'''


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
    def __init__(self, device, width, height, fps, out, *, max_seconds=20, factory=Preview):
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
        self.resume_preview()

    def resume_preview(self):
        with self.lock:
            if self.preview is None and not self.closed:
                self.preview = self.factory(self.device, self.width, self.height, self.fps,
                                            self.out/'preview.log')

    def pause_preview(self):
        with self.lock:
            preview, self.preview = self.preview, None
        if preview:
            preview.close()

    def snapshot(self):
        with self.lock:
            if self.preview is None:
                raise RuntimeError('重建期间预览已暂停，完成后自动恢复')
            return self.preview.snapshot()

    def packet(self):
        with self.lock:
            if self.preview is None:raise RuntimeError('摄像头预览尚未就绪')
            return self.preview.packet()

    def start_stream(self):
        with self.lock:
            if self.closed or self.maintenance or self.state in ('starting','recording','stopping'):
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
            return dict(state=self.state, elapsed_seconds=max(0.,elapsed), max_seconds=self.max_seconds,
                        free_mib=shutil.disk_usage(self.out).free/1024**2, clip=clip, error=self.error,
                        build=build,maintenance=self.maintenance)

    def start(self):
        with self.lock:
            if self.closed or self.maintenance or self.state in ('starting','recording','stopping'):
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
            self.stop_event.clear()
            self.worker=threading.Thread(target=self.record,args=(folder,min(256*1024**2,available)),daemon=True)
            self.worker.start()

    def stop(self):
        with self.lock:
            if self.state not in ('starting','recording','stopping'):
                raise ValueError('当前没有正在录制的短片')
            self.state='stopping'
            self.stop_event.set()

    def record(self, folder, limit):
        started=time.monotonic()
        partial, final=folder/'capture.partial.mkv',folder/'capture.mkv'
        record=dict(complete=False, requested=dict(width=self.width,height=self.height,fps=self.fps),
                    storage_limit_bytes=limit,timestamp_scope='demuxed PTS; not exposure timestamps')
        preview=None
        try:
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
            preview.finish_recording()
            if preview.process.returncode not in (0,255):
                raise RuntimeError('摄像头或录像写入异常，请检查日志')
            with self.lock: self.state='stopping'
            stream,times=inspect_clip(partial)
            duration=times[-1]-times[0]
            measured=(len(times)-1)/duration
            if duration<1.:
                raise ValueError('短片不足 1 秒，请重新录制并保持缓慢平移')
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

    def reconstruct(self):
        with self.lock:
            if self.maintenance:raise ValueError('正在清理记录，请稍候')
            if not self.build: raise ValueError('重建入口尚未配置')
            if self.state!='recorded' or not self.clip: raise ValueError('请先结束录制并等待短片检查完成')
            self.build.start(self.clip['path'])

    def restore_clip(self,metadata):
        """Restore a completed owned clip across a controlled UI restart."""
        metadata=Path(metadata).resolve()
        if self.out.resolve() not in metadata.parents:
            raise ValueError('恢复录像必须位于当前录制目录中')
        record=json.loads(metadata.read_text(encoding='utf-8'))
        path=metadata.parent/'capture.mkv'
        mode=record.get('requested',{})
        if (not record.get('complete') or str(path.resolve())!=record.get('path')
                or sha(path)!=record.get('sha256')
                or (mode.get('width'),mode.get('height'),mode.get('fps'))!=(self.width,self.height,self.fps)):
            raise ValueError('恢复录像的文件校验或摄像头模式不一致')
        with self.lock:
            if self.state in ('starting','recording','stopping'):raise ValueError('请先结束当前录制')
            self.clip,self.state,self.error=record,'recorded',None
            self.elapsed=record['duration_seconds']

    def cleanup_history(self, *, stream_only=False):
        """Delete only owned inactive sessions; retain current and last usable result."""
        with self.lock:
            if self.closed or self.maintenance or self.state in ('starting','recording','stopping'):
                raise ValueError('请先结束录制并等待保存完成，再清理历史记录')
            build=self.build.status() if self.build else {}
            if build.get('phase') in ('initializing','reconstructing','streaming','stopping'):
                raise ValueError('请等待当前重建完成，再清理历史记录')
            root=self.storage_root
            # Refuse a substituted root. Deletion must stay inside the configured
            # session directory, including on Windows junction/reparse surfaces.
            if self.out.is_symlink() or self.out.resolve()!=root:
                raise ValueError('录制目录边界不一致，未执行清理')
            pattern=r'(clip|scene|stream)_\d{8}_\d{6}_[0-9a-f]{6}'
            entries={p.name:p for p in root.iterdir() if re.fullmatch(pattern,p.name)
                     and p.is_dir() and not p.is_symlink() and p.resolve().parent==root
                     and (p/('capture.json' if p.name.startswith('clip_') else ('stream.json' if p.name.startswith('stream_') else 'events.json'))).is_file()}
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
                    if name.startswith('clip_'):
                        data=json.loads((path/'capture.json').read_text(encoding='utf-8'))
                        if data.get('complete') and data.get('path')==str(path/'capture.mkv') and (path/'capture.mkv').is_file():
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
            if self.closed or self.maintenance or self.state in ('starting','recording','stopping'):
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
                pattern=r'(clip|scene|stream)_\d{8}_\d{6}_[0-9a-f]{6}'
                selected=[];skipped=[]
                for path in sorted(root.iterdir()):
                    if not re.fullmatch(pattern,path.name):continue
                    marker='capture.json' if path.name.startswith('clip_') else ('stream.json' if path.name.startswith('stream_') else 'events.json')
                    if not path.is_dir() or not (path/marker).is_file():continue
                    if checked_path(path,root):selected.append(path)
                    else:skipped.append(path.name)
                # Pair receipts/logs also cover failed launches with no folder.
                for path in sorted(root.iterdir()):
                    if re.fullmatch(pattern+r'\.(log|measurement\.json)',path.name):
                        if checked_path(path,root):selected.append(path)
                        else:skipped.append(path.name)
                self.clip=None;self.state='idle';self.error=None;self.started=None;self.elapsed=0.
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


def make_server(camera, host='127.0.0.1', port=8081):
    token=secrets.token_urlsafe(24)
    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(5)
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
            if self.path=='/':
                self.reply(200,PAGE.replace('__TOKEN__',json.dumps(token)).encode(),'text/html; charset=utf-8')
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
            if not secrets.compare_digest(self.headers.get('X-Session-Token',''),token):
                self.reply(403,b'{"error":"Invalid session"}'); return
            try:
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
            except (OSError,ValueError,RuntimeError) as exc:
                self.reply(409,json.dumps(dict(error=str(exc)),ensure_ascii=False).encode())
    return ThreadingHTTPServer((host,port),Handler)
