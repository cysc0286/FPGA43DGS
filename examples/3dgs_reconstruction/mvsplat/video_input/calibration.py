"""Pinhole calibration and rectification into the centered frozen-camera ABI."""
import argparse
import json
from pathlib import Path

import numpy as np

from common import save, sha


class Calibration:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.record = json.loads(self.path.read_text(encoding="utf-8"))
        if self.record.get("schema") != "uvc_pinhole_v1":
            raise ValueError("Expected uvc_pinhole_v1 calibration")
        self.size = tuple(self.record["image_size"])
        self.k = np.asarray(self.record["K"], dtype=np.float64)
        self.d = np.asarray(self.record["D"], dtype=np.float64).reshape(-1)
        if (len(self.size) != 2 or any(type(v) is not int or v < 64 for v in self.size)
                or self.k.shape != (3, 3) or not np.isfinite(self.k).all()
                or not np.isfinite(self.d).all() or self.d.size not in (4, 5, 8, 12, 14)
                or min(self.k[0, 0], self.k[1, 1]) <= 0
                or not np.allclose(self.k[2], [0, 0, 1])
                or abs(self.k[0, 1]) > 1e-10 or abs(self.k[1, 0]) > 1e-10):
            raise ValueError("Invalid pinhole calibration")
        self.digest = sha(self.path)
        self.maps = None
        self.intrinsic = None

    def rectify(self, frame, width):
        import cv2  # Board startup must load Torch before OpenCV/OpenMP.
        h, w = frame.shape[:2]
        if (w, h) != self.size:
            raise ValueError(f"Calibration mode {self.size} differs from capture {(w, h)}; recalibrate")
        scale = min(1., width / w)
        size = (round(w * scale), round(h * scale))
        if self.maps is None:
            # cv.resize uses half-pixel centers, not a simple principal-point scale.
            sx, sy = size[0]/w, size[1]/h
            affine = np.array([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]])
            scaled = affine @ self.k
            self.intrinsic, _ = cv2.getOptimalNewCameraMatrix(
                scaled, self.d, size, 0, size, centerPrincipalPoint=True)
            self.maps = cv2.initUndistortRectifyMap(scaled, self.d, None,
                self.intrinsic, size, cv2.CV_32FC1)
            self.output_size = size
        if size != self.output_size:
            raise ValueError("Rectification dimensions changed")
        resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        return cv2.remap(resized, *self.maps, interpolation=cv2.INTER_LINEAR)

    def metadata(self):
        return dict(kind="calibrated_centered_rectification", schema="uvc_pinhole_v1",
                    source=str(self.path), sha256=self.digest, source_image_size=list(self.size),
                    source_intrinsic=self.k.tolist(), source_distortion=self.d.tolist(),
                    intrinsic=self.intrinsic.tolist(), distortion="rectified to zero",
                    image_size=list(self.output_size))


def calibrate(images, pattern, square_mm, out):
    """Calibrate from a directory of fixed-mode images; never open a camera."""
    import cv2
    if min(pattern) < 3 or not np.isfinite(square_mm) or square_mm <= 0:
        raise ValueError("Invalid inner-corner grid or square size")
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    object_grid = np.zeros((pattern[0]*pattern[1], 3), np.float32)
    object_grid[:, :2] = np.mgrid[:pattern[0], :pattern[1]].T.reshape(-1, 2)*square_mm
    objects, points, names = [], [], []
    size = None
    for path in sorted(Path(images).iterdir()):
        if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp"):
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot decode calibration image: {path}")
        current = image.shape[::-1]
        if size is not None and current != size:
            raise ValueError("Calibration images mix capture modes")
        size = current
        found, corners = cv2.findChessboardCornersSB(image, pattern)
        if found:
            objects.append(object_grid.copy())
            points.append(corners)
            names.append(dict(path=str(path.resolve()), sha256=sha(path)))
    if len(points) < 10:
        raise ValueError("Need at least 10 detected boards; recommend 20–30 varied views")
    rms, k, d, rotations, translations = cv2.calibrateCamera(objects, points, size, None, None)
    errors = []
    for obj, pixel, r, t in zip(objects, points, rotations, translations):
        projected, _ = cv2.projectPoints(obj, r, t, k, d)
        errors.append(float(np.sqrt(np.mean(np.sum((projected-pixel)**2, axis=2)))))
    record = dict(schema="uvc_pinhole_v1", image_size=list(size), K=k.tolist(), D=d.ravel().tolist(),
                  rms_px=float(rms), per_view_rms_px=errors, pattern_inner_corners=list(pattern),
                  square_mm=square_mm, images=names,
                  validation_scope="fit images only; inspect independent images and edge residuals")
    # Do not publish an unusable fit; quality still needs independent validation.
    if not np.isfinite(rms) or not np.isfinite(k).all() or not np.isfinite(d).all():
        raise ValueError("Nonfinite calibration fit")
    out.parent.mkdir(parents=True, exist_ok=True)
    save(out, record)
    return record


CAPTURE_PAGE = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>摄像头标定采集</title>
<style>*{box-sizing:border-box}body{font:16px system-ui;color:#17212f;background:#eef2f6;max-width:1080px;margin:24px auto;padding:0 16px}
.card{background:white;padding:20px;border-radius:14px;margin:16px 0}h1{font-size:27px}img{width:100%;display:block;background:#111;border-radius:8px;aspect-ratio:16/9;object-fit:contain}
button{font:inherit;padding:13px 19px;border:0;border-radius:8px;margin:6px 8px 6px 0;cursor:pointer;background:#e4eaf2}#capture,#solve{background:#1764cf;color:white}button:disabled{opacity:.4;cursor:default}
.hint{line-height:1.7;color:#526175}#guide{font-size:21px;font-weight:600}#error{color:#b21e2f;white-space:pre-wrap;overflow-wrap:anywhere}#quality{font-weight:600}pre{white-space:pre-wrap;overflow-wrap:anywhere}progress{width:100%;height:16px}</style>
<h1>摄像头标定采集</h1><p class="hint">9×6 内角点 · 方格 <span id="square"></span> mm · 原始 1280×720 图像保存在电脑。
固定摄像头，移动贴平的棋盘格；也可以移动摄像头。整块棋盘和白边应在画面内，手不要遮挡方格。
保持焦点与重建时相同；移动后停稳，再点击保存。此页面不启动重建。</p>
<section class="card"><p id="guide">正在连接…</p><p id="count"></p><progress id="progress" max="25" value="0"></progress>
<img id="preview" alt="棋盘格角点识别预览"><p id="quality" role="status"></p>
<button id="capture" disabled>保存这一姿态</button><button id="undo" disabled>撤销上一张</button><button id="solve" disabled>计算并验证标定</button>
<p id="error" role="alert"></p></section><section class="card"><h2>标定结果</h2><p class="hint">前 20 张拟合内参，后 5 张只验证，不参与内参拟合。
只拍正面或重复姿态无法得到可靠焦距。完成后还需检查边缘校正效果，再应用到开发板。</p><pre id="result">尚未计算。页面显示“通过”仅代表棋盘格几何检查通过，不代表重建画质验收。</pre></section>
<script>const token=__TOKEN__,el=id=>document.getElementById(id);let pending=false,last=null,actionError='';
async function command(action){if(pending)return;pending=true;actionError='';render();try{const r=await fetch('/api/'+action,{method:'POST',headers:{'X-Session-Token':token}});const d=await r.json();if(!r.ok)throw Error(d.error)}catch(e){actionError=e.message}finally{pending=false;await update()}}
function render(){if(!last)return;el('square').textContent=last.square_mm;el('guide').textContent=last.guide;el('count').textContent='拟合 '+last.fit_count+'/20 张 · 独立验证 '+last.validation_count+'/5 张';el('progress').value=last.count;
el('quality').textContent=last.message;el('quality').style.color=last.can_capture?'#157a42':'#85550a';el('capture').disabled=pending||last.solving||!last.can_capture||last.count>=25;
el('undo').disabled=pending||last.solving||!last.count;el('solve').disabled=pending||last.solving||last.count<25;el('error').textContent=actionError||last.error||'';
if(last.result)el('result').textContent=JSON.stringify(last.result,null,2);else if(last.solving)el('result').textContent='正在拟合和验证，请稍候…';else el('result').textContent='尚未计算。'}
async function update(){try{const r=await fetch('/api/status',{cache:'no-store'});if(!r.ok)throw Error('状态连接失败');last=await r.json();render()}catch(e){el('error').textContent=e.message;el('capture').disabled=true;el('solve').disabled=true}}
async function frame(){try{const r=await fetch('/frame.jpg',{cache:'no-store'});if(r.ok){const old=el('preview').src;el('preview').src=URL.createObjectURL(await r.blob());if(old.startsWith('blob:'))URL.revokeObjectURL(old)}}catch(e){}setTimeout(frame,650)}
el('capture').onclick=()=>command('capture');el('undo').onclick=()=>command('undo');el('solve').onclick=()=>command('solve');update();setInterval(update,700);frame();</script></html>'''


CAPTURE_GUIDES = [
    '中心、正面，棋盘宽度占画面约一半',
    '中心、靠近一些，棋盘完整入镜',
    '中心、稍远一些，方格仍应清晰可见',
    '移到画面左侧，停稳', '移到画面右侧，停稳',
    '移到画面上方，停稳', '移到画面下方，停稳',
    '左上区域，保留完整棋盘及白边', '右上区域，保留完整棋盘及白边',
    '左下区域，保留完整棋盘及白边', '右下区域，保留完整棋盘及白边',
    '回到中心，绕竖轴左倾约 20 度', '中心，绕竖轴右倾约 20 度',
    '中心，上缘向后倾约 20 度', '中心，下缘向后倾约 20 度',
    '左侧，轻微倾斜，保持完整入镜', '右侧，轻微倾斜，保持完整入镜',
    '上方，轻微倾斜，保持完整入镜', '下方，轻微倾斜，保持完整入镜',
    '中心，斜倾约 25 度，停稳',
    '验证 1/5：中心，选择与前面不同的距离和倾角',
    '验证 2/5：左上区域，稍微倾斜', '验证 3/5：右上区域，稍微倾斜',
    '验证 4/5：左下区域，稍微倾斜', '验证 5/5：右下区域，稍微倾斜',
]


def inspect_board(image, pattern):
    """Detect at source resolution; keep image quality gates separate from fit."""
    import cv2
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 20:
        return None, '画面过暗，请取下镜头盖并增加照明', None
    found, corners = cv2.findChessboardCornersSB(gray, pattern)
    if not found:
        return None, '未识别完整棋盘格：请完整入镜、调清焦点、避免反光', None
    points = corners.reshape(-1, 2)
    low, high = points.min(axis=0), points.max(axis=0)
    h, w = gray.shape
    grid = points.reshape(pattern[1], pattern[0], 2)
    step = min(np.linalg.norm(np.diff(grid, axis=0), axis=2).min(),
               np.linalg.norm(np.diff(grid, axis=1), axis=2).min())
    x0, y0 = np.maximum(0, low.astype(int))
    x1, y1 = np.minimum([w, h], high.astype(int)+1)
    sharpness = float(cv2.Laplacian(gray[y0:y1, x0:x1], cv2.CV_64F).var())
    info = dict(bounds=[low.tolist(), high.tolist()], square_pixels=float(step),
                sharpness=sharpness, center=(points.mean(axis=0)/[w, h]).tolist(),
                area_fraction=float(cv2.contourArea(cv2.convexHull(points)) / (w*h)))
    if step < 12:
        return corners, '棋盘太小或倾角过大：请靠近一些', info
    if min(low) < step*.65 or np.any(high > [w-1-step*.65, h-1-step*.65]):
        return corners, '棋盘靠边太近：请让整块棋盘和白边完整入镜', info
    if sharpness < 40:
        return corners, '棋盘不够清晰：调焦并停稳后再保存', info
    return corners, '已识别 54 个内角点，可以保存', info


def validate_fit(record, validation_images):
    """Held-out intrinsics check: solve only each test board's pose, never refit K/D."""
    import cv2
    k, d = np.array(record['K']), np.array(record['D'])
    pattern = tuple(record['pattern_inner_corners'])
    grid = np.zeros((pattern[0]*pattern[1], 3), np.float32)
    grid[:, :2] = np.mgrid[:pattern[0], :pattern[1]].T.reshape(-1, 2)*record['square_mm']
    results = []
    for path in validation_images:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or list(image.shape[::-1]) != record['image_size']:
            raise ValueError('独立验证图像的分辨率或文件无效')
        found, corners = cv2.findChessboardCornersSB(image, pattern)
        if not found:
            raise ValueError('独立验证图像未识别完整棋盘')
        ok, rotation, translation = cv2.solvePnP(grid, corners, k, d, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise ValueError('独立验证棋盘位姿拟合失败')
        projected, _ = cv2.projectPoints(grid, rotation, translation, k, d)
        errors = np.linalg.norm(projected.reshape(-1, 2)-corners.reshape(-1, 2), axis=1)
        results.append(dict(path=str(path), sha256=sha(path), rms_px=float(np.sqrt(np.mean(errors**2))),
                            max_px=float(errors.max())))
    return dict(scope='K/D fixed, each held-out board pose fitted; no reconstruction quality measurement',
                views=results, mean_rms_px=float(np.mean([v['rms_px'] for v in results])),
                max_view_rms_px=max(v['rms_px'] for v in results))


class CalibrationCapture:
    """PC-side assistant: shares existing preview HTTP, never opens board UVC/FPGA."""
    def __init__(self, source_url, out, pattern=(9, 6), square_mm=20., *, resume=False):
        import threading
        from urllib.parse import urlsplit
        source = urlsplit(source_url)
        if source.scheme != 'http' or source.hostname not in ('127.0.0.1', 'localhost'):
            raise ValueError('标定助手只读取本机摄像头隧道 HTTP')
        if tuple(pattern) != (9, 6) or not np.isfinite(square_mm) or square_mm <= 0:
            raise ValueError('此采集页面使用 9×6 内角点和正数方格毫米尺寸')
        self.source_url, self.out, self.pattern, self.square_mm = source_url, Path(out).resolve(), tuple(pattern), square_mm
        previous = None
        if resume:
            previous = json.loads((self.out/'capture.json').read_text(encoding='utf-8'))
            if (previous.get('schema') != 'checkerboard_capture_v1' or previous['pattern'] != list(pattern)
                    or previous['square_mm'] != square_mm or previous['source_url'] != source_url):
                raise ValueError('原采集清单与当前标定配置不同')
        else:
            self.out.mkdir(parents=True, exist_ok=False)
            (self.out/'fit').mkdir(); (self.out/'validation').mkdir()
        self.lock = threading.RLock(); self.stopped = threading.Event()
        self.samples = []; self.latest = None; self.preview = None
        self.message = '正在读取原始摄像头…'; self.error = None
        self.solving = False; self.result = None; self.worker = None
        self.size = None; self.source_id = None; self.source_received = 0.
        if previous is not None:
            self.samples = previous['samples']
            if len(self.samples) > 25:
                raise ValueError('采集清单样本数量无效')
            for i, sample in enumerate(self.samples, 1):
                role = 'fit' if i <= 20 else 'validation'
                path = Path(sample['path'])
                if (sample['index'] != i or sample['role'] != role or path.is_symlink()
                        or path.resolve().parent != self.out/role or sha(path) != sample['sha256']):
                    raise ValueError('原采集样本的路径或哈希校验失败')
            self.size = previous['image_size']; self.result = previous['result']
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def _persist(self):
        save(self.out/'capture.json', dict(schema='checkerboard_capture_v1', pattern=list(self.pattern),
             square_mm=self.square_mm, source_url=self.source_url, image_size=self.size,
             samples=self.samples, result=self.result))

    def _poll(self):
        import cv2
        import time
        from urllib.request import urlopen
        while not self.stopped.is_set():
            try:
                with urlopen(self.source_url, timeout=4) as response:
                    jpeg = response.read(4*1024**2+1)
                    sequence = response.headers.get('X-Frame-Id')
                if not sequence or len(jpeg) > 4*1024**2:
                    raise ValueError('摄像头帧号或 JPEG 长度无效')
                image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError('摄像头 JPEG 解码失败')
                size = image.shape[1::-1]
                corners, message, info = inspect_board(image, self.pattern)
                rendered = image.copy()
                if corners is not None:
                    cv2.drawChessboardCorners(rendered, self.pattern, corners, True)
                rendered = cv2.resize(rendered, (960, round(960*size[1]/size[0])))
                ok, encoded = cv2.imencode('.jpg', rendered, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    raise ValueError('角点预览编码失败')
                with self.lock:
                    if self.size is not None and tuple(self.size) != size:
                        raise ValueError('相机分辨率变化，请重新开始标定')
                    self.size = list(size)
                    if sequence != self.source_id:
                        self.source_received = time.monotonic(); self.source_id = sequence
                    self.latest = dict(jpeg=jpeg, corners=corners, info=info, sequence=sequence,
                                       detected_at=time.monotonic(), valid=message.endswith('可以保存'))
                    self.preview = encoded.tobytes(); self.message = message
            except Exception as exc:
                with self.lock:
                    self.latest = None; self.message = '采集等待：'+str(exc)
            self.stopped.wait(.25)

    def status(self):
        import time
        with self.lock:
            count = len(self.samples)
            fresh = self.latest and time.monotonic()-min(self.latest['detected_at'], self.source_received) < 3
            result = dict(self.result) if self.result else None
            application = self.out/'board_application.json'
            if result and application.is_file():
                applied = json.loads(application.read_text(encoding='utf-8'))
                result['applied_to_board'] = bool(applied.get('verified') and
                    applied.get('sha256') == sha(result['path']))
            return dict(count=count, fit_count=min(count, 20), validation_count=max(0, count-20),
                        guide=CAPTURE_GUIDES[count] if count < 25 else '25 张已保存，点击“计算并验证标定”',
                        square_mm=self.square_mm, size=self.size,
                        message='采集已完成；参数已接入开发板后续重建' if result and result.get('applied_to_board') else self.message,
                        can_capture=bool(fresh and self.latest['valid']), solving=self.solving,
                        error=self.error, result=result)

    def capture(self):
        with self.lock:
            if self.solving or len(self.samples) >= 25:
                raise ValueError('正在计算或采集已完成')
            if not self.status()['can_capture']:
                raise ValueError('当前没有清晰、完整且新鲜的棋盘帧，请停稳再试')
            item = self.latest
            if self.samples and self.samples[-1]['source_sequence'] == item['sequence']:
                raise ValueError('这张图像已经保存，请调整姿态')
            points = item['corners'].reshape(-1, 2)
            for previous in self.samples:
                other = np.array(previous['corners']).reshape(-1, 2)
                if min(np.linalg.norm(points-other, axis=1).mean(),
                       np.linalg.norm(points-other[::-1], axis=1).mean()) < 12:
                    raise ValueError('与已保存姿态太相近，请改变位置、距离或倾角')
            index = len(self.samples)+1
            role = 'fit' if index <= 20 else 'validation'
            path = self.out/role/f'board_{index:02d}.jpg'
            with path.open('xb') as stream: stream.write(item['jpeg'])
            self.samples.append(dict(index=index, role=role, path=str(path), sha256=sha(path),
                 source_sequence=item['sequence'], corners=points.tolist(), **item['info']))
            self.result = None; self.error = None; self._persist()

    def undo(self):
        with self.lock:
            if self.solving or not self.samples:
                raise ValueError('正在计算或没有可撤销样本')
            # Preserve the original file: undo only the active manifest entry.
            sample = self.samples[-1]; path = Path(sample['path'])
            destination = self.out/'unused'
            destination.mkdir(exist_ok=True)
            import secrets
            path.rename(destination/(path.stem+'_'+secrets.token_hex(3)+path.suffix))
            self.samples.pop(); self.result = None; self.error = None; self._persist()

    def solve(self):
        import threading
        with self.lock:
            if self.solving or len(self.samples) != 25:
                raise ValueError('需要先采集 20 张拟合图和 5 张独立验证图')
            self.solving = True; self.error = None; self.result = None
            self.worker = threading.Thread(target=self._solve, daemon=True)
            self.worker.start()

    def _solve(self):
        import cv2
        import secrets
        try:
            for sample in self.samples:
                if sha(sample['path']) != sample['sha256']:
                    raise ValueError('样本文件校验失败，未计算')
            path = self.out/('calibration_'+secrets.token_hex(3)+'.json')
            record = calibrate(self.out/'fit', self.pattern, self.square_mm, path)
            validation = validate_fit(record, [Path(s['path']) for s in self.samples if s['role']=='validation'])
            warnings = []
            k = np.array(record['K']); w, h = record['image_size']
            if not (.2*w < k[0, 0] < 3*w and .2*w < k[1, 1] < 3*w and 0 < k[0, 2] < w and 0 < k[1, 2] < h):
                warnings.append('焦距或主点不合理，需重新采集')
            if record['rms_px'] > .8 or validation['mean_rms_px'] > 1. or validation['max_view_rms_px'] > 2.:
                warnings.append('重投影误差偏大，请检查纸张平整度、反光及模糊')
            centers = np.array([s['center'] for s in self.samples[:20]])
            if np.ptp(centers[:, 0]) < .2 or np.ptp(centers[:, 1]) < .16:
                warnings.append('样本位置覆盖不足，请补充四周视角')
            grid = np.zeros((54, 3), np.float32)
            grid[:, :2] = np.mgrid[:9, :6].T.reshape(-1, 2)*self.square_mm
            normals = []
            for sample in self.samples[:20]:
                ok, rotation, _ = cv2.solvePnP(grid, np.array(sample['corners'], np.float32), k, np.array(record['D']))
                if not ok: raise ValueError('训练图像位姿复核失败')
                normals.append(cv2.Rodrigues(rotation)[0][:, 2])
            tilt_span = float(np.degrees(np.arccos(np.clip(np.array(normals) @ np.array(normals).T, -1, 1))).max())
            if tilt_span < 20:
                warnings.append('倾角变化不足，请补充不同方向倾斜的棋盘')
            record['independent_validation'] = validation
            record['quality_review'] = dict(status='needs_review' if warnings else 'geometric_checks_passed',
                  warnings=warnings, tilt_span_degrees=tilt_span, applied_to_board=False,
                  scope='checkerboard geometry only; lens flatness, edge residuals and scene quality require review')
            save(path, record)
            with self.lock:
                self.result = dict(path=str(path), status=record['quality_review']['status'],
                    fit_rms_px=record['rms_px'], validation_mean_rms_px=validation['mean_rms_px'],
                    validation_max_view_rms_px=validation['max_view_rms_px'],
                    K=record['K'], D=record['D'], warnings=warnings, applied_to_board=False)
                self._persist()
        except Exception as exc:
            with self.lock: self.error = str(exc)
        finally:
            with self.lock: self.solving = False

    def close(self):
        self.stopped.set(); self.thread.join(8)
        if self.worker: self.worker.join(20)


def capture_server(session, host='127.0.0.1', port=18435):
    import secrets
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    if host not in ('127.0.0.1', 'localhost'):
        raise ValueError('标定助手仅监听本机回环地址')
    token = secrets.token_urlsafe(24)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup(); self.connection.settimeout(5)

        def log_message(self, *args): pass

        def reply(self, code, data, content='application/json; charset=utf-8'):
            self.send_response(code); self.send_header('Content-Type', content)
            self.send_header('Cache-Control', 'no-store'); self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            try: self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError): pass

        def do_GET(self):
            if self.path == '/':
                self.reply(200, CAPTURE_PAGE.replace('__TOKEN__', json.dumps(token)).encode('utf-8'), 'text/html; charset=utf-8')
            elif self.path == '/api/status':
                self.reply(200, json.dumps(session.status(), ensure_ascii=False, allow_nan=False).encode('utf-8'))
            elif self.path == '/frame.jpg':
                with session.lock: jpeg = session.preview
                self.reply(200 if jpeg else 503, jpeg or b'', 'image/jpeg')
            else: self.reply(404, b'{}')

        def do_POST(self):
            if not secrets.compare_digest(self.headers.get('X-Session-Token', ''), token):
                self.reply(403, b'{}'); return
            try:
                action = {'/api/capture': session.capture, '/api/undo': session.undo, '/api/solve': session.solve}.get(self.path)
                if action is None: self.reply(404, b'{}'); return
                action(); self.reply(202, b'{"accepted":true}')
            except (ValueError, OSError, RuntimeError) as exc:
                self.reply(409, json.dumps(dict(error=str(exc)), ensure_ascii=False).encode('utf-8'))

    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", type=Path)
    source.add_argument("--source-url", help="Serve a PC-side calibration assistant using existing preview HTTP")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--pattern", type=int, nargs=2, default=[9, 6], help="INNER corners")
    p.add_argument("--square-mm", type=float, required=True)
    p.add_argument('--port', type=int, default=18435)
    p.add_argument('--resume-capture', action='store_true', help='Verify and reopen an existing capture manifest')
    a = p.parse_args(argv)
    if a.source_url:
        import cv2
        cv2.setNumThreads(2)
        session = CalibrationCapture(a.source_url, a.out, tuple(a.pattern), a.square_mm, resume=a.resume_capture)
        server = None
        try:
            server = capture_server(session, port=a.port)
            print(json.dumps(dict(event='CALIBRATION_CAPTURE_READY', url=f'http://127.0.0.1:{server.server_port}/', out=str(a.out))), flush=True)
            server.serve_forever()
        except KeyboardInterrupt: pass
        finally:
            if server: server.server_close()
            session.close()
        return
    result = calibrate(a.images, tuple(a.pattern), a.square_mm, a.out)
    print(json.dumps(dict(images=len(result["images"]), rms_px=result["rms_px"], out=str(a.out))))


if __name__ == "__main__":
    main()
