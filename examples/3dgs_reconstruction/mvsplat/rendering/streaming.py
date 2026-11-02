"""Serialize model installation and camera commands through one FPGA owner."""
import threading
import time
import numpy as np
from gaussian_generation.adapter import camera_bytes
from rendering.viewer import orbit_view


class SceneExchange:
    def __init__(self, renderer):
        self.renderer=renderer;self.lock=threading.RLock();self.base=None;self.focus=None
        self.rows=None;self.version=0;self.source_time=None;self.failure=None
        self.state=dict(phase='warming',points=[],trajectory=[],updates=0,failures=0)

    def update(self, **values):
        with self.lock:self.state.update(values)

    def snapshot(self):
        with self.lock:
            return dict(self.state,version=self.version,has_scene=self.base is not None,
                        model_age_seconds=None if self.source_time is None else max(0,time.monotonic()-self.source_time),
                        fatal_error=self.failure)

    def preview(self):
        """Bounded selector data in the exact orbit reference used by render_view."""
        with self.lock:
            points = [] if self.rows is None else self.rows[::max(1,(len(self.rows)+1499)//1500),:3].tolist()
            return dict(base=self.base,focus_distance=self.focus,points=points,version=self.version,
                        generation=self.state.get('generation',{}))

    def publish(self, rows, base, means, source_time, commit=None):
        rows=np.ascontiguousarray(rows,dtype='<f4')
        if rows.ndim!=2 or rows.shape[1]!=62 or not 0<len(rows)<=1000000 or not np.isfinite(rows).all():
            raise ValueError('Invalid candidate Gaussian rows')
        camera=camera_bytes(base)
        c2w=np.asarray(base['c2w']);depth=(means-c2w[:3,3])@c2w[:3,2]
        positive=depth[np.isfinite(depth)&(depth>0)]
        if not len(positive) or not np.isfinite(source_time):raise ValueError('Invalid candidate depth/time')
        focus=float(np.median(positive))
        with self.lock:
            if self.failure:raise RuntimeError(self.failure)
            try:
                self.renderer.load_rows(rows)
                frame=self.renderer.render_camera(camera,include_raw=True)
                frame.metadata['host_complete_monotonic']=time.monotonic()
                if not frame.metadata.get('complete'):raise RuntimeError('Candidate first frame incomplete')
                if commit:commit(frame,self.version+1)
            except (RuntimeError,TimeoutError,OSError,ValueError) as exc:
                # Never reuse a pipe after an uncertain framed reply. Saved
                # previous model/files and the browser's last image survive.
                self.failure='FPGA scene exchange stopped: '+str(exc)
                raise RuntimeError(self.failure) from exc
            self.rows=rows
            if self.base is None:self.base=base;self.focus=focus
            self.source_time=source_time;self.version+=1
            self.state.update(updates=self.version,phase='sampling')
            return frame

    def validate_controls(self, controls):
        with self.lock:
            # Validation does not touch the framed renderer pipe.
            base=self.base or dict(c2w=np.eye(4).tolist())
            orbit_view(base,controls,self.focus or 10.)

    def render_view(self, controls):
        with self.lock:
            if self.failure:raise RuntimeError(self.failure)
            if self.base is None:raise ValueError('等待首个有效场景，请缓慢横向移动摄像头')
            camera=camera_bytes(orbit_view(self.base,controls,self.focus))
            try:frame=self.renderer.render_camera(camera)
            except (RuntimeError,TimeoutError,OSError,ValueError) as exc:
                self.failure='FPGA render stopped: '+str(exc)
                raise RuntimeError(self.failure) from exc
            frame.metadata['live_version']=self.version
            return frame
