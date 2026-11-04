"""Synthetic RGB server for browser tests only; never open an FPGA device."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from rendering.runtime import LiveFrame
from web.viewer import make_server


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument('--look', action='store_true', help='Synthetic fixed-centre coverage fixture')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    class SyntheticRenderer:
        def render_camera(self, camera):
            (a.out/'last_camera.bin').write_bytes(camera)
            rgb = np.zeros((128, 128, 3), np.uint8)
            rgb[:, :, 0] = np.arange(128, dtype=np.uint8)[None, :]*2
            rgb[:, :, 1] = np.arange(128, dtype=np.uint8)[:, None]*2
            return LiveFrame(rgb, dict(wall_ms=.5))
    base = dict(c2w=np.eye(4).tolist(), intrinsics_normalized=[[1, 0, .5], [0, 1, .5], [0, 0, 1]],
                width=128, height=128)
    points=np.random.default_rng(1).uniform([-5,-5,5],[5,5,15],(800,3))
    metadata = dict(capture_center=[2.,3.,4.],default_view_mode='look',depth_observable=False,
                    coverage=dict(directions=[[0,0,1],[1,0,0]],crop_horizontal_fov_degrees=60,
                                  observed_yaw_span_degrees=90,pair_groups=2)) if a.look else None
    if a.look:
        from video_input.ring import horizontal_frustum_coverage
        second=np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]],np.float64)
        metadata['coverage'].update(horizontal_frustum_coverage([np.eye(4),second],
            base['intrinsics_normalized']))
        metadata['coverage'].update(coverage_target=.95,target_met=False)
    server = make_server(SyntheticRenderer(), base, 10, port=0,points=points,
                         generation=dict(backend='synthetic',npu_executed=False),view_metadata=metadata)
    (a.out/"server.json").write_text(json.dumps(dict(
        url=f"http://127.0.0.1:{server.server_port}/#token={server.session_token}")), encoding="utf-8")
    try:
        while not server.stopping:
            server.handle_request()
    finally:
        server.server_close()
        (a.out/"stats.json").write_text(json.dumps(server.stats), encoding="utf-8")


if __name__ == "__main__":
    main()
