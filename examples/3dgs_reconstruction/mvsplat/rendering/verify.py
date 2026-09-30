"""Board regression for scene replacement, mixed viewport sizes and archival adapter."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from initialize.session import render_environment
from rendering.runtime import LiveRenderer
from rendering.pipeline_adapter import PipelineRenderer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene",type=Path,required=True)
    p.add_argument("--binary",type=Path,required=True)
    p.add_argument("--renderer",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    data=(a.scene/"model.ply").read_bytes()
    offset=data.index(b"end_header\n")+len(b"end_header\n")
    rows=np.frombuffer(data,dtype="<f4",offset=offset).reshape(-1,62)
    meta=json.loads((a.scene/"manifest.json").read_text())
    names=[c["file"] for c in meta["cameras"] if c["role"]=="target"]
    base=(a.scene/names[0]).read_bytes()
    cameras=[(name,(a.scene/name).read_bytes()) for name in names]
    for w,h in ((64,64),(129,65),(128,128)):
        c=bytearray(base);struct.pack_into("<II",c,8,w,h)
        cameras.append(("%dx%d"%(w,h),bytes(c)))
    sequences=[]
    for cached in (False,True):
        outputs=[]
        with LiveRenderer(a.binary,render_environment(),a.out/("cached_%s.log"%cached),
                          threads=4,cached=cached) as runtime:
            runtime.load_rows(rows)
            for name,cam in cameras:
                f=runtime.render_camera(cam,include_raw=True)
                outputs.append(dict(name=name,hash=hashlib.sha256(f.raw).hexdigest()))
            runtime.load_rows(rows[:1024])
            small=runtime.render_camera(base,include_raw=True)
            runtime.load_rows(rows)
            restored=runtime.render_camera(base,include_raw=True)
            if hashlib.sha256(restored.raw).hexdigest()!=outputs[0]["hash"]:
                raise ValueError("Reload did not restore scene")
            outputs.append(dict(name="small_scene",hash=hashlib.sha256(small.raw).hexdigest()))
        sequences.append(outputs)
    if sequences[0]!=sequences[1]:
        raise ValueError("Cached and frozen projections differ across scene/viewport changes")
    with PipelineRenderer(a.renderer,a.binary,render_environment(),a.out/"adapter.log") as adapter:
        scene=adapter.load_rows(rows)
        if scene["rows_sha256"]!=hashlib.sha256(rows.tobytes()).hexdigest():
            raise ValueError("Adapter rows contract")
        frame=adapter.render_camera(base,a.out/"adapter_frame")
        if frame["frame_sha256"]!=sequences[0][0]["hash"]:
            raise ValueError("First-frame adapter output changed")
    result=dict(complete=True,board_executed=True,checks=sequences[1],
                restored_after_scene_change=True,adapter_complete=frame["complete"],
                adapter_total_ms=frame["total_ms"],adapter_live_frame_ms=frame["live_frame_ms"])
    (a.out/"verification.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=="__main__":
    main()
