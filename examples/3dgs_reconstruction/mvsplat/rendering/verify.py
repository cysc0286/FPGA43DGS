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
    p.add_argument("--radix-bits",type=int,choices=(8,11,16),default=11)
    p.add_argument("--serial-tiles",action="store_false",dest="parallel_tiles",default=True)
    p.add_argument("--parallel-collect",action="store_true")
    p.add_argument("--direct-collect",action="store_true")
    p.add_argument("--neon-pack",action="store_true")
    p.add_argument("--reference-binary",type=Path)
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
    # Every point goes behind the camera. Reusing buffers must not retain
    # records or colors from the previous nonempty frame.
    empty=bytearray(base)
    rotation=np.array(struct.unpack_from("<9d",base,24)).reshape(3,3)
    position=np.array(struct.unpack_from("<3d",base,96))+rotation[:,2]*1e6
    struct.pack_into("<3d",empty,96,*position)
    cameras.append(("all_culled",bytes(empty)))
    sequences=[]
    configurations=[(a.binary,False,True),(a.binary,True,True)]
    if a.reference_binary:
        configurations.append((a.reference_binary,True,False))
    for index,(binary,cached,candidate) in enumerate(configurations):
        outputs=[]
        with LiveRenderer(binary,render_environment(),a.out/("config_%d.log"%index),
                          threads=4,cached=cached,radix_bits=a.radix_bits,
                          parallel_tiles=a.parallel_tiles,
                          parallel_collect=candidate and a.parallel_collect,
                          direct_collect=candidate and a.direct_collect,
                          neon_pack=candidate and a.neon_pack) as runtime:
            runtime.load_rows(rows)
            for name,cam in cameras:
                f=runtime.render_camera(cam,include_raw=True)
                if name=="all_culled" and (f.metadata['active']!=0 or np.any(f.rgb)):
                    raise ValueError("Empty frustum contains stale Gaussians or RGB")
                outputs.append(dict(name=name,hash=hashlib.sha256(f.raw).hexdigest()))
            runtime.load_rows(rows[:1024])
            small=runtime.render_camera(base,include_raw=True)
            runtime.load_rows(rows)
            restored=runtime.render_camera(base,include_raw=True)
            if hashlib.sha256(restored.raw).hexdigest()!=outputs[0]["hash"]:
                raise ValueError("Reload did not restore scene")
            outputs.append(dict(name="small_scene",hash=hashlib.sha256(small.raw).hexdigest()))
            # The first 1024 rows can all lie outside this view. Also exercise
            # a spatially spread subset that must produce a nonempty image.
            runtime.load_rows(rows[::32])
            spread=runtime.render_camera(base,include_raw=True)
            if not spread.metadata['active'] or not np.any(spread.rgb):
                raise ValueError("Spread-scene regression did not exercise visible pixels")
            outputs.append(dict(name="spread_scene",hash=hashlib.sha256(spread.raw).hexdigest()))
            runtime.load_rows(rows)
            if hashlib.sha256(runtime.render_camera(base,include_raw=True).raw).hexdigest()!=outputs[0]['hash']:
                raise ValueError("Spread-scene reload did not restore full scene")
        sequences.append(outputs)
    if any(sequence!=sequences[0] for sequence in sequences[1:]):
        raise ValueError("Projection or reference differs across scene/viewport changes")
    with PipelineRenderer(a.renderer,a.binary,render_environment(),a.out/"adapter.log",
                          parallel_collect=a.parallel_collect,direct_collect=a.direct_collect,
                          neon_pack=a.neon_pack) as adapter:
        scene=adapter.load_rows(rows)
        if scene["rows_sha256"]!=hashlib.sha256(rows.tobytes()).hexdigest():
            raise ValueError("Adapter rows contract")
        frame=adapter.render_camera(base,a.out/"adapter_frame")
        if frame["frame_sha256"]!=sequences[0][0]["hash"]:
            raise ValueError("First-frame adapter output changed")
    result=dict(complete=True,board_executed=True,checks=sequences[1],
                reference_compared=bool(a.reference_binary),
                configuration=dict(parallel_collect=a.parallel_collect,direct_collect=a.direct_collect,
                                   neon_pack=a.neon_pack),
                restored_after_scene_change=True,adapter_complete=frame["complete"],
                adapter_total_ms=frame["total_ms"],adapter_live_frame_ms=frame["live_frame_ms"])
    (a.out/"verification.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=="__main__":
    main()
