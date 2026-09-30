"""Compare saved board renders to held-out video frames; no timing is rerun."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results",type=Path,required=True)
    p.add_argument("--images",type=Path,required=True)
    p.add_argument("--profiles",default="exact4,keep24k,keep16k")
    a=p.parse_args()
    spec=importlib.util.spec_from_file_location("mvsplat_metrics",Path(__file__).resolve().parents[1]/"evaluate.py")
    metrics=importlib.util.module_from_spec(spec);spec.loader.exec_module(metrics)
    profiles=a.profiles.split(",")
    names=["frame_000007","frame_000015","frame_000022"]
    report=dict(board_rendered=True,domain="clipped float RGB against held-out video RGB",profiles={})
    cell,height=270,300
    width=cell*(len(profiles)+1)+20
    panel=Image.new("RGB",(width,80+height*len(names)),"#f4f5f7")
    draw=ImageDraw.Draw(panel)
    try:font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",18)
    except OSError:font=ImageFont.load_default()
    labels={"exact4":"All 32,768 Gaussians","batch2":"All 32,768 Gaussians",
            "keep24k":"Keep 24,576 (lossy)","keep16k":"Keep 16,384 (lossy)",
            "preview24k":"Preview 24,576","preview16k":"Preview 16,384","preview8k":"Preview 8,192"}
    for col,title in enumerate(["Held-out video"]+[labels.get(name,name) for name in profiles]):
        draw.text((12+col*cell,12),title,fill="#16243a",font=font)
    for row,name in enumerate(names):
        gt=np.asarray(Image.open(a.images/(name+".png")).convert("RGB"))/255.
        images=[(gt,None)]
        for profile in profiles:
            actual=metrics.framebuffer(a.results/profile/name/"frame.bin")
            q=dict(metrics.quality(actual,gt),local_error=metrics.local_error(actual,gt))
            report["profiles"].setdefault(profile,{})[name]=q
            image=np.asarray(Image.open(a.results/profile/name/"frame.ppm").convert("RGB"))/255.
            images.append((image,q))
        for col,(image,q) in enumerate(images):
            y=70+row*height;x=12+col*cell
            preview=Image.fromarray(np.rint(image*255).astype(np.uint8)).resize((256,256),Image.Resampling.NEAREST)
            panel.paste(preview,(x,y))
            text=name if q is None else "%.2f dB / SSIM %.4f"%(q["psnr_db"],q["ssim_rgb_gaussian_11_sigma1p5_valid"])
            draw.text((x,y+262),text,fill="#16243a",font=font)
    panel.save(a.results/"comparison.png")
    (a.results/"quality.json").write_text(json.dumps(report,indent=2))
    print(json.dumps({p:{n:dict(psnr_db=q["psnr_db"],ssim=q["ssim_rgb_gaussian_11_sigma1p5_valid"])
                           for n,q in views.items()} for p,views in report["profiles"].items()},indent=2))


if __name__=="__main__":
    main()
