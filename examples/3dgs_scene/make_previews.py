"""PNG previews from actual board readbacks, with a source hash manifest."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from check_output import pixels,sha

ROOT=Path(__file__).resolve().parent


def main():
    out=ROOT/'previews';out.mkdir(exist_ok=True)
    evidence=[ROOT/'evidence/calibrate_20260926T140258',ROOT/'evidence/calibrate_20260926T140613']
    manifest=[]
    for ev in evidence:
        report=json.loads((ev/'result.json').read_text())
        for r in report['runs']:
            if r['variant']!='fpga1':continue
            p=ev/'received'/(r['stem']+'.bin')
            if sha(p)!=r['files']['.bin']:raise ValueError('Readback drift')
            w,h,a=pixels(p)
            rgb=np.rint(np.clip(a['rgba'][:,:3],0,1)*255).astype(np.uint8).reshape(h,w,3)
            png=out/(r['case']['name']+'.png');Image.fromarray(rgb).save(png)
            manifest.append(dict(png=png.name,source=str(p.relative_to(ROOT)),source_sha256=sha(p),
                png_sha256=sha(png),backend='CPU evaluation + physical FPGA composition',
                official_strict_pass=r['official']['passed'],image_note='Clamped and rounded to 8-bit; raw float errors remain in result.json'))
    fontpath=Path('C:/Windows/Fonts/arial.ttf')
    font=ImageFont.truetype(str(fontpath),20) if fontpath.exists() else ImageFont.load_default()
    small=ImageFont.truetype(str(fontpath),15) if fontpath.exists() else font
    canvas=Image.new('RGB',(1000,680),'#101b26');d=ImageDraw.Draw(canvas)
    d.text((20,14),'30TAI Lite | Actual board render outputs',font=font,fill='white')
    d.text((20,46),'Same view, 320 x 178. CPU Gaussian evaluation + GSB1 FPGA composition.',font=small,fill='#c2cbd3')
    for i,n in enumerate([10000,50000,100000,559263]):
        x=20+(i%2)*490;y=84+(i//2)*285
        im=Image.open(out/f'n{n}_v0_w320.png').resize((460,256),Image.Resampling.NEAREST)
        canvas.paste(im,(x,y+24));d.text((x,y),f'{n:,} distinct input Gaussians',font=small,fill='white')
    d.text((20,660),'GPU projection / SH / sorted lists prepared offline. Fixed-point boundary mismatches are reported.',font=small,fill='#c2cbd3')
    canvas.save(out/'scale_comparison.png')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(out)


if __name__=='__main__':main()
