"""Report quantization risks and render unaltered FP16-reference/official previews."""
import json
import struct
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / 'examples/3dgs_scene'))
from check_output import pixels, sha


def main():
    report = {'scope': 'HLS C FP16 versus official CUDA, not board or ground-truth quality', 'views': []}
    canvas = Image.new('RGB', (1000, 460), '#11202c')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 17)
    for row, view in enumerate([0, 10]):
        case = f'n559263_v{view}_w320'
        data = REPO / 'examples/3dgs_scene/data/20260926T135717' / case
        reference = ROOT / 'evidence' / f'full_reference_v{view}'
        raw = (data / 'scene.bin').read_bytes()
        w, h, tile, total, active, entries, tiles = struct.unpack_from('<7I', raw, 8)
        gs = np.frombuffer(raw, dtype=[('id','<u4'), ('q','<f4',(10,))], count=active, offset=48)['q']
        half = gs[:, :9].astype(np.float16).astype(np.float64)
        determinant = half[:, 2] * half[:, 4] - half[:, 3] ** 2
        _, _, actual = pixels(reference / 'frame.bin')
        _, _, official = pixels(data / 'official.bin')
        a = np.clip(actual['rgba'][:, :3].reshape(h, w, 3), 0, 1)
        b = np.clip(official['rgba'][:, :3].reshape(h, w, 3), 0, 1)
        errors = np.max(np.abs(a-b), axis=2)
        summary = {'case': case, 'active': active, 'fp16_nonpositive_conic_determinants': int((determinant <= 0).sum()),
                   'nonfinite_fp16_parameters': int((~np.isfinite(half)).sum()),
                   'pixels_clamped_rgb_error_above_0_01': int((errors > .01).sum()),
                   'pixels_clamped_rgb_error_above_0_05': int((errors > .05).sum()),
                   'reference_sha256': sha(reference / 'frame.bin'), 'official_sha256': sha(data / 'official.bin'),
                   'quality': json.loads((reference / 'quality.json').read_text())['official_comparison']}
        report['views'].append(summary)
        y = row * 225 + 10
        for col, (label, rgb) in enumerate([('Official FP32', b), ('HLS C FP16', a), ('Absolute difference x 10', np.clip(np.abs(a-b)*10,0,1))]):
            x = col * 330 + 10
            draw.text((x,y), f'View {view}: {label}', font=font, fill='white')
            canvas.paste(Image.fromarray(np.rint(rgb*255).astype(np.uint8)), (x,y+28))
    out = ROOT / 'evidence' / 'fp16_quality_audit'
    out.mkdir(exist_ok=True)
    canvas.save(out / 'comparison.png')
    (out / 'result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
