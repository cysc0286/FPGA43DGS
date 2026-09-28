"""Quality and paired speed comparisons from completed physical FLK1 runs."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / 'examples/3dgs_flicker_cat'))
from analyze_results import quality, read_rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('evidence', type=Path)
    args = parser.parse_args()
    evidence = args.evidence
    record = json.loads((evidence / 'result.json').read_text())
    summary = json.loads((evidence / 'summary.json').read_text())
    if not record['complete'] or len(record['runs']) != len(summary['runs']):
        raise ValueError('Complete, registered physical measurements required')
    rows = []
    for row in summary['runs']:
        actual = evidence / 'received' / (row['stem'] + '.bin')
        official = (REPO / 'examples/3dgs_scene/data/20260926T135717' /
                    row['case'] / 'official.bin')
        if not row['backend'].startswith('cpu') and not row['fp16_hls']['bitwise_equal']:
            raise ValueError('Physical output does not match its mode reference')
        rows.append({**row, 'display_quality_vs_official': quality(actual, official),
                     'readback_sha256': hashlib.sha256(actual.read_bytes()).hexdigest()})
    comparisons = []
    cases = sorted({row['case'] for row in rows})
    for case in cases:
        group = {row['backend']: row for row in rows if row['case'] == case}
        for backend, row in group.items():
            if backend.startswith('cpu'):
                continue
            item = {'case': case, 'backend': backend}
            for base in ['cpu1', 'cpu4', 'cpu_dense4', 'mode0', 'mode1']:
                if base in group:
                    item['speedup_vs_' + base] = group[base]['mean_ms'] / row['mean_ms']
            comparisons.append(item)
    output = {'version': summary['version'], 'scope': record['scope'],
              'schedule': record['schedule'], 'runs': rows, 'comparisons': comparisons,
              'precision_note': 'CPU FP32 versus FPGA FP16 is not an equal-precision hardware-only comparison. FLK1 modes share one physical design, but AABB/CAT changes the accepted contribution set. The frozen CPU Dense baseline has mini-tile CAT without the new sub-tile AABB stage; this algorithm difference must also be retained in comparisons.',
              'quality_note': 'PSNR/SSIM compare clipped RGB to the same official render, not to held-out photographs. Gaussian 11x11 SSIM sigma 1.5, population covariance, valid interior.',
              'tail_note': 'P95/P99 are descriptive sample quantiles, not validated tail latency.',
              'power': 'not measured', 'lpips': 'not measured',
              'input_result_sha256': hashlib.sha256((evidence / 'result.json').read_bytes()).hexdigest(),
              'input_summary_sha256': hashlib.sha256((evidence / 'summary.json').read_bytes()).hexdigest()}
    target = evidence / 'analysis.json'
    if target.exists():
        raise ValueError('Analysis already exists; preserve previous evidence')
    # Actual board readbacks only; the first column is explicitly the reference.
    panels = []
    for case in cases:
        group = {r['backend']: r for r in rows if r['case'] == case}
        official = REPO / 'examples/3dgs_scene/data/20260926T135717' / case / 'official.bin'
        panels.append((case + ' official reference', official))
        for backend in ['cpu4', 'mode0', 'mode2']:
            if backend not in group:
                raise ValueError('Comparison montage requires cpu4, mode0, mode2')
            row = group[backend]
            panels.append((f'{backend} {row["mean_ms"]:.3f} ms',
                           evidence / 'received' / (row['stem'] + '.bin')))
    width, height = 640, 356
    image = Image.new('RGB', (4 * width, len(cases) * (height + 40)), 'white')
    draw = ImageDraw.Draw(image)
    for i, (label, path) in enumerate(panels):
        _, rgb = read_rgb(path)
        frame = Image.fromarray(np.rint(rgb * 255).astype(np.uint8)).resize((width, height))
        x, y = (i % 4) * width, (i // 4) * (height + 40)
        draw.text((x + 8, y + 10), label, fill='black')
        image.paste(frame, (x, y + 40))
    image.save(evidence / 'board_comparison.png')
    target.write_text(json.dumps(output, indent=2))
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
