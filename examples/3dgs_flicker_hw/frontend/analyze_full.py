"""Summarize collected ARM measurements, verify hashes and render readback images."""
import argparse
import csv
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
sys.path.insert(0, str(REPO / 'examples/3dgs_scene'))
from check_output import compare


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    x = np.asarray(values, dtype=float)
    if not len(x) or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('Invalid measured samples')
    return dict(mean=float(x.mean()), median=float(np.median(x)),
                minimum=float(x.min()), maximum=float(x.max()), samples=len(x))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('evidence', type=Path)
    a = p.parse_args()
    ev = a.evidence
    board = json.loads((ev / 'board_result.json').read_text())
    host = json.loads((ev / 'result.json').read_text())
    config = json.loads((ev / 'config.json').read_text())
    if not board['complete'] or not host['complete']:
        raise ValueError('Incomplete benchmark')
    if sha(ev / 'board_result.json') != host['board_result_sha256']:
        raise ValueError('Collected result drift')
    report = dict(scope=board['scope'], source_result_sha256=sha(ev / 'board_result.json'),
                  sample_note=f'one warmup, {config["repeats"]} timed samples per view/backend; no tail or steady-state claims',
                  quality_reference='official CUDA images, not photographs',
                  p95=None, p99=None, lpips=None, power_watts=None, rows=[])
    preview = Image.new('RGB', (1280, 2 * (178 + 36)), 'white')
    draw = ImageDraw.Draw(preview)
    for view_index, view in enumerate(('0', '10')):
        official = REPO / 'examples/3dgs_scene/data/20260926T135717' / f'n559263_v{view}_w320/official.bin'
        _, ref_rgb = read_rgb(official)
        images = [('Official CUDA', ref_rgb)]
        for backend in config['backends']:
            rows = [r for r in board['rows'] if r['view'] == view and r['backend'] == backend]
            if len(rows) != config['repeats'] or sorted(r['sample'] for r in rows) != list(range(config['repeats'])):
                raise ValueError('Missing/duplicate samples')
            frame = ev / f'v{view}_{backend}.bin'
            if any(r['frame_sha256'] != sha(frame) or r['scene_sha256'] != config['scene_hashes'][view] for r in rows):
                raise ValueError('Frame or scene drift')
            row = dict(view=view, backend=backend, frame_sha256=sha(frame),
                       quality=quality(frame, official), strict_official=compare(frame, official))
            for key in ('attributes_ms', 'group_sort_ms', 'render_wall_ms', 'total_ms'):
                row[key] = stats([r[key] for r in rows])
            row['fps_from_mean_whole_invocation'] = 1000 / row['total_ms']['mean']
            timings = []
            for r in rows:
                with (ev / f'v{view}_{backend}_{r["sample"]}_render_timing.csv').open() as f:
                    inner = list(csv.DictReader(f))
                if len(inner) != 1:
                    raise ValueError('Benchmark must render exactly one frame')
                timings.append({k: float(v) for k, v in inner[0].items() if k != 'sample'})
            row['render_inner_counters'] = {k: stats([t[k] for t in timings]) for k in timings[0]}
            report['rows'].append(row)
            if backend != 'cpu_base':
                _, rgb = read_rgb(frame)
                images.append((backend, rgb))
                if backend == 'fpga':
                    difference = np.clip(np.abs(rgb-ref_rgb) * 8, 0, 1)
        images.append(('FPGA absolute RGB error x8', difference))
        for col, (label, rgb) in enumerate(images):
            x, y = col * 320, view_index * 214
            draw.text((x + 6, y + 10), f'v{view}: {label}', fill='black')
            preview.paste(Image.fromarray(np.rint(rgb * 255).astype(np.uint8)), (x, y + 36))
    for row in report['rows']:
        peers = {r['backend']: r for r in report['rows'] if r['view'] == row['view']}
        row['whole_speedup_vs_cpu_dense'] = peers['cpu_dense']['total_ms']['mean'] / row['total_ms']['mean']
        row['whole_speedup_vs_cpu_base'] = peers['cpu_base']['total_ms']['mean'] / row['total_ms']['mean']
    (ev / 'analysis.json').write_text(json.dumps(report, indent=2))
    columns = ['view', 'backend', 'mean_ms', 'median_ms', 'min_ms', 'max_ms', 'fps', 'psnr_db', 'ssim']
    with (ev / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in report['rows']:
            t, q = r['total_ms'], r['quality']
            writer.writerow(dict(view=r['view'], backend=r['backend'], mean_ms=t['mean'],
                                 median_ms=t['median'], min_ms=t['minimum'], max_ms=t['maximum'],
                                 fps=r['fps_from_mean_whole_invocation'], psnr_db=q['psnr_clamped_rgb_db'],
                                 ssim=q['ssim_gaussian11_valid']))
    preview.save(ev / 'render_comparison.png')
    print((ev / 'summary.csv').read_text())


if __name__ == '__main__':
    main()
