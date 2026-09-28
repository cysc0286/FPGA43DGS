"""Summarize frozen board CAT readbacks; never infer FPGA results from CPU runs."""
import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import convolve1d

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / 'examples/3dgs_scene'))
from check_output import pixels


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rgb(path):
    width, height, data = pixels(path)
    raw = data['rgba'][:, :3].reshape(height, width, 3).astype(np.float64)
    return raw, np.clip(raw, 0, 1)


def ssim(a, b):
    # Explicit convention: Gaussian 11x11, sigma 1.5, population covariance,
    # data range 1; discard five pixels at every border, then average RGB.
    kernel = np.exp(-np.arange(-5, 6, dtype=np.float64)**2 / (2 * 1.5**2))
    kernel /= kernel.sum()

    def smooth(x):
        return convolve1d(convolve1d(x, kernel, axis=0, mode='constant'),
                          kernel, axis=1, mode='constant')[5:-5, 5:-5]

    ma, mb = smooth(a), smooth(b)
    va, vb, cov = smooth(a*a)-ma*ma, smooth(b*b)-mb*mb, smooth(a*b)-ma*mb
    return float(np.mean(((2*ma*mb+.01**2)*(2*cov+.03**2)) /
                         ((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))))


def quality(actual, expected):
    raw_a, a = read_rgb(actual)
    raw_b, b = read_rgb(expected)
    diff = raw_a-raw_b
    mse = float(np.mean((a-b)**2))
    return {
        'psnr_clamped_rgb_db': None if mse == 0 else -10*math.log10(mse),
        'ssim_gaussian11_valid': ssim(a, b),
        'raw_rgb_max': float(np.max(np.abs(diff))),
        'raw_rgb_mean': float(np.mean(np.abs(diff))),
        'raw_rgb_error_p99': float(np.quantile(np.abs(diff), .99)),
        'lpips': None, 'lpips_status': 'not measured',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('evidence', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'results_summary.json')
    args = parser.parse_args()
    rows = []
    reports = []
    for evidence in args.evidence:
        report = json.loads((evidence / 'result.json').read_text(encoding='utf-8'))
        if not report['completed']:
            raise RuntimeError(f'incomplete evidence: {evidence}')
        case = report['case']
        baseline = evidence / f'{case}_base.bin'
        baseline_ms = report['runs']['base']['timing']['mean_ms']
        official = REPO / 'examples/3dgs_scene/data/20260926T135717' / case / 'official.bin'
        scene = official.with_name('scene.bin')
        if sha(official) != report['official_sha256'] or sha(scene) != report['scene_sha256']:
            raise RuntimeError('scene or Golden hash drift')
        with scene.open('rb') as file:
            if file.read(8) != b'GSSCN001':
                raise ValueError('input ABI')
            header = np.frombuffer(file.read(28), dtype='<u4')
        group = []
        for mode, run in report['runs'].items():
            path = evidence / f'{case}_{mode}.bin'
            if sha(path) != run['output_sha256']:
                raise RuntimeError(f'output hash drift: {path}')
            with (evidence / f'{case}_{mode}_timing.csv').open(newline='') as file:
                timings = [float(r['total_ms']) for r in csv.DictReader(file)]
            counts = run['timing']['means']
            row = {
                'evidence': str(evidence.resolve().relative_to(REPO)),
                'report_sha256': sha(evidence / 'result.json'),
                'case': case, 'mode': mode, 'threads': report['threads'],
                'source_sha256': report['source_sha256'],
                'input_sha256': report['scene_sha256'],
                'output_sha256': run['output_sha256'],
                'model_gaussians': int(header[3]), 'active_gaussians': int(header[4]),
                'tile_entries': int(header[5]), 'input_bytes': scene.stat().st_size,
                'samples': len(timings), 'warmup': report['warmup'],
                'mean_ms': float(np.mean(timings)), 'median_ms': float(np.median(timings)),
                'min_ms': min(timings), 'max_ms': max(timings),
                'p95_ms': None, 'p99_ms': None,
                'tail_status': '2 or 5 samples insufficient for tail-latency claims',
                'raster_fps': 1000/float(np.mean(timings)),
                'speedup_vs_matching_base': baseline_ms/float(np.mean(timings)),
                'counts': counts,
                'rejected_fraction': counts['rejected_pairs']/counts['candidate_pairs'],
                'strict_official_passed': run['vs_official']['passed'],
                'official_transmittance_max': run['vs_official']['transmittance_max'],
                'official_last_mismatches': run['vs_official']['last_mismatches'],
                'vs_base': quality(path, baseline), 'vs_official': quality(path, official),
                'audit': run.get('audit'),
                'audit_output_matches_timed': run.get('audit_output_matches_timed'),
                'vs_dense_bitwise': run.get('vs_dense', {}).get('bitwise_equal'),
                'fpga_executed': False, 'npu_executed': False,
                'power_watts': None, 'hardware_resources': None,
            }
            group.append(row)
            rows.append(row)
        reports.append({'evidence':str(evidence.resolve().relative_to(REPO)), 'rows':group})
    summary = {
        'scope': 'ARM CPU complete prepared-frame rasterization; GPU projection/sorting, file IO and SSH excluded',
        'quality_reference': 'matching unfiltered CPU or official CUDA render; not captured photographs',
        'ssim_definition': 'Gaussian 11x11 sigma=1.5, population covariance, valid interior, RGB mean, [0,1] clipped float64',
        'lpips_status': 'not measured',
        'approximation_gate': 'no approximate acceptance threshold has been approved; original strict gates retained',
        'reports': reports,
    }
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    csv_path = args.output.with_suffix('.csv')
    columns = ['evidence','case','mode','threads','samples','mean_ms','median_ms','min_ms','max_ms',
               'raster_fps','speedup_vs_matching_base','rejected_fraction','candidate_pairs',
               'full_evaluations','missed_qualified','process_cpu_ms','rss_kib',
               'psnr_vs_base_db','ssim_vs_base','raw_rgb_max','raw_rgb_mean',
               'official_transmittance_max','official_last_mismatches',
               'strict_official_passed','vs_dense_bitwise','source_sha256']
    with csv_path.open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flat = {key:row.get(key) for key in columns}
            flat.update({key:row['counts'][key] for key in
                         ('candidate_pairs','full_evaluations','process_cpu_ms','rss_kib')})
            flat['missed_qualified'] = (row['audit']['means']['missed_qualified']
                                        if row['audit'] else None)
            q = row['vs_base']
            flat.update(psnr_vs_base_db=q['psnr_clamped_rgb_db'],ssim_vs_base=q['ssim_gaussian11_valid'],
                        raw_rgb_max=q['raw_rgb_max'],raw_rgb_mean=q['raw_rgb_mean'])
            writer.writerow(flat)
    latest = args.evidence[-1]
    report = json.loads((latest / 'result.json').read_text(encoding='utf-8'))
    case = report['case']
    modes = [m for m in ['base','dense','dense_pr','adaptive','sparse'] if m in report['runs']]
    base_raw, base_rgb = read_rgb(latest / f'{case}_base.bin')
    h, w, _ = base_rgb.shape
    preview = Image.new('RGB', (w*len(modes), 2*h+62), 'white')
    draw = ImageDraw.Draw(preview)
    for index, mode in enumerate(modes):
        raw, rgb = read_rgb(latest / f'{case}_{mode}.bin')
        heat = np.clip(np.max(np.abs(raw-base_raw), axis=2)*20,0,1)
        heat_rgb = np.stack([heat,np.zeros_like(heat),np.zeros_like(heat)],axis=2)
        preview.paste(Image.fromarray(np.uint8(rgb*255+.5)),(index*w,24))
        preview.paste(Image.fromarray(np.uint8(heat_rgb*255+.5)),(index*w,h+54))
        draw.text((index*w+5,5),mode,fill='black')
        draw.text((index*w+5,h+32),'raw RGB max error x20',fill='black')
    preview.save(ROOT / 'comparison.png')
    print(args.output)
    for row in rows:
        if row['threads']==4:
            print(row['case'],row['mode'],round(row['mean_ms'],3),
                  round(row['speedup_vs_matching_base'],3),row['vs_base'])


if __name__ == '__main__':
    main()
