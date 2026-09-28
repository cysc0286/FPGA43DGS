"""Summarize a completed measurement and append its immutable version records."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def measurement_scope(backend):
    """Keep historical FLK0 and FLK1/CAT scopes distinct in the registry."""
    if backend in ('cpu1', 'cpu4'):
        return 'prepared-frame FP32 without CAT'
    if backend == 'cpu_dense4':
        return 'prepared-frame FP32 Dense CAT, four CPU threads'
    if backend in ('serial', 'pipeline'):
        return 'prepared-frame FP16 full evaluation without CAT; all host stages included'
    modes = {'mode0': 'AABB/CAT off', 'mode1': 'AABB only',
             'mode2': 'AABB + Dense CAT', 'mode3': 'AABB + Sparse CAT',
             'mode4': 'AABB + Smooth-Focused CAT', 'mode5': 'AABB + Spiky-Focused CAT'}
    if backend in modes:
        return 'prepared-frame FLK1 FP16 ' + modes[backend] + '; all host stages included'
    raise ValueError('Unregistered backend scope: ' + backend)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('evidence', type=Path)
    p.add_argument('--version', required=True)
    args = p.parse_args()
    result = json.loads((args.evidence / 'result.json').read_text())
    if not result['complete']:
        raise ValueError('Measurements incomplete; cannot register a finished version')
    rows, summaries = [], []
    for item in result['runs']:
        scope = measurement_scope(item['backend'])
        if item['backend'].startswith('mode'):
            if result.get('schedule') not in ('serial', 'pipeline'):
                raise ValueError('FLK1 host schedule is not recorded')
            scope += '; host schedule=' + result['schedule']
        path = args.evidence / 'received' / (item['stem'] + '_timing.csv')
        values = np.atleast_1d(np.genfromtxt(path, names=True, delimiter=','))
        cpu = item['backend'].startswith('cpu')
        total = values['total_ms'] if cpu else values['total_us'] / 1000
        if not np.isfinite(total).all() or (total <= 0).any():
            raise ValueError('Invalid total durations')
        cpu_time_ms = values['process_cpu_ms'] if cpu else values['cpu_us'] / 1000
        summary = {**item, 'samples': len(total), 'mean_ms': float(total.mean()),
                   'median_ms': float(np.median(total)), 'min_ms': float(total.min()), 'max_ms': float(total.max()),
                   'p95_ms': float(np.quantile(total,.95)), 'p99_ms': float(np.quantile(total,.99)),
                   'percentile_note': f'Descriptive quantiles of {len(total)} samples; this small repeated-view set does not establish tail latency',
                   'fps': float(1000/total.mean()), 'cpu_percent': float(100*cpu_time_ms.sum()/total.sum()),
                   'means': {name: float(values[name].mean()) for name in values.dtype.names if name != 'sample'},
                   'timing_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if not cpu:
            if not item.get('fp16_hls', {}).get('bitwise_equal', False):
                raise ValueError('Hardware has not passed its selected FP16 reference')
            trace = np.atleast_1d(np.genfromtxt(args.evidence / 'received' / (item['stem']+'_trace.csv'), names=True, delimiter=','))
            first = trace['busy_before_upload'].astype(np.uint32)
            last = trace['busy_after_upload'].astype(np.uint32)
            inside = ((first & 1) != 0) & ((last & 1) != 0) & ((first & 4) == 0) & ((last & 4) == 0)
            summary['uploads_entirely_during_active_job'] = int(inside.sum())
            summary['upload_intervals_total'] = len(trace)
            summary['proven_upload_overlap_us'] = float((trace['upload1_us'][inside]-trace['upload0_us'][inside]).sum())
            summary['overlap_note'] = 'Busy and no completion observed before AND after upload, prior completion unacknowledged; partial overlaps excluded'
        summaries.append(summary)
        rows.append({key: summary[key] for key in ['samples','mean_ms','median_ms','p95_ms','p99_ms','fps','timing_sha256']})
        rows[-1].update(version=args.version, evidence=args.evidence.name, case=item['case'], backend=item['backend'],
                        scope=scope,
                        official_strict=item['official']['passed'])
    (args.evidence/'summary.json').write_text(json.dumps({'version':args.version,'runs':summaries},indent=2))
    path = ROOT/'versions.csv'
    with path.open(newline='') as f:
        reader=csv.DictReader(f);fields=reader.fieldnames;old=list(reader)
    keys={(r['evidence'],r['case'],r['backend']) for r in old}
    with path.open('a',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        for row in rows:
            if (row['evidence'],row['case'],row['backend']) not in keys:
                writer.writerow(row)
    print(json.dumps({'version':args.version,'runs':summaries},indent=2))


if __name__=='__main__':
    main()
