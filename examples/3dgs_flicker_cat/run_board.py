"""Run the paper-derived CAT reference on the board's ARM CPU."""
import argparse
import csv
import hashlib
import json
import shlex
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / 'examples/3dgs_compositor/board'))
sys.path.insert(0, str(REPO / 'examples/3dgs_scene'))
import remote
from check_output import compare


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_timing(path):
    with Path(path).open(newline='') as file:
        rows = list(csv.DictReader(file))
    return {
        'samples': len(rows),
        'mean_ms': sum(float(row['total_ms']) for row in rows) / len(rows),
        'min_ms': min(float(row['total_ms']) for row in rows),
        'max_ms': max(float(row['total_ms']) for row in rows),
        'means': {key: sum(float(row[key]) for row in rows) / len(rows)
                  for key in rows[0] if key not in ('sample', 'total_ms')},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('case', help='scene case directory name')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--modes', nargs='+', default=['base', 'dense', 'dense_pr', 'sparse', 'adaptive'],
                        choices=['base', 'dense', 'dense_pr', 'sparse', 'adaptive'])
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20 or not 0 <= args.warmup <= 3 or not 1 <= args.threads <= 4:
        parser.error('invalid repeat/thread count')
    data = REPO / 'examples/3dgs_scene/data/20260926T135717' / args.case
    scene = data / 'scene.bin'
    official = data / 'official.bin'
    if not scene.is_file() or not official.is_file():
        parser.error(f'missing scene or official image in {data}')
    stamp = datetime.now().strftime('%Y%m%dT%H%M%S')
    evidence = ROOT / 'evidence' / f'{stamp}_{args.case}'
    evidence.mkdir(parents=True)
    source = ROOT / 'cat_reference.cpp'
    (evidence / 'sources').mkdir()
    for path in (source, Path(__file__), REPO / 'examples/3dgs_scene/check_output.py'):
        shutil.copy2(path, evidence / 'sources' / path.name)
    modes = list(dict.fromkeys(['base'] + args.modes))
    dest = f'/root/fpga43dgs_flicker_cat/{stamp}_{args.case}'
    report = {
        'schema': 'flicker-cat-board-cpu-v2',
        'scope': 'prepared-scene ARM CPU CAT reference; no FPGA or NPU',
        'paper_basis': 'FLICKER DATE 2026 Section II-III, mini-tile CAT',
        'case': args.case, 'threads': args.threads,
        'repeats': args.repeats, 'warmup': args.warmup,
        'scene_sha256': sha(scene), 'official_sha256': sha(official),
        'source_sha256': sha(source), 'runs': {}, 'completed': False,
        'modes': modes, 'remote_directory': dest,
        'timing_scope': 'resident projected input -> full output in memory, including CAT setup and mask allocation; excludes file IO, SSH, GPU projection/sorting',
        'audit_timing_is_not_performance': True,
    }
    report_path = evidence / 'result.json'

    def save():
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    save()
    client = remote.connect()
    try:
        report['board_identity'] = remote.run(client, 'uname -a; cat /proc/sys/kernel/random/boot_id')[1]
        report['cpu_info'] = remote.run(client, 'cat /proc/cpuinfo')[1]
        report['compiler'] = remote.run(client, 'g++ --version')[1]
        remote.run(client, f'mkdir -p {shlex.quote(dest)}')
        sftp = client.open_sftp()
        try:
            sftp.put(str(source), f'{dest}/cat_reference.cpp')
            sftp.put(str(scene), f'{dest}/scene.bin')
        finally:
            sftp.close()
        flags = '-O3 -std=gnu++17 -ffp-contract=off -fopenmp -Wall -Wextra'
        report['compile_flags'] = flags
        remote.run(client, f'cd {shlex.quote(dest)} && g++ cat_reference.cpp -o cat_reference {flags}',
                   timeout=180, log=evidence / 'build.log')
        report['binary_sha256'] = remote.run(client, f'sha256sum {shlex.quote(dest)}/cat_reference')[1].split()[0]
        test_output = remote.run(client, f'{shlex.quote(dest)}/cat_reference --self-test',
                                 timeout=30, log=evidence / 'self_test.log')[1]
        report['self_test'] = json.loads(test_output.strip())
        if not report['self_test']['passed']:
            raise RuntimeError('CAT unit verification failed')
        report['remote_input_sha256'] = remote.run(client, f'sha256sum {shlex.quote(dest)}/scene.bin')[1].split()[0]
        if report['remote_input_sha256'] != report['scene_sha256']:
            raise RuntimeError('board input hash mismatch')
        save()
        for mode in modes:
            stem = f'{args.case}_{mode}'
            cmd = (f'cd {shlex.quote(dest)} && ./cat_reference scene.bin {stem} {mode} '
                   f'{args.threads} {args.repeats} {args.warmup} 0')
            remote.run(client, cmd, timeout=900, log=evidence / f'{stem}.log')
            sftp = client.open_sftp()
            try:
                for suffix in ('.bin', '_timing.csv'):
                    sftp.get(f'{dest}/{stem}{suffix}', str(evidence / f'{stem}{suffix}'))
            finally:
                sftp.close()
            record = {'timing': read_timing(evidence / f'{stem}_timing.csv'),
                      'output_sha256': sha(evidence / f'{stem}.bin')}
            record['vs_official'] = compare(evidence / f'{stem}.bin', official)
            if mode == 'base' and not record['vs_official']['passed']:
                report['runs'][mode] = record
                save()
                raise RuntimeError('unfiltered reference failed frozen official gates')
            if mode != 'base':
                record['vs_base'] = compare(evidence / f'{stem}.bin', evidence / f'{args.case}_base.bin')
            if mode == 'dense_pr' and 'dense' in report['runs']:
                record['vs_dense'] = compare(evidence / f'{stem}.bin', evidence / f'{args.case}_dense.bin')
            if args.audit and mode != 'base':
                audit_stem = f'{stem}_audit'
                audit_cmd = (f'cd {shlex.quote(dest)} && ./cat_reference scene.bin {audit_stem} '
                             f'{mode} {args.threads} 1 0 1')
                remote.run(client, audit_cmd, timeout=900, log=evidence / f'{audit_stem}.log')
                sftp = client.open_sftp()
                try:
                    sftp.get(f'{dest}/{audit_stem}_timing.csv', str(evidence / f'{audit_stem}_timing.csv'))
                    sftp.get(f'{dest}/{audit_stem}.bin', str(evidence / f'{audit_stem}.bin'))
                finally:
                    sftp.close()
                record['audit'] = read_timing(evidence / f'{audit_stem}_timing.csv')
                record['audit_output_matches_timed'] = sha(evidence / f'{audit_stem}.bin') == record['output_sha256']
            report['runs'][mode] = record
            save()
        report['completed'] = True
        save()
    except Exception as error:
        report['error'] = str(error)
        save()
        raise
    finally:
        client.close()
    print(report_path)
    for mode, record in report['runs'].items():
        print(mode, f"{record['timing']['mean_ms']:.3f} ms",
              f"PSNR {record['vs_official']['psnr_vs_matching_official_db']}",
              f"rejected {record['timing']['means']['rejected_pairs']:.0f}")


if __name__ == '__main__':
    main()
