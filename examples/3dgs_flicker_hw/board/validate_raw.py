"""Upload only real inputs; compare returned hardware bytes with a local HLS golden."""
import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / 'examples/3dgs_compositor/board'))
from remote import connect, run


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    args = parser.parse_args()
    stage = json.loads((args.stage / 'result.json').read_text())
    meta = json.loads((args.reference / 'input.json').read_text())
    source, golden = args.reference / 'input.flk', args.reference / 'expected.raw'
    tiles = len(meta['tiles'])
    if not stage['compiled'] or sha(source) != meta['input_sha256']:
        raise ValueError('Unbuilt stage or input drift')
    if golden.stat().st_size != tiles * 4096:
        raise ValueError('Incomplete HLS reference')
    ev = ROOT / 'evidence' / ('raw_' + datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
    ev.mkdir(parents=True)
    dest = stage['remote'] + '/' + ev.name
    report = {'stage': str(args.stage), 'reference': str(args.reference),
              'input_sha256': sha(source), 'golden_sha256': sha(golden),
              'binary_sha256': stage['binary_sha256'], 'tiles': tiles,
              'input_records': meta['records'], 'passed': False, 'remote': dest}
    client = connect()
    try:
        if run(client, 'sha256sum ' + stage['remote'] + '/render')[1].split()[0] != stage['binary_sha256']:
            raise ValueError('Executable drift')
        run(client, 'mkdir -p ' + dest)
        run(client, 'uname -a; cat /proc/sys/kernel/random/boot_id', log=ev / 'environment.txt')
        with client.open_sftp() as sftp:
            sftp.put(str(source), dest + '/input.flk')
        if run(client, 'sha256sum ' + dest + '/input.flk')[1].split()[0] != report['input_sha256']:
            raise ValueError('Upload drift')
        run(client, f'timeout 90 {stage["remote"]}/render raw {dest}/input.flk {tiles} {dest}/output.raw',
            timeout=100, log=ev / 'run.txt')
        with client.open_sftp() as sftp:
            sftp.get(dest + '/output.raw', str(ev / 'output.raw'))
        actual, expected = (ev / 'output.raw').read_bytes(), golden.read_bytes()
        report['output_sha256'] = sha(ev / 'output.raw')
        report['output_bytes'] = len(actual)
        report['differing_bytes'] = sum(a != b for a, b in zip(actual, expected)) + abs(len(actual) - len(expected))
        report['first_mismatch_byte'] = next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b), None)
        report['passed'] = actual == expected
        print(json.dumps(report, indent=2))
        if not report['passed']:
            raise RuntimeError('Hardware/HLS mismatch: retain evidence and diagnose before frame timing')
    finally:
        (ev / 'result.json').write_text(json.dumps(report, indent=2))
        client.close()
        print('EVIDENCE=' + str(ev))


if __name__ == '__main__':
    main()
