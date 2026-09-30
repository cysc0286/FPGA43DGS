"""Repeat a frozen board camera benchmark and compare every archived frame.

Uses the exact prior command/binary/scene. Does not install firmware or reboot.
Authentication comes from the existing remote helper's environment contract.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / 'examples/3dgs_compositor/board'))
from remote import connect, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--remote-out', required=True)
    parser.add_argument('--expected-boot-sha256', required=True)
    args = parser.parse_args()
    if not args.remote_out.startswith('/root/fpga43dgs_reconstruction/'):
        parser.error('Remote output must be a new reconstruction experiment')
    if len(args.expected_boot_sha256) != 64 or any(
            c not in '0123456789abcdef' for c in args.expected_boot_sha256):
        parser.error('Expected BOOT digest must be SHA-256')
    args.out.mkdir(parents=True, exist_ok=False)
    before = json.loads((args.baseline / 'results.json').read_text())
    if not before.get('complete') or not before.get('builds'):
        raise ValueError('Baseline is incomplete or empty')
    prior = json.loads((args.baseline / 'provenance.json').read_text())
    command = json.loads((args.baseline / 'command.json').read_text())
    command[command.index('--out') + 1] = args.remote_out
    (args.out / 'command.json').write_text(json.dumps(command, indent=2))
    result = dict(complete=False, board_executed=False,
                  baseline=str(args.baseline.resolve()),
                  scope='loaded scene camera to complete RGB; no video or display')
    client = None
    try:
        client = connect()
        _, boot_id = run(client, 'cat /proc/sys/kernel/random/boot_id')
        boot_id = boot_id.strip()
        # Read the SD boot partition without leaving it mounted. Refuse an
        # existing mount, rather than changing another process's mount state.
        script = ('set -eu; test "$(uname -m)" = aarch64; '
                  'if findmnt -rn -S /dev/mmcblk0p1 >/dev/null; then exit 7; fi; '
                  'mkdir -p /mnt/fpga_resource_readonly; '
                  'mount -t vfat -o ro /dev/mmcblk0p1 /mnt/fpga_resource_readonly; '
                  "trap 'umount /mnt/fpga_resource_readonly' EXIT; "
                  'sha256sum /mnt/fpga_resource_readonly/BOOT.bin')
        _, boot = run(client, 'bash -c ' + shlex.quote(script))
        actual_boot = boot.split()[0]
        if actual_boot != args.expected_boot_sha256:
            raise ValueError('Unexpected SD firmware; benchmark not started')
        if actual_boot != prior['boot_sha256']:
            if not prior.get('boot_id') or boot_id == prior['boot_id']:
                raise ValueError('Firmware changed without a verified new boot ID')
        result.update(boot_id=boot_id, boot_sha256=actual_boot)
        run(client, 'test ! -e ' + shlex.quote(args.remote_out))
        run(client, shlex.join(command), timeout=120, log=args.out / 'console.txt')
        result['board_executed'] = True
        with client.open_sftp() as sftp:
            sftp.get(args.remote_out + '/results.json', str(args.out / 'results.json'))
        after = json.loads((args.out / 'results.json').read_text())
        if not after.get('complete') or before['scene_sha256'] != after['scene_sha256']:
            raise ValueError('Incomplete run or scene changed')
        if set(before['builds']) != set(after.get('builds', {})):
            raise ValueError('Benchmark build set changed')
        comparisons = {}
        for name, build in after['builds'].items():
            base = before['builds'][name]
            if build['binary_sha256'] != base['binary_sha256']:
                raise ValueError('Binary changed during hardware comparison')
            expected = {c['camera']: (c['frame_sha256'], c['rgb_sha256'])
                        for c in base['runs'][0]['checks']}
            if not expected or len(build['runs']) != len(base['runs']):
                raise ValueError('Missing cameras or rounds')
            for old_run, new_run in zip(base['runs'], build['runs']):
                if ({c['camera'] for c in new_run['checks']} != set(expected)
                        or len(new_run['records']) != len(old_run['records'])
                        or not new_run['records']):
                    raise ValueError('Benchmark coverage changed')
            exact = all(expected[c['camera']] == (c['frame_sha256'], c['rgb_sha256'])
                        for row in build['runs'] for c in row['checks'])
            if not exact or not all(row['all_timed_frames_match'] for row in build['runs']):
                raise ValueError('New hardware frame mismatch')
            b, a = base['summary'], build['summary']
            comparisons[name] = dict(before=b, after=a, frames_exact=True,
                latency_reduction_percent=100 * (1 - a['mean_ms'] / b['mean_ms']),
                hardware_cycle_reduction_percent=100 * (
                    1 - a['mean_hardware_cycles'] / b['mean_hardware_cycles']))
            with client.open_sftp() as sftp:
                for check in build['runs'][0]['checks']:
                    camera = Path(check['camera']).stem
                    local = args.out / name / camera
                    local.mkdir(parents=True)
                    for filename in ['frame.bin', 'frame.ppm', 'result.json']:
                        sftp.get(f'{args.remote_out}/{name}_r0/{camera}/{filename}',
                                 str(local / filename))
                    raw = (local / 'frame.bin').read_bytes()
                    ppm = (local / 'frame.ppm').read_bytes()
                    header = f"P6\n{check['width']} {check['height']}\n255\n".encode()
                    if (hashlib.sha256(raw).hexdigest() != check['frame_sha256']
                            or not ppm.startswith(header)
                            or len(ppm) - len(header) != check['rgb_bytes']
                            or hashlib.sha256(ppm[len(header):]).hexdigest()
                            != check['rgb_sha256']):
                        raise ValueError('Downloaded frame failed digest verification')
        result.update(complete=True, comparisons=comparisons)
    except Exception as error:
        result['error'] = str(error)
        raise
    finally:
        if client is not None:
            client.close()
        result['files_sha256'] = {
            str(p.relative_to(args.out)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in args.out.rglob('*') if p.is_file() and p.name != 'comparison.json'}
        (args.out / 'comparison.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result['comparisons'], indent=2))


if __name__ == '__main__':
    main()
