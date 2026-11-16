"""Check published PWM candidate bytes and BOOT partitions; never open a device."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'examples/3dgs_reconstruction/mvsplat/video_input/firmware/j17_23_21_20261115'


def verified_file(base, entry):
    relative = Path(entry['path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Manifest path escapes its root')
    path = base / relative
    if path.is_symlink() or not path.resolve().is_relative_to(base.resolve()):
        raise ValueError('Unsafe firmware/source path')
    raw = path.read_bytes()
    if len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
        raise ValueError('Manifest mismatch: ' + entry['path'])
    return raw


def main():
    data = json.loads((FOLDER / 'manifest.json').read_text(encoding='utf-8'))
    if data['schema'] != 'camera_servo_firmware_v1' or data['default_enabled'] is not False:
        raise ValueError('PWM image must remain an explicitly selected candidate')
    if data['status'] != 'compiled_candidate_not_board_qualified' or data['validation']['board_tested']:
        raise ValueError('Published validation status changed without board qualification')
    for entry in data['files']:
        verified_file(FOLDER, entry)
    sources = [entry for entry in data['sources'] if entry.get('packaged', True)]
    for entry in sources:
        verified_file(ROOT, entry)
    boot = (FOLDER / 'BOOT.bin').read_bytes()
    bit = (FOLDER / 'ai7030_edif_top_disable_icap.bit').read_bytes()
    if hashlib.sha256(boot).hexdigest() != data['boot_sha256']:
        raise ValueError('BOOT identity differs from firmware manifest')
    if hashlib.sha256(bit).hexdigest() != data['bit_sha256']:
        raise ValueError('Bitstream identity differs from firmware manifest')
    comparison = json.loads((FOLDER / 'boot_comparison.json').read_text(encoding='utf-8'))
    if comparison['boot_sha256'] != data['boot_sha256'] or comparison['bit_sha256'] != data['bit_sha256']:
        raise ValueError('BOOT comparison is for different artifacts')
    for entry in comparison['partitions']:
        begin, size = entry['offset'], entry['bytes']
        if begin < 0 or size <= 0 or begin + size > len(boot):
            raise ValueError('BOOT partition is outside the image')
        if hashlib.sha256(boot[begin:begin + size]).hexdigest() != entry['sha256']:
            raise ValueError('BOOT partition hash mismatch')
    print(f"PASS camera PWM: {len(data['files'])} artifacts, {len(sources)} matching RTL/XDC sources, "
          f"{len(comparison['partitions'])} BOOT partitions")
    print('Compiled candidate only; no flashing, servo motion or board qualification performed.')


if __name__ == '__main__':
    main()
