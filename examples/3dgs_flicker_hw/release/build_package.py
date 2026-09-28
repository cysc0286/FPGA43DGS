"""Freeze the verified rendering path as a relocatable board runtime package."""
import hashlib
import json
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO/'examples/3dgs_compositor/board'))
from remote import connect


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    output = REPO/'releases/3dgs_renderer_v1_20260928'
    output.mkdir(parents=True, exist_ok=False)
    provenance = {}

    def copy(source, target, expected=None):
        if expected is not None and sha(source) != expected:
            raise ValueError('Frozen source drift: ' + str(source))
        dest = output/target
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        provenance[target] = str(source.relative_to(REPO))

    stages = [('attributes', 'frontend_attr_20260927T231255', 'attributes.cpp'),
              ('group_sort', 'frontend_group_20260927T233920', 'group_sort.cpp'),
              ('render', 'stagecat_20260927T122707', 'render.cpp')]
    client = connect()
    try:
        with client.open_sftp() as sftp:
            for binary, name, source in stages:
                ev = ROOT/'evidence'/name
                record = json.loads((ev/'result.json').read_text())
                (output/'bin').mkdir(exist_ok=True)
                sftp.get(record['remote']+'/'+binary, str(output/'bin'/binary))
                if sha(output/'bin'/binary) != record['binary_sha256']:
                    raise ValueError('Board executable drift: ' + binary)
                provenance['bin/'+binary] = record['remote']+'/'+binary
                expected = record['source_hashes'][source] if binary == 'render' else record['source_sha256']
                copy(ev/source, 'src/cpu/'+source, expected)
                copy(ev/'result.json', 'provenance/'+name+'.json')
                if binary == 'render':
                    copy(ev/'cat_reference.cpp', 'src/cpu/cat_reference.cpp', record['source_hashes']['cat_reference.cpp'])
    finally:
        client.close()
    model_record = json.loads((ROOT/'evidence/frontend_attr_20260927T231255/result.json').read_text())
    copy(REPO/'npu_3dgs/data/official/train/point_cloud/iteration_7000/point_cloud.ply',
         'data/model.ply', model_record['model_sha256'])
    copy(REPO/'npu_3dgs/data/official/train/cameras.json', 'data/cameras.json')
    for view in (0, 10):
        copy(ROOT/f'evidence/frontend_camera_20260927/v{view}.bin', f'data/v{view}.bin')
        copy(ROOT/f'evidence/full_benchmark_20260927T234919/v{view}_fpga.bin', f'reference/v{view}_fpga.bin')
        copy(ROOT/f'evidence/frontend_hls_v{view}_20260927/timed_frame_check.json', f'provenance/v{view}_exact_check.json')
    for name in ('render.py', 'README.md', 'INTERFACE.md', 'rebuild_cpu.sh'):
        copy(ROOT/'release'/name, name)
    copy(ROOT/'frontend/pack_camera.py', 'pack_camera.py')
    copy(ROOT/'frontend/LICENSE-GraphDECO.md', 'LICENSE-GraphDECO.md')
    baseline = ROOT/'evidence/full_benchmark_20260927T234919'
    for name in ('analysis.json', 'timing.csv', 'summary.csv', 'render_comparison.png', 'physical_provenance.json'):
        copy(baseline/name, 'baseline/'+name)
    board = ROOT/'build/board_cat_20260927_210933'
    copy(board/'boot/BOOT.bin', 'firmware/BOOT.bin', '9918f9b69525bad88d8a9f06c5ab0940582df341ca32edc786d4377d3ba137cc')
    for name in ('timing_review.json', 'utilization.rpt', 'renderer_clock_and_paths.txt'):
        copy(board/name, 'hardware_reports/'+name)
    hls = ROOT/'evidence/hls_pipeline_splitopt_20260927T210845'
    for category, target in ((hls/'sources', 'src/hls'), (board/'frozen_artifacts/sources/rtl', 'src/rtl')):
        for source in sorted(category.rglob('*')):
            if source.is_file():
                copy(source, target+'/'+source.relative_to(category).as_posix())
    copy(board/'frozen_artifacts/flicker_manifest.json', 'provenance/flicker_manifest.json')
    (output/'provenance/origins.json').write_text(json.dumps(provenance, indent=2))
    files = {path.relative_to(output).as_posix(): dict(bytes=path.stat().st_size, sha256=sha(path))
             for path in sorted(output.rglob('*')) if path.is_file()}
    manifest = dict(name=output.name, purpose='3DGS rendering only; trained PLY and camera to framebuffer',
                    board='30TAI Lite', sdk='ICraft 3.36.1 ARM64 transport runtime',
                    fpga='FLK1 ABI2 split II16 at 200 MHz', files=files)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    for suffix in ('.zip', '.tar.gz'):
        if output.with_suffix(suffix).exists():
            raise ValueError('Preserve prior archive')
    with zipfile.ZipFile(output.with_suffix('.zip'), 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                z.write(path, output.name+'/'+path.relative_to(output).as_posix())
    def permissions(info):
        info.mode = 0o755 if info.isdir() or '/bin/' in info.name or info.name.endswith('.sh') else 0o644
        return info
    with tarfile.open(output.with_suffix('.tar.gz'), 'w:gz', compresslevel=3) as t:
        t.add(output, arcname=output.name, filter=permissions)
    archive_records = {output.with_suffix(ext).name: dict(bytes=output.with_suffix(ext).stat().st_size,
                                                        sha256=sha(output.with_suffix(ext)))
                       for ext in ('.zip', '.tar.gz')}
    (output.parent/(output.name+'_archives.json')).write_text(json.dumps(archive_records, indent=2))
    print(json.dumps({'output': str(output), 'files': len(files), 'archives': archive_records}, indent=2))


if __name__ == '__main__':
    main()
