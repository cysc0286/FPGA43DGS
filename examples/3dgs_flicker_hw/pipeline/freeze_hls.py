"""Freeze one completed HLS candidate without replacing earlier evidence."""
import argparse
import datetime
import hashlib
import json
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', required=True)
    parser.add_argument('--console', required=True, type=Path)
    parser.add_argument('--source-root', type=Path, default=ROOT,
                        help='Directory containing the exact pipeline/ctu/hls sources used by this candidate')
    args = parser.parse_args()
    project = ROOT / 'build' / args.project
    report = project / 'solution1/sim/report/flicker_render_pipeline_cosim.rpt'
    console = args.console.read_text(errors='replace')
    if not re.search(r'\|\s*Verilog\|\s*Pass\|', report.read_text()):
        raise ValueError('C/RTL has not passed')
    if '// ERROR : Due to pragma' in console:
        raise ValueError('Unresolved dependence diagnostics')
    dest = ROOT / 'evidence' / (args.project + '_' + datetime.datetime.now().strftime('%Y%m%dT%H%M%S'))
    dest.mkdir()
    hashes = {}

    def copy(source, name):
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(name)] = sha(target)

    for folder in ['pipeline', 'ctu', 'hls']:
        for pattern in ['*.cpp', '*.hpp']:
            for source in (args.source_root / folder).glob(pattern):
                copy(source, Path('sources') / folder / source.name)
    copy(args.source_root / 'pipeline/build.tcl', Path('sources/pipeline/build.tcl'))
    # Optional exact exponent ROM is part of the hardware source, not a tool
    # installation detail. Preserve its tables and generation/tests for replay.
    rom = args.source_root / 'pipeline/exp_rom'
    if rom.exists():
        for source in rom.iterdir():
            if source.suffix in {'.cpp', '.hpp', '.tcl', '.md'}:
                copy(source, Path('sources/pipeline/exp_rom') / source.name)
    if (project/'hls.app').exists():
        copy(project/'hls.app', Path('hls.app'))
    for source in (project / 'solution1/syn/report').iterdir():
        if source.is_file():
            copy(source, Path('reports') / source.name)
    copy(report, Path('reports') / report.name)
    copy(args.console, Path('console.txt'))
    hls_rtl = project / 'solution1/impl/verilog'
    for source in hls_rtl.iterdir():
        if source.is_file():
            copy(source, Path('rtl') / source.name)
    (dest / 'result.json').write_text(json.dumps({
        'passed': True, 'project': str(project), 'source_root': str(args.source_root.resolve()), 'created_local': datetime.datetime.now().isoformat(),
        'sha256': hashes, 'scope': 'HLS C/RTL evidence only; not physical or board acceptance'
    }, indent=2))
    print('FROZEN=' + str(dest))


if __name__ == '__main__':
    main()
