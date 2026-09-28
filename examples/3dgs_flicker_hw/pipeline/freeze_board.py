"""Keep the routed FLK1 design and exact sources before another build."""
import argparse
import datetime
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
PLATFORM = REPO / 'platform/flicker_cat_fpga'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    global PLATFORM
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    report = args.report.resolve()
    PLATFORM = Path(next(line.split('=',1)[1] for line in (report/'status.txt').read_text().splitlines() if line.startswith('PROJECT=')))
    if 'BITSTREAM_GENERATION=PASS' not in (report / 'status.txt').read_text():
        raise ValueError('Physical build incomplete')
    manifest = json.loads((PLATFORM / 'flicker_manifest.json').read_text())
    for name, digest in manifest['sources'].items():
        if sha(PLATFORM / name) != digest:
            raise ValueError('Source drift: ' + name)
    dest = report / 'frozen_artifacts'
    dest.mkdir(exist_ok=False)
    hashes = {}

    def copy(source, relative):
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = sha(target)

    for name in manifest['sources']:
        copy(PLATFORM / name, Path('sources') / name)
    copy(PLATFORM / 'flicker_manifest.json', Path('flicker_manifest.json'))
    for name in ['ai7030_edif_top_routed.dcp', 'ai7030_edif_top.bit',
                 'ai7030_edif_top_disable_icap.bit', 'runme.log']:
        copy(PLATFORM / 'fpai_demo_vivado.runs/impl_1' / name, Path('implementation') / name)
    copy(PLATFORM / 'fpai_demo_vivado.runs/synth_1/runme.log', Path('synthesis/runme.log'))
    for source in report.iterdir():
        if source.is_file():
            copy(source, Path('reports') / source.name)
    if (report / 'boot').exists():
        for name in ['BOOT.bin', 'comparison.json']:
            copy(report / 'boot' / name, Path('boot') / name)
    result = {'created_local': datetime.datetime.now().isoformat(),
              'platform': str(PLATFORM), 'report': str(report), 'sha256': hashes}
    (dest / 'manifest.json').write_text(json.dumps(result, indent=2))
    print('FROZEN=' + str(dest))


if __name__ == '__main__':
    main()
