"""Export prepared frames on the existing cloud GPU and freeze them locally."""
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path
import paramiko

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / 'examples/3dgs_compositor/board'))
from remote import run


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    stamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    local = ROOT / 'data' / stamp
    local.mkdir(parents=True)
    base = '/root/autodl-tmp/3dgs_reproduction'
    dest = base + '/scene_' + stamp
    c = paramiko.SSHClient()
    c.load_system_host_keys()
    c.set_missing_host_key_policy(paramiko.RejectPolicy())
    c.connect('connect.bjb1.seetacloud.com', port=32310, username='root',
              password=os.environ['GS_CLOUD_PASSWORD'], look_for_keys=False,
              allow_agent=False, timeout=8, banner_timeout=8, auth_timeout=8)
    try:
        run(c, 'mkdir -p ' + dest)
        sources = [ROOT / 'export_scene.py', REPO / 'examples/3dgs_reference/results/provenance/source_files_sha256.json']
        with c.open_sftp() as s:
            for p in sources:
                s.put(str(p), dest + '/' + p.name)
                if run(c, 'sha256sum ' + dest + '/' + p.name)[1].split()[0] != sha(p):
                    raise ValueError('Source upload drift')
            command = (f'cd {base} && ./venv/bin/python {dest}/export_scene.py '
                       f'--source source_pinned --model train_model '
                       f'--source-hashes {dest}/source_files_sha256.json --output {dest}/frames')
            print('Cloud export started: ' + dest, flush=True)
            _, content = run(c, command, timeout=240, log=local / 'export.log')
            print(content, flush=True)
            s.get(dest + '/frames/manifest.json', str(local / 'manifest.json'))
            manifest = json.loads((local / 'manifest.json').read_text())
            for case in manifest['cases']:
                target = local / case['name']
                target.mkdir()
                for name in ('scene.bin', 'official.bin', 'case.json'):
                    s.get(dest + '/frames/' + case['name'] + '/' + name, str(target / name))
                if sha(target / 'scene.bin') != case['input_sha256'] or sha(target / 'official.bin') != case['official_sha256']:
                    raise ValueError('Downloaded frame drift')
        receipt = dict(remote=dest, manifest_sha256=sha(local / 'manifest.json'),
            source_hashes={p.name: sha(p) for p in sources}, downloaded=True)
        (local / 'receipt.json').write_text(json.dumps(receipt, indent=2))
        (ROOT / 'data/current.json').write_text(json.dumps({'directory': str(local.resolve())}))
        print('PREPARED=' + str(local), flush=True)
    finally:
        c.close()


if __name__ == '__main__':
    main()
