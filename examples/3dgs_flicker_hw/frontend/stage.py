"""Compile and run the camera-dependent 3DGS attribute stage on the ARM board."""
import argparse
import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / 'examples/3dgs_compositor/board'))
from remote import connect, run

MODEL = REPO / 'npu_3dgs/data/official/train/point_cloud/iteration_7000/point_cloud.ply'
EXPECTED = '8212d03d01edc66423fdd8a6e9884f5f73838cccdda1c82dbf2aea514eb0dce7'


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cameras', type=Path, nargs='+', required=True)
    a = p.parse_args()
    if sha(MODEL) != EXPECTED:
        raise ValueError('Official model drift')
    source = ROOT / 'frontend/attributes.cpp'
    stamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    evidence = ROOT / 'evidence' / ('frontend_attr_' + stamp)
    evidence.mkdir()
    shutil.copy2(source,evidence/'attributes.cpp')
    remote_root = '/root/fpga43dgs_flicker/frontend_' + stamp
    model_remote = '/root/fpga43dgs_flicker/model_8212d03d/point_cloud.ply'
    record = {'model_sha256': EXPECTED, 'source_sha256': sha(source),
              'remote': remote_root, 'camera_runs': [], 'complete': False}
    client = connect()
    try:
        run(client, 'mkdir -p ' + remote_root + ' /root/fpga43dgs_flicker/model_8212d03d')
        run(client, 'cat /proc/sys/kernel/random/boot_id; uname -a', log=evidence/'environment.txt')
        remote_hash = run(client, 'sha256sum ' + model_remote, check=False)[1].split()
        if not remote_hash or remote_hash[0] != EXPECTED:
            with client.open_sftp() as sftp:
                sftp.put(str(MODEL), model_remote + '.uploading')
            if run(client, 'sha256sum ' + model_remote + '.uploading', timeout=40)[1].split()[0] != EXPECTED:
                raise ValueError('Uploaded PLY hash mismatch')
            run(client, 'mv ' + model_remote + '.uploading ' + model_remote)
        with client.open_sftp() as sftp:
            sftp.put(str(source), remote_root + '/attributes.cpp')
            for camera in a.cameras:
                sftp.put(str(camera), remote_root + '/' + camera.name)
        if run(client, 'sha256sum ' + remote_root + '/attributes.cpp')[1].split()[0] != record['source_sha256']:
            raise ValueError('Source drift after upload')
        command = ('g++ -O2 -std=gnu++17 -ffp-contract=off -Wall -Wextra '
                   + remote_root + '/attributes.cpp -o ' + remote_root + '/attributes')
        run(client, command, timeout=180, log=evidence/'compile.txt')
        record['binary_sha256'] = run(client, 'sha256sum ' + remote_root + '/attributes')[1].split()[0]
        for camera in a.cameras:
            name = camera.stem
            remote_camera = remote_root + '/' + camera.name
            if run(client, 'sha256sum ' + remote_camera)[1].split()[0] != sha(camera):
                raise ValueError('Camera drift after upload')
            output = remote_root + '/' + name + '.attr'
            command = (f'timeout 120 {remote_root}/attributes {model_remote} '
                       f'{remote_camera} {output}')
            run(client, command, timeout=135, log=evidence/(name + '_run.txt'))
            with client.open_sftp() as sftp:
                sftp.get(output, str(evidence/(name + '.attr')))
            item = {'camera': str(camera.resolve()), 'camera_sha256': sha(camera),
                    'output': name + '.attr', 'output_sha256': sha(evidence/(name + '.attr')),
                    'output_bytes': (evidence/(name + '.attr')).stat().st_size,
                    'timing': (evidence/(name + '_run.txt')).read_text().strip()}
            record['camera_runs'].append(item)
            (evidence/'result.json').write_text(json.dumps(record, indent=2))
            print(json.dumps(item), flush=True)
        record['complete'] = True
    finally:
        (evidence/'result.json').write_text(json.dumps(record, indent=2))
        client.close()
        print('EVIDENCE=' + str(evidence))


if __name__ == '__main__':
    main()
