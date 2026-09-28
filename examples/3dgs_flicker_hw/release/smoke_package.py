"""Extract the shipped tarball into a fresh board directory and compare real frames."""
import datetime
import hashlib
import json
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO/'examples/3dgs_compositor/board'))
from remote import connect, run


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    name = '3dgs_renderer_v1_20260928'
    archive = REPO/'releases'/(name+'.tar.gz')
    digest = sha(archive)
    with zipfile.ZipFile(archive.with_name(name+'.zip')) as z:
        if z.testzip() is not None:
            raise ValueError('Zip CRC check failed')
    with tarfile.open(archive, 'r:gz') as t:
        for m in t.getmembers():
            if m.name.startswith('/') or '..' in Path(m.name).parts or m.issym() or m.islnk():
                raise ValueError('Unexpected archive member')
    stamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    ev = ROOT/'evidence'/('package_smoke_'+stamp)
    ev.mkdir()
    dest = '/root/fpga43dgs_releases/'+stamp
    package = dest+'/'+name
    result = dict(package=name, tar_sha256=digest, remote=dest, cases=[], complete=False)
    c = connect()
    try:
        run(c, 'mkdir -p /root/fpga43dgs_releases && mkdir '+dest)
        run(c, 'cat /proc/sys/kernel/random/boot_id; uname -a', log=ev/'environment.txt')
        with c.open_sftp() as s:
            s.put(str(archive), dest+'/renderer.tar.gz')
        if run(c, 'sha256sum '+dest+'/renderer.tar.gz', timeout=45)[1].split()[0] != digest:
            raise ValueError('Archive upload mismatch')
        run(c, 'tar -xzf '+dest+'/renderer.tar.gz -C '+dest, timeout=90)
        run(c, 'python3 '+package+'/render.py --verify-only', timeout=90, log=ev/'verify.txt')
        for view, backend in ((0, 'fpga'), (10, 'fpga'), (0, 'cpu_dense')):
            label = 'v%d_%s' % (view, backend)
            output = dest+'/'+label
            # Run from /tmp; no source-repository working directory is present on the board.
            cmd = ('cd /tmp && python3 '+package+'/render.py --camera '+package+'/data/v%d.bin'%view+
                   ' --backend '+backend+' --out '+output)
            run(c, cmd, timeout=180, log=ev/(label+'.log'))
            with c.open_sftp() as s:
                for remote_name, local_name in (('result.json', label+'.json'), ('frame.bin', label+'.bin'),
                                              ('frame.ppm', label+'.ppm'), ('frame_timing.csv', label+'_timing.csv')):
                    s.get(output+'/'+remote_name, str(ev/local_name))
            record = json.loads((ev/(label+'.json')).read_text())
            baseline = ROOT/'evidence/full_benchmark_20260927T234919'/(label+'.bin')
            actual = sha(ev/(label+'.bin'))
            passed = record['complete'] and actual == record['frame_sha256'] == sha(baseline)
            result['cases'].append(dict(view=view, backend=backend, frame_sha256=actual,
                                        baseline_sha256=sha(baseline), passed=passed,
                                        single_invocation_ms=record['total_ms']))
            if not passed:
                raise ValueError('Packaged runtime output changed')
            print(label+' EXACT_PASS', flush=True)
        run(c, package+'/bin/render status; free -m; df -h /dev/shm', log=ev/'health.txt')
        result['complete'] = True
    finally:
        (ev/'result.json').write_text(json.dumps(result, indent=2))
        c.close()
        print('EVIDENCE='+str(ev))


if __name__ == '__main__':
    main()
