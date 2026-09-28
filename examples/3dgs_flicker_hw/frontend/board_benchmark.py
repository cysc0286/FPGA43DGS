"""Run on ARM: bounded repeatable whole-pipeline benchmark with tmpfs intermediates."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('config',type=Path);a=p.parse_args()
    conf=json.loads(a.config.read_text());dest=a.config.parent
    for path,digest in conf['hashes'].items():
        if sha(Path(path))!=digest:raise ValueError('Input/binary drift: '+path)
    rows=[];expected={}
    header=['view','backend','sample','attributes_ms','group_sort_ms','render_wall_ms','total_ms','frame_sha256','scene_sha256']
    with (dest/'timing.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=header);writer.writeheader()
        with tempfile.TemporaryDirectory(prefix='flk-full-',dir='/dev/shm') as temporary:
            work=Path(temporary)
            for sample in range(-1,conf['repeats']):
                # Alternate backend order to reduce a fixed ordering bias.
                backends=conf['backends'] if sample%2 else list(reversed(conf['backends']))
                for view,camera in conf['cameras'].items():
                    for backend in backends:
                        log_name='v{}_{}_{}'.format(view,backend,sample)
                        with (dest/(log_name+'.txt')).open('w') as log:
                            begin=time.monotonic_ns()
                            subprocess.run([conf['attributes'],conf['model'],camera,str(work/'attr.bin')],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=60)
                            projected=time.monotonic_ns()
                            subprocess.run([conf['group'],str(work/'attr.bin'),str(work/'scene.bin')],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=60)
                            sorted_time=time.monotonic_ns()
                            if backend=='fpga':
                                args=['mode','2','scene.bin','frame','pipeline','8','1','0']
                            else:
                                args=['cpu','scene.bin','frame','dense' if backend=='cpu_dense' else 'base','4','1','0','0']
                            subprocess.run([conf['renderer']]+args,cwd=str(work),check=True,stdout=log,stderr=subprocess.STDOUT,timeout=90)
                            rendered=time.monotonic_ns()
                        frame_hash=sha(work/'frame.bin');scene_hash=sha(work/'scene.bin')
                        if scene_hash!=conf['scene_hashes'][view]:raise ValueError('Scene changed across full executions')
                        key=view+'_'+backend
                        if key in expected and expected[key]!=frame_hash:raise ValueError('Repeated frame output drift')
                        expected[key]=frame_hash
                        row=dict(view=view,backend=backend,sample=sample,
                                 attributes_ms=(projected-begin)/1e6,group_sort_ms=(sorted_time-projected)/1e6,
                                 render_wall_ms=(rendered-sorted_time)/1e6,total_ms=(rendered-begin)/1e6,
                                 frame_sha256=frame_hash,scene_sha256=scene_hash)
                        if sample>=0:
                            writer.writerow(row);f.flush();rows.append(row)
                        if sample==0:shutil.copyfile(str(work/'frame.bin'),str(dest/('v'+view+'_'+backend+'.bin')))
                        shutil.copyfile(str(work/'frame_timing.csv'),str(dest/(log_name+'_render_timing.csv')))
                        print(json.dumps(row),flush=True)
    (dest/'result.json').write_text(json.dumps({'complete':True,'rows':rows,'scope':'monotonic wall time from PLY read through one frame; tmpfs intermediates and output, process startup and SDK initialization included','power':'not measured'},indent=2))


if __name__=='__main__':main()
