"""Hash-check completed scene evidence and keep large scenes in a separate table."""
import csv
import json
from pathlib import Path
from check_output import sha
ROOT=Path(__file__).resolve().parent
DEST=ROOT.parents[1]/'npu_3dgs/benchmarks'


def main():
    rows=[];runs=[]
    for path in sorted((ROOT/'evidence').glob('*/result.json')):
        r=json.loads(path.read_text())
        if r.get('schema')!='gs-large-prepared-frame-v1':continue
        if not r['completed']:raise ValueError('Unfinished scene run: '+str(path))
        data=Path(r['data'])
        if sha(data/'manifest.json')!=r['manifest_sha256']:raise ValueError('Manifest drift')
        for name,digest in r['source_hashes'].items():
            if sha(path.parent/'sources'/name)!=digest:raise ValueError('Measured source drift')
        runs.append(dict(run=path.parent.name,evidence=path.relative_to(ROOT.parents[1]).as_posix(),sha256=sha(path),
            completed=r['completed'],official_strict_pass=r['correctness_passed'],binary_sha256=r['binary_sha256'],
            scope=r['scope'],source_hashes=r['source_hashes']))
        for item in r['runs']:
            case=item['case'];t=item['timing'];m=t['means'];o=item['official'];meta=item['metadata']
            for suffix,digest in item['files'].items():
                if sha(path.parent/'received'/(item['stem']+suffix))!=digest:raise ValueError('Readback drift')
            for filename,key in [('scene.bin','input_sha256'),('official.bin','official_sha256')]:
                if sha(data/case['name']/filename)!=case[key]:raise ValueError('Input/Golden drift')
            row=dict(run=path.parent.name,phase=r['phase'],case=case['name'],variant=item['variant'],
                selected_gaussians=meta['scene_gaussians'],projected_unique_gaussians=case['projected_unique_gaussians'],
                active_gaussians=meta['active'],tile_entries=case['tile_entries'],tiles=case['tiles'],
                max_tile_candidates=case['max_tile_candidates'],pixels=meta['pixels'],
                samples=t['samples'],warmup=meta['warmup'],threads=meta['threads'],rss_kib=meta['rss_kib'],
                **{k:t[k] for k in ['mean_ms','median_ms','min_ms','max_ms','p95_ms','p99_ms','raster_fps','cpu_percent']},
                **{k:v for k,v in m.items() if k not in ('total_us','process_cpu_us')},
                logical_register_bytes=4*(m['register_reads']+m['register_writes']),
                command_payload_bytes=24*m['commands'],input_bytes=(data/case['name']/'scene.bin').stat().st_size,
                fpga_execution_ms=m['execution_cycles']/100000,
                host_outside_sdk_ms=(m['render_us']-m['load_us']-m['wait_us']-m['readback_us'])/1000,
                fixed1_integer_exact=item.get('fixed1_integer_exact',''),npu=False,dma=False,
                input_sha256=case['input_sha256'],output_sha256=item['files']['.bin'])
            row.update({('official_'+k):v for k,v in o.items()})
            rows.append(row)
    with (DEST/'scene_versions.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=list(dict.fromkeys(k for row in rows for k in row))
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    registry=dict(schema='gs-large-scene-registry-v1',runs=runs,
        note='Official gate failures retained. Raster FPS excludes GPU preprocessing, loading, and file I/O. Three repeats do not establish tail latency.',
        table_sha256=sha(DEST/'scene_versions.csv'))
    (DEST/'scene_registry.json').write_text(json.dumps(registry,indent=2))
    quality_dirs=sorted((ROOT/'evidence').glob('quality_*/quality.json'))
    if quality_dirs:
        qpath=quality_dirs[-1];quality=json.loads(qpath.read_text())
        calibrated={x['stem']:x['files']['.bin'] for path in sorted((ROOT/'evidence').glob('calibrate_*/result.json'))
                    for x in json.loads(path.read_text())['runs']}
        for item in quality['runs']:
            stem=item['case']+'_'+item['variant']
            if calibrated.get(stem)!=item['actual_sha256']:raise ValueError('Quality/readback mismatch: '+stem)
        quality['raw_evidence']=qpath.relative_to(ROOT.parents[1]).as_posix()
        quality['raw_evidence_sha256']=sha(qpath)
        (DEST/'scene_quality.json').write_text(json.dumps(quality,indent=2))
    print(json.dumps(dict(runs=len(runs),rows=len(rows),table=str(DEST/'scene_versions.csv'))))


if __name__=='__main__':main()
