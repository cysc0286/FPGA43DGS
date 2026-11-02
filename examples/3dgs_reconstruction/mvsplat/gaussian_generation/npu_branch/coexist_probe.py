"""Bounded real-device coexistence probe; no deployment acceptance or speed claim."""
import argparse
from contextlib import ExitStack
from pathlib import Path
import json
import platform

from common import load_gaussians, new_directory, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--renderer', type=Path, required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--library', type=Path, help='Candidate bridge; defaults to the historical library')
    p.add_argument('--order', choices=('npu-first', 'renderer-first'), required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if platform.machine().lower() not in ('aarch64', 'arm64'):
        p.error('Requires the actual ARM NPU/FPGA board')
    out = new_directory(a.out)
    record = dict(complete=False, diagnostic_only=True, order=a.order, stages={})
    from gaussian_generation.npu_branch.runtime import PartitionRuntime
    from gaussian_generation.npu_branch.readiness import probe_oracles
    from initialize.session import worker_environment, render_environment
    from rendering.pipeline_adapter import PipelineRenderer
    from gaussian_generation.adapter import camera_bytes, gaussian_rows
    try:
        with ExitStack() as stack:
            def npu():
                return stack.enter_context(PartitionRuntime('npu', a.assets/'candidate/compiled',
                    ['backbone_cnn'], out/'worker', library=a.library or a.assets/'arm_bridge/libmgs_npu.so',
                    environment=worker_environment()))
            def renderer():
                return stack.enter_context(PipelineRenderer(a.renderer, a.binary,
                    render_environment(), out/'renderer.log'))
            def probe(name):
                result = probe_oracles(worker, a.assets/'candidate/graphs', out/(name+'.json'))
                record['stages'][name] = result
                print(name, result['complete'], result['strict_passed'], flush=True)
            if a.order == 'npu-first':
                worker = npu()
                probe('before_renderer')
                native = renderer()
            else:
                native = renderer()
                worker = npu()
            probe('after_both_initialized')
            values = load_gaussians(a.root/'qualification_v2/cpu/gaussians.npz')
            rows, _ = gaussian_rows(values)
            native.load_rows(rows)
            metadata = json.loads((a.assets/'candidate/input/input.json').read_text())
            view = next(v for v in metadata['views'] if v['role'] == 'target')
            frame = native.runtime.render_camera(camera_bytes(view), include_raw=True)
            record['fpga'] = frame.archive(out/'frame')
            probe('after_fpga_render')
            hashes = [v['checks'][-1]['output_sha256'] for v in record['stages'].values()]
            record['oracle_unchanged'] = len(set(hashes)) == 1
            record['worker'] = worker.report()
            record['complete'] = record['oracle_unchanged']
    except BaseException as exc:
        record['error'] = type(exc).__name__+': '+str(exc)
        raise
    finally:
        save(out/'probe.json', record)


if __name__ == '__main__':
    main()
