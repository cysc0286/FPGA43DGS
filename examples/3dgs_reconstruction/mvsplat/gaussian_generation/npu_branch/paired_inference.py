"""Alternate full board inference paths on one retained model and NPU worker.

This is a timing diagnostic, not a native image acceptance receipt or a video
benchmark. CPU originals are retained only for this controlled comparison.
"""
import argparse
import hashlib
from pathlib import Path
import platform
import time

from common import new_directory, save, sha


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'weights', 'vendor', 'bundle', 'graphs', 'library', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--partitions', nargs='+', required=True)
    p.add_argument('--threads', type=int, default=3)
    p.add_argument('--rounds', type=int, default=2)
    a = p.parse_args(argv)
    if platform.machine().lower() not in ('aarch64', 'arm64') or a.rounds < 2:
        p.error('Requires the ARM board and at least two alternating rounds')
    out = new_directory(a.out)
    record = dict(complete=False, diagnostic_only=True, cases=[], threads=a.threads,
                  scope='Full network only, fixed prepared input; no video preparation or FPGA rendering')
    try:
        from gaussian_generation.runtime import ModelRuntime
        from gaussian_generation.npu_branch.catalog import locate, replace
        from gaussian_generation.npu_branch.runtime import PartitionRuntime
        from gaussian_generation.npu_branch.readiness import probe_oracles
        from initialize.session import worker_environment
        model = ModelRuntime(a.weights, a.vendor, a.threads, 16, out/'initialize')
        np, torch = model.np, model.torch
        originals = {name: locate(model.model, name) for name in a.partitions}
        with np.load(a.input/'context.npz', allow_pickle=False) as data:
            context = {k: torch.from_numpy(data[k].copy()).unsqueeze(0) for k in data.files}
        with PartitionRuntime('npu', a.bundle, a.partitions, out/'worker', library=a.library,
                              environment=worker_environment()) as worker:
            record['oracle_before'] = probe_oracles(worker, a.graphs, out/'oracle_before.json')
            worker.attach(model.model)
            replacements = {name: locate(model.model, name) for name in a.partitions}
            cases = [('cpu', []), ('backbone_only', ['backbone_cnn']), ('expanded', a.partitions)]
            if 'backbone_cnn' not in a.partitions:
                raise ValueError('Paired baseline requires backbone_cnn in selected partitions')
            for index in range(a.rounds):
                for label, enabled in (cases if index % 2 == 0 else list(reversed(cases))):
                    for name in a.partitions:
                        replace(model.model, name, replacements[name] if name in enabled else originals[name])
                    first = worker.sequence
                    start = time.perf_counter()
                    with torch.inference_mode():
                        g = model.model(context, global_step=300000, deterministic=True)
                    elapsed = time.perf_counter()-start
                    arrays = {k: getattr(g,k).detach().numpy() for k in
                              ('means','covariances','harmonics','opacities')}
                    finite = all(np.isfinite(v).all() for v in arrays.values())
                    calls = worker.report(first)['calls']
                    if [c['name'] for c in calls] != list(enabled) or not finite:
                        raise ValueError('Actual NPU call set or Gaussian finite check failed')
                    result = dict(round=index, label=label, inference_seconds=elapsed,
                        npu_executed=bool(calls), enabled=list(enabled), finite=finite, calls=calls,
                        gaussian_hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()})
                    record['cases'].append(result)
                    save(out/'paired.json', record)
                    print(label, index, elapsed, flush=True)
                    del g, arrays
            record['oracle_after'] = probe_oracles(worker, a.graphs, out/'oracle_after.json')
            record['provenance'] = worker.provenance
            record['repeat_exact'] = all(len({tuple(sorted(c['gaussian_hashes'].items()))
                for c in record['cases'] if c['label'] == label}) == 1 for label,_ in cases)
            if not record['repeat_exact']:
                raise ValueError('Repeated full network output changed')
            record['complete'] = True
            record['context_sha256'] = sha(a.input/'context.npz')
    except BaseException as exc:
        record['error'] = type(exc).__name__+': '+str(exc)
        raise
    finally:
        save(out/'paired.json', record)


if __name__ == '__main__':
    main()
