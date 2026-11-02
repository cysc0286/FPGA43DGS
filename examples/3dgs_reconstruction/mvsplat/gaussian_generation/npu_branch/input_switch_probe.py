"""Detect retained NPU output lag using distinct inputs; no speedup claims.

Repeated identical oracles can hide a result belonging to the previous input.
Keep full outputs only in RAM, compare A,A,B,B,A,A,B,B and save hashes/errors.
"""
import argparse
import hashlib
from pathlib import Path
import numpy as np
from common import new_directory, save, sha
from gaussian_generation.npu_branch.artifacts import relative_file, validate_bundle
from gaussian_generation.npu_branch.runtime import PartitionRuntime
from initialize.session import worker_environment


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'graphs', 'library', 'out'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--partitions', nargs='+', required=True)
    a = p.parse_args(argv)
    meta, _ = validate_bundle(a.graphs, a.partitions, 'onnx_reference')
    out = new_directory(a.out)
    record = dict(complete=False, diagnostic_only=True, calls=[],
                  scope='Retained-session input freshness; not arithmetic or image acceptance')
    try:
        with PartitionRuntime('npu', a.bundle, a.partitions, out/'worker',
                library=a.library, environment=worker_environment()) as worker:
            if worker.provenance['oracle_manifest_sha256'] != sha(a.graphs/'manifest.json'):
                raise ValueError('Compiled/oracle provenance mismatch')
            for name in a.partitions:
                path = relative_file(a.graphs, name+'/oracle.npz')
                if sha(path) != meta['partitions'][name]['oracle_sha256']:
                    raise ValueError('Changed oracle')
                with np.load(path, allow_pickle=False) as data:
                    values = {'A':data['input'].copy(), 'B':np.zeros_like(data['input'])}
                outputs = []
                for index, label in enumerate('AABBAABB'):
                    value = worker.execute(name, values[label])
                    outputs.append(value.copy())
                    record['calls'].append(dict(partition=name, index=index, input_label=label,
                        input_sha256=hashlib.sha256(values[label].tobytes()).hexdigest(),
                        output_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
                        finite=bool(np.isfinite(value).all()),
                        call=worker.report(worker.sequence-1)['calls'][0]))
                    save(out/'switch.json', record)
                expected = {'A':outputs[1], 'B':outputs[3]}
                for row, value in zip(record['calls'][-8:], outputs):
                    row['same_input_repeat_exact'] = bool(np.array_equal(value, expected[row['input_label']]))
                    row['max_abs_vs_settled'] = float(np.max(np.abs(value-expected[row['input_label']])))
                del outputs, expected, values
            record['worker'] = worker.report()
            record['complete'] = all(c['finite'] and c['same_input_repeat_exact'] for c in record['calls'])
            if not record['complete']:
                raise ValueError('First result after input change differs from settled result')
    except BaseException as exc:
        record['error'] = type(exc).__name__+': '+str(exc)
        raise
    finally:
        save(out/'switch.json', record)


if __name__ == '__main__':
    main()
