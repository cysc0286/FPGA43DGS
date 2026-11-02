"""Fixed numerical gate on the actual retained worker before public READY."""
from pathlib import Path
import hashlib
import numpy as np
from common import save, sha
from gaussian_generation.npu_branch.artifacts import relative_file, validate_bundle
from gaussian_generation.npu_branch.verify import errors


def probe_oracles(worker, graphs, output):
    graphs = Path(graphs)
    record = dict(complete=False, tolerances=dict(rtol=2e-3, atol=2e-4), checks=[],
                  diagnostic_only=True, input_freshness_checks=[],
                  scope="finite/repeatable oracle probe; strict failures retained, not deployment acceptance")
    try:
        names = list(worker.parts)
        meta, _ = validate_bundle(graphs, names, "onnx_reference")
        if sha(graphs/"manifest.json") != worker.provenance["oracle_manifest_sha256"]:
            raise ValueError("Compiled/oracle provenance mismatch")
        # Run all partitions, then revisit them. Immediate same-partition repeats
        # alone cannot expose state overwritten by a different network session.
        previous = {}
        for round_index in range(2):
            for name in names:
                path = relative_file(graphs, name+"/oracle.npz")
                if sha(path) != meta["partitions"][name]["oracle_sha256"]:
                    raise ValueError("Oracle checksum changed")
                with np.load(path, allow_pickle=False) as data:
                    actual = worker.execute(name, data["input"])
                    comparison = errors(actual, data["expected"], 2e-3, 2e-4)
                    if actual.shape != data['expected'].shape or not np.isfinite(actual).all():
                        raise ValueError('NPU oracle shape/finite contract failed: ' + name)
                digest = hashlib.sha256(actual.tobytes()).hexdigest()
                exact = previous.get(name, digest) == digest
                record["checks"].append(dict(partition=name, round=round_index, **comparison,
                    oracle_sha256=sha(path), output_sha256=digest, repeat_exact=exact))
                if not exact:
                    raise ValueError('NPU repeated oracle output changed: ' + name)
                previous[name] = digest
        # Identical inputs alone cannot detect a one-request lag. Change the
        # input, repeat it, then return to the saved oracle before publishing
        # READY. Keep the original oracle checks intact for receipt identity.
        for name in names:
            path = relative_file(graphs, name+"/oracle.npz")
            with np.load(path, allow_pickle=False) as data:
                original = data['input'].copy()
            hashes = []
            for label, value in (('B', np.zeros_like(original)),
                                 ('B', np.zeros_like(original)), ('A', original), ('A', original)):
                actual = worker.execute(name, value)
                if not np.isfinite(actual).all():
                    raise ValueError('NPU input-switch nonfinite output: '+name)
                digest = hashlib.sha256(actual.tobytes()).hexdigest()
                hashes.append(digest)
                record['input_freshness_checks'].append(dict(partition=name,
                    input_label=label, output_sha256=digest))
            if hashes[0] != hashes[1] or hashes[2] != previous[name] or hashes[3] != previous[name]:
                raise ValueError('NPU stale output after input switch: '+name)
            # These checkpoint-backed partitions must react to the real-vs-zero
            # probe. A permanently cached oracle must not qualify as fresh.
            if hashes[0] == previous[name]:
                raise ValueError('NPU unchanged output after input switch: '+name)
        record["complete"] = True
        record['strict_passed'] = all(c['passed'] for c in record['checks'])
    except BaseException as exc:
        record["error"] = type(exc).__name__+": "+str(exc)
        raise
    finally:
        save(output, record)
    return record


def validate_oracles(worker, graphs, output, *, acceptance=None):
    record = probe_oracles(worker, graphs, output)
    record.update(complete=False, diagnostic_only=False)
    try:
        if acceptance is None:
            if not record['strict_passed']:
                raise ValueError('NPU numerical readiness failed: ' +
                                 next(c['partition'] for c in record['checks'] if not c['passed']))
            record['validation'] = dict(policy='strict_tensor', rtol=2e-3, atol=2e-4)
        else:
            from gaussian_generation.npu_branch.acceptance import validate_worker
            record['validation'] = validate_worker(acceptance, worker,
                [c for c in record['checks'] if c['round'] == 1])
        record['complete'] = True
    except BaseException as exc:
        record['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        save(output, record)
    return record
