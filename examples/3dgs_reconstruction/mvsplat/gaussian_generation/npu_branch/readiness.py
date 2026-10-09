"""Fixed numerical gate on the actual retained worker before public READY."""
from pathlib import Path
import numpy as np
from common import save, sha
from gaussian_generation.npu_branch.artifacts import relative_file, validate_bundle
from gaussian_generation.npu_branch.verify import errors


def validate_oracles(worker, graphs, output):
    graphs = Path(graphs)
    record = dict(complete=False, tolerances=dict(rtol=2e-3, atol=2e-4), checks=[],
                  scope="real retained partition outputs; not full model/image acceptance")
    try:
        names = list(worker.parts)
        meta, _ = validate_bundle(graphs, names, "onnx_reference")
        if sha(graphs/"manifest.json") != worker.provenance["oracle_manifest_sha256"]:
            raise ValueError("Compiled/oracle provenance mismatch")
        # Run all partitions, then revisit them. Immediate same-partition repeats
        # alone cannot expose state overwritten by a different network session.
        for round_index in range(2):
            for name in names:
                path = relative_file(graphs, name+"/oracle.npz")
                if sha(path) != meta["partitions"][name]["oracle_sha256"]:
                    raise ValueError("Oracle checksum changed")
                with np.load(path, allow_pickle=False) as data:
                    actual = worker.execute(name, data["input"])
                    comparison = errors(actual, data["expected"], 2e-3, 2e-4)
                record["checks"].append(dict(partition=name, round=round_index, **comparison))
                if not comparison["passed"]:
                    raise ValueError("NPU numerical readiness failed: "+name)
        record["complete"] = True
    except BaseException as exc:
        record["error"] = type(exc).__name__+": "+str(exc)
        raise
    finally:
        save(output, record)
    return record
