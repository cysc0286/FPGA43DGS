"""Real descriptor-pair oracle, block-score and final-match validation on CPU."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from matcher import blocks, match, scalar_reference, SCORE_ATOL
from backends import NumpyBackend, OnnxCPUBackend


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", required=True, type=Path)
    p.add_argument("--graph", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    graph = json.loads((a.graph/"manifest.json").read_text())
    model = a.graph/"sift_pair_dot.onnx"
    if hashlib.sha256(model.read_bytes()).hexdigest() != graph["graph_sha256"]:
        raise ValueError("Graph hash mismatch")
    tile = graph["tile"]
    onnx = OnnxCPUBackend(model)
    records = []
    for case in json.loads((a.cases/"manifest.json").read_text())["cases"]:
        path = a.cases/(case["name"]+".npz")
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["npz_sha256"]:
            raise ValueError("Descriptor case hash mismatch")
        with np.load(path, allow_pickle=False) as d:
            left, right = d["a"], d["b"]
        start = time.perf_counter()
        reference = scalar_reference(left, right)
        oracle_s = time.perf_counter()-start
        cpu_matches, cpu_stats = match(left, right, NumpyBackend(), tile)
        onnx_matches, onnx_stats = match(left, right, onnx, tile)
        # Independent integer dot oracle, including all padding entries.
        error = 0.
        for i, j, ni, nj, aa, bb in blocks(left, right, tile):
            expected = np.zeros((tile, tile), dtype=np.float32)
            expected[:ni, :nj] = (left[i:i+ni].astype(np.int64) @ right[j:j+nj].astype(np.int64).T).astype(np.float32)/np.float32(512**2)
            actual = onnx.dot(aa, bb)[0, 0]
            error = max(error, float(np.abs(actual-expected).max()))
        passed = bool(error <= SCORE_ATOL and np.array_equal(reference, cpu_matches) and np.array_equal(reference, onnx_matches))
        np.save(a.output/(case["name"]+"_matches.npy"), onnx_matches, allow_pickle=False)
        records.append({"case": case, "oracle_s": oracle_s, "numpy": cpu_stats, "onnx_cpu": onnx_stats,
                        "score_max_abs_error": error, "match_indices_exact": bool(np.array_equal(reference, onnx_matches)), "passed": passed})
    result = {"backend": "ONNX Runtime CPU only", "npu_executed": False, "graph": graph,
              "onnx_session_init_s": onnx.init_s, "cases": records, "passed": all(x["passed"] for x in records),
              "boundary": "matching module only; three image pairs; no pose/training/rendering/board speed claim"}
    (a.output/"result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
