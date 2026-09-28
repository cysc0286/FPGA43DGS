"""Portable little-endian block files for a single persistent ICraft session."""
import argparse
import hashlib
import json
import struct
from pathlib import Path
import numpy as np
from matcher import blocks, descriptors, match, scalar_reference, SCORE_ATOL

HEADER = struct.Struct("<8s4I")
INDEX = struct.Struct("<4I")


def pack(case, out, tile):
    out.mkdir(parents=True, exist_ok=False)
    with np.load(case, allow_pickle=False) as d:
        a, b = descriptors(d["a"]), descriptors(d["b"])
    count = ((len(a)+tile-1)//tile)*((len(b)+tile-1)//tile)
    with (out/"jobs.bin").open("wb") as f:
        f.write(HEADER.pack(b"HGSJOB01", tile, count, len(a), len(b)))
        for i, j, ni, nj, aa, bb in blocks(a, b, tile):
            f.write(INDEX.pack(i, j, ni, nj))
            f.write(aa.astype("<f4").tobytes())
            f.write(bb.astype("<f4").tobytes())
    # A self-contained copy binds verification to the exact source descriptors.
    np.savez(out/"descriptors.npz", a=a, b=b)
    meta = {"abi": 1, "tile": tile, "blocks": count, "rows": [len(a), len(b)],
            "jobs_sha256": hashlib.sha256((out/"jobs.bin").read_bytes()).hexdigest(),
            "descriptors_sha256": hashlib.sha256((out/"descriptors.npz").read_bytes()).hexdigest(),
            "npu_executed": False, "logical_score_output_bytes": count*tile*tile*4}
    (out/"manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


class RecordedBackend:
    name = "recorded_blocks_unattributed"
    npu_executed = False

    def __init__(self, bundle, output, validate_scores=True):
        self.validate_scores = validate_scores
        self.meta = json.loads((bundle/"manifest.json").read_text())
        for filename, key in [("jobs.bin", "jobs_sha256"), ("descriptors.npz", "descriptors_sha256")]:
            if hashlib.sha256((bundle/filename).read_bytes()).hexdigest() != self.meta[key]:
                raise ValueError("Input bundle hash mismatch: "+filename)
        with np.load(bundle/"descriptors.npz", allow_pickle=False) as d:
            self.a, self.b = descriptors(d["a"]), descriptors(d["b"])
        self.file = output.open("rb")
        header = self.file.read(HEADER.size)
        want = (b"HGSOUT01", self.meta["tile"], self.meta["blocks"], *self.meta["rows"])
        if len(header) != HEADER.size or HEADER.unpack(header) != want:
            self.file.close()
            raise ValueError("Output header mismatch")
        self.iterator = iter(blocks(self.a, self.b, self.meta["tile"]))
        self.seen, self.error = 0, 0.

    def dot(self, aa, bb):
        i, j, ni, nj, left, right = next(self.iterator)
        if not np.array_equal(aa, left) or not np.array_equal(bb, right):
            raise ValueError("Block input/order mismatch")
        raw = self.file.read(INDEX.size)
        if len(raw) != INDEX.size or INDEX.unpack(raw) != (i, j, ni, nj):
            raise ValueError("Missing/reordered/duplicated output block")
        tile = self.meta["tile"]
        raw = self.file.read(tile*tile*4)
        if len(raw) != tile*tile*4:
            raise ValueError("Truncated score block")
        output = np.frombuffer(raw, "<f4").reshape(1, 1, tile, tile)
        if not np.isfinite(output).all():
            raise ValueError("Non-finite output")
        if self.validate_scores:
            expected = np.zeros((tile, tile), np.float32)
            expected[:ni, :nj] = (self.a[i:i+ni].astype(np.int64) @ self.b[j:j+nj].astype(np.int64).T).astype(np.float32)/np.float32(512**2)
            self.error = max(self.error, float(np.abs(output[0, 0]-expected).max()))
        self.seen += 1
        return output

    def finish(self):
        if self.seen != self.meta["blocks"] or self.file.read(1):
            raise ValueError("Wrong result count or trailing data")


def verify(bundle, output, report):
    if report.exists():
        raise ValueError("Report exists")
    backend = RecordedBackend(bundle, output)
    try:
        candidate, timing = match(backend.a, backend.b, backend, backend.meta["tile"])
        backend.finish()
        exact = np.array_equal(candidate, scalar_reference(backend.a, backend.b))
        passed = bool(exact and backend.error <= SCORE_ATOL)
        record = {"passed": passed, "score_max_abs_error": backend.error, "score_atol": SCORE_ATOL,
                  "match_indices_exact": bool(exact), "matches": len(candidate),
                  "npu_executed": False, "execution_attribution": "requires independent runner/profile evidence",
                  "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
                  "jobs_sha256": backend.meta["jobs_sha256"],
                  "verification_time_not_npu_time": timing}
        report.write_text(json.dumps(record, indent=2))
        np.save(report.with_suffix(".matches.npy"), candidate, allow_pickle=False)
    finally:
        backend.file.close()
    if not passed:
        raise ValueError("Recorded output failed score or match gate")
    return record


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    packer = sub.add_parser("pack")
    packer.add_argument("--case", type=Path, required=True)
    packer.add_argument("--output", type=Path, required=True)
    packer.add_argument("--tile", type=int, choices=[32, 64, 128, 256], default=128)
    checker = sub.add_parser("verify")
    checker.add_argument("--bundle", type=Path, required=True)
    checker.add_argument("--output", type=Path, required=True)
    checker.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    if a.action == "pack":
        pack(a.case, a.output, a.tile)
    else:
        print(json.dumps(verify(a.bundle, a.output, a.report), indent=2))


if __name__ == "__main__":
    main()
