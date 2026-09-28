"""Independent database branch: CPU reference or inject validated candidate matches."""
import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--database", type=Path, required=True)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--matches", type=Path, required=True, help="Directory written by validate.py")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["reference", "inject"], default="reference")
    p.add_argument("--verify-geometry", action="store_true")
    a = p.parse_args()
    import pycolmap as pc
    if pc.__version__ != "4.2.0":
        raise ValueError("Reference is pinned to pycolmap 4.2.0")
    manifest = json.loads((a.cases/"manifest.json").read_text())
    before = hashlib.sha256(a.database.read_bytes()).hexdigest()
    if before != manifest["database_sha256"]:
        raise ValueError("Source DB differs from descriptor evidence")
    a.output.mkdir(parents=True, exist_ok=False)
    database = a.output/"database.db"
    with sqlite3.connect(a.database.resolve().as_uri()+"?mode=ro", uri=True) as src, sqlite3.connect(database) as dst:
        src.backup(dst)
    cases = manifest["cases"]
    for c in cases:
        if hashlib.sha256((a.cases/(c["name"]+".npz")).read_bytes()).hexdigest() != c["npz_sha256"]:
            raise ValueError("Descriptor case hash differs from source manifest")
    pairs_file = a.output/"pairs.txt"
    pairs_file.write_text("".join(" ".join(c["images"])+"\n" for c in cases))
    with pc.Database.open(database) as db:
        db.clear_matches()
        db.clear_two_view_geometries()
        if a.mode == "inject":
            # Recompute the independent oracle before admitting data to a SfM branch.
            from matcher import scalar_reference
            for c in cases:
                with np.load(a.cases/(c["name"]+".npz"), allow_pickle=False) as data:
                    expected = scalar_reference(data["a"], data["b"])
                candidate = np.load(a.matches/(c["name"]+"_matches.npy"), allow_pickle=False)
                if candidate.dtype != np.uint32 or not np.array_equal(candidate, expected):
                    raise ValueError("Candidate indices fail exact gate")
                db.write_matches(*c["image_ids"], candidate)
    if a.mode == "reference":
        options = pc.FeatureMatchingOptions()
        options.use_gpu = False
        options.num_threads = 2
        options.skip_geometric_verification = True
        options.sift.cpu_brute_force_matcher = True
        pairing = pc.ImportedPairingOptions(match_list_path=str(pairs_file.resolve()))
        pc.match_image_pairs(database, matching_options=options, pairing_options=pairing, device=pc.Device.cpu)
    records = []
    with pc.Database.open(database) as db:
        for c in cases:
            actual = db.read_matches(*c["image_ids"])
            candidate = np.load(a.matches/(c["name"]+"_matches.npy"), allow_pickle=False)
            records.append({"case": c["name"], "colmap_matches": len(actual), "candidate_matches": len(candidate),
                            "indices_exact": bool(np.array_equal(actual, candidate))})
    if a.verify_geometry:
        # Matching with skip_geometric_verification can leave empty geometry rows.
        # Remove them so verify_matches actually estimates geometry rather than skips.
        with pc.Database.open(database) as db:
            db.clear_two_view_geometries()
        pc.verify_matches(database, pairs_file)
        with pc.Database.open(database) as db:
            for record, c in zip(records, cases):
                record["geometric_inliers"] = len(db.read_two_view_geometry(*c["image_ids"]).inlier_matches)
                record["geometry_passed"] = record["geometric_inliers"] >= 15
    unchanged = before == hashlib.sha256(a.database.read_bytes()).hexdigest()
    result = {"mode": a.mode, "source_database_unchanged": unchanged, "npu_executed": False,
              "matching_policy": "COLMAP CPU brute force, angular ratio .8, max distance .7, bidirectional ratio",
              "cases": records, "passed": unchanged and all(r["indices_exact"] and r.get("geometry_passed", True) for r in records),
              "boundary": "selected image pairs only, all previous match tables cleared in copy; no full SfM/training claim"}
    (a.output/"result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
