"""Archive this offline v1 run, including failed attempts, without replacing history."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    def copy(src, dst):
        target = out/dst
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/src, target)
    for source, target in [("generated/cases", "cases"), ("generated/graph128_nhwc", "onnx"),
                           ("runs/offline_v1", "offline"), ("runs/cpp_cpu_pair0", "cpp_cpu"),
                           ("runs/colmap_reference_v2", "colmap_reference"), ("runs/colmap_injection", "colmap_injection"),
                           ("runs/build_cpu", "cpp_cpu_build"), ("runs/build_cpu_v2", "unit_cpp_build"),
                           ("runs/build_npu_syntax_v3", "npu_header_check"), ("runs/board_plan_v1", "board_plan")]:
        for file in (ROOT/source).iterdir():
            if file.is_file() and file.suffix in (".json", ".log", ".csv", ".tsv", ".npz", ".npy", ".onnx"):
                copy(file.relative_to(ROOT), Path(target)/file.name)
    compile_dir = ROOT/"runs/compile_tf32_nhwc"
    for file in compile_dir.glob("*.json"):
        copy(file.relative_to(ROOT), Path("compile")/file.name)
    for file in compile_dir.glob("*.log"):
        copy(file.relative_to(ROOT), Path("compile")/file.name)
    for file in (compile_dir/"compiled").glob("*_ZG.*"):
        copy(file.relative_to(ROOT), Path("compiled")/file.name)
    failed = ["compile_tf32_v1", "compile_tf32_fd", "compile_tf32_fd_v2", "compile_tf32_fd_v3", "compile_tf32_fd_v4",
              "build_npu_syntax", "build_npu_syntax_v2", "colmap_reference"]
    for attempt in failed:
        for file in (ROOT/"runs"/attempt).iterdir():
            if file.is_file() and file.suffix in (".json", ".log"):
                copy(file.relative_to(ROOT), Path("earlier_attempts")/attempt/file.name)
    copy("runs/unit_tests_v1.log", "unit_tests.log")
    copy("generated/board_job_pair0/manifest.json", "cpp_cpu/job_manifest.json")
    for pattern in ("*.py", "*.md", "board/*.cpp"):
        for source in ROOT.glob(pattern):
            copy(source.relative_to(ROOT), Path("source")/source.relative_to(ROOT))
    result = json.loads((out/"offline/result.json").read_text())
    geometry = json.loads((out/"colmap_reference/result.json").read_text())["cases"]
    fields = ["version", "case", "descriptor_pairs", "blocks", "matches", "geometric_inliers", "score_max_abs_error",
              "onnx_cpu_total_ms_single_run", "logical_input_bytes", "logical_output_bytes", "npu_executed"]
    with (out/"versions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item, geom in zip(result["cases"], geometry):
            stat = item["onnx_cpu"]
            writer.writerow(dict(version="v1_offline", case=item["case"]["name"], descriptor_pairs=item["case"]["descriptor_pairs"],
                blocks=stat["calls"], matches=stat["matched"], geometric_inliers=geom["geometric_inliers"],
                score_max_abs_error=item["score_max_abs_error"], onnx_cpu_total_ms_single_run=stat["total_s"]*1000,
                logical_input_bytes=stat["logical_input_bytes"], logical_output_bytes=stat["logical_output_bytes"], npu_executed=False))
    summary = {"version": "v1_offline", "npu_executed": False, "tests_passed": 8,
               "real_cases": 3, "descriptor_comparisons": sum(c["case"]["descriptor_pairs"] for c in result["cases"]),
               "onnx_scores_and_matches_passed": result["passed"], "compiler_stages_passed": 5,
               "sdk_header_syntax_passed": True, "arm_linked": False,
               "board_speedup": None, "board_precision": None, "new_reconstruction_quality": None,
               "boundary": "offline preparation of descriptor matching only; no board access or default pipeline replacement"}
    (out/"summary.json").write_text(json.dumps(summary, indent=2))
    hashes = {f.relative_to(out).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.rglob("*")) if f.is_file()}
    (out/"SHA256SUMS.json").write_text(json.dumps(hashes, indent=2))
    for relative, expected in hashes.items():
        if hashlib.sha256((out/relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError("Archive read-back mismatch")
    print(json.dumps({"output": str(out), "files": len(hashes)+1, "bytes": sum(f.stat().st_size for f in out.rglob("*") if f.is_file()),
                      "hashes_verified": True, **summary}, indent=2))


if __name__ == "__main__":
    main()
