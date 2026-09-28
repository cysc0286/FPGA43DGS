"""Prepare a portable source/test-input bundle, without any ARM deployment claim."""
import argparse
import json
import shutil
import tarfile
from pathlib import Path
from common import ROOT, save, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True, type=Path, help="New package directory; sibling .tar.gz also created")
    p.add_argument("--source", type=Path, default=ROOT.parent/"vendor/OpenSplat_bounded_v1")
    p.add_argument("--video", type=Path, default=ROOT.parent/"data/nyu_snippet_curl.mp4")
    p.add_argument("--reference-run", type=Path, default=ROOT.parent/"runs/nyu_cpu_v1")
    a = p.parse_args()
    if a.output.exists() or a.output.with_suffix(".tar.gz").exists():
        raise ValueError("Package target exists; use a fresh name")
    manifest = json.loads((a.source/"heterogs_source.json").read_text())
    for n, h in manifest["modified_files"].items():
        if sha256(a.source/n) != h:
            raise ValueError("Source differs from its prepared manifest: " + n)
    a.output.mkdir(parents=True)
    shutil.copytree(ROOT, a.output/"bounded", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ["stages.py", "evaluate.py", "collect_evidence.py", "fetch_video.py"]:
        shutil.copy2(ROOT.parent/name, a.output/name)
    shutil.copy2(ROOT.parent/"pipeline.py", a.output/"pipeline.py")
    shutil.copytree(ROOT.parent/"modules", a.output/"modules", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(ROOT/"README.md", a.output/"README.md")
    shutil.copytree(a.source, a.output/"source/OpenSplat",
                    ignore=shutil.ignore_patterns("build*", ".git", "__pycache__"))
    # Copy small reference data only. Never bundle credentials or the Windows runtime.
    ref = a.output/"reference"
    ref.mkdir()
    for name in ["project", "sfm.json", "project.json", "splat.ply"]:
        src = a.reference_run/name
        if src.is_dir():
            shutil.copytree(src, ref/name)
        else:
            shutil.copy2(src, ref/name)
    (a.output/"data").mkdir()
    (a.output/"evidence").mkdir()
    shutil.copy2(ROOT.parent/"evidence/versions.csv", a.output/"evidence/versions.csv")
    shutil.copy2(ROOT.parent/"evidence/sources.json", a.output/"evidence/baseline_sources.json")
    preparation = ROOT.parent/"evidence/bounded_v1_preparation"
    if preparation.exists():
        shutil.copytree(preparation, a.output/"evidence/bounded_v1_preparation")
    shutil.copy2(a.video, a.output/"data/video.mp4")
    save(a.output/"manifest.json", {"kind": "CPU reconstruction candidate source bundle",
         "resource_abi": 1, "source": manifest, "arm_binary_included": False,
         "board_tested": False, "video_sha256": sha256(a.video),
         "reference_scope": "SfM project from PC; only for isolated trainer tests, not evidence of board SfM",
         "dependencies_included": False, "license": "OpenSplat AGPL-3.0; see source/OpenSplat/LICENSE.txt"})
    hashes = {f.relative_to(a.output).as_posix(): sha256(f) for f in sorted(a.output.rglob("*")) if f.is_file()}
    save(a.output/"sha256.json", hashes)
    archive = a.output.with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(a.output, arcname=a.output.name)
    with tarfile.open(archive, "r:gz") as tar:
        import hashlib
        for name, expected in hashes.items():
            f = tar.extractfile(a.output.name+"/"+name)
            if f is None:
                raise ValueError("Missing archive member: " + name)
            h = hashlib.sha256()
            for block in iter(lambda: f.read(1024*1024), b""):
                h.update(block)
            if h.hexdigest() != expected:
                raise ValueError("Archive hash mismatch: " + name)
    save(archive.with_suffix(archive.suffix+".json"), {"archive": archive.name, "sha256": sha256(archive),
         "verified_files": len(hashes), "bytes": archive.stat().st_size})
    print(archive)


if __name__ == "__main__":
    main()
