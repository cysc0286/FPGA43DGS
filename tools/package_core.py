"""Create a source-only collaboration ZIP from an explicit manifest; no board access."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def source_files(include_working_tree=False):
    cfg = json.loads((ROOT/"tools/core_sources.json").read_text(encoding="utf-8"))
    selected = {ROOT/p for p in cfg["files"]}
    skipped = set(cfg["skip_directories"])
    excluded = set(cfg.get("exclude_files", []))
    # Git's ignore rules apply even in the explicit unstaged-refactor mode.
    # Never recursively collect user inputs, generated assets or ignored trees.
    command = ["git", "ls-files", "--cached", "-z"]
    if include_working_tree:
        command += ["--others", "--exclude-standard"]
    tracked = subprocess.check_output(command, cwd=ROOT).decode("utf-8")
    for name in tracked.rstrip("\0").split("\0"):
        if name in excluded:
            continue
        relative = Path(name)
        if not any(name.startswith(folder.rstrip("/") + "/") for folder in cfg["trees"]):
            continue
        if any(part in skipped or part.startswith("build_") for part in relative.parts[:-1]):
            continue
        p = ROOT / relative
        if not p.exists():
            continue  # An unstaged deletion must not resurrect the old path.
        if (p.suffix in cfg["extensions"] or p.name.startswith("requirements") and p.suffix == ".txt"
                or p.name == ".gitignore" or "LICENSE" in p.name):
            selected.add(p)
    for path in selected:
        if not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()) or path.is_symlink():
            raise ValueError("Missing/unsafe source: " + str(path))
        if path.stat().st_size > 5*1024*1024:
            raise ValueError("Unexpected large source: " + str(path))
    return sorted(selected)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--include-working-tree", action="store_true",
                   help="Include non-ignored new sources during an unstaged refactor; review the source manifest")
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    files = source_files(a.include_working_tree)
    hashes = {f.relative_to(ROOT).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    with zipfile.ZipFile(a.output, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, "FPGA43DGS/" + f.relative_to(ROOT).as_posix())
        z.writestr("FPGA43DGS/SOURCE_SHA256.json", json.dumps(hashes, indent=2))
    with zipfile.ZipFile(a.output) as z:
        for name, expected in hashes.items():
            if hashlib.sha256(z.read("FPGA43DGS/"+name)).hexdigest() != expected:
                raise ValueError("Archive verification failed: " + name)
    print(json.dumps(dict(files=len(files), bytes=a.output.stat().st_size,
                         sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
                         scope="source only; no new training or board execution"), indent=2))


if __name__ == "__main__":
    main()
