"""Create a source-only collaboration ZIP from an explicit manifest; no board access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def source_files():
    cfg = json.loads((ROOT/"tools/core_sources.json").read_text(encoding="utf-8"))
    selected = {ROOT/p for p in cfg["files"]}
    skipped = set(cfg["skip_directories"])
    for folder in cfg["trees"]:
        for parent, dirs, files in os.walk(ROOT/folder):
            dirs[:] = [d for d in dirs if d not in skipped and not d.startswith("build_")]
            for name in files:
                p = Path(parent)/name
                if p.suffix in cfg["extensions"] or name.startswith("requirements") and p.suffix == ".txt" or name == ".gitignore" or "LICENSE" in name:
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
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    files = source_files()
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
