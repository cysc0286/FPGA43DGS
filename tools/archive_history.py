"""Pack selected tracked history and restore it without overwriting a checkout.

This is a file/archive operation, never a board or algorithm test. Creation does
not delete source files. The external index records every original path/hash.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT).decode("utf-8")


def selected_group(path):
    # Frozen packages are independently audited and must remain byte-identical.
    if path.startswith("releases/") or "/rendering/package/" in path or "/render_branch/" in path:
        return None
    if path.startswith("examples/3dgs_reconstruction/mvsplat/results/"):
        return "frontend_history"
    hardware = "examples/3dgs_flicker_hw/"
    if any(path.startswith(hardware + prefix) for prefix in (
            "physical_opt/", "pipeline/exp_rom/", "pipeline/resource_balance/",
            "pipeline/lane_workset/")):
        return "fpga_history"
    if path.startswith("examples/") and "/results/" in path:
        return "other_history"
    return None


def create(destination):
    destination.mkdir(parents=True, exist_ok=False)
    dirty = set(git("diff", "--name-only", "HEAD", "-z").rstrip("\0").split("\0"))
    groups = {}
    for name in git("ls-files", "-z").rstrip("\0").split("\0"):
        group = selected_group(name)
        if group:
            groups.setdefault(group, []).append(name)
    index = {"schema": "heterogs_history_archive_v1",
             "source_commit": git("rev-parse", "HEAD").strip(),
             "scope": "Tracked historical source/evidence; no models, SDKs, credentials or local builds",
             "archives": [], "retained_working_changes": []}
    for group, names in sorted(groups.items()):
        entries = []
        target = destination / (group + ".zip")
        with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in sorted(names):
                source = ROOT / name
                if source.is_symlink() or not source.resolve().is_relative_to(ROOT):
                    raise ValueError("Unsafe archive source: " + name)
                # Preserve unrelated working changes locally, but do not publish
                # them hidden inside a historical-source archive.
                payload = (subprocess.check_output(["git", "show", "HEAD:" + name], cwd=ROOT)
                           if name in dirty else source.read_bytes())
                entry = {"path": name, "bytes": len(payload), "sha256": digest(payload),
                         "retain_in_worktree": name in dirty}
                entries.append(entry)
                if name in dirty:
                    index["retained_working_changes"].append(name)
                archive.writestr(name, payload)
            archive.writestr("ARCHIVE_MANIFEST.json", json.dumps(entries, indent=2) + "\n")
        item = {"file": target.name, "bytes": target.stat().st_size,
                "sha256": digest(target.read_bytes()), "files": entries}
        verify_one(destination, item)
        index["archives"].append(item)
    (destination / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"archives": len(index["archives"]),
                      "files": sum(len(a["files"]) for a in index["archives"]),
                      "bytes": sum(a["bytes"] for a in index["archives"]),
                      "retained_working_changes": index["retained_working_changes"]}, indent=2))


def safe_relative(name):
    relative = PurePosixPath(name)
    if (relative.is_absolute() or ".." in relative.parts or "\\" in name
            or ":" in name or str(relative) != name):
        raise ValueError("Unsafe archive path: " + name)
    return relative


def verify_one(base, item):
    archive_name = safe_relative(item["file"])
    if len(archive_name.parts) != 1:
        raise ValueError("Archive filename must be local to the index")
    path = base / archive_name
    if path.stat().st_size != item["bytes"] or digest(path.read_bytes()) != item["sha256"]:
        raise ValueError("Archive digest mismatch: " + str(path))
    with zipfile.ZipFile(path) as archive:
        expected = {e["path"] for e in item["files"]} | {"ARCHIVE_MANIFEST.json"}
        if len(archive.namelist()) != len(expected) or set(archive.namelist()) != expected:
            raise ValueError("Unexpected or duplicate archive members")
        if json.loads(archive.read("ARCHIVE_MANIFEST.json")) != item["files"]:
            raise ValueError("Embedded manifest differs from index")
        for entry in item["files"]:
            safe_relative(entry["path"])
            payload = archive.read(entry["path"])
            if len(payload) != entry["bytes"] or digest(payload) != entry["sha256"]:
                raise ValueError("Archived file mismatch: " + entry["path"])


def restore(base, destination):
    index = json.loads((base / "index.json").read_text(encoding="utf-8"))
    if destination.exists():
        raise ValueError("Restore destination must be new; never overwrite active source")
    for item in index["archives"]:
        verify_one(base, item)
    destination.mkdir(parents=True)
    for item in index["archives"]:
        with zipfile.ZipFile(base / item["file"]) as archive:
            for entry in item["files"]:
                target = destination / safe_relative(entry["path"])
                if not target.resolve().is_relative_to(destination.resolve()):
                    raise ValueError("Restore path escapes destination")
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as output:
                    output.write(archive.read(entry["path"]))
    print("Restored original relative paths into " + str(destination))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("create", "verify", "restore"))
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "create":
        create(args.archive.resolve())
    elif args.action == "restore":
        if args.output is None:
            parser.error("restore requires --output")
        restore(args.archive.resolve(), args.output.resolve())
    else:
        index = json.loads((args.archive / "index.json").read_text(encoding="utf-8"))
        for item in index["archives"]:
            verify_one(args.archive, item)
        print("Archive members, lengths, CRCs and SHA256 digests verified")


if __name__ == "__main__":
    main()
