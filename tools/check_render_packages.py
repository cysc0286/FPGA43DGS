"""Verify frozen renderer bytes and ensure the experimental package is not invoked.

Offline only: no SSH, firmware installation, or timing measurements.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MVS = ROOT / "examples/3dgs_reconstruction/mvsplat"


def verify_package(folder):
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = (folder / entry["path"]).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError("Manifest path escapes package")
        data = path.read_bytes()
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError(f"Package hash mismatch: {path}")
    boot = folder / "hardware/BOOT.bin"
    if hashlib.sha256(boot.read_bytes()).hexdigest() != manifest["boot_sha256"]:
        raise ValueError("BOOT identity mismatch")
    print(f"PASS {folder.name}: {len(manifest['files'])} hash-verified files")
    return manifest


def main():
    stable = verify_package(MVS / "rendering/package")
    candidate = verify_package(MVS / "render_branch")
    profile = json.loads((MVS / "rendering/mainline.json").read_text(encoding="utf-8"))
    if stable["boot_sha256"] != profile["required_hardware"]["accepted_boot_sha256"]:
        raise ValueError("Default profile does not select frozen four-lane hardware")
    if not stable["default_enabled"] or profile["required_hardware"]["lanes"] != 4 or candidate["default_enabled"]:
        raise ValueError("Incorrect default renderer selection")
    frozen_profile = json.loads((MVS / "rendering/package/profile.json").read_text(encoding="utf-8"))
    if frozen_profile != profile:
        raise ValueError("Frozen mainline profile has drifted from the runtime profile")
    if candidate["boot_sha256"] == stable["boot_sha256"]:
        raise ValueError("Mainline and candidate must have different firmware identities")
    checked = 0
    for base in (MVS / "rendering", MVS / "initialize", MVS / "video_input",
                 MVS.parent / "modules"):
        for path in base.rglob("*.py"):
            if any(part in {"results", "package", "__pycache__"} for part in path.parts):
                continue
            source = path.read_text(encoding="utf-8-sig")
            ast.parse(source, filename=str(path))
            # Reject both imports and string-based launch/load references.
            if "render_branch" in source or "pipegs_hgr_v4_coord_2lane" in source:
                raise ValueError(f"Default source references experimental branch: {path}")
            checked += 1
    for name in ("pipeline.py", "mvsplat/run.py"):
        path = MVS.parent / name
        if path.exists() and "render_branch" in path.read_text(encoding="utf-8-sig"):
            raise ValueError(f"Pipeline entry invokes candidate: {path}")
    print(f"PASS default isolation: {checked} runtime/module sources, four-lane profile")
    print("No board test performed; saved validation dates remain unchanged.")


if __name__ == "__main__":
    main()
