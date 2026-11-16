"""Archive source and hash-verified firmware from explicit manifests; no board access."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MAX_ARCHIVE_BYTES = 200 * 1024 * 1024
MAX_UNVERIFIED_SOURCE_BYTES = 5 * 1024 * 1024


def frozen_manifest_files(manifest_names):
    """Load and verify exact frozen-package assets, including ROMs and board images."""
    selected = set()
    for name in manifest_names:
        manifest_path = (ROOT / name).resolve()
        if not manifest_path.is_relative_to(ROOT.resolve()) or not manifest_path.is_file():
            raise ValueError("Missing/unsafe frozen asset manifest: " + name)
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
        for entry in document.get("files", []):
            relative = Path(entry["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe path in frozen asset manifest: " + entry["path"])
            asset = (manifest_path.parent / relative).resolve()
            if not asset.is_relative_to(ROOT.resolve()) or not asset.is_file() or asset.is_symlink():
                raise ValueError("Missing/unsafe frozen asset: " + str(asset))
            content = asset.read_bytes()
            if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
                raise ValueError("Frozen asset differs from manifest: " + entry["path"])
            selected.add(asset)
    return selected


def source_files(include_working_tree=False):
    cfg = json.loads((ROOT/"tools/core_sources.json").read_text(encoding="utf-8"))
    selected = {ROOT/p for p in cfg["files"] + cfg.get("release_files", [])}
    runtime_files = cfg.get("runtime_files", {})
    frozen_assets = frozen_manifest_files(cfg.get("frozen_asset_manifests", []))
    selected.update(frozen_assets)
    skipped = set(cfg["skip_directories"]) | {"tests", "__pycache__", "docs", "evidence", "validation"}
    excluded = set(cfg.get("exclude_files", []))
    # Git's ignore rules apply even in the explicit unstaged-refactor mode.
    # Never recursively collect user inputs, generated assets or ignored trees.
    command = ["git", "ls-files", "--cached", "-z"]
    if include_working_tree:
        command += ["--others", "--exclude-standard"]
    tracked = subprocess.check_output(command, cwd=ROOT).decode("utf-8")
    strict_manifest = bool(runtime_files)
    for name in tracked.rstrip("\0").split("\0"):
        if name in excluded:
            continue
        relative = Path(name)
        roots = [folder.rstrip("/") for folder in cfg["trees"]]
        if not any(name.startswith(folder + "/") for folder in roots):
            continue
        if any(part in skipped or part.startswith("build_") for part in relative.parts[:-1]):
            continue
        if relative.name.startswith("test_") or relative.name.endswith("_test.cpp"):
            continue
        if not strict_manifest:
            p = ROOT / relative
            if (p.suffix in cfg["extensions"] or p.name.startswith("requirements") and p.suffix == ".txt"
                    or p.name == ".gitignore" or "LICENSE" in p.name):
                selected.add(p)
            continue
        if len(relative.parts) >= 5 and relative.parts[:3] == Path("examples/3dgs_reconstruction/mvsplat").parts:
            area = relative.parts[3]
            if area == "rendering" and relative.parts[4] == "render_main":
                continue  # The complete exact asset closure comes from its frozen manifest.
            if area == "gaussian_generation" and len(relative.parts) >= 5 and relative.parts[4] == "npu_branch":
                if relative.name not in runtime_files.get("npu_branch", []):
                    continue
            elif area in runtime_files:
                if relative.name not in runtime_files[area]:
                    continue
            else:
                continue
        elif len(relative.parts) == 4 and relative.parts[:3] == Path("examples/3dgs_reconstruction/mvsplat").parts:
            if relative.name != "common.py":
                continue
        else:
            continue
        p = ROOT / relative
        if not p.exists():
            continue  # An unstaged deletion must not resurrect the old path.
        if (p.suffix in cfg["extensions"] or p.name.startswith("requirements") and p.suffix == ".txt"
                or p.name == ".gitignore" or "LICENSE" in p.name):
            selected.add(p)
    for root_name in cfg.get("vendor_source_roots", []):
        vendor_root = (ROOT / root_name).resolve()
        if not vendor_root.is_relative_to(ROOT.resolve()) or not vendor_root.is_dir():
            raise ValueError("Missing/unsafe vendored source root: " + root_name)
        vendor_manifest = ROOT / "third_party/MVSplat-reference-manifest.json"
        vendor_data = json.loads(vendor_manifest.read_text(encoding="utf-8"))
        expected_files = {item["path"]: item for item in vendor_data.get("included_files", [])}
        actual_files = {}
        for p in vendor_root.rglob("*"):
            if not p.is_file():
                continue
            if p.is_symlink() or not p.resolve().is_relative_to(vendor_root):
                raise ValueError("Unsafe vendored source: " + str(p))
            rel = p.relative_to(vendor_root).as_posix()
            actual_files[rel] = p
            if rel not in expected_files:
                raise ValueError("Unmanifested vendored source: " + rel)
            entry = expected_files[rel]
            data = p.read_bytes()
            if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
                raise ValueError("Pinned vendored source changed: " + rel)
            if p.suffix in cfg["extensions"] or "LICENSE" in p.name or p.name == ".gitignore":
                selected.add(p)
        if set(actual_files) != set(expected_files):
            raise ValueError("Pinned vendor source file set is incomplete")
    for area, names in runtime_files.items():
        if area == "npu_branch":
            base = ROOT / "examples/3dgs_reconstruction/mvsplat/gaussian_generation/npu_branch"
        elif area == "common":
            base = ROOT / "examples/3dgs_reconstruction/mvsplat"
        else:
            base = ROOT / "examples/3dgs_reconstruction/mvsplat" / area
        for name in names:
            if not (base / name).is_file():
                raise ValueError("Required runtime module is missing: " + str(base / name))
    selected.update(frozen_manifest_files(cfg.get("frozen_asset_manifests", [])))
    for path in selected:
        if not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()) or path.is_symlink():
            raise ValueError("Missing/unsafe source: " + str(path))
        if path.stat().st_size > MAX_UNVERIFIED_SOURCE_BYTES and path not in frozen_assets:
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
    filelist = ["FPGA43DGS final implementation-chain source bundle",
                f"Payload files: {len(files)} (SHA-256 listed in SOURCE_SHA256.json)",
                "", "Files:"]
    filelist.extend("  " + f.relative_to(ROOT).as_posix() for f in files)
    filelist += ["", "External restoration requirements are documented in docs/BOARD_DEPLOYMENT.md."]
    with zipfile.ZipFile(a.output, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, "FPGA43DGS/" + f.relative_to(ROOT).as_posix())
        hashes["FILELIST.txt"] = hashlib.sha256(("\n".join(filelist) + "\n").encode("utf-8")).hexdigest()
        z.writestr("FPGA43DGS/FILELIST.txt", "\n".join(filelist) + "\n")
        z.writestr("FPGA43DGS/SOURCE_SHA256.json", json.dumps(hashes, indent=2))
    archive_size = a.output.stat().st_size
    if archive_size > MAX_ARCHIVE_BYTES:
        a.output.unlink()
        raise ValueError(f"Compressed archive exceeds 200 MiB: {archive_size} bytes")
    with zipfile.ZipFile(a.output) as z:
        for name, expected in hashes.items():
            archived = "FPGA43DGS/" + name if name != "FILELIST.txt" else "FPGA43DGS/FILELIST.txt"
            if hashlib.sha256(z.read(archived)).hexdigest() != expected:
                raise ValueError("Archive verification failed: " + name)
        if z.testzip() is not None:
            raise ValueError("Archive CRC test failed")
        archive_names = set(z.namelist())
        if "FPGA43DGS/examples/3dgs_reconstruction/pipeline.py" not in archive_names:
            raise ValueError("Unified pipeline entry is missing")
        if "FPGA43DGS/examples/3dgs_reconstruction/vendor/MVSplat_reference/src/model/encoder/encoder_costvolume.py" not in archive_names:
            raise ValueError("Pinned MVSplat upstream implementation is missing")
        required = ("hardware/BOOT.bin", "hardware/ai7030_edif_top_disable_icap.bit",
                    "hardware/reports/timing_summary.rpt", "hardware/reports/utilization.rpt")
        prefix = "FPGA43DGS/examples/3dgs_reconstruction/mvsplat/rendering/render_main/"
        if any(prefix + item not in archive_names for item in required):
            raise ValueError("Frozen FPGA deployment or implementation reports are missing")
    print(json.dumps(dict(files=len(files), bytes=a.output.stat().st_size,
                         sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
                         scope="mainline source plus manifest-verified frozen FPGA board assets; no private media, weights, SDK, or environment"), indent=2))


if __name__ == "__main__":
    main()
