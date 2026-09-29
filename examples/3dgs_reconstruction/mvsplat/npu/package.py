"""Prepare a relocatable NPU candidate bundle locally; does not upload or execute."""
import argparse
import json
from pathlib import Path
import shutil
import tarfile
from common import new_directory, save, sha
from npu.artifacts import validate_bundle, relative_file


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compiled", required=True, type=Path)
    p.add_argument("--graphs", required=True, type=Path)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--partitions", required=True, nargs="+")
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args(argv)
    compiled, _ = validate_bundle(a.compiled, a.partitions, "npu")
    graphs, _ = validate_bundle(a.graphs, a.partitions, "onnx_reference")
    if compiled["source_manifest_sha256"] != sha(a.graphs/"manifest.json"):
        raise ValueError("Compiled and source graph bundles differ")
    if graphs["context_sha256"] != sha(a.input/"context.npz"):
        raise ValueError("Numerical-oracle input differs from exported graph input")
    out = new_directory(a.out)
    bundle = new_directory(out/"candidate")
    for kind, manifest, source in (("compiled", compiled, a.compiled), ("graphs", graphs, a.graphs)):
        folder = new_directory(bundle/kind)
        manifest = dict(manifest, partitions={name:manifest["partitions"][name] for name in a.partitions})
        for name, item in manifest["partitions"].items():
            names = [item["graph"], item["raw"]] if kind == "compiled" else [name+"/model.onnx", name+"/oracle.npz"]
            for filename in names:
                src = relative_file(source, filename)
                if filename.endswith("oracle.npz") and sha(src) != item["oracle_sha256"]:
                    raise ValueError("Oracle hash mismatch")
                target = folder/filename
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
        save(folder/"manifest.json", manifest)
    input_meta = json.loads((a.input/"input.json").read_text())
    for name in ["context.npz", "input.json"]+["images/"+v["name"] for v in input_meta["views"]]:
        src = relative_file(a.input, name)
        target = bundle/"input"/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    files = {str(f.relative_to(bundle)).replace("\\", "/"):dict(bytes=f.stat().st_size, sha256=sha(f))
             for f in sorted(bundle.rglob("*")) if f.is_file()}
    save(bundle/"files.json", files)
    archive = out/"mvsplat_npu_candidate.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for f in sorted(bundle.rglob("*")):
            if f.is_file():
                tar.add(f, arcname="candidate/"+f.relative_to(bundle).as_posix(), recursive=False)
    # Re-validate the relocated paths before marking the package ready.
    validate_bundle(bundle/"compiled", a.partitions, "npu")
    validate_bundle(bundle/"graphs", a.partitions, "onnx_reference")
    save(out/"package.json", dict(complete=True, board_uploaded=False, npu_executed=False,
        partitions=a.partitions, archive_sha256=sha(archive), archive_bytes=archive.stat().st_size,
        files=len(files), weight_sha256=graphs["weight_sha256"],
        note="Requires matching runtime code, official checkpoint and ARM bridge; no SDK binaries included"))
    print(archive)


if __name__ == "__main__":
    main()
