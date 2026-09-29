"""Stage the CPU MVSplat runtime in an isolated board workspace."""
import argparse
from pathlib import Path, PurePosixPath
import shlex
import sys
import tarfile

from common import new_directory, save, sha

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / "3dgs_compositor/board"))
import remote


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--remote-root", required=True)
    a = p.parse_args()
    dest = PurePosixPath(a.remote_root)
    if not str(dest).startswith("/root/fpga43dgs_reconstruction/") or ".." in dest.parts:
        p.error("Use an isolated reconstruction directory")
    out = new_directory(a.out)
    files = {}
    for subfolder, suffixes in (("mvsplat", (".py", ".json")), ("modules", (".py",))):
        for f in sorted((ROOT/subfolder).rglob("*")):
            if f.is_file() and f.suffix in suffixes and "__pycache__" not in f.parts:
                files[str(f.relative_to(ROOT)).replace("\\", "/")] = f
    files["pipeline.py"] = ROOT/"pipeline.py"
    for f in sorted((ROOT/"vendor/MVSplat_reference").rglob("*")):
        if f.is_file() and (f.suffix in (".py", ".yaml") or f.name in ("LICENSE", "LICENSE.txt")):
            files["vendor/MVSplat_reference/"+str(f.relative_to(ROOT/"vendor/MVSplat_reference")).replace("\\", "/")] = f
    for f in sorted((ROOT/"models/mvsplat/arm_wheels").glob("*.whl")):
        files["wheels/"+f.name] = f
    if not any("antlr4_python3_runtime" in key for key in files):
        raise ValueError("Include the pure-Python ANTLR wheel used by OmegaConf")
    files["weights/re10k.ckpt"] = ROOT/"models/mvsplat/re10k.ckpt"
    for f in sorted(a.input.rglob("*")):
        if f.is_file():
            files["input/"+str(f.relative_to(a.input)).replace("\\", "/")] = f
    manifest = {name:dict(bytes=f.stat().st_size, sha256=sha(f)) for name,f in files.items()}
    save(out/"files.json", manifest)
    archive = out/"runtime.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name,f in files.items():
            tar.add(f, arcname=name, recursive=False)
        tar.add(out/"files.json", arcname="files.json", recursive=False)
    client = remote.connect()
    sftp = client.open_sftp()
    try:
        remote.run(client, "mkdir " + shlex.quote(str(dest)), timeout=15)
        sftp.put(str(archive), str(dest/"runtime.tar.gz"))
        _, actual = remote.run(client, "sha256sum " + shlex.quote(str(dest/"runtime.tar.gz")), timeout=30)
        if actual.split()[0] != sha(archive):
            raise ValueError("Deployment archive hash mismatch")
        remote.run(client, shlex.join(["tar", "-xzf", str(dest/"runtime.tar.gz"), "-C", str(dest)]), timeout=90)
        script = "set -eu\n" + "cd " + shlex.quote(str(dest)) + "\n"
        script += "export PATH=/root/fpga43dgs_reconstruction/arm_env/bin:$PATH\n"
        script += "export LD_LIBRARY_PATH=/root/fpga43dgs_reconstruction/arm_env/lib\n"
        script += "export LD_PRELOAD=/root/fpga43dgs_reconstruction/arm_env/lib/libgomp.so.1\n"
        script += "export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2\n"
        script += "python -m pip install --no-index --no-deps --target deps wheels/*.whl\n"
        script += "export PYTHONPATH=" + shlex.quote(str(dest/"deps")) + "\n"
        script += "python - <<'VERIFY'\nimport json,pathlib,hashlib\nroot=pathlib.Path('.')\nfor name,value in json.loads(pathlib.Path('files.json').read_text()).items():\n assert hashlib.sha256((root/name).read_bytes()).hexdigest()==value['sha256'],name\nimport torch,einops,e3nn,scipy,omegaconf,jaxtyping,pycolmap,cv2\nprint('ARM_IMPORT_OK',torch.__version__,torch.cuda.is_available())\nVERIFY\n"
        (out/"install.sh").write_text(script, encoding="utf-8")
        remote.run(client, script, timeout=240, log=out/"install.log")
        save(out/"deployment.json", dict(complete=True, remote_root=str(dest), archive_sha256=sha(archive),
             files=len(files), dependencies_isolated=True))
        print("Board runtime staged and imports verified:", dest, flush=True)
    finally:
        sftp.close()
        client.close()


if __name__ == "__main__":
    main()
