"""Small standard-library helpers shared by preparation and execution tools."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def cpu_env(threads, resources=None):
    # Do not inherit stale candidate settings into a baseline/profile.
    env = {k: v for k, v in os.environ.items() if not k.startswith("HGS_")}
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
               OPENBLAS_NUM_THREADS=str(threads), NUMEXPR_NUM_THREADS=str(threads))
    if resources:
        env.update({"HGS_"+k.upper(): str(v) for k, v in resources.items()})
    runtime = ROOT.parent / "runtime_paths.json"
    if os.name == "nt" and runtime.exists():
        env["PATH"] = os.pathsep.join(json.loads(runtime.read_text())) + os.pathsep + env.get("PATH", "")
    return env
