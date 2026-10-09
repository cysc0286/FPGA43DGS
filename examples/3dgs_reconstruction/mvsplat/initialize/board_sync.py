"""Stage changed warm-pipeline source in a separate board candidate directory."""
import argparse
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT.parent / "3dgs_compositor/board"))
import remote

SOURCE_FILES = ("common.py", "export.py", "initialize/model_runtime.py",
                "initialize/renderer_runtime.py", "initialize/attributes_resident.cpp",
                "initialize/validate_resident.py", "initialize/group_sort_resident.cpp",
                "initialize/run.py", "initialize/session.py",
                "rendering/__init__.py", "rendering/runtime.py",
                "rendering/pipeline_adapter.py", "rendering/mainline.json",
                "video_input/prepare.py", "warm_pipeline.py")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="/root/fpga43dgs_reconstruction/npu_board_v2_20260929")
    p.add_argument("--dest", required=True)
    p.add_argument("--renderer", default="/root/fpga43dgs_releases/20260928T004334/3dgs_renderer_v1_20260928")
    a = p.parse_args()
    if not a.dest.startswith("/root/fpga43dgs_reconstruction/") or a.dest == a.base:
        p.error("Destination must be a new reconstruction candidate")
    client = remote.connect()
    try:
        qbase, qdest = shlex.quote(a.base), shlex.quote(a.dest)
        remote.run(client, "test ! -e " + qdest)
        remote.run(client, "mkdir -p " + qdest)
        for name in ("mvsplat", "modules", "pipeline.py"):
            remote.run(client, "cp -a " + qbase + "/" + name + " " + qdest + "/" + name,
                       timeout=30)
        for name in ("deps", "input", "vendor", "weights"):
            remote.run(client, "ln -s " + qbase + "/" + name + " " + qdest + "/" + name)
        files = SOURCE_FILES
        directories = sorted({str(Path(name).parent).replace("\\", "/") for name in files})
        remote.run(client, "mkdir -p " + " ".join(
            shlex.quote(a.dest + "/mvsplat/" + directory) for directory in directories))
        sftp = client.open_sftp()
        try:
            for name in files:
                sftp.put(str(ROOT / "mvsplat" / name), a.dest + "/mvsplat/" + name)
        finally:
            sftp.close()
        script = " ".join(shlex.quote(a.dest + "/mvsplat/" + name)
                          for name in files if name.endswith(".py"))
        status, output = remote.run(client,
            "/root/fpga43dgs_reconstruction/arm_env/bin/python3 -m py_compile " + script,
            timeout=30)
        build = ("g++ -O3 -std=gnu++17 -ffp-contract=off -DPACKED_SORT -Wall -Wextra "
                 + shlex.quote(a.dest + "/mvsplat/initialize/group_sort_resident.cpp")
                 + " -o " + shlex.quote(a.dest + "/mvsplat/initialize/group_sort_resident")
                 + " -I" + shlex.quote(a.renderer + "/src/cpu"))
        remote.run(client, build, timeout=120)
        attributes_build = ("g++ -O3 -std=gnu++17 -ffp-contract=off -Wall -Wextra "
                 + shlex.quote(a.dest + "/mvsplat/initialize/attributes_resident.cpp")
                 + " -o " + shlex.quote(a.dest + "/mvsplat/initialize/attributes_resident")
                 + " -I" + shlex.quote(a.renderer + "/src/cpu"))
        remote.run(client, attributes_build, timeout=120)
        print(output or "Board candidate source compiled", flush=True)
        print(a.dest, flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
