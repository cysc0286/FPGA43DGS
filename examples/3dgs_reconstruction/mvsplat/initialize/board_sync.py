"""Stage changed warm-pipeline source in a separate board candidate directory."""
import argparse
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mvsplat"))
from initialize import transport as remote
# The six owners supply one deployment manifest; tests and frozen assets do not.
OWNERS = ('initialize', 'video_input', 'pose_estimation', 'gaussian_generation', 'rendering', 'web')
def deployment_sources():
    base = ROOT / 'mvsplat'
    selected = ['common.py']
    for owner in OWNERS:
        for path in sorted((base / owner).rglob('*')):
            if (not path.is_file() or path.name.startswith('test_')
                    or any(part in {'tests', '__pycache__', 'render_main', 'docs', 'validation'}
                           for part in path.relative_to(base).parts)):
                continue
            # Historical validation receipts are documentation, not runtime assets.
            if path.suffix == '.json' and path.name not in {'mainline.json', 'acceptance.json'}:
                continue
            if path.suffix in {'.py', '.cpp', '.h', '.hpp', '.html', '.css', '.js', '.json', '.sh'}:
                selected.append(path.relative_to(base).as_posix())
    return tuple(dict.fromkeys(selected))

SOURCE_FILES = deployment_sources()
NPU_SOURCE_FILES = tuple(name for name in SOURCE_FILES
                         if name.startswith('gaussian_generation/'))


def upload_sources(client, board_mvsplat, files=SOURCE_FILES):
    """Upload current package paths into a separate candidate directory."""
    directories = sorted({str(Path(name).parent).replace("\\", "/") for name in files})
    remote.run(client, "mkdir -p " + " ".join(
        shlex.quote(board_mvsplat + "/" + directory) for directory in directories))
    sftp = client.open_sftp()
    try:
        for name in files:
            sftp.put(str(ROOT / "mvsplat" / name), board_mvsplat + "/" + name)
    finally:
        sftp.close()


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
        remote.run(client, "mkdir -p " + qdest + "/mvsplat")
        for name in ("deps", "input", "vendor", "weights"):
            remote.run(client, "ln -s " + qbase + "/" + name + " " + qdest + "/" + name)
        files = SOURCE_FILES
        upload_sources(client, a.dest + "/mvsplat", files)
        # The public CLI is copied once, outside the six implementation directories.
        sftp = client.open_sftp()
        try:
            sftp.put(str(ROOT / 'pipeline.py'), a.dest + '/pipeline.py')
        finally:
            sftp.close()
        script = " ".join(shlex.quote(a.dest + "/mvsplat/" + name)
                          for name in files if name.endswith(".py"))
        status, output = remote.run(client,
            "/root/fpga43dgs_reconstruction/arm_env/bin/python3 -m py_compile " + script,
            timeout=30)
        build = ("g++ -O3 -std=gnu++17 -ffp-contract=off -DPACKED_SORT -Wall -Wextra "
                 + shlex.quote(a.dest + "/mvsplat/rendering/group_sort_resident.cpp")
                 + " -o " + shlex.quote(a.dest + "/mvsplat/rendering/group_sort_resident")
                 + " -I" + shlex.quote(a.renderer + "/src/cpu"))
        remote.run(client, build, timeout=120)
        attributes_build = ("g++ -O3 -std=gnu++17 -ffp-contract=off -Wall -Wextra "
                 + shlex.quote(a.dest + "/mvsplat/rendering/attributes_resident.cpp")
                 + " -o " + shlex.quote(a.dest + "/mvsplat/rendering/attributes_resident")
                 + " -I" + shlex.quote(a.renderer + "/src/cpu"))
        remote.run(client, attributes_build, timeout=120)
        print(output or "Board candidate source compiled", flush=True)
        print(a.dest, flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
