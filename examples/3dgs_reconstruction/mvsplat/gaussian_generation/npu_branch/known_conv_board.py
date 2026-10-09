"""Transfer and run the isolated known-convolution diagnostic on the lab board."""
import argparse
import json
from pathlib import Path
import shlex
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graphs", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--bridge-source", type=Path,
                   default=Path(__file__).with_name("bridge.cpp"),
                   help="Bridge source to build on the ARM board")
    p.add_argument("--board-library", help="Existing board bridge library; skip rebuilding")
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--board-root", default="/root/fpga43dgs_reconstruction/npu_known_conv_detailed_20260930")
    a = p.parse_args()
    if a.repeats < 1:
        p.error("--repeats must be positive")
    if a.out.exists():
        p.error("Output evidence directory already exists")
    a.out.mkdir(parents=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]/"3dgs_compositor/board"))
    import remote
    from initialize.board_sync import NPU_SOURCE_FILES, upload_sources
    client = remote.connect()
    root = a.board_root
    try:
        status, output = remote.run(client,
            "test ! -e "+shlex.quote(root)+" && mkdir -p "+shlex.quote(root), timeout=10)
        upload_sources(client, root + "/mvsplat", NPU_SOURCE_FILES)
        sftp = client.open_sftp()
        manifest = json.loads((a.bundle/"manifest.json").read_text())
        files = {
            "graphs": [a.graphs/"manifest.json"] + [
                a.graphs/name/"oracle.npz" for name in manifest["partitions"]],
            "bundle": [a.bundle/"manifest.json"] + [
                a.bundle/item[key] for item in manifest["partitions"].values()
                for key in ("graph", "raw")],
        }
        for kind, source in (("graphs", a.graphs), ("bundle", a.bundle)):
            for file in files[kind]:
                relative = file.relative_to(source).as_posix()
                target = root+"/"+kind+"/"+relative
                remote.run(client, "mkdir -p "+shlex.quote(target.rsplit("/", 1)[0]), timeout=10)
                sftp.put(str(file), target)
        sftp.put(str(Path(__file__).with_name("known_conv.py")), root+"/known_conv.py")
        if a.board_library is None:
            sftp.put(str(a.bridge_source.resolve()), root+"/bridge.cpp")
        # Keep the command explicit because remote shells otherwise treat a
        # typo in the SDK path as a successful no-op when logging is redirected.
        build = (
            "cd "+shlex.quote(root)+" && sdk=/root/heterogs_npu/sdk_3.36.1/usr && "
            "timeout 180s g++ -O3 -std=gnu++17 -ffp-contract=off -shared -fPIC bridge.cpp "
            "-o libmgs_npu_detailed.so -I\"$sdk/include\" -L\"$sdk/lib/aarch64-linux-gnu\" "
            "-Wl,-rpath,\"$sdk/lib/aarch64-linux-gnu\" -licraft_zg330backend -licraft_hostbackend "
            "-licraft_xrt -licraft_xir -licraft_utils -ldw -ldl -pthread > build.log 2>&1"
        )
        if a.board_library is None:
            status, output = remote.run(client, build, timeout=200, check=False)
            (a.out/"bridge_build.log").write_text(output, encoding="utf-8")
            (a.out/"bridge_build_exit.txt").write_text(str(status)+"\n", encoding="ascii")
            if status:
                raise SystemExit(status)
            library = root+"/libmgs_npu_detailed.so"
        else:
            library = a.board_library
            status, output = remote.run(client, "test -f "+shlex.quote(library), timeout=10, check=False)
            if status:
                raise FileNotFoundError(library)
        command = (
            "cd /root/fpga43dgs_reconstruction/npu_board_v2_20260929/mvsplat && "
            "timeout 90s env "
            "LD_LIBRARY_PATH=/root/fpga43dgs_reconstruction/arm_env/lib:"
            "/root/heterogs_npu/sdk_3.36.1/usr/lib/aarch64-linux-gnu "
            "PYTHONPATH=" + shlex.quote(root + "/mvsplat") + " "
            "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "
            "/root/fpga43dgs_reconstruction/arm_env/bin/python3 "
            +shlex.quote(root+"/known_conv.py")+" run --graphs "+shlex.quote(root+"/graphs")
            +" --bundle "+shlex.quote(root+"/bundle")
            +" --library "+shlex.quote(library)
            +" --out "+shlex.quote(root+"/result")
            +" --repeats "+str(a.repeats))
        status, output = remote.run(client, command, timeout=100, check=False)
        (a.out/"board.log").write_text(output, encoding="utf-8")
        (a.out/"exit.txt").write_text(str(status)+"\n", encoding="ascii")
        try:
            sftp.get(root+"/result/result.json", str(a.out/"result.json"))
        except OSError:
            pass
        for name in json.loads((a.bundle/"manifest.json").read_text())["partitions"]:
            for index in range(a.repeats):
                try:
                    sftp.get(root+"/result/"+name+f"_{index}.npy", str(a.out/(name+f"_{index}.npy")))
                except OSError:
                    pass
        print(output)
        print("Board exit:", status)
        if status:
            raise SystemExit(status)
    finally:
        client.close()


if __name__ == "__main__":
    main()
