"""Back up four completed, generated diagnostics before reclaiming SD space."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import sys
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    a = p.parse_args()
    if a.out.exists() or a.receipt.exists():
        raise ValueError("Use fresh archive and receipt paths")
    base = "/root/fpga43dgs_reconstruction"
    names = ["npu_cost_probe_20260930", "depth_coverage_20260930",
             "depth_coverage_20260930_v2", "npu_backbone_prefix_20260930"]
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]/"3dgs_compositor/board"))
    import remote
    client = remote.connect()
    try:
        # Generated directories only; reject symlinks and any changed parent.
        paths = [base+"/"+name for name in names]
        for path in paths:
            status, result = remote.run(client, "test -d " + shlex.quote(path) +
                " && test ! -L " + shlex.quote(path) + " && realpath " + shlex.quote(path))
            if result.strip() != path:
                raise ValueError("Unexpected archive root")
        _, running = remote.run(client, "ps -eo args")
        if any(name in running for name in names):
            raise RuntimeError("A process still references a diagnostic directory")
        joined = " ".join(shlex.quote(name) for name in names)
        # Hash every regular file before transfer. This map is compared to the
        # contents of the resulting local archive, before any remote removal.
        hash_command = "cd " + shlex.quote(base) + " && find " + joined + " -type f -exec sha256sum {} +"
        _, hashes = remote.run(client, hash_command, timeout=60)
        expected = {line[66:]: line[:64] for line in hashes.splitlines()}
        a.out.parent.mkdir(parents=True, exist_ok=True)
        transport = client.get_transport()
        channel = transport.open_session(timeout=10)
        channel.exec_command("cd " + shlex.quote(base) + " && tar -czf - " + joined)
        try:
            channel.settimeout(60)
            with a.out.open("xb") as stream:
                while True:
                    block = channel.recv(1024*1024)
                    if not block:
                        break
                    stream.write(block)
            errors = channel.makefile_stderr().read().decode("utf-8", "replace")
            if channel.recv_exit_status():
                raise RuntimeError("Board archive failed: "+errors)
        finally:
            channel.close()
        actual = {}
        with tarfile.open(a.out, "r:gz") as archive:
            for item in archive:
                if item.isfile():
                    digest = hashlib.sha256()
                    with archive.extractfile(item) as stream:
                        for block in iter(lambda: stream.read(1024*1024), b""):
                            digest.update(block)
                    actual[item.name] = digest.hexdigest()
        if actual != expected:
            raise ValueError("Archive contents or hashes differ; board copies retained")
        # A second read protects against a completed test being restarted.
        _, running = remote.run(client, "ps -eo args")
        if any(name in running for name in names):
            raise RuntimeError("Diagnostic restarted; board copies retained")
        _, after = remote.run(client, hash_command, timeout=60)
        if {line[66:]:line[:64] for line in after.splitlines()} != expected:
            raise ValueError("Diagnostic files changed; board copies retained")
        a.receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt = dict(archive=str(a.out.resolve()), files=expected, roots=paths,
                       per_file_hashes_verified=True, remote_removed=False)
        a.receipt.write_text(json.dumps(receipt, indent=2)+"\n")
        for path in paths:
            # All values above are a fixed allowlist and their realpaths were
            # checked; no shell glob or dynamically discovered deletion target.
            remote.run(client, "rm -rf -- " + shlex.quote(path), timeout=30)
        receipt["remote_removed"] = True
        a.receipt.write_text(json.dumps(receipt, indent=2)+"\n")
        print(remote.run(client, "df -h /")[1])
        print("Verified archive:", a.out, "files:", len(actual))
    finally:
        client.close()


if __name__ == "__main__":
    main()
