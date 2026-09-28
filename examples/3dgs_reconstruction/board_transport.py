"""Pinned-host deployment helper; credentials are supplied only through environment."""
import argparse
import hashlib
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/"3dgs_compositor/board"))
import remote


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    run.add_argument("--script", type=Path, required=True)
    run.add_argument("--log", type=Path, required=True)
    run.add_argument("--timeout", type=int, default=120)
    put = sub.add_parser("put")
    put.add_argument("--local", type=Path, required=True)
    put.add_argument("--remote", required=True)
    get = sub.add_parser("get")
    get.add_argument("--remote", required=True)
    get.add_argument("--local", type=Path, required=True)
    a = p.parse_args()
    client = remote.connect()
    try:
        if a.action == "run":
            if a.log.exists():
                raise ValueError("Log already exists")
            a.log.parent.mkdir(parents=True, exist_ok=True)
            code, output = remote.run(client, a.script.read_text(encoding="utf-8"), timeout=a.timeout,
                                      log=a.log, check=False)
            a.log.with_suffix(a.log.suffix+".json").write_text(json.dumps({"exit_code": code,
                "script_sha256": hashlib.sha256(a.script.read_bytes()).hexdigest(), "host": remote.HOST}))
            print(output[-7000:])
            raise SystemExit(code)
        sftp = client.open_sftp()
        if a.action == "put":
            sftp.put(str(a.local), a.remote)
            _, output = remote.run(client, "sha256sum "+shlex.quote(a.remote))
            if output.split()[0] != hashlib.sha256(a.local.read_bytes()).hexdigest():
                raise ValueError("Upload hash mismatch")
            print("Uploaded and verified", a.remote)
        else:
            if a.local.exists():
                raise ValueError("Download destination exists")
            a.local.parent.mkdir(parents=True, exist_ok=True)
            _, output = remote.run(client, "sha256sum "+shlex.quote(a.remote))
            sftp.get(a.remote, str(a.local))
            if output.split()[0] != hashlib.sha256(a.local.read_bytes()).hexdigest():
                raise ValueError("Download hash mismatch")
            print("Downloaded and verified", a.local)
        sftp.close()
    finally:
        client.close()


if __name__ == "__main__":
    main()
