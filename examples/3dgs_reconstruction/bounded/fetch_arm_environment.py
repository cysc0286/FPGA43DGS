"""Download a solved ARM conda environment on the connected PC, checking hashes.

No system installation, GPU use or board access. The explicit list is consumed
on the board with micromamba --offline in an isolated prefix.
"""
import argparse
import concurrent.futures
import hashlib
import json
import tarfile
import time
import urllib.request
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--board-cache", required=True)
    a = p.parse_args()
    lock = json.loads(a.lock.read_text(encoding="utf-8-sig"))
    if not lock.get("success"):
        raise ValueError("Dependency solve did not succeed")
    packages = lock["actions"]["LINK"]
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    def fetch(record):
        target = out/record["fn"]
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == record["sha256"]:
            return record["fn"]
        part = target.with_suffix(target.suffix+".part")
        for attempt in range(3):
            try:
                digest = hashlib.sha256()
                with urllib.request.urlopen(record["url"], timeout=25) as response, part.open("wb") as f:
                    while True:
                        chunk = response.read(1024*1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        digest.update(chunk)
                if digest.hexdigest() != record["sha256"]:
                    raise ValueError("Downloaded hash differs: "+record["fn"])
                part.replace(target)
                return record["fn"]
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1)
    done, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fetch, r): r for r in packages}
        for future in concurrent.futures.as_completed(futures):
            try:
                done.append(future.result())
            except Exception as exc:
                failures.append({"file": futures[future]["fn"], "error": str(exc)})
            if (len(done)+len(failures)) % 10 == 0:
                print("Downloaded", len(done), "/", len(packages), "failures", len(failures), flush=True)
    report = {"downloaded": len(done), "expected": len(packages), "failures": failures,
              "packages": [{k: r[k] for k in ("name", "version", "fn", "url", "sha256", "size")} for r in packages]}
    (out/"downloads.json").write_text(json.dumps(report, indent=2))
    if failures:
        raise RuntimeError("Downloads incomplete; rerun reuses verified packages")
    (out/"explicit.txt").write_text("@EXPLICIT\n"+"".join("file://"+a.board_cache.rstrip("/")+"/"+r["fn"]+"#"+r["md5"]+"\n" for r in packages))
    micromamba = next(r for r in packages if r["name"] == "micromamba")
    with tarfile.open(out/micromamba["fn"]) as archive:
        (out/"micromamba").write_bytes(archive.extractfile("bin/micromamba").read())
    print("Verified environment downloads ready", flush=True)


if __name__ == "__main__":
    main()
