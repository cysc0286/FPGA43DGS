"""Fetch the unprocessed RGB test video; checksum pins the dataset bytes."""
import hashlib
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URL = "https://raw.githubusercontent.com/magicleap/SuperPointPretrainedNetwork/1fda796addba9b6f8e79d586a3699700a86b1cea/assets/nyu_snippet.mp4"
SHA256 = "0a7e3c684e917dd937ae1ec6c64f770c111b939f08f020b66965c1fccf9ae86a"


def main():
    dst = ROOT/"data/nyu_snippet_curl.mp4"
    dst.parent.mkdir(exist_ok=True)
    if not dst.exists():
        part = dst.with_suffix(".download")
        curl = shutil.which("curl.exe") or shutil.which("curl")
        if not curl:
            raise RuntimeError("curl is required to fetch the test video")
        subprocess.run([curl, "--fail", "--location", "--retry", "2", "--connect-timeout", "15",
                        "--max-time", "180", "--output", str(part), URL], check=True)
        if hashlib.sha256(part.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError("Source video hash changed; do not silently change the baseline")
        part.rename(dst)
    if hashlib.sha256(dst.read_bytes()).hexdigest() != SHA256:
        raise RuntimeError("Existing video hash mismatch")
    print(dst)


if __name__ == "__main__":
    main()
