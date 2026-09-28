#!/usr/bin/env bash
# Install only the verified offline candidate environment, never system Python.
set -euo pipefail
base="${1:-/root/fpga43dgs_reconstruction}"
test "$(uname -m)" = aarch64
export MAMBA_ROOT_PREFIX="$base/arm_mamba"
export MAMBA_EXTRACT_THREADS=1
export MAMBA_DOWNLOAD_THREADS=1
export CONDA_OVERRIDE_GLIBC=2.31
python3 - "$base/arm_packages" <<'PY'
import hashlib, json, pathlib, sys
folder = pathlib.Path(sys.argv[1])
report = json.loads((folder / 'downloads.json').read_text())
for record in report['packages']:
    digest = hashlib.sha256()
    with (folder / record['fn']).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != record['sha256']:
        raise ValueError('Invalid package: ' + record['fn'])
print('Verified', len(report['packages']), 'offline packages')
PY
chmod +x "$base/arm_packages/micromamba"
timeout 1200 "$base/arm_packages/micromamba" create --offline --no-rc --no-pyc -y \
  -p "$base/arm_env" -f "$base/arm_packages/explicit.txt"
"$base/arm_env/bin/python" -m pip install --no-index --find-links "$base/arm_packages" torch==2.7.1
