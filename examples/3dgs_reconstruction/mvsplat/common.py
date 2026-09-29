"""Versioned, file-based MVSplat bridge; no board or training side effects."""
import hashlib
import json
from pathlib import Path

SOURCE_COMMIT = "01f9a28edb5eb68416e7e63b01f8d90c3bdfbf01"
FULL_WEIGHT_SHA256 = "83d0d9eaa753fa4a1f925288dc1f90b8c3297fad0ab0f6ed1f11a1c5946da25a"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def new_directory(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


def load_gaussians(path):
    import numpy as np
    with np.load(path, allow_pickle=False) as f:
        result = {key: f[key] for key in ("means", "covariances", "harmonics", "opacities")}
    return validate_gaussians(result)


def validate_gaussians(result):
    import numpy as np
    n = len(result["means"])
    expected = {"means": (n, 3), "covariances": (n, 3, 3), "harmonics": (n, 3, 25), "opacities": (n,)}
    if not 0 < n <= 1000000:
        raise ValueError("Gaussian count outside renderer ABI")
    for key, shape in expected.items():
        if result[key].shape != shape or not np.isfinite(result[key]).all():
            raise ValueError("Invalid Gaussian field: " + key)
    if np.any(result["opacities"] < 0) or np.any(result["opacities"] > 1):
        raise ValueError("Opacity must already be activated")
    return result
