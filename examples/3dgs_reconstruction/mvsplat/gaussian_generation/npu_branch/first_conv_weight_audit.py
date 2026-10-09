"""Audit ICraft's quantized first-convolution weight layout against ONNX."""
import argparse
import json
from pathlib import Path

import numpy as np


def audit(root):
    root = Path(root)
    metadata = json.loads((root / "real_weight_quantized.json").read_text(encoding="utf-8"))
    conv = next(op for op in metadata["ops"] if op["_type_key"] == "icraft::xir::Conv2d")
    parameter = conv["inputs"][1]
    dtype = parameter["dtype"]
    if (dtype["layout"] != "@layout(HWIO)" or dtype["shape"] != [7, 7, 3, 64]
            or dtype["element_dtype"]["storage_dtype"] != "@tf(32)"):
        raise ValueError("Unexpected ICraft quantized weight contract")

    payload = (root / "real_weight_quantized.raw").read_bytes()
    header_size = 28
    packed_shape = (7, 7, 3, 4, 16)
    expected_size = header_size + int(np.prod(packed_shape)) * 4
    if len(payload) != expected_size or metadata["params_bytes"] != expected_size:
        raise ValueError("Unexpected quantized weight payload length")
    if payload[:16] != bytes.fromhex("a54943524146545f524157a501000000"):
        raise ValueError("Unexpected ICraft raw header")

    packed = np.frombuffer(payload, dtype="<f4", offset=header_size).reshape(packed_shape)
    decoded = packed.transpose(3, 4, 2, 0, 1).reshape(64, 3, 7, 7)
    with np.load(root / "real_weight_oracle.npz", allow_pickle=False) as oracle:
        original = np.asarray(oracle["weight"], dtype=np.float32)
    if original.shape != decoded.shape:
        raise ValueError("Decoded and ONNX weight shapes differ")
    rounded = original.astype(np.float16).astype(np.float32)
    delta_original = np.abs(decoded.astype(np.float64) - original.astype(np.float64))
    delta_rounded = np.abs(decoded.astype(np.float64) - rounded.astype(np.float64))
    return {
        "scope": "ICraft quantized intermediate file, not uploaded PLDDR state",
        "layout": parameter["dtype"]["layout"],
        "packed_layout": "HWIO with output channels in groups of 16",
        "weight_shape": list(original.shape),
        "file_bytes": len(payload),
        "max_abs_vs_onnx": float(delta_original.max()),
        "mean_abs_vs_onnx": float(delta_original.mean()),
        "rmse_vs_onnx": float(np.sqrt(np.mean(delta_original ** 2))),
        "max_abs_vs_fp16_rounded": float(delta_rounded.max()),
        "exact_fraction_vs_fp16_rounded": float(np.mean(decoded == rounded)),
        "all_finite": bool(np.all(np.isfinite(decoded))),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
