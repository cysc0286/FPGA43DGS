"""Compare first-convolution NPU output against explicit precision controls."""
import argparse
import json
from pathlib import Path

import numpy as np


def tf32_round(array):
    value = np.asarray(array, dtype=np.float32)
    bits = value.view(np.uint32).copy()
    # Keep the sign, exponent and ten explicit mantissa bits. Round ties to even.
    bits += np.uint32(0xFFF) + ((bits >> 13) & np.uint32(1))
    bits &= np.uint32(0xFFFFE000)
    return bits.view(np.float32)


def compare(actual, expected):
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    limit = 2e-4 + 2e-3 * np.abs(expected.astype(np.float64))
    return dict(max_abs=float(delta.max()), rmse=float(np.sqrt(np.mean(delta * delta))),
                mean_abs=float(delta.mean()), outside_percent=float(100 * np.mean(delta > limit)))


def run(root):
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper

    out = {}
    for name in ("synthetic", "real_input", "real_weight_only", "real_weight"):
        with np.load(root / (name + "_oracle.npz"), allow_pickle=False) as oracle:
            original_input = oracle["input"]
            original_weight = oracle["weight"]
            reference = oracle["expected"]
        actual = np.load(root / (name + "_0.npy"), allow_pickle=False)
        if actual.shape != reference.shape:
            raise ValueError("First-convolution output shape mismatch")
        cases = {}
        for weight_mode in ("fp32", "fp16", "tf32"):
            weight = original_weight if weight_mode == "fp32" else (
                original_weight.astype(np.float16).astype(np.float32)
                if weight_mode == "fp16" else tf32_round(original_weight))
            graph = onnx.load(str(root / "first_conv.onnx"))
            for index, initializer in enumerate(graph.graph.initializer):
                if initializer.name == "inner.conv1.weight":
                    graph.graph.initializer[index].CopyFrom(
                        numpy_helper.from_array(weight, initializer.name))
                    break
            else:
                raise ValueError("First-convolution weight initializer missing")
            session = ort.InferenceSession(graph.SerializeToString(),
                                           providers=["CPUExecutionProvider"])
            for input_mode in ("fp32", "fp16", "tf32"):
                value = original_input if input_mode == "fp32" else (
                    original_input.astype(np.float16).astype(np.float32)
                    if input_mode == "fp16" else tf32_round(original_input))
                expected = session.run(None, {"features": value})[0]
                for output_mode in ("fp32", "fp16"):
                    candidate = expected if output_mode == "fp32" else expected.astype(np.float16).astype(np.float32)
                    cases[f"{input_mode}/{weight_mode}/{output_mode}"] = compare(actual, candidate)
        ranked = sorted(cases.items(), key=lambda item: item[1]["rmse"])
        out[name] = dict(npu_against_reference=compare(actual, reference),
                         npu_output_fp16_exact=bool(np.array_equal(actual, actual.astype(np.float16).astype(np.float32))),
                         best=[dict(mode=mode, **metrics) for mode, metrics in ranked[:6]])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.data)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
