"""Check whether first-convolution discrepancies follow a simple layout error."""
import argparse
from itertools import permutations
import json
from pathlib import Path

import numpy as np


def metrics(actual, expected):
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
    }


def run(graph, oracle, actual):
    import onnxruntime as ort

    with np.load(oracle, allow_pickle=False) as data:
        x = data["input"]
        reference = data["expected"]
    output = np.load(actual, allow_pickle=False)
    if output.shape != reference.shape or x.shape[1] != 3:
        raise ValueError("Expected the three-channel first convolution and matching output")
    session = ort.InferenceSession(str(graph), providers=["CPUExecutionProvider"])
    channel = {}
    for order in permutations(range(3)):
        prediction = session.run(None, {"features": x[:, order].copy()})[0]
        channel["".join(map(str, order))] = metrics(output, prediction)
    spatial = {}
    for height_shift in range(-2, 3):
        for width_shift in range(-2, 3):
            h0 = max(0, height_shift)
            h1 = min(output.shape[2], output.shape[2] + height_shift)
            w0 = max(0, width_shift)
            w1 = min(output.shape[3], output.shape[3] + width_shift)
            observed = output[:, :, h0:h1, w0:w1]
            target = reference[:, :, h0-height_shift:h1-height_shift,
                               w0-width_shift:w1-width_shift]
            spatial[f"{height_shift},{width_shift}"] = metrics(observed, target)
    return {
        "reference": metrics(output, reference),
        "channel_orders": channel,
        "spatial_shifts": spatial,
        "best_channel_order": min(channel, key=lambda key: channel[key]["rmse"]),
        "best_spatial_shift": min(spatial, key=lambda key: spatial[key]["rmse"]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", type=Path, required=True)
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--actual", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    result = run(a.graph, a.oracle, a.actual)
    a.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("reference", "best_channel_order", "best_spatial_shift")}, indent=2))


if __name__ == "__main__":
    main()
