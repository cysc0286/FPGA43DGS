"""Two dynamic descriptor inputs; never bake each new video frame into weights."""
import argparse
import hashlib
import json
from pathlib import Path
import onnx
from onnx import TensorProto as T, helper as h


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tile", type=int, choices=[32, 64, 128, 256], default=128)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    t = a.tile
    model = h.make_model(h.make_graph([h.make_node("MatMul", ["left", "right"], ["scores"], name="sift_dynamic_dot")],
        "sift_pair_dot", [h.make_tensor_value_info("left", T.FLOAT, [1, 1, t, 128]),
                          h.make_tensor_value_info("right", T.FLOAT, [1, 1, 128, t])],
        [h.make_tensor_value_info("scores", T.FLOAT, [1, 1, t, t])]),
        producer_name="HeteroGS reconstruction candidate", ir_version=8, opset_imports=[h.make_opsetid("", 11)])
    onnx.checker.check_model(model, full_check=True)
    path = a.output/"sift_pair_dot.onnx"
    onnx.save(model, path)
    manifest = {"schema": "hgs-sift-dot-v1", "tile": t, "descriptor_dim": 128,
                "graph_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "inputs": {"left": [1, 1, t, 128], "right": [1, 1, 128, t]}, "output": [1, 1, t, t],
                "compiler_inputs_nhwc": [[1, t, 128, 1], [1, 128, t, 1]],
                "layout": "ONNX NCHW-shaped matrices, ICraft NHWC host interface; singleton C means identical flat buffers",
                "input_encoding": "COLMAP uint8 divided by 512, represented exactly as FP32; right is transposed",
                "dynamic_inputs": 2, "constant_scene_weights": False, "npu_executed": False,
                "icraft_dynamic_matmul_support": "unverified", "score_atol": 2e-4, "match_indices": "exact",
                "logical_input_bytes_per_call": 2*t*128*4, "logical_output_bytes_per_call": t*t*4}
    (a.output/"manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
