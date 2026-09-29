"""First-backbone-convolution controls for ICraft weight/layout diagnosis."""
import argparse
import json
from pathlib import Path

import numpy as np
from common import new_directory, save, sha


def export(source, real_oracle, out):
    import onnx
    import onnxruntime as ort
    from onnx import numpy_helper

    original = onnx.shape_inference.infer_shapes(onnx.load(str(source)))
    first = original.graph.node[0]
    if first.op_type != "Conv" or list(first.input) != ["features", "inner.conv1.weight"]:
        raise ValueError("Unexpected first backbone operation")
    weights = {tensor.name: numpy_helper.to_array(tensor) for tensor in original.graph.initializer}
    real_weight = np.asarray(weights[first.input[1]], np.float32)
    if real_weight.shape != (64, 3, 7, 7):
        raise ValueError("Unexpected first convolution weight shape")
    with np.load(real_oracle, allow_pickle=False) as payload:
        real_input = np.asarray(payload["input"], np.float32)
    if real_input.shape != (2, 3, 128, 128):
        raise ValueError("Unexpected backbone input shape")

    out = new_directory(out)
    manifest = dict(schema="mvsplat_first_conv_diagnostic_v1", weight_sha256="diagnostic_not_production",
                    input_shape=list(real_input.shape), source_sha256=sha(source),
                    real_oracle_sha256=sha(real_oracle), partitions={})
    synthetic_input = ((np.arange(real_input.size, dtype=np.float32).reshape(real_input.shape) % 31)-15)/32
    synthetic_weight = ((np.arange(real_weight.size, dtype=np.float32).reshape(real_weight.shape) % 11)-5)/128
    for name, x, weight in (("synthetic", synthetic_input, synthetic_weight),
                            ("real_input", real_input, synthetic_weight),
                            ("real_weight_only", synthetic_input, real_weight),
                            ("real_weight", real_input, real_weight)):
        folder = new_directory(out/name)
        graph = onnx.ModelProto()
        graph.CopyFrom(original)
        del graph.graph.node[:]
        graph.graph.node.extend([first])
        del graph.graph.initializer[:]
        graph.graph.initializer.extend([numpy_helper.from_array(weight, first.input[1])])
        del graph.graph.output[:]
        for output in list(original.graph.value_info)+list(original.graph.output):
            if output.name == first.output[0]:
                graph.graph.output.extend([output])
                break
        if not graph.graph.output:
            raise ValueError("Missing inferred first-layer output shape")
        onnx.checker.check_model(graph)
        onnx.save(graph, folder/"model.onnx")
        session = ort.InferenceSession(str(folder/"model.onnx"), providers=["CPUExecutionProvider"])
        expected = session.run(None, {"features": x})[0]
        np.savez(folder/"oracle.npz", input=x, expected=expected, weight=weight)
        manifest["partitions"][name] = dict(complete=True, input_shape=list(x.shape),
            output_shape=list(expected.shape), graph_sha256=sha(folder/"model.onnx"),
            oracle_sha256=sha(folder/"oracle.npz"), exact_required=False,
            weight_kind=name, weight_shape=list(weight.shape))
        print(name, list(x.shape), list(expected.shape), flush=True)
    save(out/"manifest.json", manifest)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--real-oracle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    export(a.source, a.real_oracle, a.out)


if __name__ == "__main__":
    main()
