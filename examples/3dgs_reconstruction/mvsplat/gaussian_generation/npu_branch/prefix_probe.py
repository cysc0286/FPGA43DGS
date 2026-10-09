"""Export consecutive ONNX backbone prefixes against the same real input."""
import argparse
import json
from pathlib import Path

import numpy as np
from common import new_directory, save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--oracle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ends", type=int, nargs="+", default=[0, 16, 28, 56, 84, 85])
    a = p.parse_args()
    import onnx
    import onnxruntime as ort
    graph = onnx.shape_inference.infer_shapes(onnx.load(a.source))
    with np.load(a.oracle, allow_pickle=False) as oracle:
        x = oracle["input"].copy()
    if len(graph.graph.input) != 1 or graph.graph.input[0].name != "features":
        raise ValueError("Expected original unary backbone ONNX graph")
    shapes = {value.name: value for value in list(graph.graph.value_info)+list(graph.graph.output)}
    out = new_directory(a.out)
    manifest = dict(schema="mvsplat_prefix_diagnostic_v1", weight_sha256="known_not_mvsplat",
                    input_shape=list(x.shape), source_sha256=sha(a.source),
                    oracle_sha256=sha(a.oracle), partitions={})
    for end in a.ends:
        if end < 0 or end >= len(graph.graph.node) or end > 85:
            raise ValueError("Invalid source node index")
        name = "prefix_"+str(end).zfill(2)
        folder = new_directory(out/name)
        subgraph = onnx.ModelProto()
        subgraph.CopyFrom(graph)
        del subgraph.graph.node[:]
        subgraph.graph.node.extend(graph.graph.node[:end+1])
        used = {value for node in subgraph.graph.node for value in node.input}
        initializers = [value for value in subgraph.graph.initializer if value.name in used]
        del subgraph.graph.initializer[:]
        subgraph.graph.initializer.extend(initializers)
        del subgraph.graph.output[:]
        output = subgraph.graph.node[-1].output[0]
        subgraph.graph.output.extend([shapes[output]])
        onnx.checker.check_model(subgraph)
        onnx.save(subgraph, folder/"model.onnx")
        session = ort.InferenceSession(str(folder/"model.onnx"),
            providers=["CPUExecutionProvider"])
        expected = session.run(None, {"features":x})[0]
        np.savez(folder/"oracle.npz", input=x, expected=expected)
        manifest["partitions"][name] = dict(complete=True,
            input_shape=list(x.shape), output_shape=list(expected.shape),
            graph_sha256=sha(folder/"model.onnx"),
            oracle_sha256=sha(folder/"oracle.npz"), exact_required=False,
            source_end_node=end, source_end_op=graph.graph.node[end].op_type,
            source_end_output=output)
        print(name, end, graph.graph.node[end].op_type, list(expected.shape), flush=True)
    save(out/"manifest.json", manifest)


if __name__ == "__main__":
    main()
