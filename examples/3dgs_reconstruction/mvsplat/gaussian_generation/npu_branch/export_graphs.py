"""Export real, checkpoint-backed MVSplat partitions and validate on host CPU.

No synthetic inputs, retraining, hardware access, or silent ONNX fallback.
"""
import argparse
import json
from pathlib import Path
import time
import numpy as np

from common import new_directory, save, sha
from gaussian_generation.runtime import ModelRuntime
from gaussian_generation.npu_branch.catalog import PARTITIONS, locate


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--vendor", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--partitions", nargs="+", choices=list(PARTITIONS), default=list(PARTITIONS))
    a = p.parse_args(argv)
    out = new_directory(a.out)
    runtime = ModelRuntime(a.weights, a.vendor, 2, 16, out / "runtime")
    torch = runtime.torch
    import onnx
    import onnxruntime as ort
    samples, hooks = {}, []
    def capture(name):
        def hook(module, args, value):
            if name in samples or len(args) != 1:
                raise ValueError("Partition is not called exactly once with a unary input: " + name)
            sequence = isinstance(value, (list, tuple))
            if sequence and len(value) != 1:
                raise ValueError("Multiple outputs need an explicit ABI")
            tensor = value[0] if sequence else value
            if args[0].ndim != 4 or tensor.ndim != 4:
                raise ValueError("Only static 4D partitions are supported")
            samples[name] = (args[0].detach().clone(), tensor.detach().clone(), sequence)
        return hook
    for name in a.partitions:
        hooks.append(locate(runtime.model, name).register_forward_hook(capture(name)))
    try:
        reference = runtime.infer(a.input, out / "reference", sha(a.input / "context.npz"))
    finally:
        for hook in hooks:
            hook.remove()
    class Unary(torch.nn.Module):
        def __init__(self, module, sequence):
            super().__init__()
            self.inner, self.sequence = module, sequence
        def forward(self, x):
            result = self.inner(x)
            return result[0] if self.sequence else result
    manifest = dict(schema="mvsplat_partitions_v1", npu_executed=False,
                    weight_sha256=sha(a.weights), context_sha256=reference["input_sha256"],
                    source_commit=reference["source_commit"], input_shape=reference["input_shape"],
                    torch_version=torch.__version__, onnx_version=onnx.__version__,
                    onnxruntime_version=ort.__version__, partitions={})
    for name in a.partitions:
        folder = new_directory(out / name)
        item = dict(path=PARTITIONS[name], complete=False)
        try:
            x, expected, sequence = samples[name]
            graph = folder / "model.onnx"
            module = Unary(locate(runtime.model, name), sequence).eval()
            with torch.inference_mode():
                torch.onnx.export(module, (x,), str(graph), input_names=["features"],
                                  output_names=["result"], opset_version=17, dynamo=False)
            onnx.checker.check_model(str(graph))
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            session = ort.InferenceSession(str(graph), sess_options=options, providers=["CPUExecutionProvider"])
            t = time.perf_counter()
            actual = session.run(None, {"features": x.numpy()})[0]
            seconds = time.perf_counter()-t
            error = np.abs(actual-expected.numpy())
            passed = bool(np.allclose(actual, expected.numpy(), rtol=2e-3, atol=2e-4))
            np.savez(folder / "oracle.npz", input=x.numpy(), expected=expected.numpy())
            item.update(complete=passed, input_shape=list(x.shape), output_shape=list(expected.shape),
                        returns_list=sequence, logical_layout="NCHW", dtype="float32",
                        graph_sha256=sha(graph), oracle_sha256=sha(folder / "oracle.npz"),
                        parameters=sum(v.numel() for v in module.parameters()),
                        host_onnx_seconds=seconds, max_abs_error=float(error.max()),
                        mean_abs_error=float(error.mean()), host_onnx_passed=passed,
                        tolerance=dict(rtol=2e-3, atol=2e-4), npu_executed=False)
        except Exception as exc:
            item["error"] = type(exc).__name__+": "+str(exc)
        manifest["partitions"][name] = item
        save(out / "manifest.json", manifest)
        print(json.dumps(dict(name=name, **item)), flush=True)
    if not all(v["complete"] for v in manifest["partitions"].values()):
        raise RuntimeError("One or more partition exports failed; see manifest.json")


if __name__ == "__main__":
    main()
