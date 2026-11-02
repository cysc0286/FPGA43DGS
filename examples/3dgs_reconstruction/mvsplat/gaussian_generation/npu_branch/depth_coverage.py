"""Probe checkpoint-backed depth modules without changing the production graph."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from common import save, sha
from gaussian_generation.runtime import ModelRuntime


MODULES = ("corr_refine_net", "refine_unet")


def profile_cpu(runtime, context_path, out, repeats):
    """Time actual nested stages; inclusive parent times must not be added."""
    from gaussian_generation.npu_branch.catalog import PARTITIONS, locate
    stages = {name: locate(runtime.model, name) for name in PARTITIONS}
    stages.update(backbone=runtime.model.backbone,
                  corr_refine_net=runtime.model.depth_predictor.corr_refine_net,
                  refine_unet=runtime.model.depth_predictor.refine_unet,
                  depth=runtime.model.depth_predictor,
                  gaussian_adapter=runtime.model.gaussian_adapter)
    rounds, hooks, starts, current = [], [], {}, {}
    def begin(name):
        def hook(module, inputs):
            starts[name] = time.perf_counter()
        return hook
    def end(name):
        def hook(module, inputs, output):
            current[name] = time.perf_counter()-starts[name]
        return hook
    for name, module in stages.items():
        hooks.extend((module.register_forward_pre_hook(begin(name)),
                      module.register_forward_hook(end(name))))
    try:
        with np.load(context_path, allow_pickle=False) as data:
            context = {k: runtime.torch.from_numpy(data[k].copy()).unsqueeze(0) for k in data.files}
        for index in range(repeats):
            current.clear()
            start = time.perf_counter()
            with runtime.torch.inference_mode():
                g = runtime.model(context, global_step=300000, deterministic=True)
            seconds = time.perf_counter()-start
            arrays = {k: getattr(g, k).detach().numpy() for k in
                      ('means', 'covariances', 'harmonics', 'opacities')}
            rounds.append(dict(round=index, inference_seconds=seconds, stages_seconds=dict(current),
                finite=all(np.isfinite(v).all() for v in arrays.values()),
                gaussian_hashes={k: hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()}))
            save(out/'cpu_profile.json', dict(npu_executed=False, threads=runtime.threads,
                scope='Actual board CPU reference; nested stage durations are inclusive', rounds=rounds))
            print(json.dumps(rounds[-1]), flush=True)
            del g, arrays
    finally:
        for hook in hooks:
            hook.remove()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--vendor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--profile-only", action="store_true", help="Measure actual CPU stage costs; do not export")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    result = {"weight_sha256": sha(args.weights), "context_sha256": sha(args.context),
              "modules": {}, "npu_executed": False}
    save(args.out / "result.json", result)
    if args.repeats < 1:
        parser.error('At least one repetition required')
    runtime = ModelRuntime(args.weights, args.vendor, args.threads, 16, args.out / "runtime")
    if args.profile_only:
        profile_cpu(runtime, args.context, args.out, args.repeats)
        return
    torch = runtime.torch
    import onnx
    captured = {}
    hooks = []

    def capture(name):
        def hook(module, inputs, output):
            if name in captured or len(inputs) != 1 or not isinstance(output, torch.Tensor):
                raise ValueError("Depth module is not a one-call unary tensor partition: " + name)
            captured[name] = (inputs[0].detach().clone(), output.detach().clone())
        return hook

    for name in MODULES:
        hooks.append(getattr(runtime.model.depth_predictor, name).register_forward_hook(capture(name)))
    try:
        with np.load(args.context, allow_pickle=False) as data:
            context = {key: torch.from_numpy(data[key].copy()).unsqueeze(0) for key in data.files}
        with torch.inference_mode():
            runtime.model(context, global_step=300000, deterministic=True)
    finally:
        for hook in hooks:
            hook.remove()

    for name in MODULES:
        record = {"exported": False, "prefixes": {}}
        result["modules"][name] = record
        try:
            x, expected = captured[name]
            module = getattr(runtime.model.depth_predictor, name).eval()
            record.update(input_shape=list(x.shape), output_shape=list(expected.shape),
                          parameters=sum(p.numel() for p in module.parameters()))
            graph = args.out / (name + ".onnx")
            start = time.perf_counter()
            with torch.inference_mode():
                torch.onnx.export(module, (x,), str(graph), input_names=["features"],
                                  output_names=["result"], opset_version=17, dynamo=False)
            onnx.checker.check_model(str(graph))
            model = onnx.load(str(graph), load_external_data=False)
            ops = {}
            for node in model.graph.node:
                key = node.domain + ":" + node.op_type if node.domain else node.op_type
                ops[key] = ops.get(key, 0) + 1
            np.savez(args.out / (name + "_oracle.npz"), input=x.numpy(), expected=expected.numpy())
            record.update(exported=True, seconds=time.perf_counter()-start,
                          graph_sha256=sha(graph), ops=ops,
                          oracle_sha256=sha(args.out / (name + "_oracle.npz")))
        except Exception as exc:
            record["error"] = type(exc).__name__ + ": " + str(exc)
        finally:
            save(args.out / "result.json", result)
            print(json.dumps({"name": name, **record}), flush=True)
        x, _ = captured[name]
        children = list(getattr(runtime.model.depth_predictor, name).children())
        for end in (1, 3, len(children)):
            if end > len(children):
                continue
            prefix_name = name + "_prefix_" + str(end)
            item = {"exported": False, "end": end,
                    "module_types": [type(child).__name__ for child in children[:end]]}
            record["prefixes"][prefix_name] = item
            try:
                prefix = torch.nn.Sequential(*children[:end]).eval()
                with torch.inference_mode():
                    expected = prefix(x)
                graph = args.out / (prefix_name + ".onnx")
                with torch.inference_mode():
                    torch.onnx.export(prefix, (x,), str(graph), input_names=["features"],
                                      output_names=["result"], opset_version=17, dynamo=False)
                onnx.checker.check_model(str(graph))
                model = onnx.load(str(graph), load_external_data=False)
                ops = {}
                for node in model.graph.node:
                    key = node.domain + ":" + node.op_type if node.domain else node.op_type
                    ops[key] = ops.get(key, 0) + 1
                np.savez(args.out / (prefix_name + "_oracle.npz"), input=x.numpy(),
                         expected=expected.numpy())
                item.update(exported=True, output_shape=list(expected.shape),
                            graph_sha256=sha(graph), ops=ops,
                            oracle_sha256=sha(args.out / (prefix_name + "_oracle.npz")))
            except Exception as exc:
                item["error"] = type(exc).__name__ + ": " + str(exc)
            finally:
                save(args.out / "result.json", result)
                print(json.dumps({"name": prefix_name, **item}), flush=True)


if __name__ == "__main__":
    main()
