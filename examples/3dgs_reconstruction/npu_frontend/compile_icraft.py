"""Offline compiler entry with an explicit plan/execute boundary and failure logs."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--graph", required=True, type=Path, help="Directory produced by export_graph.py")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--compiler", type=Path, default=ROOT.parents[2]/"npu_3dgs/.vendor/icraft-3.36.1/bin")
    p.add_argument("--dtype", choices=["fp16", "tf32", "fp32"], default="tf32")
    p.add_argument("--execute", action="store_true", help="Actually run the PC compiler; never accesses a board")
    a = p.parse_args()
    meta = json.loads((a.graph/"manifest.json").read_text())
    model = (a.graph/"sift_pair_dot.onnx").resolve()
    if hashlib.sha256(model.read_bytes()).hexdigest() != meta["graph_sha256"]:
        raise ValueError("Graph hash differs from manifest")
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    compiled = a.output/"compiled"
    compiled.mkdir()
    t, name = meta["tile"], "sift_pair_dot"
    commands = [("parse", ["--net_name", name, "--network", model, "--jr_path", compiled,
        "--framework", "onnx", "--target", "zhuge", "--inputs", f"1,{t},128,1;1,128,{t},1",
        "--inputs_layout", "NHWC;NHWC", "--inputs_dtype", "FP32;FP32", "--pre_method", "nop;nop",
        "--pre_scale", "1;1", "--pre_mean", "0;0", "--channel_swap", "0;0"])]
    for stage, source in [("optimize", "parsed"), ("quantize", "optimized"), ("adapt", "quantized"), ("generate", "adapted")]:
        args = ["--json", compiled/(name+"_"+source+".json"), "--raw", compiled/(name+"_"+source+".raw"), "--jr_path", compiled]
        if stage != "generate":
            args += ["--target", "zhuge"]
        if stage == "quantize":
            args += ["--qdtype", a.dtype, "--no_transinput", "true", "--no_imagemake", "true"]
        commands.append((stage, args))
    plan = {"graph": meta, "compiler": str(a.compiler.resolve()), "dtype": a.dtype, "npu_executed": False,
            "status": "plan_only", "commands": {s: [str(a.compiler.resolve()/("icraft-"+s+".exe"))]+list(map(str, args)) for s, args in commands}, "stages": []}
    result_path = a.output/"result.json"
    def save():
        result_path.write_text(json.dumps(plan, indent=2))
    save()
    if not a.execute:
        print(result_path)
        return
    env = os.environ.copy()
    env["PATH"] = str(a.compiler.resolve())+os.pathsep+env.get("PATH", "")
    try:
        for stage, cmd in plan["commands"].items():
            plan["status"] = "running_"+stage
            save()
            with (a.output/(stage+".log")).open("wb") as log:
                run = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=180, cwd=a.output)
            plan["stages"].append({"stage": stage, "exit": run.returncode,
                                   "compiler_sha256": hashlib.sha256(Path(cmd[0]).read_bytes()).hexdigest()})
            if run.returncode:
                raise RuntimeError("ICraft "+stage+" failed; inspect "+str(a.output/(stage+".log")))
        plan["status"] = "compiled_not_executed"
        plan["artifacts"] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in compiled.iterdir() if f.is_file()}
        from audit_graph import audit
        plan["placement_audit"] = audit(compiled/(name+"_ZG.json"), compiled/(name+"_ZG.raw"), t)
    except Exception as exc:
        plan["status"] = "compile_failed"
        plan["error"] = str(exc)
        raise
    finally:
        save()
    print(result_path)


if __name__ == "__main__":
    main()
