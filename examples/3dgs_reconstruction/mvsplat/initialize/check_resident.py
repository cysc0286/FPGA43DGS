"""Local native check: cached Gaussian projection equals the frozen executable."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from common import new_directory, save, sha


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--renderer", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--cameras", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    out = new_directory(a.out.resolve())
    source = a.renderer.resolve()/"src/cpu/attributes.cpp"
    candidate = Path(__file__).with_name("attributes_resident.cpp")
    env = dict(os.environ)
    if os.name == "nt":
        setup = Path(__file__).resolve().parents[2]/"vendor/export_env.bat"
        if setup.is_file():
            lines = subprocess.check_output(["cmd.exe", "/d", "/c", str(setup)], text=True)
            env.update(line.split("=", 1) for line in lines.splitlines() if "=" in line and not line.startswith("="))
        env = {k.upper():v for k,v in env.items()}
        compiler = shutil.which("cl.exe", path=env["PATH"])
        if not compiler:
            raise ValueError("MSVC missing; run from an x64 Visual Studio Developer shell")
        base = [compiler, "/nologo", "/std:c++17", "/EHsc", "/O2", "/fp:strict", "/MD", "/I"+str(source.parent)]
    else:
        base = ["g++", "-std=c++17", "-O3", "-ffp-contract=off", "-I"+str(source.parent)]
    executables = []
    for name, src in (("frozen_attributes", source), ("resident_attributes", candidate)):
        binary = out/(name+(".exe" if os.name == "nt" else ""))
        args = base+[str(src)]+(["/Fe"+str(binary), "/Fo"+str(out/(name+".obj"))]
                              if os.name == "nt" else ["-o", str(binary)])
        with (out/(name+".build.log")).open("wb") as log:
            subprocess.run(args, env=env, cwd=out, check=True, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        executables.append(binary)
    # Two views plus the first again exercise view changes without a new LOAD.
    cameras = a.cameras + a.cameras[:1]
    commands = ["LOAD "+json.dumps(str(a.model.resolve()))]
    for i, camera in enumerate(cameras):
        subprocess.run([str(executables[0]), str(a.model.resolve()), str(camera.resolve()), str(out/f"reference_{i}.bin")],
                       check=True, capture_output=True, timeout=120)
        commands.append("PROJECT "+json.dumps(str(camera.resolve()))+" "+json.dumps(str(out/f"cached_{i}.bin")))
    commands.append("QUIT")
    result = subprocess.run([str(executables[1])], input="\n".join(commands)+"\n", text=True,
                            capture_output=True, timeout=180, check=True)
    (out/"protocol.txt").write_text(result.stdout, encoding="utf-8")
    if result.stdout.count("SCENE_LOADED") != 1 or result.stdout.count("PROJECT_COMPLETE") != len(cameras):
        raise ValueError("Unexpected resident projection protocol")
    checks = [dict(camera=str(cam), reference_sha256=sha(out/f"reference_{i}.bin"),
                   resident_sha256=sha(out/f"cached_{i}.bin")) for i, cam in enumerate(cameras)]
    passed = all(x["reference_sha256"] == x["resident_sha256"] for x in checks)
    save(out/"result.json", dict(complete=passed, board_executed=False, model_sha256=sha(a.model),
        frozen_source_sha256=sha(source), candidate_source_sha256=sha(candidate),
        loads=1, projections=len(cameras), checks=checks))
    if not passed:
        raise ValueError("Cached projections differ from frozen executable")
    print(json.dumps(dict(complete=passed, loads=1, projections=len(cameras), board_executed=False)))


if __name__ == "__main__":
    main()
