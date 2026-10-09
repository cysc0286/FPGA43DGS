"""Compare generated Gaussians through the same independent CPU renderer."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from common import new_directory, save, load_gaussians, sha
from evaluate import quality, local_error, passes
from reference import render


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--verification", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    out = new_directory(a.out)
    receipt = json.loads((a.verification/"verification.json").read_text())
    if not receipt["complete"]:
        raise ValueError("Numerical verification must pass before image evaluation")
    meta = json.loads((a.input/"input.json").read_text())
    sources = {}
    for kind in ("cpu", "hybrid"):
        if sha(a.verification/kind/"gaussians.npz") != receipt[kind]["gaussian_sha256"]:
            raise ValueError("Gaussian artifact changed")
        if meta["context_sha256"] != receipt[kind]["input_sha256"]:
            raise ValueError("Camera/context provenance mismatch")
        sources[kind] = load_gaussians(a.verification/kind/"gaussians.npz")
    gate = json.loads(Path(__file__).resolve().parents[2].joinpath("acceptance.json").read_text())["reconstruction_vs_rgb"]
    result = dict(backend=receipt["backend"], npu_executed=receipt["npu_executed"], board_renderer_executed=False,
                  scene_preparation_seconds=None, quality_gate=gate, views={},
                  scope="Same independent CPU SH3 renderer; not FPGA or official CUDA output")
    panels = []
    for view in meta["views"]:
        if view["role"] != "target":
            continue
        image_path = a.input/"images"/view["name"]
        if sha(image_path) != view["prepared_sha256"]:
            raise ValueError("Ground truth image changed")
        truth = np.asarray(Image.open(image_path).convert("RGB"), np.float64)/255
        images = {name: np.clip(render(g, view)[0], 0, 1) for name, g in sources.items()}
        record = dict(cpu_vs_rgb=quality(images["cpu"], truth),
                      hybrid_vs_rgb=quality(images["hybrid"], truth),
                      hybrid_vs_cpu=quality(images["hybrid"], images["cpu"]),
                      hybrid_vs_rgb_local_error=local_error(images["hybrid"], truth))
        record["reconstruction_passed"] = passes(record["hybrid_vs_rgb"], gate)
        result["views"][view["name"]] = record
        row = Image.new("RGB", (3*256, 282), "#20242c")
        draw = ImageDraw.Draw(row)
        for i, (label, array) in enumerate((("Video target", truth), ("CPU Gaussian", images["cpu"]),
                                           (receipt["backend"]+" Gaussian", images["hybrid"]))):
            im = Image.fromarray(np.rint(array*255).astype(np.uint8))
            im.save(out/(Path(view["name"]).stem+"_"+str(i)+".png"))
            row.paste(im.resize((256,256)), (256*i,26))
            draw.text((256*i+4,7), label, fill="white")
        panels.append(row)
        save(out/"quality.json", result)
        print(view["name"], json.dumps(record), flush=True)
    if not result["views"]:
        raise ValueError("No target views available for image validation")
    result["all_views_passed"] = all(v["reconstruction_passed"] for v in result["views"].values())
    save(out/"quality.json", result)
    panel = Image.new("RGB", (768,282*len(panels)))
    for i, row in enumerate(panels):
        panel.paste(row, (0,282*i))
    panel.save(out/"comparison.png")


if __name__ == "__main__":
    main()
