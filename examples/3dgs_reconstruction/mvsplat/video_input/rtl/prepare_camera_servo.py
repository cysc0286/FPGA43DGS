"""Copy the original 30TAI platform and add PWM without changing its sources.

Creates a new directory only, verifies every copied file, and records the source
hashes. Building and programming are separate operations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relocate_roms(destination: Path) -> list[dict]:
    """Relocate HLS literal ROM paths; preserve the ROM data and RTL logic."""
    relocations = []
    for rtl in sorted((destination / "rtl/flicker").glob("*.v")):
        data = rtl.read_bytes()
        for match in re.finditer(rb'\$readmem[hb]\s*\(\s*"([^"]+)"', data):
            old = match[1]
            rom = rtl.parent / Path(old.decode("utf-8").replace("\\", "/")).name
            if not rom.is_file():
                raise FileNotFoundError(f"HLS ROM data is missing: {rom}")
            new = rom.as_posix().encode("utf-8")
            if old != new:
                data = data.replace(old, new)
                relocations.append(dict(rtl=rtl.relative_to(destination).as_posix(),
                                        old=old.decode("utf-8"), new=new.decode("utf-8"),
                                        rom_sha256=sha(rom)))
        if sha(rtl) != hashlib.sha256(data).hexdigest():
            rtl.write_bytes(data)
    return relocations


def prepare(source: Path, destination: Path) -> dict:
    source, destination = source.resolve(strict=True), destination.resolve()
    if destination.name != "camera_servo_platform_j17_23_21":
        raise ValueError("Use the explicitly named isolated PWM candidate directory")
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError("Source and candidate must be separate directory trees")
    if destination.exists():
        raise FileExistsError(destination)
    project = source / "fpai_demo_vivado.xpr"
    if not project.is_file():
        raise FileNotFoundError(project)
    root = ET.parse(project).getroot()
    # All design files must resolve inside the tree that will be copied.
    variables = {"$PPRDIR": source, "$PSRCDIR": source / "fpai_demo_vivado.srcs"}
    for entry in root.findall(".//File"):
        name = entry.attrib["Path"]
        for variable, base in variables.items():
            name = name.replace(variable, base.as_posix())
        path = Path(name).resolve()
        if source not in path.parents:
            raise ValueError(f"External design dependency: {path}")
        if not path.is_file():
            container = path.parent.with_suffix(".xcix")
            if path.suffix != ".xci" or not container.is_file():
                raise FileNotFoundError(path)
            with zipfile.ZipFile(container) as archive:
                if not any(Path(n).name == path.name for n in archive.namelist()):
                    raise ValueError(f"Missing XCI in core container: {container}")
    files = sorted(p for p in source.rglob("*") if p.is_file())
    if any(p.is_symlink() or p.is_junction() for p in source.rglob("*")):
        raise ValueError("Linked platform entries require explicit review")
    original = {p.relative_to(source).as_posix(): sha(p) for p in files}
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    for relative, expected in original.items():
        if sha(destination / relative) != expected:
            raise RuntimeError(f"Copy verification failed: {relative}")

    top = destination / "rtl/ai7030_edif_top.v"
    text = top.read_text(encoding="utf-8")
    marker = "adder_top U_adder_top("
    if text.count(marker) != 1 or "camera_servo_bus" in text:
        raise ValueError("Unexpected original top-level layout")
    port_marker = "  output                 PJTAG_TDO"
    if text.count(port_marker) != 1:
        raise ValueError("Expected top-level port insertion point")
    text = text.replace(port_marker,
        "  output                 camera_pan_pwm,\n"
        "  output                 camera_tilt_pwm,\n" + port_marker, 1)
    start = text.index(marker)
    end = text.index("\n);", start) + len("\n);")
    instance = text[start:end]
    names = ["awwaddr", "awwdata", "awwvalid", "awwready", "araddr",
             "arvalid", "arready", "rdata", "rvalid", "rready"]
    for name in names:
        old = "User_Reg_" + name
        if instance.count(old) != 1:
            raise ValueError(f"Unexpected renderer connection: {old}")
        instance = instance.replace(old, "Renderer_Reg_" + name)
    router = """// PWM is routed only at 0x300--0x33f; renderer data paths stay unchanged.
wire [31:0] Renderer_Reg_awwaddr, Renderer_Reg_awwdata;
wire Renderer_Reg_awwvalid, Renderer_Reg_awwready;
wire [31:0] Renderer_Reg_araddr, Renderer_Reg_rdata;
wire Renderer_Reg_arvalid, Renderer_Reg_arready, Renderer_Reg_rvalid, Renderer_Reg_rready;
camera_servo_bus #(.CLOCK_HZ(100000000)) U_camera_servo_bus (
  .clk(gp_clk), .rst_n(~gp_clk_rst),
  .awaddr(User_Reg_awwaddr), .awdata(User_Reg_awwdata),
  .awvalid(User_Reg_awwvalid), .awready(User_Reg_awwready),
  .araddr(User_Reg_araddr), .arvalid(User_Reg_arvalid), .arready(User_Reg_arready),
  .rdata(User_Reg_rdata), .rvalid(User_Reg_rvalid), .rready(User_Reg_rready),
  .downstream_awaddr(Renderer_Reg_awwaddr), .downstream_awdata(Renderer_Reg_awwdata),
  .downstream_awvalid(Renderer_Reg_awwvalid), .downstream_awready(Renderer_Reg_awwready),
  .downstream_araddr(Renderer_Reg_araddr), .downstream_arvalid(Renderer_Reg_arvalid),
  .downstream_arready(Renderer_Reg_arready), .downstream_rdata(Renderer_Reg_rdata),
  .downstream_rvalid(Renderer_Reg_rvalid), .downstream_rready(Renderer_Reg_rready),
  .pan_pwm(camera_pan_pwm), .tilt_pwm(camera_tilt_pwm)
);

"""
    top.write_text(text[:start] + router + instance + text[end:], encoding="utf-8")
    local = Path(__file__).resolve().parent
    additions = {"camera_servo_pwm.v": "rtl/camera_servo_pwm.v",
                 "camera_servo_bus.v": "rtl/camera_servo_bus.v",
                 "camera_servo_j17_23_21.xdc": "constrs_1/camera_servo_j17_23_21.xdc"}
    for name, relative in additions.items():
        shutil.copyfile(local / name, destination / relative)
    candidate_project = destination / project.name
    project_text = candidate_project.read_text(encoding="utf-8")
    project_text, changes = re.subn(r'(<Project\b[^>]*\bPath=")[^"]*(")',
        lambda m: m[1] + candidate_project.as_posix() + m[2], project_text, count=1)
    if changes != 1:
        raise ValueError("Expected project location attribute")
    candidate_project.write_text(project_text, encoding="utf-8")
    rom_relocations = relocate_roms(destination)
    for relative, expected in original.items():
        if sha(source / relative) != expected:
            raise RuntimeError(f"Original changed during preparation: {relative}")
    changed = [p for p, expected in original.items() if sha(destination / p) != expected]
    expected_changes = {project.name, "rtl/ai7030_edif_top.v"} | {r["rtl"] for r in rom_relocations}
    if set(changed) != expected_changes:
        raise RuntimeError(f"Unexpected modifications: {changed}")
    manifest = {"source": str(source), "candidate": str(destination),
                "original_unchanged": True, "source_files": original,
                "modified_files": {p: sha(destination / p) for p in changed},
                "rom_relocations": rom_relocations,
                "added_files": {p: sha(destination / p) for p in additions.values()},
                "board_programmed": False}
    (destination / "servo_preparation.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return {k: v for k, v in manifest.items() if k != "source_files"} | {
        "copied_files": len(original)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(prepare(arguments.source, arguments.destination), indent=2))
