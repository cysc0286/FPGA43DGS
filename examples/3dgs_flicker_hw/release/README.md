# 3DGS Renderer v1 — 30TAI Lite

This package freezes the verified CPU+FPGA rendering stage. Input: a trained GraphDECO Gaussian PLY and camera pose/intrinsics. Output: RGB framebuffer. Video decoding, camera-pose estimation and Gaussian optimization are not included. Training is a separate upstream module.

## Run on the board

Requirements: the currently verified FLK1 ABI2 split-II16 firmware, ARM64 Linux, Python 3.8+, and the installed ICraft 3.36.1 ARM runtime. SDK provides transport only; no NPU model is executed. Default SDK root is `/root/heterogs_npu/sdk_3.36.1/usr`; set `ICRAFT_SDK_ROOT` for another compatible installation.

```sh
tar -xzf 3dgs_renderer_v1_20260928.tar.gz
cd 3dgs_renderer_v1_20260928
python3 render.py --verify-only
python3 render.py --camera data/v0.bin --out /root/gs-render-v0
python3 render.py --camera data/v10.bin --out /root/gs-render-v10
```

Each output directory must be new. The package checks SHA256 before execution, then renders exactly one frame. Intermediate files use `/dev/shm` and are removed when the invocation exits. Outputs are `frame.bin` (float RGB/T and last contributor), `frame.ppm` (viewable RGB), `frame_timing.csv`, `run.log` and `result.json`. Timing excludes checksum verification, final output copy and PPM conversion; it includes attribute/group/render child processes, model read and SDK setup. Use `--backend cpu_dense` or `--backend cpu_base` for the frozen four-thread CPU renderer.

The default model is official `train`, iteration 7000, 559,263 Gaussians. Both bundled cameras render 320×178. Other models/cameras must satisfy [INTERFACE.md](INTERFACE.md); only the bundled model/two views have full recorded acceptance. `pack_camera.py` produces the binary camera input from a GraphDECO `cameras.json` list.

## Frozen baseline (2026-09-27/28)

One warmup and three timed whole-pipeline samples per backend; same board CPU-generated input, tmpfs intermediates, no display/SSH time:

| Backend | View 0 | View 10 |
|---|---:|---:|
| CPU base4 | 8.250 s | 6.929 s |
| CPU Dense4 | 4.569 s | 3.917 s |
| CPU+FPGA Dense | 3.164 s | 2.962 s |

FPGA/CPU Dense speedups are 1.444x/1.322x. CPU FP32 and FPGA FP16/AABB differ, so these are system comparisons, not equal-numerics hardware-only gains. Matching HLS FP16 references and repeat outputs are byte-exact. Against official CUDA renders: PSNR 46.2075/47.9167 dB and SSIM 0.9982361/0.9989669; official FP32 strict equality gates fail. Quality is against rendered images, not held-out photographs. LPIPS/power unmeasured. Three samples do not establish tail latency or video throughput. Raw records and readback comparison are in `baseline/`.

## Hardware/source boundary

Firmware SHA256: `9918f9b69525bad88d8a9f06c5ab0940582df341ca32edc786d4377d3ba137cc`. `firmware/BOOT.bin` is a frozen recovery artifact; the runner does not write the boot partition. The current board already runs this version. Keep the existing verified backup/install procedure for firmware changes.

Whole-board LUT/FF/DSP/BRAM: 78.08%/62.57%/65.00%/75.66%; Slice 99.99%. New 200 MHz path setup/hold +0.310/+0.054 ns. Existing vendor AI pulse-width violations and I/O/CDC limits remain; this is not unconditional whole-board signoff.

`src/cpu` contains the exact compiled sources; `rebuild_cpu.sh` writes into a separate `rebuilt` folder. `src/hls` and `src/rtl` retain the installed renderer's HLS/custom RTL sources. A full vendor FPGA rebuild additionally needs the existing licensed 30TAI Lite platform, Vivado 2018.3, Procise, IP and pin constraints; those installations are not embedded here. `hardware_reports/` and `provenance/` bind the package to that build.

Original GraphDECO formulas/source derivatives remain under [LICENSE-GraphDECO.md](LICENSE-GraphDECO.md); vendor artifacts retain their upstream terms. This is a local engineering package, not a claim of complete FLICKER replication or a complete video reconstruction product.
