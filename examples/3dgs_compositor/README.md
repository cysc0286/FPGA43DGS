# Ordered 3DGS alpha compositor: first PL migration

Scope frozen on 2026-09-25: migrate ordered RGB/transmittance compositing only.
Projection, covariance, sorting and Gaussian exp evaluation remain outside PL.
The input replay is prepared from the existing official 16x16 tile, not a full
on-board renderer. No training, NPU or WSR is involved.

## Acceptance and metrics (fixed before implementation results)

1. RTL and board output must match the independent integer oracle exactly for
   every RGB, T, last-contributor and status field. Test order, alpha rejection,
   early termination, background once, reset, malformed commands, sequence
   commit and read-response backpressure. A failed branch/ID is never hidden by
   an image-quality average.
2. Real-tile output versus the frozen official CUDA output retains the CPU
   baseline limits: RGB max absolute <= 1e-4, RGB mean absolute <= 1e-5,
   T max absolute <= 1e-5, last contributor exact. This is a separate quantized
   implementation comparison, not a change to the CPU baseline.
3. Report PL command cycles and implemented clock, command service time,
   CPU same-stage replay time, and host-observed PL replay time including register
   writes/polling/readback. Parsing/file IO are excluded and stated separately.
   The current register transport may be slower than CPU; record that result.
   A cycle estimate is not measured board time or full-render FPS.
4. Report LUT/FF/DSP/BRAM, setup WNS/TNS, hold WHS and pulse-width WPWS for the
   implemented whole design, plus paths through this module. A generated bitstream
   with inherited timing violations is not timing acceptance.
5. Later full-frame tests will measure camera-submit to framebuffer-complete
   latency (mean/P95/P99), sustained FPS, quality PSNR/SSIM/LPIPS, memory/DDR
   bandwidth and measured power/energy per frame. They are not this tile test's
   measured results. Model, camera, precision, resolution and workload must match
   when reporting speedup.

## Numeric and register contract v1 (GSC1)

Unsigned T and alpha use 30 fractional bits (ONE=1073741824). RGB uses
24 fractional bits, with supported inputs 0..16 inclusive; out-of-range input
returns an error without changing pixel state. All multiplications use a full
64-bit product and round-half-up. There is no silent saturation or wraparound.

- CLEAR (opcode 0): C=0, T=ONE, last=seen=0, terminated=finished=false.
- ACCUM (1): require increasing positive ordinal, alpha<=ONE, RGB<=16.
  Clamp alpha to round(0.99*ONE)=1063004406. Skip alpha<ceil(ONE/255)=4210753.
  weight=round(T*alpha/ONE), nextT=T-weight. If nextT<ceil(0.0001*ONE)=107375,
  set terminated and reject that contribution, retaining previous C/T/last.
  Otherwise C += round(RGB*weight/ONE), T=nextT, last=ordinal.
  Subsequent contributions after termination do not change C/T/last.
- BACKGROUND (2): once only, C += round(background_RGB*T/ONE), set finished.
  Reject a second BACKGROUND or ACCUM after finished. CLEAR opens a new pixel.
- Other opcodes: error without changing pixel state.

RFU byte offsets relative to 0x400c0000, request sequence committed last:

| Write | 0x0c | 0x10 | 0x14 | 0x20 | 0x24 | 0x28 | 0x2c |
|---|---|---|---|---|---|---|---|
| Field | alpha | R | G | B | ordinal | opcode | request sequence |

| Read | 0x84 | 0x88 | 0x8c | 0x90 | 0x94 | 0x98 | 0x9c | 0xa0 | 0xa4 | 0xa8 |
|---|---|---|---|---|---|---|---|---|---|---|
| Field | C_R | C_G | C_B | T | completed sequence | status | 0x47534331 | last | last cycles | pixel cycles |

Status bits: 0 busy, 1 command error, 2 terminated, 3 finished. One command may
be outstanding. Payload is captured when request differs from completed; keep
request fixed until completion. Repeated sequence never re-executes. Reset clears
request/completion and pixel state. Pixel cycles reset on CLEAR and include CLEAR,
skips, rejected commands and background, but exclude idle/host stalls. Last cycles
includes capture and commit clock edges. ra_clk is the existing 100 MHz clock.

Large vendor projects/build output remain ignored under platform/ and build/.
Original CPU/ALU sources are not overwritten. `program.ps1` loads temporary PL
over JTAG. The separate `package_boot.ps1` / `update_boot.sh` route explicitly
backs up and replaces SD BOOT over SSH, preserving its non-PL partitions and
the Linux Image/DTB/uEnv files; it supports hash-guarded rollback.

## Reproduce the verified software and simulation stages

Run from the repository root with Python + NumPy and Vivado 2018.3 installed:

```powershell
python examples/3dgs_compositor/generate_vectors.py
powershell -NoProfile -ExecutionPolicy Bypass -File examples/3dgs_compositor/run_sim.ps1
# Run prepare_project once; it refuses to replace an existing project.
python examples/3dgs_compositor/board/prepare_project.py
powershell -NoProfile -ExecutionPolicy Bypass -File examples/3dgs_compositor/run_sim.ps1 -Registers
powershell -NoProfile -ExecutionPolicy Bypass -File examples/3dgs_compositor/board/build.ps1
# Supply FPGA_BOARD_PASSWORD privately in your environment; it is not stored.
python examples/3dgs_compositor/board/deploy.py --backend cpu
```

After reviewing implementation timing and loading the matching GSC1 design:

```powershell
python examples/3dgs_compositor/board/deploy.py --backend fpga
```

The board receives input commands and source only; golden states stay on the
PC. `trace` validates every command. `tile` reads final pixels only and times
one warm-up plus measured repeats; repeated RGB/T/index/status fingerprints
must agree. Golden generation uses offline NumPy float32 exp, whose agreement
with the frozen official image is checked; it is not timed as board Gaussian
evaluation. The replay includes post-termination commands to exercise ignoring
late work. It should not be used as an optimized full-render benchmark.

CPU-only compilation uses GNU C++17 -O2. The FPGA client also requires GNU
C++17 (the installed ICraft headers depend on GNU variadic macro behavior),
`icraft_xrt`, `icraft_xir`, `icraft_utils`, `dw`, pthread and dl. Both modes use
the same C++ source and optimization level. SDK initialization and text parsing
occur outside the timed loop; all per-command PL register operations are inside.
