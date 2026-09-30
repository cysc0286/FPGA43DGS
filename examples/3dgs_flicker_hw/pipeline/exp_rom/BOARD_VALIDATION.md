# Physical implementation and board validation — 2026-09-30

## Current status

The compact ROM has passed exhaustive exponent tests, full-renderer C/RTL,
and six-mode DMA integration. Full placement, routing, post-route physical
optimization, vendor bitstream generation, network installation and actual
board tests are complete. **The new candidate is running on the board**.
The original four-lane release remains unchanged and backed up for rollback.

With the identical Cortex-A53 native binary, 32,768 Gaussians and 128×128 output,
mean camera-to-RGB latency is **49.059 ms**, versus **50.294 ms** before the
hardware replacement. The hardware engine cycles fall **6.37%**. This is a
measured incremental improvement, not an order-of-magnitude acceleration.

## Actual board comparison

Three held-out cameras, ten repetitions each, two rounds: 60 timed frames per
backend. CPU/FPGA order is reversed in round two. First renders and archival
checks are recorded separately. The old and new FPGA tests use the same binary
SHA-256 `db4c05413c4f57198a6f46a5069ae6820faf935db048a150ea767f26bfaa5317`.
They were run before/after reboot, not interleaved hardware A/B; wall-time
differences therefore include normal run-to-run variation.

| Metric | Original FPGA | Compact-ROM FPGA | Change |
|---|---:|---:|---:|
| Camera → complete RGB, mean | 50.294 ms | 49.059 ms | −2.46% |
| Engine wall time, mean | 28.450 ms | 26.684 ms | −6.21% |
| Hardware cycles/frame, mean | 5,570,456 | 5,215,446 | −6.37% |
| P95 camera → RGB | 51.460 ms | 50.204 ms | |
| P99 camera → RGB | 51.546 ms | 51.620 ms | no tail-latency gain |

New median is 48.945 ms, maximum 53.593 ms; round means are 49.248/48.870 ms.
Mean latency corresponds to 20.38 frames/s for serialized requests, excluding
physical display latency. The first request after scene load was 56.894 ms.
The current CPU Dense control measured 162.830 ms, so this mixed-precision
CPU+FPGA path is 3.32× faster than that CPU path; CPU uses different arithmetic,
so this is **not** a same-arithmetic pure-hardware speedup. All old/new FPGA
raw and RGB outputs match for all three cameras in both rounds, and every timed
RGB frame matches its later archived view.

New-hardware regressions also pass cached/uncached projection, 64×64 and 129×65
viewports, switching to a 1,024-row scene and restoring the complete scene, and
the first-frame archival adapter. Their seven raw hashes equal the original
hardware's results. The software source/contract check passes 306 Python files
and 71 selected tests; those offline checks are separate from the board proof.

New stage means: projection 7.754 ms; grouping/sorting 12.359 ms; engine
26.684 ms; RGB packing 1.230 ms. Upload 2.496 ms, wait 15.174 ms and readback
0.674 ms are engine submetrics and must not be added again. First-view logical
payload is 3,441,984 input bytes and 262,144 output bytes per frame; these are
interface byte counts, not measured total DDR traffic. Renderer-only peak RSS
is 39,184 KiB (38.27 MiB). Per-stage CPU utilization and physical power were not
measured. More than 20 ms still lies in projection, sorting and host-side work,
which explains why a 6.37% cycle gain produces only a 2.46% full-frame gain.

## Whole video and visual quality

The complete board chain was rerun with the same video, weights and scheduling.
**VIDEO_COMPLETE → FRAME_COMPLETE is 24.693683 s**, compared with the new
before-replacement run's 24.337347 s. Both are single runs. This experiment does
**not** show a reduction in scene-preparation latency and does not meet 20 s.
SCENE_READY occurs at 24.607823 s; first-render/compatibility archival takes
85.860 ms. The raw first frame remains identical. Model inference still runs
on the ARM CPU (16.511 s); context is ready at 5.738 s, with later PnP overlapping
inference. Generation and rendering must not be combined into a misleading FPS.

Prewarming is recorded separately at 46.878 s and is outside that performance
clock. Input is an already closed local video file; physical acquisition time
is not measured. Peak sampled process-tree RSS is 610.24 MiB, minimum available
memory 232.86 MiB. Tree RSS can count shared pages more than once. The board RTC
is incorrect; host date is 2026-09-30 and durations use monotonic clocks.

Fresh quality recomputation from new-hardware raw frames against held-out video:

| Camera | PSNR | SSIM | Mean absolute RGB error |
|---|---:|---:|---:|
| Frame 7 | 19.4519 dB | 0.8045 | 0.06570 |
| Frame 15 | 20.2223 dB | 0.7851 | 0.04668 |
| Frame 22 | 21.3486 dB | 0.7842 | 0.04337 |

Metrics use clipped float RGB; SSIM is Gaussian 11×11, sigma 1.5, valid window,
averaged over channels. Results are unchanged from the original FPGA. Visible
blur and boundary holes remain; frame 7 is still below the former 20 dB guide.
This ROM change introduces no extra image loss, but does not improve reconstruction
quality or establish full-video coverage or downstream task accuracy.

## Physical experiments

The first implementation used the old routed checkpoint as an incremental seed.
About 18.47% of cells could not be reused. It spent over half an hour in
`Place Remaining` and was deliberately stopped after an independent full
placement completed. This is an interrupted experiment, not proof of a placer
error or an impossible design. Its source, log and interruption reason are
preserved under `build/rom_incremental_interrupted_20260930`.

Independent full placement of the **same post-opt netlist** completed in
575 seconds (609 seconds including saving/reporting). It uses the same
`ExtraNetDelay_high` directive as the original flow. Standalone checkpoint loading
does not retain the vendor hook's session DRC enable flags; the exact four
existing proxy-device settings were restored after checking their declarations
in the unchanged vendor hook. No new timing exception or new DRC waiver was added.
The first standalone attempt's missing-session-setting error and the corrected
run are retained separately.

| Resource | Original routed board | New routed and physically optimized board |
|---|---:|---:|
| LUT | 61,369 / 78.08% | 60,398 / 76.84% |
| FF | 98,364 / 62.57% | 97,523 / 62.04% |
| Slice | 19,649 / 99.99% | 19,484 / 99.16% |
| BRAM36-equivalent tile | 200.5 / 75.66% | 226.5 / 85.47% |
| DSP | 260 / 65.0% | 252 / 63.0% |

These are whole-board numbers and differ from the HLS kernel estimates.
The original board has inherited vendor pulse-width/CDC limitations; successful
renderer checks do not constitute whole-board timing signoff.

Initial routing had one −0.012 ns setup violation in vendor video-buffer enable
logic. `phys_opt_design -directive AggressiveExplore` repaired it without any
constraint or clock change. Final whole-design setup/hold slack is +0.030/+0.022 ns;
the shared 200 MHz domain setup slack is +0.034 ns. Paths through the renderer/DMA
have setup/hold slack +0.316/+0.050 ns. All 149,247 routable nets are fully routed,
with zero routing errors. The inherited AI pulse-width checks remain −0.409 ns
(two endpoints), so this remains a prototype under the documented vendor flow,
not complete whole-board timing signoff.

The uncorrected implementation was frozen before finalizing the optimized
checkpoint. `physical_opt/finalize_candidate.tcl` requires preserved originals,
unchanged Lite pins and shared clock, and nonnegative setup/hold before invoking
the original vendor bitstream hooks. This is a new hardware implementation at
200 MHz, not an overclock. ROM area is deliberately traded for fewer cycles.

## Reproducible locations

- HLS project: `build/hls_pipeline_exp_compact_20260930`.
- Frozen HLS proof: `evidence/hls_pipeline_exp_compact_20260930_20260930T151335`.
- Real RTL integration: `evidence/pipeline_integration_20260930T145219`.
- Platform: `platform/flicker_rom_fpga` at repository root.
- Original incremental attempt: `build/board_cat_20260930_145221` (interrupted).
- Independent placement: `build/rom_fullplace_v3_20260930/full_placed.dcp`.
- Resumed implementation: `build/board_cat_20260930_155540`.
- Post-route repair: `build/rom_post_route_20260930`.
- Finalized candidate: `build/board_rom_final_20260930`.

Paths starting with `build/` or `evidence/` are relative to `examples/3dgs_flicker_hw`.
The resumed build uses `FLK_RESUME_PLACE=1` and the new fully placed checkpoint
as `FLK_INCREMENTAL_DCP`. It reuses completed synthesis and new placement,
then executes the normal vendor physical-opt, route and bitstream hooks.

Before any replacement, current board BOOT was read from the SD partition
mounted read-only and matched the frozen release:
`9918f9b69525bad88d8a9f06c5ab0940582df341ca32edc786d4377d3ba137cc`.
The boot partition has ample space; root storage has about 412 MiB free, so
only the small firmware, logs and required test outputs should be transferred.

## Installation, evidence and next work

Installed BOOT SHA-256:
`0528e019a5fb19c7f281b773ddf5ab5a8a524ddfebdc4fc9acf74273db31d048`.
Non-PL boot partitions are exact; SD kernel, DTB and uEnv hashes remain exact.
The previous BOOT was backed up locally and on the board, and the written BOOT
was checked after read-only remount. Boot ID changed and SDK/FLK1 status passed.
The rollback command is retained in `evidence/boot_20260930T162317/result.json`.
This report's `results/20260930/boot_install.json` is a compact copy.

Detailed measurements, command lines, raw/RGB hashes and the comparison image:
`examples/3dgs_reconstruction/mvsplat/results/20260930/fpga_exp_rom`.
Whole-chain events and memory records: adjacent `compiler_full_before` and
`compiler_full_after`. ROM reports and final timing/resource/package proofs are
under this directory's `results/20260930/physical`. Large checkpoints, original
bitstreams and boot backups remain local, identified by recorded SHA-256.

Next larger renderer candidate: interleave independent mini-tile contexts to
reduce repeated finite-loop fill/drain. Preserve the per-pixel contribution
order; the historical direct-index design's II=72 must not be reintroduced by
simply removing state copies. Separately, optimize the now-visible ~20 ms CPU
projection/sort work. Raising the shared DDR/render clock is not justified by
renderer-only timing. Achieving a <20 s first frame still requires front-end
and network-inference optimization; NPU was not used in these results.
