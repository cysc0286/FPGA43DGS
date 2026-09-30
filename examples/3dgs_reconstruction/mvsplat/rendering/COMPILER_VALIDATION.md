# ARM compiler and data-layout experiments — 2026-09-30

Subsequent hardware measurement: the exact-ROM four-lane candidate is now
installed, with 49.059 ms mean complete-RGB latency and unchanged images.
See `../../../3dgs_flicker_hw/pipeline/exp_rom/BOARD_VALIDATION.md` for hardware,
physical and whole-chain results. The software A/B results below continue to
refer to the original FPGA and have not been overwritten.

These are new measurements on the four-core Cortex-A53 board. The FPGA was
the frozen four-lane, 200 MHz split pipeline throughout these software tests.
This document does not count an unprogrammed HLS candidate as board acceleration.

## Method

All variants use the same 32,768 Gaussians, 128×128 output, held-out cameras
7/15/22, four CPU threads and two-Tile batches. Each variant has 60 timed frames
(three cameras × ten repetitions × two rounds); the second round reverses
variant order. Scene load and first-use latency are separate. Timing starts
before sending the camera and ends after receiving and checking the RGB frame.
Archival and physical display are excluded. This is one scene, not 60 independent
scenes. Full records and compiler logs remain in `../results/20260930/compiler_v1`.

## Compiler result

| Same-input pair | Portable mean | A53 mean | Latency reduction |
|---|---:|---:|---:|
| Pure CPU, identical C++ source | 177.553 ms | 161.341 ms | 9.13% |
| CPU + existing FPGA | 50.604 ms | 49.995 ms | 1.20% |

The A53 build uses `-O3 -mcpu=cortex-a53 -ffp-contract=off
-fno-math-errno -fno-trapping-math`. Finite-value checks remain active. Neither
`-ffast-math` nor FMA reassociation is enabled in the selected default. The two
compiler variants produce identical raw and RGB outputs **within each backend**.
CPU-versus-FPGA outputs use different arithmetic and are not claimed bit-identical.

The native C++ hot path already replaced per-frame Python/process/file work in
the preceding increment. Thus this experiment measures target-specific code
generation, not a second Python-to-C++ speedup. The much smaller heterogeneous
gain is consistent with most raster work already being in hardware.

For this paired run the CPU A53 P95/P99 were 163.448/163.666 ms; heterogeneous
A53 P95/P99 were 51.154/51.169 ms. Native peak RSS was about 38.3 MiB. This is
renderer memory, not the MVSplat process-tree peak.

An earlier comparison of old portable and new-default source measured
179.750→161.358 ms (10.24%). Since that pair also included disabled layout
switches, `cpu_isolated_results.json` repeats the experiment with the exact
same pre-layout source and only compiler flags changed. The 9.13% result is
the cleaner compiler attribution and is used in the table.

## Negative and inconclusive experiments

| Experiment | Paired reference | Candidate mean | Decision |
|---|---:|---:|---|
| Allow fused multiply-add | 50.046 ms | 50.684 ms | Slower; keep optional |
| Prepack every active Gaussian into a cached 64-byte word | 50.097 ms | 52.511 ms | Extra memory pass loses; disabled |
| Store active Gaussians in depth order | 50.097 ms | 49.878 ms | Small, unstable gain; optional |
| Depth order + cached words | 50.097 ms | 51.775 ms | Disabled |

FMA changed at most one RGB level, with 8/16/24 changed channel values across the
three images. PSNR versus the exact renderer was 86.015/83.005/81.244 dB; these
are implementation differences, **not quality against photographs**. All layout
variants were raw/RGB exact. A separate final repeat measured selected default
50.294 ms and depth layout 50.331 ms, so depth layout is not promoted on a noisy
0.2 ms apparent gain. Both switches remain explicit comparison tools.

The selected default in that repeat took 7.568 ms projection, 12.039 ms grouping
and sorting, and 28.450 ms engine wall time. Pack/upload/wait/read are submetrics;
they must not be added again to engine time. Hardware cycle count still averaged
about 5.57 million per frame. Compiler tuning cannot change those RTL cycles.

## Integration and whole-chain check

Real-board checks covered cached/uncached projection, 64×64 and 129×65 output,
scene replacement, returning to 32,768 rows, and the first-frame archival adapter.
Three full-scene raw hashes match the frozen path. See `verification.json`.

The complete preheated video chain, using the new A53 binary with original FPGA,
measured **24.337347 s VIDEO_COMPLETE → FRAME_COMPLETE**. SCENE_READY occurred at
24.260588 s; rendering plus first-frame archival then took 76.760 ms. This is
`../results/20260930/compiler_full_before`, one new run, not an average and not
an achievement of the 20-second target. Prewarming is excluded from this metric.

Because the full-point images are identical, recorded held-out photograph quality
is unchanged: PSNR 19.452/20.222/21.349 dB, SSIM 0.8045/0.7851/0.7842. The first
view remains below the earlier 20 dB guideline. Whole-video coverage, physical
display latency, power and NPU acceleration are not established by these tests.

## Reproduce on ARM

```sh
sh mvsplat/rendering/build.sh /path/to/frozen_renderer portable /tmp/live_portable
sh mvsplat/rendering/build.sh /path/to/frozen_renderer a53 /tmp/live_a53
PYTHONPATH=mvsplat python3 -m rendering.compare_builds \
  --scene /path/to/renderer_input --out /path/to/new_comparison \
  --build portable=/tmp/live_portable --build a53=/tmp/live_a53
```

Use `--cpu-build portable --cpu-build a53` for a pure CPU pair. Optional layout
flags are `--depth-build NAME` and `--compact-build NAME`. Output directories are
new per experiment; previous results are never overwritten.
