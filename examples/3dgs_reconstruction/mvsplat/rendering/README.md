# Live camera-to-frame renderer

2026-10-09 integration: the warm video path's `PipelineRenderer` now defaults
to `LiveRenderer.mainline(...)`, including direct collection and NEON packing.
Explicit tuning uses `--render-profile custom`; the legacy renderer remains
available when no native binary is supplied. Board staging now transfers the
entrypoints and profile together. See [verification and backend NPU scope](PIPELINE_INTEGRATION_20261009.md).
This is a locally tested integration fix, not a new video-to-frame board result.

2026-10-09 packaging: the accepted four-lane implementation is frozen in
[package/](package/README.md), including BOOT, PL bitstream, exact generated RTL,
HLS dependency closure and native CPU sources. The independently archived
[render_branch](../render_branch/README.md) is never selected by this runtime.
The October 8 paired measurement is 43.575 ms vs 53.302 ms for the candidate;
these are historical board measurements, not new packaging performance results.


2026-10-06: the accepted mainline is **four-lane grouped-shared hardware at
200 MHz, persistent native C++, full-scene ordered Dense mode 2**. Use
`LiveRenderer.mainline(binary, environment, log_path)`; the accepted options
and artifact identities are in [mainline.json](mainline.json).
The latest saved rollback run is 42.9170 ms mean / 44.0654 ms P95 (60 frames);
both original-firmware runs that day total 120 frames / 43.9201 ms mean.
These are loaded-scene, 32,768-Gaussian, 128×128 camera-to-complete-RGB results.
See [the mainline decision, quality, resources and exclusions](MAINLINE.md).
This profile/document update adds no new board performance measurement.
The profile selects options; it does not attest or install board firmware.

Historical 2026-09-30 result: three-pass depth radix plus four-core
Tile list construction. On the unchanged FPGA, 32,768 Gaussians at 128×128,
paired view-switch latency falls from 48.632 to **45.212 ms** (7.03%; P95
46.263 ms). Pixels are unchanged. See [view-switch validation](VIEW_SWITCH_VALIDATION.md).

Next-step analysis: [sort-free methods and physical optimization](SORT_FREE_AND_ROUTING_PLAN.md).
This is a sourced design comparison, not another board performance result.

Dominant-stage work: [FPGA hot path and candidate status](HOT_PATH_20260930.md).
Hardware candidates are opt-in; the installed renderer remains the baseline
until routed timing and matched board measurements are accepted.

The scene is prepared before interactive rendering. LiveRenderer.load_scene()
or load_rows() installs a scene; render_camera() accepts the existing 136-byte
camera and returns a complete RGB framebuffer. LiveFrame.archive() is explicit
and runs after delivery. No per-view model hashing, intermediate attribute/scene
file, Python pixel loop, or image archive is on this path.

## Structure

- mainline.json / MAINLINE.md: accepted options, artifact identity and decision.
- runtime.py: framed pipe interface and explicit frame archival.
- live_renderer.cpp: scene lifetime, parallel projection, ordered Tile lists,
  persistent DMA buffers and native RGB conversion.
- cached_projection.hpp: one-time covariance/opacity preparation and per-view
  projection. Screen projection, depth and SH color still change per camera.
- depth_sort.hpp: stable positive-FP32 radix sort, preserving all depth bits.
- tile_lists.hpp: contiguous sorted work blocks, private Tile counts, ordered
  prefix offsets and parallel writes to disjoint list segments.
- tests/: native differential checks for sorting and Tile list ownership/order.
- pipeline_adapter.py: compatibility with the warm video-to-first-frame path.
  This adapter still counts raw-frame archival before FRAME_COMPLETE; its time
  must not be presented as the lower archive-free interactive latency.
- benchmark.py: real-board mainline by default; historical thread, batch and
  lossy-budget profiles remain opt-in via --profiles.
- build.sh: uses the frozen CPU sources and the installed ICraft SDK. The
  existing initialize/render_resident.cpp supplies established FLK1 helpers.
- compare_builds.py: paired native compiler/layout experiments on the ARM board;
  alternates variant order and archives output differences after timing.

The model, projection/sort workspace and FPGA allocation remain in one native
process. The existing hardware two-bank queue is reused. This software profile
adds no bitstream change or cross-frame double buffering. FPGA tags, lengths, idle state and
finite pixel checks remain enforced.

## Build and use on the board

    sh mvsplat/rendering/build.sh /path/to/3dgs_renderer_v1_20260928
    PYTHONPATH=mvsplat python3 -m rendering.benchmark \
      --scene /path/to/renderer_input --binary mvsplat/rendering/live_renderer \
      --reference /path/to/previous/validation.json --out /path/to/new/results

The build defaults to the measured Cortex-A53 profile. Optional second and third
arguments select `portable`, `a53`, or experimental `a53-fma`, and the output
binary path. `COMPILER_VALIDATION.md` reports the paired ARM measurements,
including negative FMA/prepacking results. Optional `LiveRenderer` arguments
`compact_payload` and `depth_layout` default to false; no unproven layout gain
is silently enabled.

Rebuild the native binary when updating the Python runtime: its tested defaults
are now `radix_bits=11, parallel_tiles=True`. To call a preserved older binary,
pass `radix_bits=8, parallel_tiles=False`; both new flags are then omitted.
The experimental depth layout also requires `parallel_tiles=False`.

For video preparation, append --live-renderer /absolute/path/live_renderer to
the existing initialize command. Optional --render-max-gaussians 16384 is
explicitly approximate; it ranks opacity-weighted scene-space Gaussian surface
area once per scene. It can lose small objects or detail and is measured against
both the complete scene and the held-out photographs when available.

The better tested preview adds --render-uniform-preview: it samples Gaussian
rows uniformly and expands each retained world covariance by original/retained
count. This is a deliberate footprint approximation, not retraining or an exact
merge. It is sensitive to row ordering and can create stripes or blur. At this
scene the 16k version outperforms importance-only pruning visually, but all
quality loss is retained in VALIDATION.md. Full rows remain the default.

For an interactive controller, use LiveRenderer.mainline(binary, environment,
log_path), load the full scene once and submit successive camera payloads to
render_camera(). The generic constructor remains for controlled experiments.
max_gaussians=16384 and uniform_preview=True are a lossy preview experiment,
not the accepted mainline; the first view loses about 2.46 dB PSNR. The returned
frame.rgb is ready for the consumer; frame.archive(path) is an optional later
operation. The Python controller's physical display/transport latency is not
included in the board benchmark.

Interactive timing begins before sending a camera and ends after complete RGB
receipt and shape validation. It excludes preprocessing/scene loading, archival,
remote-control transport and physical display. The benchmark keeps first-use
latency separate, measures repeated different views, then archives outputs and
compares them after the timed loop. It reports actual transferred payload bytes,
not DDR bus traffic. NPU is not involved.
# 2026-09-30 FPGA 硬件更新

同一 A53 程序在新精确指数 ROM 位流上完成 60 帧实测：49.059 ms，P95
50.204 ms，三视角 raw/RGB 与原 FPGA 相同。CPU 对照 162.830 ms。
新视频首帧复测 24.694 s，仍未达到 20 s。硬件、资源、时序及完整计时见
`../../../3dgs_flicker_hw/pipeline/exp_rom/BOARD_VALIDATION.md`；本目录旧测试
保持原数据，不把新位流混入旧编译器配对结果。
