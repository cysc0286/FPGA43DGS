# Live camera-to-frame renderer

The scene is prepared before interactive rendering. LiveRenderer.load_scene()
or load_rows() installs a scene; render_camera() accepts the existing 136-byte
camera and returns a complete RGB framebuffer. LiveFrame.archive() is explicit
and runs after delivery. No per-view model hashing, intermediate attribute/scene
file, Python pixel loop, or image archive is on this path.

## Structure

- runtime.py: framed pipe interface and explicit frame archival.
- live_renderer.cpp: scene lifetime, parallel projection, ordered Tile lists,
  persistent DMA buffers and native RGB conversion.
- cached_projection.hpp: one-time covariance/opacity preparation and per-view
  projection. Screen projection, depth and SH color still change per camera.
- pipeline_adapter.py: compatibility with the warm video-to-first-frame path.
  This adapter still counts raw-frame archival before FRAME_COMPLETE; its time
  must not be presented as the lower archive-free interactive latency.
- benchmark.py: real-board full-point, thread, batch and lossy-budget profiles.
- build.sh: uses the frozen CPU sources and the installed ICraft SDK. The
  existing initialize/render_resident.cpp supplies established FLK1 helpers.
- compare_builds.py: paired native compiler/layout experiments on the ARM board;
  alternates variant order and archives output differences after timing.

The model, projection/sort workspace and FPGA allocation remain in one native
process. The existing hardware two-bank queue is reused; no new bitstream or
cross-frame double buffering is introduced. FPGA tags, lengths, idle state and
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

For an interactive controller, construct LiveRenderer with threads=4, batch=2.
Use max_gaussians=16384 and uniform_preview=True for the fast preview profile,
then submit successive camera payloads to render_camera(). The returned
frame.rgb is ready for the consumer; frame.archive(path) is an optional later
operation. The Python controller's physical display/transport latency is not
included in the board benchmark.

Interactive timing begins before sending a camera and ends after complete RGB
receipt and shape validation. It excludes preprocessing/scene loading, archival,
remote-control transport and physical display. The benchmark keeps first-use
latency separate, measures repeated different views, then archives outputs and
compares them after the timed loop. It reports actual transferred payload bytes,
not DDR bus traffic. NPU is not involved.
