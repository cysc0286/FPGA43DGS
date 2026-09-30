# Post-route experiment — 2026-09-30

`post_route.tcl` explores the actual frozen 2026-09-27 21:09:33 routed design.
The original checkpoint is read-only; each experiment gets a separate directory.
This is a physical implementation experiment, not an FPGA clock change.

## Measured result

| Case | Shared 200 MHz domain setup slack | Renderer setup slack | Renderer hold slack |
|---|---:|---:|---:|
| Original | 0.026 ns | 0.310 ns | 0.054 ns |
| Default AggressiveExplore | 0.026 ns | 0.310 ns | 0.054 ns |
| Extra 0.450 ns setup margin, before optimization | −0.424 ns | −0.140 ns | 0.054 ns |
| Extra margin, after optimization | −0.401 ns | −0.140 ns | 0.054 ns |
| Original timing budget restored | 0.049 ns | 0.310 ns | 0.054 ns |

Default physical optimization was skipped by Vivado because there was no
negative setup slack. The tighter experimental budget caused real physical
optimization; after restoring the original budget the improvement was only
**0.023 ns on a vendor video path**, with no renderer slack improvement.

The limiting shared-clock path runs from video read-enable logic to a video
buffer BRAM enable. Its 3.132 ns data delay contains 2.823 ns routing and 0.309 ns
logic; clock skew is −1.462 ns. Routing dominates that path, but it is not the
renderer arithmetic path. The renderer and DMA share `clk_pll_i` with vendor DDR
and video logic. Renderer-only slack is not permission to increase that entire
domain's clock.

Both physical candidates remained fully routed. No generated firmware was
installed and the clock stayed 200 MHz. Therefore **there is no board FPS gain
from this experiment**. At unchanged frequency and cycle count, better slack
alone does not shorten frame time. A useful next physical experiment needs an
architectural change, a separately verified clock boundary, or fewer cycles.

Whole-board timing signoff is not claimed: the original vendor AI clock retains
two pulse-width violations (−0.409 ns), and inherited video/CDC constraints need
separate review. New renderer checks do not erase these inherited limitations.

## Evidence and use

`results/20260930/summary.json` identifies the source checkpoint by SHA-256.
The surrounding reports retain timing, routing, utilization, render-path and
shared-clock-domain comparisons. The script refuses to replace an existing
experiment or overwrite a pre-existing user clock uncertainty with the optional
temporary margin. Its experimental constraint is removed before the final
native-budget reports.

```text
vivado -mode batch -source post_route.tcl -log new.log -journal new.jou \
  -tclargs original_routed.dcp new_result_directory 0.45
```

Use the installed 2018.3 tool and environment matching the frozen checkpoint.
This experiment is distinct from the new exp-ROM HLS implementation under
`../pipeline/exp_rom`.

## Full-placement control for the changed ROM netlist

`full_place.tcl POST_OPT_DCP NEW_OUTPUT ORIGINAL_VENDOR_OPT_PRE_TCL` performs
independent placement without importing old cell locations. It requires an
unused output directory and replays only four DRC enable settings explicitly
present in the original vendor hook. Checkpoint files do not preserve that
session state. The first missing-setting failure is retained in the local build
log; the corrected run completed placement in 575 seconds. The old-checkpoint
incremental attempt remained in detailed placement for over half an hour and
was deliberately interrupted, with evidence retained. Neither outcome by itself
establishes routed timing or board speed.

For a completed synthesis and a new fully placed checkpoint, the board build
supports `FLK_RESUME_PLACE=1` plus `FLK_INCREMENTAL_DCP=<new_placed.dcp>`.
It resumes the normal project flow from placement rather than redoing synthesis,
then runs the existing physical-opt, route and vendor bitstream hooks.
`restore_vendor_checks.tcl` restores only the inherited proxy-device settings
when that resumed process opens the checkpoint in a new session. Preserve the
earlier run's logs before resetting a stopped candidate; never reset a live run.

The current ROM candidate's placement, final route and board status are recorded
in `../pipeline/exp_rom/BOARD_VALIDATION.md`.

## Completed ROM candidate

The new ROM netlist initially routed with −0.012 ns setup slack on the same
vendor video-buffer enable class of path. Post-route AggressiveExplore repaired
it: global setup/hold +0.030/+0.022 ns, shared 200 MHz setup +0.034 ns, paths
through render/DMA +0.316/+0.050 ns. No timing constraints or clock frequency
changed; inherited AI pulse-width limitations remain. Final route errors are zero.

`finalize_candidate.tcl PLATFORM OPTIMIZED_DCP NEW_REPORT ORIGINAL_FROZEN_MANIFEST`
requires the preserved original, Lite pins, unchanged 5 ns shared clock and
nonnegative setup/hold, then invokes the unchanged vendor bitstream hooks.
It replaces only the candidate platform's implementation outputs after freezing
them; the original released platform is untouched. The finalization log and
source checkpoint are recorded separately from the initial project-run log.
Run the normal routed audit, package verification, freeze and guarded installer
after finalization. A successful command alone is not board verification.

This candidate was installed and tested: 5.570→5.215 million hardware cycles
and 50.294→49.059 ms complete RGB latency at unchanged 200 MHz. The speed gain
comes from the ROM architecture's reduced cycles; physical optimization makes
that architecture meet setup/hold. It is not a separate clock-frequency gain.
