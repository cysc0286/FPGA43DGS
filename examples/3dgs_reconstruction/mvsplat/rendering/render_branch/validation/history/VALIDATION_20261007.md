# Two-lane HGR physical-fit validation — 2026-10-07

## Scope

The experiment reduces the physical HGR render lanes from four to two while
retaining the four virtual-quadrant output contract.  The comparison target is
the frozen four-lane `pipegs_hgr_v4_matched_r9` source/platform.  No source in
the frozen baseline was overwritten.

## Functional evidence

- C simulation: PASS.  All 131072 coordinate conversions match the original
  HLS cast, and the existing mode-0/six-mode, seven-tile, edge/order/empty,
  drain and reset tests pass.
- RTL co-simulation: PASS.  `*** RTL co-simulation finished: PASS ***`.
- HLS project: `hls_pipegs_hgr_v4_coord_2lane_fix2`.
- HLS log: [build log](../build/hls_pipegs_hgr_v4_coord_2lane_fix2_20261007T222421.log).

The final RTL fix made the feedback stop snapshot width-dependent.  The former
hard-coded bit 63 belonged to a 64-pixel four-lane snapshot and caused one
two-lane tile to diverge.  The two-lane implementation now uses
`HGR_PIXELS_PER_LANE-1` (bit 127) and the full RTL comparison passes.

## HLS resource comparison

Reports:

- [two-lane HLS report](../build/hls_pipegs_hgr_v4_coord_2lane_fix2/solution1/syn/report/flicker_render_pipeline_csynth.rpt)
- [four-lane HLS report](../build/hls_pipegs_hgr_v4_matched_r9/solution1/syn/report/flicker_render_pipeline_csynth.rpt)

| Resource | Frozen four-lane | Two-lane | Change |
|---|---:|---:|---:|
| LUT | 34,003 | 25,585 | -8,418 (-24.76%) |
| FF | 60,439 | 44,802 | -15,637 (-25.87%) |
| BRAM18K | 203 | 155 | -48 (-23.65%) |
| DSP48E | 196 | 112 | -84 (-42.86%) |
| HLS estimated clock | 5.292 ns | 5.292 ns | unchanged |

These are the HLS kernel reports, not the whole-board utilization.

## Whole-board implementation

The isolated platform is
`platform/flicker_hgr_v4_coord_2lane_fpga`.  The board build completed
synthesis, placement, routing and bitstream generation:

- [implementation utilization](../build/board_cat_20261007_223535/utilization.rpt)
- [timing summary](../build/board_cat_20261007_223535/timing_summary.rpt)
- [DRC](../build/board_cat_20261007_223535/drc.rpt)
- [build status](../build/board_cat_20261007_223535/status.txt)
- [Vivado implementation log](../../../../platform/flicker_hgr_v4_coord_2lane_fpga/fpai_demo_vivado.runs/impl_1/runme.log)

The placed-device utilization is compared with the frozen four-lane r9 placed
report:

| Resource | Frozen four-lane r9 | Two-lane | Change |
|---|---:|---:|---:|
| Slice LUTs | 62,942 (80.08%) | 57,971 (73.75%) | -4,971 (-7.90%) |
| Slice registers | 98,329 (62.55%) | 88,719 (56.44%) | -9,610 (-9.77%) |
| BRAM Tile | 218.5 (82.45%) | 201 (75.85%) | -17.5 (-8.01%) |
| DSP48E1 | 204 (51.00%) | 128 (32.00%) | -76 (-37.25%) |

The whole-board reduction is smaller than the HLS-kernel reduction because the
PS/DDR/video shell remains unchanged and dominates part of the device.

Final routed timing in the two-lane implementation:

| Metric | Result |
|---|---:|
| WNS | +0.030 ns |
| TNS | 0.000 ns |
| WHS | +0.029 ns |
| THS | 0.000 ns |
| Failed/unrouted nets | 0 |
| Bitstream | generated successfully |

Vivado's summary still prints `Timing constraints are not met` because the
inherited design has a pulse-width check of WPWS -0.409 ns / TPWS -0.479 ns on
two endpoints.  Setup and hold timing are clean, and the same pulse-width
exception is present in the earlier four-lane platform reports; it is not a
new HGR route failure.  It must remain visible in any deployment review.

## Board test boundary at report time

No BOOT image was changed and no new bitstream was installed.  Therefore there
is no new measured board render time, 32768-Gaussian image, PSNR or SSIM for
the two-lane branch.  The required next test is the existing paired
32768-Gaussian, 128x128 camera set (views 7/15/22), with the same process and
quality checks as the frozen four-lane baseline.  Until that test is run, the
results above prove functional equivalence in C/RTL and physical fit/timing,
not an end-to-end speedup.

Follow-up: the paired 32768-Gaussian board test has since completed. Its
measured latency, image quality, resource use, rollback verification, and
limitations are documented in
[`VALIDATION_32768_20261008.md`](VALIDATION_32768_20261008.md).
