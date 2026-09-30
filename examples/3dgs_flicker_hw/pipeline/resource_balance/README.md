# Resource balance experiments

This directory contains the four-lane grouped-renderer resource and placement
work. The deployed result is documented in [BOARD_VALIDATION.md](BOARD_VALIDATION.md);
[REVIEW.md](REVIEW.md) records the RTL review and rejected/unfinished controls.

- Implementation: `../pipeline.cpp`, with opt-in build switches.
- Rebuild entry: `../lane_workset/build_variant.ps1 -Variant grouped-shared`.
- `inspect.tcl`: read-only resource, hierarchy and timing reports from a DCP.
- `route_placed.tcl`: route an independent placement without changing its source.
- `compare_board.py`: repeat a saved board command and verify firmware, binary,
  scene, camera coverage and downloaded raw/RGB digests.
- `results/20260930/summary.json`: measured frame timings and stage statistics.
- `results/20260930/physical_shared/`: final physical reports and installation receipt.
- `results/20260930/board_*`: measured board data, regression and images.
- Other result directories: frozen HLS, RTL and placement control evidence.

`packed`, `bram_fifo` and `shared` result receipts describe the original offline
test stage. They are not rewritten after deployment; final hardware status lives
in `physical_shared` and `summary.json`. Intermediate estimates are not board FPS.

Generated platforms, tool build trees, DCPs and BOOT binaries remain in the
existing ignored local directories. The repository retains sources and evidence;
rebuilding requires the local vendor platform and Vivado/Procise installation.
