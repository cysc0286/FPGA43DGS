# Board-side 3DGS front end

The installed CPU+FPGA path accepts the locked official trained PLY and a camera pose. ARM computes projection, covariance/conic, opacity and degree-3 SH color, creates Tile lists and sorts them by depth. FPGA FLK1 performs sub-tile AABB, Mini-Tile CAT, Gaussian evaluation and ordered compositing. No GPU execution is required at rendering time. Training, pose estimation, live video and NPU compute are outside this path.

The selected sorter packs positive finite depth bits and original Gaussian ID into a 64-bit key, sorts once, then distributes ordered IDs into Tile lists. The positive-depth guard makes integer and float depth order equivalent; ID resolves ties exactly as the original stable CPU sort. All four tested sorter variants produced identical scene bytes for both complete views. This is a CPU implementation optimization, not a new FLICKER algorithm.

Current results and limitations: [VALIDATION.md](VALIDATION.md). Full benchmark: [full_benchmark_20260927T234919](../evidence/full_benchmark_20260927T234919/REPORT.md). FPGA split scheduling, measured separately with unchanged input/output: [split_board_comparison_20260927](../evidence/split_board_comparison_20260927/REPORT.md).

## Reproduce

Run from the repository root. Set FPGA_BOARD_PASSWORD privately in the process environment. Scripts check model/camera/source/binary hashes and create new evidence directories. The commands below reference the completed current stages; replace evidence paths with newly printed paths when rebuilding.

```powershell
& D:/Tools/Python31210/python.exe examples/3dgs_flicker_hw/frontend/benchmark_full.py --attributes examples/3dgs_flicker_hw/evidence/frontend_attr_20260927T231255 --group examples/3dgs_flicker_hw/evidence/frontend_group_20260927T233920 --repeats 3
& D:/Tools/Python31210/python.exe examples/3dgs_flicker_hw/frontend/analyze_full.py examples/3dgs_flicker_hw/evidence/full_benchmark_20260927T234919
```

The benchmark uses one warmup and alternates FPGA Dense, CPU Dense4 and CPU base4. Each sample reads the PLY, projects, groups/sorts, then renders exactly one frame. Intermediate files and framebuffer use /dev/shm. Monotonic wall time includes process startup, reading, allocation, SDK initialization and output writing; excludes SSH transfer, analysis and display. This is a command-line invocation benchmark, not a resident interactive renderer or a cold-cache SD benchmark.

## Rebuild CPU stages

```powershell
& D:/Tools/Python31210/python.exe examples/3dgs_flicker_hw/frontend/stage.py --cameras examples/3dgs_flicker_hw/evidence/frontend_camera_20260927/v0.bin examples/3dgs_flicker_hw/evidence/frontend_camera_20260927/v10.bin
& D:/Tools/Python31210/python.exe examples/3dgs_flicker_hw/frontend/validate_attributes.py --evidence examples/3dgs_flicker_hw/evidence/frontend_attr_20260927T231255
& D:/Tools/Python31210/python.exe examples/3dgs_flicker_hw/frontend/stage_group_sort.py --attributes examples/3dgs_flicker_hw/evidence/frontend_attr_20260927T231255 --packed-sort
& D:/Tools/Python31210/python.exe examples/3dgs_flicker_hw/frontend/validate_group_sort.py --evidence examples/3dgs_flicker_hw/evidence/frontend_group_20260927T233920
```

Use new output paths with pack_camera.py for other poses; archived files are evidence. hybrid_render.py retains the earlier attribute-only test using official GPU lists. It is not the complete board-generated scene path. run_full.py preserves the earlier SD-file demonstration; use the tmpfs benchmark for the reported repeated comparison.

## Numerical and source boundary

attributes.cpp adapts GraphDECO diff-gaussian-rasterization forward.cu and auxiliary.h, pinned at commit 9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0. Original source, notices and license are retained in [frontend_source_20260927](../evidence/frontend_source_20260927/); see [LICENSE-GraphDECO.md](LICENSE-GraphDECO.md). Scalar compilation uses -ffp-contract=off. ARM/GPU float differences affect near-equal depth ties and four edge Tile memberships in view 10. These differences are disclosed and checked against a frozen image-quality gate.

verify_frame_reference.py checks the timed framebuffer against HLS for the same board input. Both complete scenes match exactly. The original official FP32 strict gate remains a separate diagnostic and fails; the accepted approximation gate does not redefine that result.
