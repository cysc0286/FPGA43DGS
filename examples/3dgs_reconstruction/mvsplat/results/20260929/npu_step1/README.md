# NPU incremental step 1: backbone numerical diagnosis

Test organization date: 2026-09-29/30. Target: 30TAI Lite, board-side ZG330 AIU, ICraft SDK 3.36.1. This step changes no production inference code, bitstream, or acceptance threshold. The computer only sent commands, compiled the candidate graph, and compared saved tensors; model inference ran on the board NPU.

## Fixed contract and versions

- Partition: `backbone_cnn`, official MVSplat re10k checkpoint and saved real-input oracle. Input: two 128 x 128 RGB context images, logical shape `[2,3,128,128]`. Output: `[2,128,32,32]`. Bundle and oracle hashes are checked by `oracle_probe.py`.
- Production candidate: existing TF32 compiled graph, FP32 host I/O, ABI 2 shared buffers. The unchanged numerical gate is `rtol=0.002, atol=0.0002` for every output element.
- Baseline library: existing `libmgs_npu.so`. Diagnostic libraries: `bridge_precheck.cpp` with lazy parameter load; `bridge_eager.cpp` with eager parameter load. Both add SDK `precheck()` after `apply()` and after each forward. `eager.sh` records the exact board build and probe commands. The diagnostic `precheck()` adds approximately 7.9 s per call and its timings are **not** performance results.
- `fp32` compiler control: only this partition was recompiled in `runs/mvsplat_npu_compile_fp32_backbone_20260929`; the original TF32 bundle was retained. The compiler's five stages ran, but the output became Quantized FP32/NCHWc16, which the current audited host ABI rejects. It was not deployed as a working replacement.

## Fresh board results

| Check | Result | Interpretation |
|---|---:|---|
| TF32 baseline, two real NPU calls | Identical output; hardware placement confirmed | Deterministic on this fixed input, not numerically accepted |
| Maximum / mean absolute error, RMSE | 0.04001218 / 0.00391551 / 0.00519463 | Both calls fail the unchanged element-wise gate |
| Elements outside the gate | 77.665% | Not an isolated outlier |
| Pearson correlation with oracle | 0.9999779 | Broad tensor arrangement agrees; does not prove every channel/layout correct |
| Output values representable exactly as FP16 | 100% | Consistent with reduced mantissa precision; does not prove it is the sole error source |
| Lazy vs eager parameter loading | Same maximum, mean, and RMSE | Loading mode alone does not fix the discrepancy |
| SDK `precheck()` with either mode | Fails at weight of opid 241 in PLDDR, after `apply()` and after forward | Weight readback is not certified. Could be a real mismatch or a precheck/compiled-graph issue; cause not established |
| Second existing partition, `proj_feature` (one Conv2d) | `precheck()` fails at its opid 22; max abs error 0.00235957, RMSE 0.00017138 | The issue is not confined to the 16-convolution backbone; this diagnostic call also fails the original gate |
| `fp32` compile control | NPU target generated, host output ABI audit failed | Cannot use this artifact through the current bridge without a separately verified layout conversion |

The production baseline calls took 40.43 ms and 24.56 ms end to end for this **single partition**; the SDK compute/wait portions were 14.81 ms and 1.14 ms, with about 10.7-11.4 ms for SFB output conversion. Logical input plus output is 1.375 MiB per call. These are two diagnostic samples with **incorrect output**, not a full-model speedup or a latency estimate for the video pipeline. Their P95/P99 would be meaningless.

Evidence: `baseline.json`, `baseline.log`, `baseline_backbone_cnn_0.npy`, `precheck.json`, `precheck_worker.log`, `eager.json`, `eager_worker.log`, `proj_feature.json`, `proj_feature_worker.log`, `build_v2.log`, `build_eager.log`, and the preserved scripts/sources in this directory. The failed `fp32` artifact and its stage logs remain under the ignored `runs/` path above. The first relative-path compiler invocation failed before creating an output directory; the second invocation produced the audited control artifact.

## Full-chain status and next increment

No new video-to-frame or image-quality run was made in this step, because the NPU subgraph has not passed its numerical gate. The most recent **historical CPU + FPGA** board baseline remains 27.876 s mean from `VIDEO_COMPLETE` to verified first frame (three runs, 128 x 128), versus the provisional 20 s target. Retained-view commands took 0.825-0.829 s, versus the 1 s target. The target view-7 quality was PSNR 19.45 dB, SSIM 0.8045; it misses the existing 20 dB PSNR gate. These are not NPU results. FPGA utilization/timing and power were not remeasured here.

Next, create a *minimal* one-convolution NPU graph with known weights and input; the existing `proj_feature` subgraph contains one Conv2d but has a large 16 MiB input and inherited compiler structure. Check the minimal graph's SDK `precheck()` and numerical output against a host oracle, then add operations only after that base case is sound. This distinguishes device/SDK upload behavior from accumulated TF32 numerical error before choosing precision or layout changes. Keep the current CPU + FPGA route as the usable reference.
