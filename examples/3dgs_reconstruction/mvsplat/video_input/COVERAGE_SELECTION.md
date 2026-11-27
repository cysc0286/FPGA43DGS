# Recorded-view coverage selection

The manual-camera deployment must use automatic measured coverage, not the
temporary fixed-one-pair smoke-test profile. Recommended launch options:

```text
--backend cpu --max-seconds 15 --ring-pairs 2 --ring-auto-coverage --ring-min-pairs 1
--ring-candidates 64 --ring-selection-policy coverage --ring-coverage-target 1.0
--ring-max-gaussians 65536 --ring-appearance-strength 0
```

Two is the current web deployment's maximum group budget, reduced from eight
after a three-pair run hit the host reserve guard on its second inference.
This cap is not proof that all two-pair inputs fit memory. A partial sweep can finish with fewer groups
when no unused validated pair adds coverage. A target of 1.0 requests full
horizontal coverage; it does not turn a partial recording into 360 degrees.
The explicit minimum of one permits short local sweeps; the legacy CLI default
minimum of eight remains available when explicitly selecting an eight-group experiment.
Generation is sequential and keeps the existing bounded point allocation and
700 MiB PSS / 96 MiB host-reserve protections in the web launcher.
Preparation drops decoded frame buffers after contexts have been persisted and
asks glibc to return unused allocator pages before inference. This is optional
on non-glibc hosts; it does not change prepared images, poses or the memory gates.

## Slow sweeps and dense candidates

The explicit `coverage` selection policy considers nonadjacent measured endpoints. With at most
64 candidates, there are at most 1,953 endpoint pairs with an intervening target.
The existing 5–55 degree separation, cropped-frustum overlap and 85%-of-FOV
limit still apply. Endpoints also require a measured unwrapped yaw separation
of 5–55 degrees, preventing a near-full turn from masquerading as a local pair.
Each pair holds out the intervening image nearest its measured yaw midpoint.
Contexts and held-out targets remain globally disjoint. Only three actual image
indices are reserved per group, and zero-gain groups are skipped once the
explicit minimum is satisfied. The fixed 1/2/4-group selector is unchanged.

This avoids rejecting every pair in a densely sampled slow sweep merely because
two *adjacent* candidate intervals span less than five degrees. It also lets
the greedy coverage selector include the ends of the observed sweep instead of
retaining only a narrow interior interval. It does not relax rotation matching,
invent a baseline, enlarge the focal field or force full-circle poses.

## Validation, 2026-11-21

114 local tests passed across coverage, decoding, rotation graphs, generation,
pipeline coordination, exposure and measured-FPS capture. Tests include dense
27-degree sweeps, reverse motion, crossing zero, out-and-back motion, held-out
target exclusion, full-circle coverage and rejection of invalid overlap. These
114 checks and the board results below are from November 21; November 27 merges
the policy without replacing main's existing `balanced`/`legacy` implementations.

For the same downloaded 729-frame USB clip and measured calibration, the old
fixed-one-pair/16-candidate result covered 42 horizontal one-degree ray samples.
Before the pairing fix, auto/64 candidates covered 46 samples from two groups,
despite all 64 registered views collectively covering 58. After the fix,
auto/64 selected endpoints 0 and 543 with held-out target 208, covering all 58
available samples in one group. These are geometric frustum estimates, not
rendered image quality or depth validation; local and board runs are recorded
separately in the task evidence under workspace tmp/coverage_fix_20261121.
The first board attempt confirmed the 58-sample coverage but stopped during
inference at the unchanged host reserve gate (minimum sampled MemAvailable
95.5 MiB, limit 96 MiB; pipeline-tree peak PSS 478.415 MiB). It is retained as a
failed attempt, not a completed rendering. Allocator release was then added to
avoid retaining preparation workspaces across that stage boundary.

The follow-up real board run completed CPU generation and the existing FPGA
first frame with the allocator release enabled. It registered 64/64 candidates,
retained 32,467 Gaussians, and confirmed the same 58-degree sampled coverage.
One broad pair already spans the available cropped directions in this clip;
additional zero-gain groups were correctly omitted. The context indices and
context content digest match the failed attempt, so releasing allocator pages
did not change this prepared input.

For this single rerun, VIDEO_COMPLETE→FRAME_COMPLETE was 85.028 s, versus
62.392 s in the original 16-candidate fixed-one-pair run. The web clock was
139.181 s because this retry also included model initialization. This is a
coverage improvement with a preparation-time cost, not a speedup. Preparation
took 61.769 s before a 0.648 s allocator-release step; model inference took
17.909 s. The first-frame renderer reported 116.072 ms host wall time,
113.170 ms native time and 84.437 ms engine time. Four viewer confirmations at
yaw 0/13/26/13 degrees returned HTTP 200 and 41.23–44.24 ms reported render
times; the repeated 13-degree image hashes matched exactly. These are a few
frame measurements, not sustained FPS or P95.

Against the new held-out target 208, single-frame PSNR was 20.460 dB and SSIM
0.7024 (RGB, Gaussian 11×11/sigma 1.5, valid crop). The old test used target 242,
so these numbers are not a paired image-quality comparison. Blur and artifacts
remain. All 152 frozen render_main files retain their pre-change hashes; no
firmware or motor change was made. The 15-second limit, measured-FPS policy,
exposure controls and upload page were verified after deployment. Full-circle
new-camera coverage, peak memory for the successful still-running viewer,
LPIPS, hardware utilization, true DDR traffic and power were not measured in
this change. The failed reserve-gate result remains part of the evidence.

The model still uses centered 128×128 crops. Multi-group coverage can accumulate
observed crop directions, but it cannot recover the discarded horizontal edges
of each original wide frame. Pure camera rotation also leaves depth unobserved.
More pairs cost inference time, and splitting a fixed Gaussian budget over more
groups can reduce detail. Power, true DDR traffic and hardware utilization are
not inferred from these geometric coverage measurements.
