# 2026-09-30 NPU known-weight stage-cost check

This directory contains a real 30TAI Lite run of three small, known-input
single-convolution graphs. The run uses one session per case and the updated
ABI bridge, which reports `input_write`, `forward_submit`, `wait`,
`output_convert`, and `sdk_total` separately.

All cases passed the unchanged numerical contract. Identity and dyadic cases
were bit-exact; the fractional case had maximum absolute error
`0.000167668`, RMSE `0.000059451`, and passed `rtol=0.002`, `atol=0.0002`.
The two-call means (milliseconds) were:

| graph | input write | submit | wait | output conversion | SDK total |
|---|---:|---:|---:|---:|---:|
| identity | 0.090 | 0.417 | 0.005 | 0.675 | 1.188 |
| dyadic | 0.066 | 0.373 | 0.005 | 0.587 | 1.030 |
| fractional | 0.067 | 0.369 | 0.006 | 0.581 | 1.022 |

These are interface diagnostics, not a MVSplat speedup. The real MVSplat
backbone prefix still fails at its first compiled node, so its timings remain
invalid for performance claims. The raw JSON and board log are kept beside
this file.
