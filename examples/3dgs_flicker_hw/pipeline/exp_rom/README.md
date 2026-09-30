# Exact negative-exponent ROM for the four-lane renderer

This candidate replaces the existing `hls::half_exp` call with a BRAM lookup.
It changes the circuit implementing the exponent, not Gaussian count, thresholds,
depth order, color arithmetic, FPGA frequency or the software/FPGA ABI.
The original HLS path remains the default when `FLK_EXACT_EXP_ROM` is unset.

## Why this is possible

`evaluate_mini` calls the exponent only when FP16 `power <= 0`. That restricts
the input to 31,746 encodings, including both zeros and negative infinity. NaNs
do not enter this branch. `generate.cpp` evaluates every such negative encoding
with the **installed Vivado HLS 2018.3 math implementation**, retaining the exact
result bits. The original rounding is retained; this is not host `libm` or a
new approximation to the mathematical exponential.

The first 3,585 magnitude codes always produce 1.0, and codes above 18,650
produce zero. Removing those constant ranges leaves 15,066 stored values
(30,132 logical bytes). Each lane has a private ROM so independent reads remain
possible. The full untrimmed lookup is retained as an area comparison.

## Completed checks

- Exponent probe C simulation and Verilog co-simulation: all 31,746 encodings
  bit-identical to the HLS implementation, including positive/negative zero and
  negative infinity.
- Full four-lane renderer C/RTL co-simulation passes seven transactions.
- Actual generated RTL plus DMA, registers and CDC passes six render modes,
  4,608 pixel comparisons, stalls, guard checks and buffer ownership checks.
  This uses 51 real records over three Tiles and is a bounded protocol test,
  not a full-scene board speed result.

| HLS estimate, renderer only | Original split pipeline | Full lookup | Compact lookup |
|---|---:|---:|---:|
| BRAM18 | 167 | 275 | 219 |
| DSP | 252 | 244 | 244 |
| FF | 67,395 | 64,215 | 64,227 |
| LUT | 38,183 | 34,775 | 34,927 |
| 16-pixel evaluate call latency | 145 cycles | 133 cycles | 133 cycles |

The exponent probe's pipelined loop latency falls from 17 to 5 cycles; its
initiation interval stays **one**. Therefore this is not a 3.4× throughput claim.
In the real six-mode integration test, original → compact cycles are:
30,888→28,966; 28,174→26,426; 5,042→4,861; 5,035→4,807;
5,017→4,781; 5,028→4,803. Output values are identical. Full-frame speed may
improve less because projection, sorting, geometry preparation and DMA remain.

HLS's reported estimated worst period remains 5.292 ns for both full renderers.
This is an estimate, not routed timing. Only a completed physical implementation
and subsequent board measurements can establish whether the candidate is usable
at the existing 200 MHz and whether it accelerates the scene.

## Reproduction

Use the installed Vivado HLS environment. Run `build.tcl` in this directory for
the original/full-lookup exponent probes, then `compact.tcl` for the compact
generator and exhaustive C/RTL test. These create separate project directories.

For the complete renderer, set `FLK_HLS_PROJECT` to a new name and
`FLK_EXACT_EXP_ROM=2`, then run the parent `pipeline/build.tcl`. Value 1 selects
the full lookup; unset the variable for the original implementation. The parent
script checks II=1 before full co-simulation. `pipeline/prepare_project.py`
packages the generated RTL into a separate platform; `pipeline/build_board.ps1`
then performs physical implementation. The optional `FLK_INCREMENTAL_DCP`
references the preserved baseline placement; it does not replace that baseline.

Small report copies, source hashes and the exact integration record live in
`results/20260930`. Original HLS projects and physical build output remain in
the ignored `../../build` tree. See the accompanying board validation record
for final physical/board status; HLS success alone is not deployment success.
