# Whole-grid refinement

All 81 rows were recomputed with the same Wishart draws/seed as the screening run. Fitted-tau curves use an 81-point grid, outer/inner orders 14/20, and vertical order 16; baseline profit uses outer/inner orders 40/28 and vertical order 72. The refined arrays are in [refined_results.json](refined_results.json), with separate `*_refined_curve.json` checkpoints.

Runtime: 309.6 seconds. Certification/root failures: 0. Largest absolute screen-to-refinement change in relative mean profit: 0.005314 percentage points (all 81 rows).

The N=512 cases with negative mean effects are concentrated in the high H-reference-noise setting (`D_H=16`). Examples:

| D_A=D_V | CAC | D_H | Mean Δ / 100k | Δ / baseline | MCSE / 100k | Clipped |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 150 | 16 | -48.56 | -16.832% | 0.28 | 4.040% |
| 16 | 75 | 16 | -3045.92 | -4.145% | 22.28 | 4.040% |
| 4 | 150 | 16 | -808.90 | -1.831% | 6.08 | 4.040% |
| 4 | 75 | 16 | -3670.32 | -0.868% | 27.60 | 4.040% |
| 1 | 150 | 16 | -937.64 | -0.418% | 6.83 | 4.040% |
| 16 | 150 | 4 | -1.47 | -0.510% | 0.09 | 0.040% |

At `D_A=D_V=16`, `CAC=150`, baseline profit is only 288.47 per 100k. Thus the large relative percentages at this edge correspond to small absolute changes (for example, at N=64 and D_H=16, Δ≈−224.96); the percentage is denominator-sensitive. These grid points are mathematical stress cases, not evidence that such settings are typical. The parameter region needs product data for empirical assessment.

Monte Carlo error is reported per row; this refinement checks quadrature/interpolation sensitivity over the whole grid but is not a universal numerical error bound or an empirical validation.
