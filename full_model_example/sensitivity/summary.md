# Sensitivity screen results

This is a low-order numerical screen, not a high-precision result or an empirical plausibility claim. The fixed grid and integration choices are in [protocol.md](protocol.md); all machine-readable outputs are in [results.json](results.json).

Elapsed time: 37.2 seconds. Rows: 81. Certification/root-bracketing failures: 0.

Rows with mean profit loss greater than 1% of the full baseline profit:

| D_A=D_V | CAC | N | D_H | Mean Δ / 100k | MCSE / 100k | Δ / baseline | Baseline / 100k | Clipped |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 150 | 64 | 16 | -224.96 | 0.66 | -77.985% | 288.47 | 41.565% |
| 16 | 150 | 64 | 4 | -108.43 | 0.47 | -37.589% | 288.47 | 14.436% |
| 16 | 75 | 64 | 16 | -14462.71 | 48.44 | -19.683% | 73478.56 | 41.565% |
| 16 | 150 | 512 | 16 | -48.56 | 0.28 | -16.833% | 288.47 | 4.040% |
| 16 | 150 | 64 | 1 | -35.45 | 0.23 | -12.289% | 288.47 | 2.588% |
| 16 | 75 | 64 | 4 | -7197.78 | 34.77 | -9.796% | 73478.56 | 14.436% |
| 4 | 150 | 64 | 16 | -3857.32 | 13.03 | -8.731% | 44181.26 | 41.565% |
| 4 | 150 | 64 | 4 | -1924.51 | 9.41 | -4.356% | 44181.26 | 14.436% |
| 4 | 75 | 64 | 16 | -17804.39 | 60.81 | -4.214% | 422513.03 | 41.565% |
| 16 | 75 | 512 | 16 | -3045.92 | 22.28 | -4.145% | 73478.56 | 4.040% |
| 16 | 75 | 64 | 1 | -2061.25 | 18.69 | -2.805% | 73478.56 | 2.588% |
| 4 | 75 | 64 | 4 | -8791.40 | 43.32 | -2.081% | 422513.03 | 14.436% |
| 1 | 150 | 64 | 16 | -4586.19 | 15.46 | -2.043% | 224462.64 | 41.565% |
| 4 | 150 | 512 | 16 | -808.91 | 6.08 | -1.831% | 44181.26 | 4.040% |
| 16 | 25 | 64 | 16 | -25487.98 | 86.92 | -1.602% | 1591221.94 | 41.565% |
| 4 | 150 | 64 | 1 | -542.62 | 5.12 | -1.228% | 44181.26 | 2.588% |

Deployment noise multiplier is applied to both activity and turnover observation-noise variances. H-reference noise is varied independently. Incremental effects are evaluated within the fixed application gate; baseline profit includes the full accepted region. All monetary amounts are in parrots per 100,000 incoming candidates.
