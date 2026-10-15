# Results: learning (local_learning)

Local model, hidden-convention benchmark. Pass rates are on the TRUE outcome of each task. Seeds = independent task orders; every condition sees the same order for a given seed.

## Headline (mean over seeds; +/- = standard deviation across seeds)

| Condition | Seeds | Pass rate | First half | Second half | Rule tasks | Controls | Memory size |
|---|---|---|---|---|---|---|---|
| No memory | 3 | 0.242 ± 0.014 | 0.255 ± 0.066 | 0.230 ± 0.042 | 0.085 ± 0.017 | 1.000 ± 0.000 | 0 |
| Static memory | 3 | 0.400 ± 0.051 | 0.333 ± 0.064 | 0.467 ± 0.042 | 0.274 ± 0.068 | 1.000 ± 0.000 | 110 |
| ATMC without trust | 3 | 0.248 ± 0.005 | 0.255 ± 0.055 | 0.242 ± 0.046 | 0.081 ± 0.006 | 1.000 ± 0.000 | 82 |
| ATMC without merging | 3 | 0.261 ± 0.046 | 0.261 ± 0.076 | 0.261 ± 0.021 | 0.115 ± 0.050 | 0.933 ± 0.058 | 76 |
| ATMC (full) | 3 | 0.285 ± 0.005 | 0.267 ± 0.028 | 0.303 ± 0.038 | 0.133 ± 0.011 | 1.000 ± 0.000 | 74 |
| ATMC, similarity-first ranking | 3 | 0.282 ± 0.016 | 0.291 ± 0.066 | 0.273 ± 0.066 | 0.122 ± 0.019 | 1.000 ± 0.000 | 74 |
| ATMC, no forgetting | 3 | 0.288 ± 0.026 | 0.285 ± 0.038 | 0.291 ± 0.073 | 0.130 ± 0.032 | 1.000 ± 0.000 | 84 |
| ATMC, sim-first + no forgetting | 3 | 0.276 ± 0.029 | 0.279 ± 0.028 | 0.273 ± 0.079 | 0.115 ± 0.036 | 1.000 ± 0.000 | 84 |
| ATMC, all pillars off + no hit boost | 3 | 0.397 ± 0.056 | 0.339 ± 0.076 | 0.455 ± 0.036 | 0.270 ± 0.074 | 1.000 ± 0.000 | 110 |
| ATMC, trust only + no hit boost | 3 | 0.261 ± 0.023 | 0.267 ± 0.038 | 0.255 ± 0.048 | 0.104 ± 0.028 | 0.967 ± 0.058 | 110 |
| ATMC (full), no hit boost | 3 | 0.270 ± 0.034 | 0.261 ± 0.042 | 0.279 ± 0.093 | 0.111 ± 0.038 | 0.967 ± 0.058 | 78 |
| ATMC + verified-evidence trust | 3 | 0.248 ± 0.005 | 0.255 ± 0.036 | 0.242 ± 0.046 | 0.085 ± 0.006 | 0.967 ± 0.058 | 77 |
| ATMC + verified-evidence trust, no hit boost | 3 | 0.309 ± 0.009 | 0.315 ± 0.056 | 0.303 ± 0.064 | 0.159 ± 0.017 | 1.000 ± 0.000 | 84 |
| Ladder 1: + trust | 3 | 0.382 ± 0.016 | 0.352 ± 0.069 | 0.412 ± 0.038 | 0.259 ± 0.023 | 0.967 ± 0.058 | 110 |
| Ladder 2: + trust + forgetting | 3 | 0.306 ± 0.023 | 0.297 ± 0.073 | 0.315 ± 0.052 | 0.156 ± 0.033 | 1.000 ± 0.000 | 109 |

## Paired comparisons (same tasks, same order)

Difference = B minus A in pass rate over all (seed, task) pairs; 95% bootstrap CI; exact McNemar p-value. Only the pairs both conditions completed are used.

| A | B | n pairs | B - A | 95% CI | B right / A right | McNemar p |
|---|---|---|---|---|---|---|
| No memory | Static memory | 330 | +0.158 | [+0.109, +0.206] | 64 / 12 | 0.0000 |
| No memory | ATMC without trust | 330 | +0.006 | [-0.021, +0.033] | 12 / 10 | 0.8318 |
| No memory | ATMC without merging | 330 | +0.018 | [-0.015, +0.052] | 18 / 12 | 0.3616 |
| No memory | ATMC (full) | 330 | +0.042 | [+0.009, +0.076] | 23 / 9 | 0.0201 |
| No memory | ATMC, similarity-first ranking | 330 | +0.039 | [+0.009, +0.070] | 20 / 7 | 0.0192 |
| No memory | ATMC, no forgetting | 330 | +0.045 | [+0.015, +0.079] | 22 / 7 | 0.0081 |
| No memory | ATMC, sim-first + no forgetting | 330 | +0.033 | [+0.003, +0.067] | 20 / 9 | 0.0614 |
| No memory | ATMC, all pillars off + no hit boost | 330 | +0.155 | [+0.106, +0.203] | 62 / 11 | 0.0000 |
| No memory | ATMC, trust only + no hit boost | 330 | +0.018 | [-0.015, +0.052] | 17 / 11 | 0.3449 |
| No memory | ATMC (full), no hit boost | 330 | +0.027 | [-0.006, +0.061] | 21 / 12 | 0.1628 |
| No memory | ATMC + verified-evidence trust | 330 | +0.006 | [-0.024, +0.036] | 13 / 11 | 0.8388 |
| No memory | ATMC + verified-evidence trust, no hit boost | 330 | +0.067 | [+0.030, +0.103] | 30 / 8 | 0.0005 |
| No memory | Ladder 1: + trust | 330 | +0.139 | [+0.091, +0.188] | 59 / 13 | 0.0000 |
| No memory | Ladder 2: + trust + forgetting | 330 | +0.064 | [+0.027, +0.100] | 30 / 9 | 0.0011 |
| Static memory | ATMC (full) | 330 | -0.115 | [-0.161, -0.070] | 13 / 51 | 0.0000 |
| ATMC without trust | ATMC (full) | 330 | +0.036 | [+0.009, +0.064] | 17 / 5 | 0.0169 |
| ATMC without merging | ATMC (full) | 330 | +0.024 | [+0.000, +0.048] | 12 / 4 | 0.0768 |
| Static memory | ATMC, similarity-first ranking | 330 | -0.118 | [-0.164, -0.073] | 12 / 51 | 0.0000 |
| Static memory | ATMC, no forgetting | 330 | -0.112 | [-0.161, -0.067] | 14 / 51 | 0.0000 |
| Static memory | ATMC, sim-first + no forgetting | 330 | -0.124 | [-0.170, -0.079] | 10 / 51 | 0.0000 |
| ATMC (full) | ATMC, similarity-first ranking | 330 | -0.003 | [-0.024, +0.018] | 6 / 7 | 1.0000 |
| ATMC (full) | ATMC, no forgetting | 330 | +0.003 | [-0.021, +0.027] | 9 / 8 | 1.0000 |
| ATMC (full) | ATMC, sim-first + no forgetting | 330 | -0.009 | [-0.030, +0.009] | 4 / 7 | 0.5488 |

## Pass rate by family (mean over seeds)

| Family | No memory | Static memory | ATMC without trust | ATMC without merging | ATMC (full) | ATMC, similarity-first ranking | ATMC, no forgetting | ATMC, sim-first + no forgetting | ATMC, all pillars off + no hit boost | ATMC, trust only + no hit boost | ATMC (full), no hit boost | ATMC + verified-evidence trust | ATMC + verified-evidence trust, no hit boost | Ladder 1: + trust | Ladder 2: + trust + forgetting |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control (control) | 1.00 | 1.00 | 1.00 | 0.93 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 | 0.97 | 0.97 | 1.00 | 0.97 | 1.00 |
| dates_dmy | 0.00 | 0.70 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.70 | 0.00 | 0.00 | 0.00 | 0.33 | 0.70 | 0.33 |
| ids_padded | 0.00 | 0.03 | 0.00 | 0.00 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| log_format | 0.00 | 0.23 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.23 | 0.10 | 0.10 | 0.10 | 0.10 | 0.23 | 0.10 |
| money_paise | 0.10 | 0.20 | 0.10 | 0.10 | 0.07 | 0.07 | 0.07 | 0.07 | 0.20 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 |
| no_mutation (known practice) | 0.90 | 0.93 | 1.00 | 0.90 | 0.93 | 1.00 | 1.00 | 1.00 | 0.93 | 0.97 | 1.00 | 1.00 | 0.97 | 0.90 | 0.97 |
| none_on_invalid | 0.10 | 0.20 | 0.13 | 0.37 | 0.37 | 0.23 | 0.33 | 0.23 | 0.20 | 0.23 | 0.23 | 0.13 | 0.23 | 0.17 | 0.23 |
| result_tuple | 0.00 | 0.03 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.00 | 0.00 | 0.00 | 0.00 | 0.03 | 0.00 |
| sorted_unique | 0.27 | 0.37 | 0.20 | 0.23 | 0.40 | 0.43 | 0.40 | 0.37 | 0.33 | 0.27 | 0.37 | 0.23 | 0.37 | 0.47 | 0.33 |
| units_mm | 0.10 | 0.43 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.10 | 0.43 | 0.10 | 0.10 | 0.10 | 0.20 | 0.43 | 0.20 |
| username_norm | 0.20 | 0.27 | 0.10 | 0.13 | 0.13 | 0.13 | 0.13 | 0.13 | 0.27 | 0.13 | 0.10 | 0.10 | 0.10 | 0.20 | 0.10 |

## Cost (local model, no API quota)

| Condition | Tokens in per run | Tokens out per run | Wall time per run (min) |
|---|---|---|---|
| No memory | 9145 | 8463 | 0.3 |
| Static memory | 40586 | 4195 | 4.2 |
| ATMC without trust | 43055 | 5427 | 3.9 |
| ATMC without merging | 38721 | 4660 | 3.1 |
| ATMC (full) | 39610 | 6047 | 3.8 |
| ATMC, similarity-first ranking | 39434 | 5896 | 3.2 |
| ATMC, no forgetting | 39798 | 5956 | 1.3 |
| ATMC, sim-first + no forgetting | 39448 | 5930 | 0.1 |
| ATMC, all pillars off + no hit boost | 40589 | 4189 | 0.5 |
| ATMC, trust only + no hit boost | 36392 | 4467 | 3.5 |
| ATMC (full), no hit boost | 41010 | 5258 | 4.0 |
| ATMC + verified-evidence trust | 41848 | 5485 | 1.9 |
| ATMC + verified-evidence trust, no hit boost | 44744 | 5423 | 5.2 |
| Ladder 1: + trust | 40341 | 4422 | 2.3 |
| Ladder 2: + trust + forgetting | 40705 | 4219 | 0.9 |
