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

## Paired comparisons (same tasks, same order)

Difference = B minus A in pass rate over all (seed, task) pairs; 95% bootstrap CI; exact McNemar p-value. Only the pairs both conditions completed are used.

| A | B | n pairs | B - A | 95% CI | B right / A right | McNemar p |
|---|---|---|---|---|---|---|
| No memory | Static memory | 330 | +0.158 | [+0.109, +0.206] | 64 / 12 | 0.0000 |
| No memory | ATMC without trust | 330 | +0.006 | [-0.021, +0.033] | 12 / 10 | 0.8318 |
| No memory | ATMC without merging | 330 | +0.018 | [-0.015, +0.052] | 18 / 12 | 0.3616 |
| No memory | ATMC (full) | 330 | +0.042 | [+0.009, +0.076] | 23 / 9 | 0.0201 |
| Static memory | ATMC (full) | 330 | -0.115 | [-0.161, -0.070] | 13 / 51 | 0.0000 |
| ATMC without trust | ATMC (full) | 330 | +0.036 | [+0.009, +0.064] | 17 / 5 | 0.0169 |
| ATMC without merging | ATMC (full) | 330 | +0.024 | [+0.000, +0.048] | 12 / 4 | 0.0768 |

## Pass rate by family (mean over seeds)

| Family | No memory | Static memory | ATMC without trust | ATMC without merging | ATMC (full) |
|---|---|---|---|---|---|
| control (control) | 1.00 | 1.00 | 1.00 | 0.93 | 1.00 |
| dates_dmy | 0.00 | 0.70 | 0.00 | 0.00 | 0.00 |
| ids_padded | 0.00 | 0.03 | 0.00 | 0.00 | 0.03 |
| log_format | 0.00 | 0.23 | 0.10 | 0.10 | 0.10 |
| money_paise | 0.10 | 0.20 | 0.10 | 0.10 | 0.07 |
| no_mutation (known practice) | 0.90 | 0.93 | 1.00 | 0.90 | 0.93 |
| none_on_invalid | 0.10 | 0.20 | 0.13 | 0.37 | 0.37 |
| result_tuple | 0.00 | 0.03 | 0.00 | 0.00 | 0.00 |
| sorted_unique | 0.27 | 0.37 | 0.20 | 0.23 | 0.40 |
| units_mm | 0.10 | 0.43 | 0.10 | 0.10 | 0.10 |
| username_norm | 0.20 | 0.27 | 0.10 | 0.13 | 0.13 |

## Cost (local model, no API quota)

| Condition | Tokens in per run | Tokens out per run | Wall time per run (min) |
|---|---|---|---|
| No memory | 9145 | 8463 | 0.3 |
| Static memory | 40586 | 4195 | 4.2 |
| ATMC without trust | 43055 | 5427 | 3.9 |
| ATMC without merging | 38721 | 4660 | 3.1 |
| ATMC (full) | 39610 | 6047 | 3.8 |
