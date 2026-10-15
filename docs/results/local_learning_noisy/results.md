# Results: learning_noisy (local_learning_noisy)

Local model, hidden-convention benchmark. Pass rates are on the TRUE outcome of each task. Seeds = independent task orders; every condition sees the same order for a given seed.

## Headline (mean over seeds; +/- = standard deviation across seeds)

| Condition | Seeds | Pass rate | First half | Second half | Rule tasks | Controls | Memory size |
|---|---|---|---|---|---|---|---|
| No memory | 3 | 0.239 ± 0.005 | 0.242 ± 0.046 | 0.236 ± 0.055 | 0.081 ± 0.006 | 1.000 ± 0.000 | 0 |
| Static memory | 3 | 0.370 ± 0.069 | 0.333 ± 0.056 | 0.406 ± 0.084 | 0.237 ± 0.090 | 1.000 ± 0.000 | 110 |
| ATMC without trust | 3 | 0.245 ± 0.009 | 0.261 ± 0.028 | 0.230 ± 0.046 | 0.081 ± 0.017 | 1.000 ± 0.000 | 82 |
| ATMC without merging | 3 | 0.248 ± 0.014 | 0.255 ± 0.018 | 0.242 ± 0.028 | 0.093 ± 0.028 | 0.967 ± 0.058 | 76 |
| ATMC (full) | 3 | 0.273 ± 0.024 | 0.279 ± 0.042 | 0.267 ± 0.073 | 0.115 ± 0.036 | 1.000 ± 0.000 | 75 |

## Paired comparisons (same tasks, same order)

Difference = B minus A in pass rate over all (seed, task) pairs; 95% bootstrap CI; exact McNemar p-value. Only the pairs both conditions completed are used.

| A | B | n pairs | B - A | 95% CI | B right / A right | McNemar p |
|---|---|---|---|---|---|---|
| No memory | Static memory | 330 | +0.130 | [+0.088, +0.176] | 52 / 9 | 0.0000 |
| No memory | ATMC without trust | 330 | +0.006 | [-0.024, +0.036] | 14 / 12 | 0.8450 |
| No memory | ATMC without merging | 330 | +0.009 | [-0.024, +0.042] | 16 / 13 | 0.7111 |
| No memory | ATMC (full) | 330 | +0.033 | [+0.003, +0.067] | 20 / 9 | 0.0614 |
| Static memory | ATMC (full) | 330 | -0.097 | [-0.136, -0.061] | 7 / 39 | 0.0000 |
| ATMC without trust | ATMC (full) | 330 | +0.027 | [+0.000, +0.055] | 14 / 5 | 0.0636 |
| ATMC without merging | ATMC (full) | 330 | +0.024 | [+0.006, +0.045] | 10 / 2 | 0.0386 |

## Pass rate by family (mean over seeds)

| Family | No memory | Static memory | ATMC without trust | ATMC without merging | ATMC (full) |
|---|---|---|---|---|---|
| control (control) | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 |
| dates_dmy | 0.00 | 0.57 | 0.00 | 0.00 | 0.00 |
| ids_padded | 0.00 | 0.03 | 0.00 | 0.00 | 0.03 |
| log_format | 0.00 | 0.20 | 0.10 | 0.10 | 0.10 |
| money_paise | 0.10 | 0.07 | 0.07 | 0.03 | 0.07 |
| no_mutation (known practice) | 0.90 | 0.93 | 0.97 | 0.93 | 0.97 |
| none_on_invalid | 0.10 | 0.23 | 0.13 | 0.27 | 0.23 |
| result_tuple | 0.00 | 0.03 | 0.00 | 0.00 | 0.00 |
| sorted_unique | 0.23 | 0.33 | 0.23 | 0.20 | 0.37 |
| units_mm | 0.10 | 0.43 | 0.10 | 0.10 | 0.10 |
| username_norm | 0.20 | 0.23 | 0.10 | 0.13 | 0.13 |

## Cost (local model, no API quota)

| Condition | Tokens in per run | Tokens out per run | Wall time per run (min) |
|---|---|---|---|
| No memory | 9145 | 8463 | 0.1 |
| Static memory | 40002 | 4138 | 2.6 |
| ATMC without trust | 43690 | 5205 | 3.2 |
| ATMC without merging | 39350 | 4598 | 2.0 |
| ATMC (full) | 41117 | 6255 | 3.7 |
