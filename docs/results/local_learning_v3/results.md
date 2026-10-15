# Results: learning_v3 (local_learning_v3)

Local model, hidden-convention benchmark. Pass rates are on the TRUE outcome of each task. Seeds = independent task orders; every condition sees the same order for a given seed.

## Headline (mean over seeds; +/- = standard deviation across seeds)

| Condition | Seeds | Pass rate | First half | Second half | Rule tasks | Controls | Memory size |
|---|---|---|---|---|---|---|---|
| atmc_v3 | 3 | 0.439 ± 0.023 | 0.424 ± 0.028 | 0.455 ± 0.063 | 0.333 ± 0.038 | 1.000 ± 0.000 | 97 |
| atmc_v3_nocards | 3 | 0.406 ± 0.050 | 0.400 ± 0.048 | 0.412 ± 0.056 | 0.285 ± 0.053 | 0.967 ± 0.058 | 137 |
| static_repair | 3 | 0.430 ± 0.023 | 0.406 ± 0.010 | 0.455 ± 0.036 | 0.330 ± 0.028 | 0.967 ± 0.058 | 136 |

## Paired comparisons (same tasks, same order)

Difference = B minus A in pass rate over all (seed, task) pairs; 95% bootstrap CI; exact McNemar p-value. Only the pairs both conditions completed are used.

| A | B | n pairs | B - A | 95% CI | B right / A right | McNemar p |
|---|---|---|---|---|---|---|

## Pass rate by family (mean over seeds)

| Family | atmc_v3 | atmc_v3_nocards | static_repair |
|---|---|---|---|
| control (control) | 1.00 | 0.97 | 0.97 |
| dates_dmy | 0.73 | 0.73 | 0.83 |
| ids_padded | 0.17 | 0.10 | 0.07 |
| log_format | 0.23 | 0.27 | 0.30 |
| money_paise | 0.40 | 0.20 | 0.47 |
| no_mutation (known practice) | 0.83 | 0.93 | 0.80 |
| none_on_invalid | 0.20 | 0.17 | 0.23 |
| result_tuple | 0.00 | 0.00 | 0.03 |
| sorted_unique | 0.53 | 0.43 | 0.23 |
| units_mm | 0.53 | 0.47 | 0.50 |
| username_norm | 0.20 | 0.20 | 0.30 |

## Cost (local model, no API quota)

| Condition | Tokens in per run | Tokens out per run | Wall time per run (min) |
|---|---|---|---|
| atmc_v3 | 78533 | 14487 | 6.0 |
| atmc_v3_nocards | 71236 | 15055 | 8.3 |
| static_repair | 57867 | 14195 | 9.1 |
