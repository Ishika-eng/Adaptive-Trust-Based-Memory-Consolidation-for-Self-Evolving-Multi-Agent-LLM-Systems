# Results: learning_v3 (local_learning_v3)

Local model, hidden-convention benchmark. Pass rates are on the TRUE outcome of each task. Seeds = independent task orders; every condition sees the same order for a given seed.

## Headline (mean over seeds; +/- = standard deviation across seeds)

| Condition | Seeds | Pass rate | First half | Second half | Rule tasks | Controls | Memory size |
|---|---|---|---|---|---|---|---|
| atmc_v3_nocards | 1 | 0.382 | 0.364 | 0.400 | 0.244 | 1.000 | 139 |
| static_repair | 1 | 0.409 | 0.400 | 0.418 | 0.300 | 0.900 | 138 |

## Paired comparisons (same tasks, same order)

Difference = B minus A in pass rate over all (seed, task) pairs; 95% bootstrap CI; exact McNemar p-value. Only the pairs both conditions completed are used.

| A | B | n pairs | B - A | 95% CI | B right / A right | McNemar p |
|---|---|---|---|---|---|---|

## Pass rate by family (mean over seeds)

| Family | atmc_v3_nocards | static_repair |
|---|---|---|
| control (control) | 1.00 | 0.90 |
| dates_dmy | 0.70 | 0.80 |
| ids_padded | 0.00 | 0.00 |
| log_format | 0.20 | 0.30 |
| money_paise | 0.30 | 0.50 |
| no_mutation (known practice) | 1.00 | 0.90 |
| none_on_invalid | 0.00 | 0.00 |
| result_tuple | 0.00 | 0.00 |
| sorted_unique | 0.40 | 0.30 |
| units_mm | 0.30 | 0.50 |
| username_norm | 0.30 | 0.30 |

## Cost (local model, no API quota)

| Condition | Tokens in per run | Tokens out per run | Wall time per run (min) |
|---|---|---|---|
| atmc_v3_nocards | 70815 | 16270 | 7.3 |
| static_repair | 58098 | 14635 | 11.8 |
