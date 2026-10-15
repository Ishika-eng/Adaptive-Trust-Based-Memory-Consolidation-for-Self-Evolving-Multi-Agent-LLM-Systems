# Results: learning_v4 (local_learning_v4)

Local model, hidden-convention benchmark. Pass rates are on the TRUE outcome of each task. Seeds = independent task orders; every condition sees the same order for a given seed.

## Headline (mean over seeds; +/- = standard deviation across seeds)

| Condition | Seeds | Pass rate | First half | Second half | Rule tasks | Controls | Memory size |
|---|---|---|---|---|---|---|---|
| Plain memory + examples + memory retry | 3 | 0.464 ± 0.040 | 0.424 ± 0.038 | 0.503 ± 0.042 | 0.356 ± 0.059 | 1.000 ± 0.000 | 135 |
| ATMC cards + memory retry (similarity) | 3 | 0.479 ± 0.028 | 0.461 ± 0.046 | 0.497 ± 0.010 | 0.367 ± 0.033 | 1.000 ± 0.000 | 100 |
| ATMC v4 (utility + signature) | 3 | 0.455 ± 0.027 | 0.400 ± 0.031 | 0.509 ± 0.031 | 0.337 ± 0.039 | 1.000 ± 0.000 | 98 |

## Paired comparisons (same tasks, same order)

Difference = B minus A in pass rate over all (seed, task) pairs; 95% bootstrap CI; exact McNemar p-value. Only the pairs both conditions completed are used.

| A | B | n pairs | B - A | 95% CI | B right / A right | McNemar p |
|---|---|---|---|---|---|---|

## Pass rate by family (mean over seeds)

| Family | Plain memory + examples + memory retry | ATMC cards + memory retry (similarity) | ATMC v4 (utility + signature) |
|---|---|---|---|
| control (control) | 1.00 | 1.00 | 1.00 |
| dates_dmy | 0.87 | 0.83 | 0.77 |
| ids_padded | 0.03 | 0.07 | 0.17 |
| log_format | 0.30 | 0.37 | 0.43 |
| money_paise | 0.40 | 0.37 | 0.37 |
| no_mutation (known practice) | 0.90 | 0.97 | 0.97 |
| none_on_invalid | 0.23 | 0.17 | 0.17 |
| result_tuple | 0.07 | 0.00 | 0.00 |
| sorted_unique | 0.50 | 0.60 | 0.30 |
| units_mm | 0.53 | 0.50 | 0.53 |
| username_norm | 0.27 | 0.40 | 0.30 |

## Cost (local model, no API quota)

| Condition | Tokens in per run | Tokens out per run | Wall time per run (min) |
|---|---|---|---|
| Plain memory + examples + memory retry | 71284 | 9354 | 8.0 |
| ATMC cards + memory retry (similarity) | 100842 | 10040 | 9.6 |
| ATMC v4 (utility + signature) | 108070 | 8231 | 9.7 |
