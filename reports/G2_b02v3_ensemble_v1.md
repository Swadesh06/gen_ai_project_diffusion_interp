# G2 — B02-v3 ensemble across 10 heads (cell 2.5)

Heads: 10 (linear + MLP × 5 hookpoints incl. concat-all)

## Headline table — AUC by (strategy × dataset)

| strategy | b02v3_val |
|---|---|
| single_best | 0.9772 |
| mean_prob | 0.9805 |
| max_prob | 0.9827 |
| vote | 0.9545 |
| stacker | 0.9760 |

## AP by (strategy × dataset)

| strategy | b02v3_val |
|---|---|
| single_best | 0.9444 |
| mean_prob | 0.9408 |
| max_prob | 0.9408 |
| vote | 0.9173 |
| stacker | 0.9403 |

## Per-dataset sample counts (n / n_pos)

| dataset | n | n_pos |
|---|---|---|
| b02v3_val | 308 | 36 |
