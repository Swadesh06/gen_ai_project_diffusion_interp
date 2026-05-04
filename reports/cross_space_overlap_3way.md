# Cross-space SAE feature overlap

Top-k features per block per attack run (k=50); Jaccard between every pair.

## hookpoint `down.2.1`

| run \ run | A01_pixel_eps4_n200 | A02_latent_eps0.1_n200 | A03_emb_eps0.5_n200 |
|---|---|---|---|
| A01_pixel_eps4_n200 | 1.000 | 0.754 | 0.613 |
| A02_latent_eps0.1_n200 | 0.754 | 1.000 | 0.639 |
| A03_emb_eps0.5_n200 | 0.613 | 0.639 | 1.000 |

## hookpoint `mid.0`

| run \ run | A01_pixel_eps4_n200 | A02_latent_eps0.1_n200 | A03_emb_eps0.5_n200 |
|---|---|---|---|
| A01_pixel_eps4_n200 | 1.000 | 0.587 | 0.538 |
| A02_latent_eps0.1_n200 | 0.587 | 1.000 | 0.471 |
| A03_emb_eps0.5_n200 | 0.538 | 0.471 | 1.000 |

## hookpoint `up.0.0`

| run \ run | A01_pixel_eps4_n200 | A02_latent_eps0.1_n200 | A03_emb_eps0.5_n200 |
|---|---|---|---|
| A01_pixel_eps4_n200 | 1.000 | 0.754 | 0.562 |
| A02_latent_eps0.1_n200 | 0.754 | 1.000 | 0.515 |
| A03_emb_eps0.5_n200 | 0.562 | 0.515 | 1.000 |

## hookpoint `up.0.1`

| run \ run | A01_pixel_eps4_n200 | A02_latent_eps0.1_n200 | A03_emb_eps0.5_n200 |
|---|---|---|---|
| A01_pixel_eps4_n200 | 1.000 | 0.786 | 0.724 |
| A02_latent_eps0.1_n200 | 0.786 | 1.000 | 0.639 |
| A03_emb_eps0.5_n200 | 0.724 | 0.639 | 1.000 |
