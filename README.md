# AD package — strain ranking + 8-distance violins on simulated sets

Self-contained analysis folder. Inputs, scripts, results and figures all live here.

## Layout

| Path | Contents |
| :--- | :------- |
| `inputs/` | 3 simulated datasets + `traits_d1/d2/d3.csv` + `strain_traits.csv` template |
| `scripts/` | `ad_8distance_violins.py`, `ad_strain_ranking.py` (run from anywhere) |
| `results/` | `strain_rankings_d*.csv`, `sensitivity_table_d*.csv` (+ `_full`) |
| `figures/` | `violins_8_d*.png`, `Fig_AD_concept.png`, `Fig_ranking_comparison_d*.png` |

## Run

```bash
cd AD_package
python scripts/ad_8distance_violins.py [--datasets d1 d2 d3] [--out DIR]
python scripts/ad_strain_ranking.py --traits inputs/traits_d1.csv --tag d1 [--top 20] [--alpha 3]
```

Fill `inputs/strain_traits.csv` with real strain rows (0–100 scale) to rank your own panel instead of the simulated demo sets.
