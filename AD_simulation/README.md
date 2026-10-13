# AD simulation — directional strain ranking proof of concept

Simulated proof-of-concept for the proposed Adaptive Directional (AD) distance
for multi-trait microbial strain prioritization. **The simulated data do not
represent experimental measurements and are not intended to demonstrate
biological efficacy.**

## Layout

| Path | Contents |
| :--- | :------- |
| `data/simulated_strains.csv` | 100 hypothetical strains x 6 traits (95 random + 5 designed edge cases) |
| `scripts/generate_simulation.py` | Builds the simulated panel (seed 42) |
| `scripts/symmetric_distance.py` | Conventional weighted Euclidean to the ideal |
| `scripts/ad_distance.py` | Directional AD (deficit penalty R, excess unpenalized) |
| `scripts/ranking_comparison.py` | Ranks both ways; `results/rankings.csv` with rank changes |
| `scripts/sensitivity_analysis.py` | Penalty ratios 1:1–10:1 + trait-noise uncertainty |
| `scripts/figure_5.py` | Composite Figure 5 (panels A–F) |
| `scripts/run_all.py` | Runs the full pipeline in order |
| `figures/Figure_5.png` | A: real ML evidence … F: sensitivity |
| `results/` | `rankings.csv`, `sensitivity.csv`, `uncertainty.csv` |

## Run

```bash
cd AD_simulation
python scripts/run_all.py
```

Requirements: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `scipy`.
