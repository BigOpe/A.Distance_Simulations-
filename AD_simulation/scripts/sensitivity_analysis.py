# sensitivity_analysis.py — (a) penalty ratios 1:1/2:1/5:1/10:1: top-10 overlap
# and Spearman vs the symmetric ranking; (b) uncertainty extension: trait
# noise SD=5, 200 Monte-Carlo draws, rank mean+/-SD per strain.
# Outputs: results/sensitivity.csv, results/uncertainty.csv

from pathlib import Path

import numpy as np
import pandas as pd

from ad_distance import ad_distance
from symmetric_distance import symmetric_distance

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "simulated_strains.csv"
RES = HERE.parent / "results"
RES.mkdir(parents=True, exist_ok=True)
RATIOS = [1, 2, 5, 10]

df = pd.read_csv(DATA)
traits = [c for c in df.columns if c != "Strain"]
X = df[traits].to_numpy(dtype=float)
strains = df["Strain"].tolist()

sym_rank = pd.Series(symmetric_distance(X)).rank(method="min").to_numpy()
rows = []
ranks = {}
for R in RATIOS:
    r = pd.Series(ad_distance(X, deficit_penalty=float(R))).rank(method="min").to_numpy()
    ranks[R] = r
    top_sym = set(np.argsort(pd.Series(symmetric_distance(X)).rank(method="min").to_numpy())[:10])
    top_ad = set(np.argsort(r)[:10])
    rows.append({"deficit_ratio": R,
                 "spearman_vs_symmetric": round(float(pd.Series(sym_rank).corr(pd.Series(r), method="spearman")), 4),
                 "top10_overlap": len(top_sym & top_ad)})
pd.DataFrame(rows).to_csv(RES / "sensitivity.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False), flush=True)

rng = np.random.default_rng(42)
NREP = 200
acc = np.zeros((len(strains), NREP))
for b in range(NREP):
    Xn = X + rng.normal(0, 5.0, X.shape)
    acc[:, b] = pd.Series(ad_distance(Xn, deficit_penalty=3.0)).rank(method="min").to_numpy()
unc = pd.DataFrame({"Strain": strains, "rank_mean": acc.mean(axis=1).round(2),
                    "rank_sd": acc.std(axis=1, ddof=1).round(2)})
unc = unc.sort_values("rank_mean").reset_index(drop=True)
unc.to_csv(RES / "uncertainty.csv", index=False)
print("uncertainty: top-5 mean ranks:", unc.head(5)["rank_mean"].round(1).tolist(), flush=True)
print("median rank-SD across strains:", round(float(unc["rank_sd"].median()), 2), flush=True)
