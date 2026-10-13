# figure_5.py — composite Figure 5 (panels A-F). Reads results/*.csv +
# ../ML_remake/metrics.csv. Output: figures/Figure_5.png

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
FIG = HERE.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

rk = pd.read_csv(RES / "rankings.csv")
sens = pd.read_csv(RES / "sensitivity.csv")
unc = pd.read_csv(RES / "uncertainty.csv")
df = pd.read_csv(HERE.parent / "data" / "simulated_strains.csv")
traits = [c for c in df.columns if c != "Strain"]
ml = pd.read_csv(HERE.parent.parent / "ML_remake" / "metrics.csv")

fig, axes = plt.subplots(2, 3, figsize=(18, 11), squeeze=False)
ax = axes.flat

# A: current evidence (real ML numbers)
m = ml.set_index("model").loc[["RandomForest", "GradBoosting", "SVR", "Lasso", "Ridge"]]
ax[0].bar(m.index, m["R2_mean"], yerr=m["R2_sd"], capsize=4, color="#0072B2")
ax[0].set(title="A — Current evidence (65 strains, repeated CV R²)", ylabel="R²")
ax[0].tick_params(axis="x", labelrotation=18, labelsize=9)

# B: simulated trait distributions
ax[1].boxplot([df[t] for t in traits], labels=[t[:8] for t in traits])
ax[1].tick_params(axis="x", labelrotation=25, labelsize=9)
ax[1].set(title="B — Simulated decision problem (100 strains x 6 traits)", ylabel="Trait value")

# C/D: top-15 bars both rankings (true distances, best at top)
for j, (rkcol, dcol, ttl) in enumerate([("sym_rank", "sym_distance", "C — Symmetric ranking"),
                                        ("ad_rank", "ad_distance", "D — AD ranking")]):
    top = rk.nsmallest(15, rkcol).iloc[::-1]
    ax[2 + j].barh(top["Strain"], top[dcol], color="#0072B2" if j == 0 else "#D55E00")
    ax[2 + j].set(title=ttl, xlabel="Distance to ideal")

# E: rank shifts
ax[4].scatter(rk["sym_rank"], rk["ad_rank"], s=28, edgecolors="black", linewidths=0.4)
mx = max(rk["sym_rank"].max(), rk["ad_rank"].max())
ax[4].plot([0.5, mx + 0.5], [0.5, mx + 0.5], "k--", lw=1)
big = rk[np.abs(rk["rank_change"]) >= 5]
for _, r in big.iterrows():
    ax[4].annotate(r["Strain"], (r["sym_rank"], r["ad_rank"]), fontsize=8)
ax[4].set(title="E — Ranking changes (AD vs symmetric)", xlabel="Symmetric rank", ylabel="AD rank")

# F: sensitivity
ax2 = ax[5].twinx()
ax[5].plot(sens["deficit_ratio"], sens["spearman_vs_symmetric"], "o-", color="#D55E00", label="Spearman")
ax2.plot(sens["deficit_ratio"], sens["top10_overlap"], "s-", color="#009E73", label="top-10 overlap")
ax[5].set(title="F — Sensitivity to deficit penalty", xlabel="Deficit:excess ratio", ylabel="Spearman")
ax2.set_ylabel("Top-10 overlap")
ax[5].set_xscale("log")

fig.tight_layout()
fig.savefig(FIG / "Figure_5.png", dpi=300)
plt.close(fig)
print("wrote Figure_5.png", flush=True)
