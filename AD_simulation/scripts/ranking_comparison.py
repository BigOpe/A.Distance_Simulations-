# ranking_comparison.py — rank all strains both ways; report rank changes.
# Output: results/rankings.csv (Strain, sym_distance, sym_rank, ad_distance,
# ad_rank, rank_change). Prints top movers (the biologically interpretable cases).

from pathlib import Path

import pandas as pd

from ad_distance import ad_distance
from symmetric_distance import symmetric_distance

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "simulated_strains.csv"
RES = HERE.parent / "results"
RES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)
traits = [c for c in df.columns if c != "Strain"]
X = df[traits].to_numpy(dtype=float)

d_sym = symmetric_distance(X)
d_ad = ad_distance(X, deficit_penalty=3.0)
out = pd.DataFrame({"Strain": df["Strain"], "sym_distance": d_sym.round(3),
                    "ad_distance": d_ad.round(3)})
out["sym_rank"] = out["sym_distance"].rank(method="min").astype(int)
out["ad_rank"] = out["ad_distance"].rank(method="min").astype(int)
out["rank_change"] = out["sym_rank"] - out["ad_rank"]  # + promoted under AD
out = out.sort_values("ad_rank").reset_index(drop=True)
out.to_csv(RES / "rankings.csv", index=False)
print(out.head(10).to_string(index=False), flush=True)
print("--- biggest promotions (AD vs symmetric) ---", flush=True)
print(out.nlargest(5, "rank_change")[["Strain", "sym_rank", "ad_rank", "rank_change"]].to_string(index=False),
      flush=True)
print("--- biggest demotions ---", flush=True)
print(out.nsmallest(5, "rank_change")[["Strain", "sym_rank", "ad_rank", "rank_change"]].to_string(index=False),
      flush=True)
print("spearman(sym_rank, ad_rank) = %.3f" % out["sym_rank"].corr(out["ad_rank"], method="spearman"),
      flush=True)
