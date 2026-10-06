# ad_strain_ranking.py — rank strains against a fitted ideal profile.
#
# Ideal = per-trait maximum across the input strains (empirical frontier,
# TOPSIS-style positive-ideal). Assumes higher-is-better for every trait
# (valid for production/tolerance/growth traits; review before applying
# to burden-type descriptors such as LOH widths, where the direction
# assumption is illustrative only).
# Directional AD to the ideal: deficits below target penalized x(alpha),
# harmless excesses unpenalized. Equal weights (override with --weights).
#
# Input:  any CSV with a Strain column + numeric trait columns
#         (default: strain_traits.csv next to this script).
# Outputs: results/strain_rankings[_TAG].csv, results/sensitivity_table[_TAG].csv
#          (Strain, mean_rank, sd_rank, rank per alpha)
#          figures/Fig_AD_concept.png, Fig_ranking_comparison[_TAG].png
#
# Usage:
#   python ad_strain_ranking.py --traits strain_traits.csv [--tag d1]
#                               [--alpha 3] [--weights w1 w2 ...]

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ALPHAS = [1.0, 2.0, 3.0, 5.0, 10.0]


def directional_to_ideal(X, t, w, alpha):
    diff = np.asarray(X, dtype=float) - np.asarray(t, dtype=float)
    f = 1.0 + (alpha - 1.0) * (diff < 0)
    return np.sum(w * f * diff ** 2, axis=1)


def fig_concept(out):
    t = np.linspace(-3, 3, 400)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t, t ** 2, lw=2.5, label="symmetric (alpha=1)", color="#0072B2")
    ax.plot(t, (1 + 2.0 * (t < 0)) * t ** 2, lw=2.5, label="directional (alpha=3)", color="#D55E00")
    ax.axvline(0, color="k", lw=1)
    ax.set(xlabel="Deviation from ideal trait value (strain - ideal)",
           ylabel="Penalty contribution",
           title="AD concept: deficits penalized, excesses tolerated")
    ax.legend(fontsize=11)
    ax.text(-2.2, 7.5, "deficit\nx3 penalty", color="#D55E00", fontsize=11, ha="center")
    ax.text(2.0, 2.5, "excess\nunpenalized", color="#0072B2", fontsize=11, ha="center")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def main(alpha=3.0, weights=None, traits_file=None, out_root=None, tag="", top=30):
    out_root = Path(out_root) if out_root else HERE.parent
    if traits_file is None:
        traits_file = HERE.parent / "inputs" / "strain_traits.csv"
    figdir = Path(out_root) / "figures"
    resdir = Path(out_root) / "results"
    figdir.mkdir(parents=True, exist_ok=True)
    resdir.mkdir(parents=True, exist_ok=True)
    suf = ("_" + tag) if tag else ""
    fig_concept(figdir / "Fig_AD_concept.png")

    src = Path(traits_file)
    if not src.exists():
        raise SystemExit("Missing %s: provide a CSV with Strain + numeric trait columns." % src)
    df = pd.read_csv(src)
    traits = [c for c in df.columns if c != "Strain"]
    X = df[traits].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise SystemExit("Non-numeric values in trait columns; clean %s first." % src)
    strains = df["Strain"].astype(str).tolist()
    ideal = np.nanmax(X, axis=0)  # fitted empirical frontier
    print("fitted ideal:", dict(zip(traits, np.round(ideal, 2))), flush=True)
    if weights is None:
        try:
            from ad_8distance_violins import D1_HIGH, D1_LOW, LOH_W, D1_TRAITS
        except ImportError:
            D1_HIGH, D1_LOW, LOH_W, D1_TRAITS = set(), set(), {}, []
        if D1_TRAITS and set(traits) == set(D1_TRAITS):
            w = np.array([3.0 if t in D1_HIGH else 0.5 if t in D1_LOW else 1.0 for t in traits], float)
        elif LOH_W and set(traits) <= set(LOH_W):
            w = np.array([LOH_W[c] for c in traits], float)
        else:
            v = np.nanvar(X, axis=0)
            w = np.sqrt(v)
    else:
        w = np.asarray(weights, dtype=float)
    assert len(w) == len(traits), "weights length must match trait count"
    w = w / w.sum()
    print("weights:", dict(zip(traits, np.round(w, 3))), flush=True)

    d_ad = directional_to_ideal(X, ideal, w, alpha)
    d_eu = np.sqrt(np.sum((X - ideal) ** 2, axis=1))
    rk_ad = pd.Series(d_ad).rank(method="min").astype(int).to_numpy()
    rk_eu = pd.Series(d_eu).rank(method="min").astype(int).to_numpy()
    out = pd.DataFrame({"Strain": strains, "AD_distance": np.round(d_ad, 3), "AD_rank": rk_ad,
                        "Euclidean_distance": np.round(d_eu, 3), "Euclidean_rank": rk_eu})
    out = out.sort_values("AD_rank").reset_index(drop=True)
    out.to_csv(resdir / ("strain_rankings%s.csv" % suf), index=False)

    rho = float(pd.Series(rk_ad).corr(pd.Series(rk_eu), method="spearman"))
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(rk_eu, rk_ad, s=60, edgecolors="black", linewidths=0.5)
    lim = [0.5, max(rk_ad.max(), rk_eu.max()) + 0.5]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set(xlabel="Euclidean rank to ideal", ylabel="AD rank to ideal (alpha=%g)" % alpha,
           title="Ranking comparison%s (Spearman rho=%.3f, n=%d)" % ((" " + tag) if tag else "", rho, len(strains)))
    fig.tight_layout()
    fig.savefig(figdir / ("Fig_ranking_comparison%s.png" % suf), dpi=300)
    plt.close(fig)

    ranks = {}
    for a in ALPHAS:
        dd = directional_to_ideal(X, ideal, w, a)
        ranks[a] = pd.Series(dd).rank(method="min").to_numpy()
    R = pd.DataFrame(ranks, index=strains)
    R.columns = ["rank_a%s" % (str(a).replace(".", "p")) for a in ALPHAS]
    sens = pd.DataFrame({"Strain": list(R.index),
                         "mean_rank": R.mean(axis=1).to_numpy().round(2),
                         "sd_rank": R.std(axis=1).to_numpy().round(2)})
    sens = pd.concat([sens, R.reset_index(drop=True)], axis=1)
    sens = sens.sort_values("mean_rank").reset_index(drop=True)
    sens.to_csv(resdir / ("sensitivity_table%s_full.csv" % suf), index=False)
    short = sens.head(top).drop(columns=[c for c in sens.columns if c.startswith("rank_a")])
    short.to_csv(resdir / ("sensitivity_table%s.csv" % suf), index=False)
    print("wrote strain_rankings%s.csv + comparison + sensitivity table top-%d (n=%d, rho=%.3f)." % (
        suf, len(short), len(strains), rho), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rank strains vs fitted ideal (directional AD).")
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--weights", nargs="*", type=float, default=None)
    ap.add_argument("--traits", default=str(HERE.parent / "inputs" / "strain_traits.csv"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--top", type=int, default=30,
                    help="rows kept in sensitivity_table (full table saved as *_full.csv)")
    a = ap.parse_args()
    main(alpha=a.alpha, weights=a.weights, traits_file=a.traits, tag=a.tag, top=a.top)
