# ad_8distance_violins.py — 8-distance violin plots for the 3 simulated sets.
#
# Reusable module (no Colab dependencies). Formulas are repo-exact (dist.py):
# squared AD aggregation, sqrt weighted Euclidean, pinv Mahalanobis,
# shifted/normalized Jensen-Shannon, 1-D Wasserstein on trait positions.
# Weights: D1 tiers (dist.py), LOH dict (dist3.py), YP sqrt-variance (dist2.py).
#
# Usage:
#   python ad_8distance_violins.py [--data DIR] [--out DIR] [--datasets d1 d2 d3]

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import pinv
from scipy.spatial import distance as ssd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    from scipy.spatial.distance import wasserstein_distance  # noqa: F401
except ImportError:  # scipy<1.8
    from scipy.stats import wasserstein_distance  # noqa: F401

HERE = Path(__file__).resolve().parent
D1_TRAITS = ["Copper", "Ethanol", "H2O2", "SO2", "Chitosan", "TFL", "Cerulenine",
             "Protease", "Sulfite_reductase", "Beta_lyase", "Tyrosine_Decarboxylase",
             "Histidine_Decarboxylase", "Ornithine_Decarboxylase", "Beta_Glycosidase"]
D1_HIGH = {"Beta_lyase", "Chitosan", "Protease", "Histidine_Decarboxylase",
           "Ornithine_Decarboxylase"}
D1_LOW = {"Sulfite_reductase", "Tyrosine_Decarboxylase", "TFL"}
LOH_W = {"number_loh": 2.0, "percentage_loh": 3.0, "size_without_loh": 2.5,
         "max_width": 1.5, "mean_width": 1.5, "median_width": 1.5, "min_width": 0.5}

ORDER = ["Euclidean", "PCA", "Weighted Euclidean", "Mahalanobis",
         "Jensen-Shannon", "Wasserstein", "AD symmetric", "AD asymmetric"]
COLORS = {"Euclidean": "#009E73", "PCA": "#999999",
          "Weighted Euclidean": "#E69F00", "Mahalanobis": "#7B68EE",
          "Jensen-Shannon": "#87CEEB", "Wasserstein": "#FFA500",
          "AD symmetric": "#0072B2", "AD asymmetric": "#D55E00"}


def euclid(X):
    return ssd.squareform(ssd.pdist(X, "euclidean"))


def symmetrize(D):
    S = 0.5 * (np.asarray(D) + np.asarray(D).T)
    np.fill_diagonal(S, 0.0)
    return S


def weighted_euclid(X, w):
    return np.sqrt(ssd.squareform(ssd.pdist(X, "seuclidean", V=np.asarray(w))))


def mahalanobis(X, VI=None):
    VI = pinv(np.cov(X.T)) if VI is None else VI
    return ssd.squareform(ssd.pdist(X, "mahalanobis", VI=VI))


def jensenshannon(X):
    Xs = X - X.min(0) + 1e-9
    Xp = Xs / Xs.sum(1, keepdims=True)
    return ssd.squareform(ssd.pdist(Xp, "jensenshannon"))


def wasserstein(X, blk=250):
    from scipy.stats import wasserstein_distance as wd
    pos = np.arange(X.shape[1])
    Xn = X - X.min(1, keepdims=True) + 1e-9
    Xn /= Xn.sum(1, keepdims=True)
    C = np.cumsum(Xn, axis=1)[:, :-1]
    gaps = np.diff(pos)
    n = len(X)
    D = np.zeros((n, n))
    for s in range(0, n, blk):
        e = min(s + blk, n)
        D[s:e] = np.abs(C[s:e, None, :] - C[None, :, :]) @ gaps
        D[:, s:e] = D[s:e].T
    return D


def ad_directed(X, w, alpha, blk=250):
    n, p = X.shape
    w = np.asarray(w).reshape(1, p)
    D = np.zeros((n, n))
    for s in range(0, n, blk):
        e = min(s + blk, n)
        d = X[s:e, None, :] - X[None, :, :]
        D[s:e] = np.sum(w * (1 + (alpha - 1) * (d < 0)) * d ** 2, axis=2)
    return D


def load(tag, data_dir):
    if tag == "d1":
        df = pd.read_csv(data_dir / "dataset1_synthetic_oenological.csv")
        return df[D1_TRAITS].to_numpy(float), D1_TRAITS
    if tag == "d2":
        df = pd.read_csv(data_dir / "dataset2_synthetic_loh.csv")
        cols = [c for c in df.columns if c not in ("Strain", "sum_loh")]
        return df[cols].apply(pd.to_numeric).to_numpy(float), cols
    df = pd.read_csv(data_dir / "dataset3_synthetic_phenotype.csv")
    cols = [c for c in df.columns if c not in ("Strain", "YPACETATE")]
    return df[cols].to_numpy(float), cols


def weights_for(tag, X, cols):
    if tag == "d1":
        w = np.array([3.0 if t in D1_HIGH else 0.5 if t in D1_LOW else 1.0 for t in cols], float)
    elif tag == "d2":
        w = np.array([LOH_W[c] for c in cols], float)
    else:
        v = X.var(0)
        w = np.sqrt(v)
    return w / w.sum()


def run_tag(tag, data_dir, out_dir):
    Xr, cols = load(tag, data_dir)
    X = StandardScaler().fit_transform(Xr)
    w = weights_for(tag, X, cols)
    Xpca = PCA(n_components=2, random_state=42).fit_transform(X)
    mats = {"Euclidean": euclid(X), "PCA": euclid(Xpca),
            "Weighted Euclidean": weighted_euclid(X, w),
            "Mahalanobis": mahalanobis(X), "Jensen-Shannon": jensenshannon(X),
            "Wasserstein": wasserstein(X),
            "AD symmetric": symmetrize(ad_directed(X, w, 1.0)),
            "AD asymmetric": symmetrize(ad_directed(X, w, 3.0))}
    tri = np.triu_indices(len(X), 1)
    fig, ax = plt.subplots(figsize=(11, 5))
    parts = ax.violinplot([mats[m][tri] for m in ORDER], showmeans=True)
    for b, m in zip(parts["bodies"], ORDER):
        b.set_facecolor(COLORS[m])
        b.set_alpha(0.85)
    for k in ("cmeans", "cmins", "cmaxes", "cbars"):
        parts[k].set_color("black")
    ax.set_xticks(range(1, 9), ORDER, rotation=20, ha="right")
    ax.set_ylabel("Pairwise distance")
    ax.set_title("Distance distributions (%s, n=%d)" % (tag, len(X)))
    fig.tight_layout()
    fig.savefig(out_dir / ("violins_8_%s.png" % tag), dpi=300)
    plt.close(fig)
    return {m: round(float(mats[m][tri].mean()), 3) for m in ORDER}


def main(data_dir=HERE.parent / "inputs", out_dir=None, tags=("d1", "d2", "d3")):
    out_dir = Path(out_dir) if out_dir else HERE.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for tag in tags:
        print(tag, "done", run_tag(tag, Path(data_dir), out_dir), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="8-distance violin plots (repo-exact formulas).")
    ap.add_argument("--data", default=str(HERE.parent / "inputs"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--datasets", nargs="+", default=["d1", "d2", "d3"])
    a = ap.parse_args()
    main(Path(a.data), a.out, tuple(a.datasets))
