# ad_distance.py — Adaptive Directional distance to the ideal profile.
#   AD_i = sqrt(sum_j w_j g_j(x_ij, T_j)), g_j = R deficit penalty if
#   x_ij < T_j else 1 (harmless excess). R=1 recovers symmetric distance.
# Standalone (no side effects on import).

import numpy as np

TARGET = np.array([100.0, 80.0, 85.0, 70.0, 75.0, 85.0])
WEIGHTS = np.array([2.0, 1.5, 2.0, 1.0, 1.0, 2.0])


def ad_distance(X, target=TARGET, w=WEIGHTS, deficit_penalty=3.0, excess_penalty=1.0):
    X = np.asarray(X, dtype=float)
    t = np.asarray(target, dtype=float)
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    diff = X - t
    g = np.where(diff < 0, deficit_penalty, excess_penalty)
    return np.sqrt(np.sum(w * g * diff ** 2, axis=1))
