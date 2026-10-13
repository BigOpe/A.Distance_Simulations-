# symmetric_distance.py — conventional weighted Euclidean to the ideal profile.
#   D_i = sqrt(sum_j w_j (x_ij - T_j)^2). Standalone (no side effects on import).

import numpy as np

TARGET = np.array([100.0, 80.0, 85.0, 70.0, 75.0, 85.0])
WEIGHTS = np.array([2.0, 1.5, 2.0, 1.0, 1.0, 2.0])


def symmetric_distance(X, target=TARGET, w=WEIGHTS):
    X = np.asarray(X, dtype=float)
    t = np.asarray(target, dtype=float)
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    return np.sqrt(np.sum(w * (X - t) ** 2, axis=1))
