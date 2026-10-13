# generate_simulation.py — 100 hypothetical strains x 6 traits (0-100+ scale).
# 97 random strains (truncated normals) + 3 designed edge cases exposing the
# symmetric-distance failure: excess-pulcherrimin (harmless), stress-deficit
# (disqualifying), balanced star (ideal). Seed 42. Output: data/simulated_strains.csv

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
DATA.mkdir(parents=True, exist_ok=True)

TRAITS = ["Pulcherrimin", "Stress", "Fermentation", "Aroma", "Growth", "Compatibility"]
MEANS = [55.0, 60.0, 65.0, 55.0, 60.0, 65.0]
SDS = [20.0, 15.0, 15.0, 15.0, 15.0, 15.0]

rng = np.random.default_rng(42)
X = np.clip(rng.normal(MEANS, SDS, (95, 6)), 5, 130)
ids = ["S%02d" % (i + 1) for i in range(95)]

edge = pd.DataFrame([
    ["S_A_excess", 120.0, 83.0, 88.0, 70.0, 80.0, 85.0],
    ["S_B_deficit", 100.0, 62.0, 88.0, 70.0, 80.0, 85.0],
    ["S_C_star", 100.0, 80.0, 85.0, 70.0, 75.0, 85.0],
    ["S_D_big_excess", 150.0, 80.0, 85.0, 70.0, 75.0, 85.0],
    ["S_E_severe_deficit", 100.0, 40.0, 85.0, 70.0, 75.0, 85.0],
], columns=["Strain"] + TRAITS)

df = pd.DataFrame(X, columns=TRAITS)
df.insert(0, "Strain", ids)
df = pd.concat([df, edge], ignore_index=True)
df.to_csv(DATA / "simulated_strains.csv", index=False)
print("wrote simulated_strains.csv", df.shape, flush=True)
