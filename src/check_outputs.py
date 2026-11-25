import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
ENC = REPO / "results" / "encoding" / "resnet50"
problems = []

# split (04)
s = np.load(REPO / "results" / "split" / "CSI1_split.npz")
if len(s["train_idx"]) != 3933 or len(s["test_idx"]) != 983:
    problems.append("split sizes changed")
if np.bincount(s["cv_fold"]).tolist() != [787, 787, 787, 786, 786]:
    problems.append("CV fold sizes changed")

# Y (03)
voxels = {"EarlyVis": 495, "LOC": 342, "OPA": 288, "PPA": 331, "RSC": 229}
for roi, nv in voxels.items():
    y = np.load(REPO / "results" / "y" / f"CSI1_{roi}.npy")
    if y.shape != (4916, nv) or np.isnan(y).any():
        problems.append(f"Y for {roi} changed or has NaN")

# features (05)
info = json.loads((REPO / "results" / "features" / "resnet50" / "CSI1_info.json").read_text())
print("feature info:", info.get("layer_shapes", info.get("layers")))

# Table 1 (06)
table1 = {
    "EarlyVis": [0.639, 0.689, 0.758, 0.666],
    "LOC":      [0.493, 0.571, 0.707, 0.671],
    "OPA":      [0.537, 0.573, 0.716, 0.714],
    "PPA":      [0.629, 0.686, 0.800, 0.783],
    "RSC":      [0.682, 0.732, 0.835, 0.906],
}
summ = pd.read_csv(ENC / "CSI1_summary.csv")
t = summ.pivot(index="ROI", columns="layer", values="median_r_over_nc")
for roi, vals in table1.items():
    got = t.loc[roi, ["layer1", "layer2", "layer3", "layer4"]].round(3).tolist()
    if got != vals:
        problems.append(f"Table 1 {roi}: expected {vals}, got {got}")
n_ok = summ[summ.layer == "layer1"]["voxels_nc_ok"].sum()
if n_ok != 1479:
    problems.append(f"voxels with NC > 0.1: expected 1479, got {n_ok}")

if problems:
    print("PROBLEMS FOUND:")
    for p in problems:
        print(" -", p)
else:
    print("All checks passed: split, Y, and Table 1 match the report.")
