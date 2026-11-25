"""
Written by: Erika Sohn

What this code does:
* Computes the Table 2 results: paired differences in r / noise ceiling
  between ResNet-50 layers, with 95% bootstrap confidence intervals
* Estimates uncertainty by resampling the existing test images with
  replacement, rather than collecting new data
* Runs two versions: a voxel-level bootstrap (resamples voxels) and an
  image-level bootstrap (resamples test images, the one reported in Table 2)

Inputs: results/encoding/resnet50/ (CSI1_voxels.csv, CSI1_layer*_model.npz)
        results/y/ (CSI1_{ROI}.npy)
Run from src/: python 07_table2_bootstrap.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
YDIR = REPO / "results/y"
ENC = REPO / "results/encoding/resnet50"
ROIS = ["EarlyVis", "LOC", "OPA", "PPA", "RSC"]
PAIRS = [("layer4", "layer1"), ("layer3", "layer4"), ("layer3", "layer1")]
PAIRS_IMG = [("layer4", "layer1"), ("layer3", "layer1"), ("layer3", "layer4")]

# == STEP 1: LOAD DATA == # 

# corr between predicted vs actual brain respnse
def colcorr(A, B):
    A = A - A.mean(0) # dev from from predicted mu
    B = B - B.mean(0) # dev from actual mu
    top = (A * B).sum(0)
    bottom = np.sqrt((A * A).sum(0) * (B * B).sum(0))
    return top / bottom

v = pd.read_csv(ENC / "CSI1_voxels.csv")

# r normalized formula rn = r / nc
# w as a voxel × CNN layer matrix
v["rn"] = v["r"] / v["nc"]
w = v.pivot_table(index=["voxel", "ROI"], columns="layer", values="rn").reset_index()
nc = v[v.layer == "layer1"].set_index("voxel")["nc"]
w = w[w["voxel"].map(nc) > 0.1]
rng = np.random.default_rng(0)
print("Voxel-level bootstrap")

# == STEP 2: POINT ESTIMATION == # 

# per voxel diff in normalized r between 2 layers 
# -> take the median per ROI
for a, b in PAIRS:
    for roi in ROIS:
        in_roi = w[w.ROI == roi]
        d = (in_roi[a] - in_roi[b]).to_numpy()

        boots = []
        for _ in range(5000):
            sample = rng.choice(d, len(d))
            boots.append(np.median(sample))

        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"  {a}-{b} {roi:8s} {np.median(d):+.3f} [{lo:+.3f}, {hi:+.3f}]")

# == STEP 3: BOOTSTRAPPING == # 

# load actual vs predicted brain responses
Ys = []
for r in ROIS:
    Ys.append(np.load(YDIR / f"CSI1_{r}.npy"))
Y = np.hstack(Ys)

m = {}
for l in ["layer1", "layer3", "layer4"]:
    m[l] = np.load(ENC / f"CSI1_{l}_model.npz")
test_idx = m["layer1"]["test_idx"]
roi_of = m["layer1"]["roi_of_voxel"]
Yt = Y[test_idx]

# filter voxels by noise-ceiling of 0.1
# voxels that don't meet this sample don't get bootstrapped
ncv = v[v.layer == "layer1"].sort_values("voxel")["nc"].to_numpy()
ok = ncv > 0.1
safe_nc = ncv.copy()
safe_nc[~ok] = 1.0
rng = np.random.default_rng(0)
n = len(test_idx)
print("Image-level bootstrap")
for a, b in PAIRS_IMG:
    boots = {}
    for r in ROIS:
        boots[r] = []

    # replacement sampling of 983 test images
    # how much layer diff would change across diff test sets 
    # https://www.geeksforgeeks.org/data-science/sampling-with-or-without-replacement/
    for _ in range(1000):
        i = rng.integers(0, n, n)
        r_a = colcorr(m[a]["Y_pred"][i], Yt[i])
        r_b = colcorr(m[b]["Y_pred"][i], Yt[i])
        d = (r_a - r_b) / safe_nc
        for r in ROIS:
            # median over viable voxels only, previously filtered by noise-ceiling threshold
            keep = (roi_of == r) & ok
            boots[r].append(np.median(d[keep]))
    for r in ROIS:
        lo, hi = np.percentile(boots[r], [2.5, 97.5])
        print(f"  {a}-{b} {r:8s} {np.median(boots[r]):+.3f} [{lo:+.3f}, {hi:+.3f}]")