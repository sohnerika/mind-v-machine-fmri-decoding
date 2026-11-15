"""Build y (brain responses) for one BOLD5000 subject (CSI1) from Release 2.0 TYPED ROI betas

For each ROI:
  1. combine LHS and RHS voxels
  2. z-score each voxel within each scan session to normalize
  3. average repeated presentations of the same image

Usage (from the repo root):
    python3 src/build_y.py          # CSI1
    python3 src/build_y.py CSI2

Inputs:
    data/<SUBJ>_trials.csv                                  (trial -> image, scan order)
    data/fmri/roi_betas/BOLD5000_GLMsingle_ROI_betas/py/    (TYPED ROI betas)

Outputs in results/y/:
    <SUBJ>_<ROI>.npy          (n_images, n_voxels)  image-averaged responses = y
    <SUBJ>_<ROI>_repeats.npy  (n_repeated_images, max_reps, n_voxels), NaN-padded:
                              single presentations of the repeated images (for noise ceilings)
    <SUBJ>_images.csv         row order shared by every y matrix
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SUBJ = sys.argv[1] if len(sys.argv) > 1 else "CSI1"
ROIS = ["EarlyVis", "LOC", "OPA", "PPA", "RSC"]

REPO = Path(__file__).resolve().parent.parent
BETAS = REPO / "data" / "fmri" / "roi_betas" / "BOLD5000_GLMsingle_ROI_betas" / "py"
TRIALS = REPO / "data" / f"{SUBJ}_trials.csv"
OUT = REPO / "results" / "y"
OUT.mkdir(parents=True, exist_ok=True)

# ---- Trial table: defines row order of the betas -------------------------------
trials = pd.read_csv(TRIALS)
n_trials = len(trials)
sessions = trials["session"].to_numpy()
codes, images = pd.factorize(trials["image"])     # images in order of first appearance
counts = np.bincount(codes)
rep_ids = np.flatnonzero(counts > 1)
max_reps = int(counts.max())
print(f"{SUBJ}: {n_trials} trials, {len(images)} unique images, "
      f"{len(rep_ids)} repeated (up to {max_reps}x), {len(np.unique(sessions))} sessions\n")


def zscore_within_session(X):
    """z-score every voxel (column) separately within each session"""
    Z = np.empty_like(X, dtype=np.float64)
    for s in np.unique(sessions):
        m = sessions == s
        mu = X[m].mean(axis=0)
        sd = X[m].std(axis=0)
        sd[sd == 0] = 1.0
        Z[m] = (X[m] - mu) / sd
    return Z


def split_half_reliability(R):
    """Corr of each voxel, between first and second presentation across repeated images"""
    a = R[:, 0] - R[:, 0].mean(axis=0)
    b = R[:, 1] - R[:, 1].mean(axis=0)
    denom = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0))
    return np.divide((a * b).sum(axis=0), denom, out=np.zeros(a.shape[1]), where=denom > 0)


rows = []
for roi in ROIS:
    # 1. Combine hemispheres
    parts, n_hemi = [], {}
    for hemi in ("LH", "RH"):
        f = BETAS / f"{SUBJ}_GLMbetas-TYPED-FITHRF-GLMDENOISE-RR_allses_{hemi}{roi}.npy"
        if not f.exists():
            sys.exit(f"Missing file: {f}")
        arr = np.load(f)
        n_hemi[hemi] = arr.shape[1]
        parts.append(arr)
    X = np.hstack(parts).astype(np.float64)
    if X.shape[0] != n_trials:
        sys.exit(f"{roi}: betas have {X.shape[0]} rows but trial table has {n_trials}")

    # Drop voxels with any non-finite value
    good = np.isfinite(X).all(axis=0)
    X = X[:, good]

    # 2. z-score within session
    Z = zscore_within_session(X)

    # 3. Avg repeated presentations -> one row per unique image
    y = np.zeros((len(images), Z.shape[1]))
    np.add.at(y, codes, Z)
    y /= counts[:, None]

    # Keep single presentations of repeated images for noise ceilings later
    R = np.full((len(rep_ids), max_reps, Z.shape[1]), np.nan, dtype=np.float32)
    for i, img in enumerate(rep_ids):
        idx = np.flatnonzero(codes == img)
        R[i, :len(idx)] = Z[idx]

    rel = split_half_reliability(R)

    np.save(OUT / f"{SUBJ}_{roi}.npy", y.astype(np.float32))
    np.save(OUT / f"{SUBJ}_{roi}_repeats.npy", R)
    rows.append({
        "ROI": roi,
        "LH": n_hemi["LH"], "RH": n_hemi["RH"],
        "voxels": Z.shape[1], "dropped": int((~good).sum()),
        "y_shape": str(y.shape),
        "median_r": round(float(np.median(rel)), 3),
        "pct_r>0": round(100 * float((rel > 0).mean()), 1),
    })

# Row order shared by all y matrices
first = trials.drop_duplicates("image").set_index("image").loc[images]
img_table = pd.DataFrame({"row": np.arange(len(images)), "image": images,
                          "n_presentations": counts})
if "ImgType" in first.columns:
    img_table["ImgType"] = first["ImgType"].to_numpy()
img_table.to_csv(OUT / f"{SUBJ}_images.csv", index=False)

print(pd.DataFrame(rows).to_string(index=False))
print(f"\nSaved to {OUT}")
print("median_r: per-voxel split-half reliability (presentation 1 vs 2 of the repeated images).")
print("Expect positive values, typically highest in EarlyVis.")