"""
Written by: Erika Sohn

What this code does:
* Fits Method 1, the encoding model: predicts CSI1's voxel responses from
  each ResNet-50 layer (layer1 to layer4) across five ROIs
* For each layer, standardizes the features and reduces them with PCA (Z),
  then fits a ridge regression per voxel, with each voxel's lambda chosen
  by 5-fold CV and the model refit on all training images
* Predicts the 983 test images and scores each voxel by its correlation r
  with CSI1's measured responses, normalized by its noise ceiling (r / NC)
* Produces the Table 1 results (median r / noise ceiling by ROI and layer)

Inputs: results/features/resnet50/ (CSI1_layer*.npy)
        results/y/ (CSI1_{ROI}.npy, CSI1_{ROI}_repeats.npy, CSI1_images.csv)
        results/split/CSI1_split.npz
Outputs: results/encoding/resnet50/ (CSI1_layer*_model.npz, CSI1_voxels.csv,
         CSI1_summary.csv)
Run from src/: python 06_fit_encoding.py
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument("--subject", default="CSI1")
parser.add_argument("--model", default="resnet50")
parser.add_argument("--layers", nargs="+", default=["layer1", "layer2", "layer3", "layer4"])
parser.add_argument("--pcs", type=int, default=1000, help="number of PCA components (k)")
args = parser.parse_args()

ROIS = ["EarlyVis", "LOC", "OPA", "PPA", "RSC"]
REPO = Path(__file__).resolve().parent.parent
YDIR = REPO / "results" / "y"
FDIR = REPO / "results" / "features" / args.model
SPLIT = REPO / "results" / "split" / f"{args.subject}_split.npz"
OUT = REPO / "results" / "encoding" / args.model
OUT.mkdir(parents=True, exist_ok=True)
NC_MIN = 0.1   # voxels below this noise ceiling are left out of normalized scores


# correlation matrix 
def colcorr(A, B):
    """Pearson r between matching columns of A and B (one r per voxel)."""
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    den = np.sqrt((A * A).sum(axis=0) * (B * B).sum(axis=0))
    return np.divide((A * B).sum(axis=0), den, out=np.zeros(A.shape[1]), where=den > 0)


# == STEP 1: SPLIT IMAGES == # 

# note: the indexes map to the same image for both datasets
# i.e., row 0 of X = image A = row of Y
# so this basically tells us that row 0 / image A goes into train, 
# while row 1 or image B goes into test
split = np.load(SPLIT)
train_idx, test_idx, cv_fold = split["train_idx"], split["test_idx"], split["cv_fold"]
n_folds = int(cv_fold.max()) + 1
images = pd.read_csv(YDIR / f"{args.subject}_images.csv")
n_pres_test = images["n_presentations"].to_numpy()[test_idx]
rep_test = n_pres_test > 1
print(f"{args.subject}: {len(train_idx)} train, {len(test_idx)} test "
      f"({rep_test.sum()} repeated), {n_folds} CV folds")

# == STEP 2: CREATE Y MATRIX == # 

# Y must be a single matrix (images, voxels)
# one row per image, one col per voxel
# previously, each ROI was its own array
Ys, Rs, roi_of_voxel = [], [], []
for roi in ROIS:
    y = np.load(YDIR / f"{args.subject}_{roi}.npy").astype(np.float64)
    Ys.append(y)
    Rs.append(np.load(YDIR / f"{args.subject}_{roi}_repeats.npy").astype(np.float64))
    roi_of_voxel += [roi] * y.shape[1]
Y = np.hstack(Ys) # (images, all voxels)
R = np.concatenate(Rs, axis=2) # (repeated images, presentations, all voxels)
roi_of_voxel = np.array(roi_of_voxel)
Y_train, Y_test = Y[train_idx], Y[test_idx]
print("voxels per ROI:", {r: int((roi_of_voxel == r).sum()) for r in ROIS},
      f"total {Y.shape[1]}")

# calculate noise ceiling using R repeat image data
# single-trial variance = signal variance + noise variance -> / n = average noise
# which can be detected in repeat images
var_noise = np.nanmean(np.nanvar(R, axis=1, ddof=1), axis=0)
var_total = np.nanvar(R.reshape(-1, R.shape[2]), axis=0)
var_signal = np.clip(var_total - var_noise, 0, None)

def noise_ceiling(n_pres):
    """Highest correlation a perfect model could reach on these images."""
    k = np.mean(1.0 / n_pres)
    den = var_signal + var_noise * k
    return np.sqrt(np.divide(var_signal, den, out=np.zeros_like(den), where=den > 0))

nc_all, nc_rep = noise_ceiling(n_pres_test), noise_ceiling(n_pres_test[rep_test])

# == STEP 3: PCA == # 
voxel_rows, summary_rows = [], []
for layer in args.layers:
    t0 = time.time()
    print(f"\n=== {layer} ===")

    # Z or weight adjusted X
    # standardize F with training stats, then PCA fit on training rows
    F = np.load(FDIR / f"{args.subject}_{layer}.npy").astype(np.float32)
    mu = F[train_idx].mean(axis=0)
    sd = F[train_idx].std(axis=0)
    sd[sd == 0] = 1.0
    F = (F - mu) / sd
    k = min(args.pcs, len(train_idx) - 1, F.shape[1])
    # randomized PCA, fast for wide matrices
    # https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    pca = PCA(n_components=k, svd_solver="randomized", random_state=0)
    Z_train = pca.fit_transform(F[train_idx]).astype(np.float64)
    Z_test = pca.transform(F[test_idx]).astype(np.float64)
    del F
    print(f"  features -> {k} PCs, {100 * pca.explained_variance_ratio_.sum():.1f}% of variance"
          f"  ({time.time() - t0:.0f} s)")

    # == STEP 4: RIDGE == # 
    # solved through the SVD of Z so that every penalty lambda costs almost nothing
    # Z = U S V^T  ->  W(lambda) = V diag(s / (s^2 + lambda)) U^T Y
    base = np.median((Z_train ** 2).sum(axis=0))
    lambdas = base * np.logspace(-4, 3, 22)

    def ridge_parts(Z, Ycentered):
        # https://numpy.org/doc/stable/reference/generated/numpy.linalg.svd.html
        U, s, Vt = np.linalg.svd(Z, full_matrices=False)
        return s, Vt, U.T @ Ycentered

    # k-fold CV
    # choose lambda per voxel via CV inside training set
    cv_score = np.zeros((len(lambdas), Y.shape[1]))
    for f in range(n_folds):
        tr, va = cv_fold != f, cv_fold == f
        Zm, Ym = Z_train[tr].mean(axis=0), Y_train[tr].mean(axis=0)
        s, Vt, UtY = ridge_parts(Z_train[tr] - Zm, Y_train[tr] - Ym)
        ZvV = (Z_train[va] - Zm) @ Vt.T
        for i, lam in enumerate(lambdas):
            pred = ZvV @ ((s / (s ** 2 + lam))[:, None] * UtY)
            cv_score[i] += colcorr(pred, Y_train[va]) / n_folds
    best = cv_score.argmax(axis=0)
    predictable = cv_score.max(axis=0) > 0.1
    at_edge = np.isin(best[predictable], [0, len(lambdas) - 1]).mean() if predictable.any() else 0.0

    # refit on all training images, each voxel with its own lambda
    Zm, Ym = Z_train.mean(axis=0), Y_train.mean(axis=0)

    # this is where both brain, resnet data join
    s, Vt, UtY = ridge_parts(Z_train - Zm, Y_train - Ym)
    W = np.empty((k, Y.shape[1]))
    for i, lam in enumerate(lambdas):
        cols = best == i
        if cols.any():
            W[:, cols] = Vt.T @ ((s / (s ** 2 + lam))[:, None] * UtY[:, cols])
    resid_var = ((Y_train - ((Z_train - Zm) @ W + Ym)) ** 2).mean(axis=0)

    # == STEP 4: PREDICT TEST IMGS AND SCORE EACH VOXEL == # 
    Y_pred = (Z_test - Zm) @ W + Ym # Ŷ_test
    r_all = colcorr(Y_pred, Y_test)
    r_rep = colcorr(Y_pred[rep_test], Y_test[rep_test])
    print(f"  ridge done, λ at grid edge for {100 * at_edge:.1f}% of predictable voxels"
          f"  ({time.time() - t0:.0f} s)")

    np.savez(OUT / f"{args.subject}_{layer}_model.npz",
             Y_pred=Y_pred.astype(np.float32), W=W.astype(np.float32),
             lam=lambdas[best], resid_var=resid_var, test_idx=test_idx,
             roi_of_voxel=roi_of_voxel)

    for roi in ROIS:
        m = roi_of_voxel == roi
        ok = m & (nc_all > NC_MIN)
        summary_rows.append({
            "layer": layer, "ROI": roi, "voxels": int(m.sum()),
            "median_r": float(np.median(r_all[m])),
            "median_r_repeats": float(np.median(r_rep[m])),
            "median_nc": float(np.median(nc_all[m])),
            "median_r_over_nc": float(np.median(r_all[ok] / nc_all[ok])) if ok.any() else np.nan,
            "voxels_nc_ok": int(ok.sum()),
        })
    voxel_rows.append(pd.DataFrame({
        "layer": layer, "voxel": np.arange(Y.shape[1]), "ROI": roi_of_voxel,
        "r": r_all, "r_repeats": r_rep, "nc": nc_all, "nc_repeats": nc_rep,
        "lambda": lambdas[best]}))

pd.concat(voxel_rows).to_csv(OUT / f"{args.subject}_voxels.csv", index=False)
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUT / f"{args.subject}_summary.csv", index=False)

table = summary.pivot(index="ROI", columns="layer", values="median_r_over_nc").loc[ROIS]
print("\nMedian r / noise ceiling (test set), rows = ROI, columns = layer:")
print(table.round(3).to_string())
print("\nBest layer per ROI:", table.idxmax(axis=1).to_dict())
raw = summary.pivot(index="ROI", columns="layer", values="median_r").loc[ROIS]
print("\nRaw median r (test set):")
print(raw.round(3).to_string())
print(f"\nSaved to {OUT}")
