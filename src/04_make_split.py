"""
Written by: Erika Sohn

What this code does:
* Splits CSI1's 4,916 images into training (80%) and test (20%) sets,
  keeping the same mix of COCO, ImageNet, and scene images in each
* Puts all 113 repeated images in the test set first (used for the noise
  ceiling and to avoid leakage), then randomly fills the rest of each
  source's 20% from images shown only once
* Assigns each training image to one of 5 cross-validation folds, also
  stratified by source, for choosing each voxel's ridge penalty
* Uses a fixed seed (42), so the split is identical on every run, and
  checks that train and test don't overlap and every repeat is in test

Inputs: results/y/CSI1_images.csv
Outputs: results/split/ (CSI1_split.npz with train_idx, test_idx, cv_fold;
         CSI1_split.csv, one row per image with its split and fold)
Run from src/: python 04_make_split.py
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--subject", default="CSI1")
parser.add_argument("--test-frac", type=float, default=0.2)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--folds", type=int, default=5)
args = parser.parse_args()

REPO = Path(__file__).resolve().parent.parent
IMAGES = REPO / "results" / "y" / f"{args.subject}_images.csv"
OUT = REPO / "results" / "split"
OUT.mkdir(parents=True, exist_ok=True)

# == STEP 1: LOAD DATA == # 

df = pd.read_csv(IMAGES)
df["source"] = df["ImgType"].str.replace("rep_", "", regex=False)
df["repeated"] = df["n_presentations"] > 1

# seeded random generator, so the split is identical on every run
rng = np.random.default_rng(args.seed)

# == STEP 2: CREATE TEST SET == #

# all repeats first, 
# then random fill within each source
test = set(df.index[df["repeated"]])

for src, group in df.groupby("source"): # once each for coco, imagenet, scenes
    target = round(args.test_frac * len(group))  # 20% * 2,000 = 400 test images needed
    have = int(group["repeated"].sum()) # 45 coco repeats
    pool = group.index[~group["repeated"]].to_numpy() # 2,000-45 repx = 1,955 coco shown once, draw from here

    # how many more test images we need 
    # 400-45 = 355 coco images drawn randomly from 1,955 pool 
    test |= set(rng.choice(pool, target - have, replace=False).tolist()) 

test_idx = np.sort(np.fromiter(test, dtype=int))
train_idx = np.setdiff1d(df.index.to_numpy(), test_idx) # every row not in test = train row

# == STEP 3: K-FOLD CV == #

# CV folds inside the training set,
# stratified by image source
cv_fold = np.empty(len(train_idx), dtype=int)
train_sources = df.loc[train_idx, "source"].to_numpy()
for src in np.unique(train_sources):
    pos = np.flatnonzero(train_sources == src)
    rng.shuffle(pos)
    cv_fold[pos] = np.arange(len(pos)) % args.folds    # deal images into folds in turn

assert len(np.intersect1d(train_idx, test_idx)) == 0, "train and test overlap"
assert len(train_idx) + len(test_idx) == len(df), "some images are unassigned"
assert df.loc[df["repeated"]].index.isin(test_idx).all(), "a repeat is not in test"

np.savez(OUT / f"{args.subject}_split.npz",
         train_idx=train_idx, test_idx=test_idx, cv_fold=cv_fold,
         seed=args.seed, test_frac=args.test_frac)
df["split"] = "train"
df.loc[test_idx, "split"] = "test"
df["cv_fold"] = -1
df.loc[train_idx, "cv_fold"] = cv_fold
df[["row", "image", "source", "repeated", "split", "cv_fold"]].to_csv(
    OUT / f"{args.subject}_split.csv", index=False)

summary = pd.crosstab(df["source"], df["split"], margins=True, margins_name="total")
summary["test %"] = (100 * summary["test"] / summary["total"]).round(1)
print(f"{args.subject}: {len(train_idx)} train, {len(test_idx)} test (seed {args.seed})\n")
print(summary.to_string())
print("\nRepeated images in test:", int(df.loc[test_idx, "repeated"].sum()), "of",
      int(df["repeated"].sum()))
print("Training images per CV fold:", np.bincount(cv_fold).tolist())
print(f"\nSaved to {OUT}")
