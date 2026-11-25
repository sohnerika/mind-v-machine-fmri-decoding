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
Run from the repo root: python src/04_make_split.py
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
df["source"] = df["ImgType"].str.replace("rep_", "", regex=False)   # "rep_coco" -> "coco"
df["repeated"] = df["n_presentations"] > 1

# seeded random generator, so the split is identical on every run
rng = np.random.default_rng(args.seed)

# == STEP 2: CREATE TEST SET == #

# start the test set with every repeated image
test = set()
for i in df.index:
    if df.loc[i, "repeated"]:
        test.add(i)

# then, for each source, top up to 20% with randomly chosen single-showing images
for src, group in df.groupby("source"):           # coco, imagenet, scenes (in that order)
    target = round(args.test_frac * len(group))   # e.g. 20% of 2,000 coco = 400
    have = int(group["repeated"].sum())           # repeats from this source already in test
    pool = group[group["repeated"] == False].index.to_numpy()   # images shown once

    picked = rng.choice(pool, target - have, replace=False)     # draw the rest at random
    for i in picked:
        test.add(int(i))

test_idx = np.array(sorted(test))
train_idx = np.array([i for i in df.index if i not in test])   # every other image is training

# == STEP 3: K-FOLD CV == #

# deal the training images into folds, separately for each source
cv_fold = np.empty(len(train_idx), dtype=int)
train_sources = df.loc[train_idx, "source"].to_numpy()
for src in np.unique(train_sources):              # coco, imagenet, scenes
    pos = np.flatnonzero(train_sources == src)    # positions of this source's training images
    rng.shuffle(pos)                              # shuffle them
    for j, p in enumerate(pos):
        cv_fold[p] = j % args.folds               # fold 0, 1, 2, 3, 4, 0, 1, ... like dealing cards

# == STEP 4: CHECKS AND SAVE == #

assert len(set(train_idx) & set(test_idx)) == 0, "train and test overlap"
assert len(train_idx) + len(test_idx) == len(df), "some images are unassigned"
for i in df.index:
    if df.loc[i, "repeated"]:
        assert i in test, "a repeat is not in test"

np.savez(OUT / f"{args.subject}_split.npz",
         train_idx=train_idx, test_idx=test_idx, cv_fold=cv_fold,
         seed=args.seed, test_frac=args.test_frac)

df["split"] = "train"
df.loc[test_idx, "split"] = "test"
df["cv_fold"] = -1
df.loc[train_idx, "cv_fold"] = cv_fold
df[["row", "image", "source", "repeated", "split", "cv_fold"]].to_csv(
    OUT / f"{args.subject}_split.csv", index=False)

# == STEP 5: SUMMARY == #

summary = pd.crosstab(df["source"], df["split"], margins=True, margins_name="total")
summary["test %"] = (100 * summary["test"] / summary["total"]).round(1)
print(f"{args.subject}: {len(train_idx)} train, {len(test_idx)} test (seed {args.seed})\n")
print(summary.to_string())
print("\nRepeated images in test:", int(df.loc[test_idx, "repeated"].sum()), "of",
      int(df["repeated"].sum()))
print("Training images per CV fold:", np.bincount(cv_fold).tolist())
print(f"\nSaved to {OUT}")