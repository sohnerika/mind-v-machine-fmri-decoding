"""Inspect BOLD5000 Release 2.0 ROI betas and check trial order.

Usage (inside the folder with CSI1_trials.csv):
    python3 inspect_roi_betas.py

What it does:
  1. Unzips BOLD5000_GLMsingle_ROI_betas.zip into roi_betas/ (once).
  2. Lists every file with its shape (npy, mat, h5, nii, txt/csv).
  3. Compares CSI1_imgnames.txt with the image column of CSI1_trials.csv.
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
ZIP = BASE / "BOLD5000_GLMsingle_ROI_betas.zip"
OUT = BASE / "roi_betas"


def describe(p):
    s = p.name.lower()
    try:
        if s.endswith(".npy"):
            a = np.load(p, mmap_mode="r")
            return f"npy {a.shape} {a.dtype}"
        if s.endswith(".mat"):
            try:
                from scipy.io import loadmat
                m = loadmat(p)
                return "mat " + ", ".join(f"{k}{getattr(v, 'shape', '')}"
                                          for k, v in m.items() if not k.startswith("__"))
            except NotImplementedError:  # MATLAB v7.3 files are HDF5
                return describe_h5(p, "mat(v7.3)")
        if s.endswith((".h5", ".hdf5")):
            return describe_h5(p, "h5")
        if s.endswith((".nii", ".nii.gz")):
            import nibabel as nib
            return f"nifti {nib.load(p).shape}"
        if s.endswith((".txt", ".csv", ".tsv")):
            lines = p.read_text(errors="replace").splitlines()
            return f"text, {len(lines)} lines, first: {lines[0][:60] if lines else ''!r}"
    except Exception as e:  # keep going if one file is odd
        return f"could not read ({type(e).__name__}: {e})"
    return ""


def describe_h5(p, label):
    import h5py
    items = []
    with h5py.File(p, "r") as f:
        f.visititems(lambda n, o: items.append(f"{n}{o.shape}") if hasattr(o, "shape") else None)
    return f"{label}: " + ", ".join(items[:12]) + (" ..." if len(items) > 12 else "")


# 1. Unzip once
if not OUT.exists():
    if not ZIP.exists():
        raise SystemExit(f"Not found: {ZIP}. Move the zip into {BASE} first.")
    print("Unzipping ...")
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(OUT)

# 2. List contents
files = sorted(p for p in OUT.rglob("*") if p.is_file() and not p.name.startswith("._"))
print(f"\n{len(files)} files in {OUT.name}/")
for p in files:
    mb = p.stat().st_size / 1e6
    print(f"  {str(p.relative_to(OUT)):70s} {mb:8.1f} MB  {describe(p)}")

# 3. Image-order check
trials = BASE / "CSI1_trials.csv"
names_file = next((p for p in [BASE / "CSI1_imgnames.txt", *OUT.rglob("CSI1_imgnames.txt")]
                   if p.exists()), None)
if trials.exists() and names_file:
    table = pd.read_csv(trials)["image"].tolist()
    names = [l.strip().removeprefix("rep_")
             for l in names_file.read_text().splitlines() if l.strip()]
    same = sum(a == b for a, b in zip(table, names))
    print(f"\nImage order check ({names_file.name} vs CSI1_trials.csv):")
    print(f"  {len(names)} names vs {len(table)} trials, {same} positions identical")
    if same == len(table) == len(names):
        print("  PERFECT MATCH: trial order confirmed")
    else:
        bad = next((i for i, (a, b) in enumerate(zip(table, names)) if a != b), None)
        if bad is not None:
            print(f"  first difference at trial {bad + 1}: table={table[bad]!r} names={names[bad]!r}")
        print("  first 3 names in file:", names[:3])
else:
    print("\nSkipped order check (need CSI1_trials.csv and CSI1_imgnames.txt here).")
