"""
Written by: Erika Sohn

What this code does:
* Builds a BOLD5000 trial table for one subject: one row per trial, in the
  true order the images were shown in the scanner
* Reads the trial timing from the OpenNeuro events.tsv files
* Cross-checks the table against the presentation lists in
  BOLD5000_Stimuli, to confirm every trial matches the image recorded
  as shown

Inputs: OpenNeuro events.tsv files and BOLD5000_Stimuli presentation lists
Output: <SUBJ>_trials.csv (one row per trial, in scan order; 5,254 trials
        for CSI1)
Run from Terminal, inside ~/bold5000:
    python3 build_trial_table_events.py          CSI1 by default
    python3 build_trial_table_events.py CSI2     another subject
"""

import re
import sys
from pathlib import Path

import pandas as pd

SUBJ = sys.argv[1] if len(sys.argv) > 1 else "CSI1"

# Use the folder this script sits in; fall back to ~/bold5000 and ~/Documents/bold5000.
candidates = [Path(__file__).resolve().parent, Path.home() / "bold5000",
              Path.home() / "Documents" / "bold5000"]
BASE = next((c for c in candidates if (c / "ds001499" / f"sub-{SUBJ}").is_dir()), None)
if BASE is None:
    sys.exit("Could not find ds001499/sub-" + SUBJ + " in any of:\n  "
             + "\n  ".join(str(c) for c in candidates)
             + "\nPut this script in the folder that contains ds001499 and BOLD5000_Stimuli.")
print("Using data folder:", BASE)
EVENTS = BASE / "ds001499" / f"sub-{SUBJ}"
STIM = BASE / "BOLD5000_Stimuli"
OUT = BASE / f"{SUBJ}_trials.csv"


def num(pattern, text):
    return int(re.search(pattern, text).group(1))


# main task events files, in scan order (session, then run)
all_events = list(EVENTS.glob("ses-*/func/*_events.tsv"))
files = sorted(
    EVENTS.glob("ses-*/func/*task-5000scenes*_events.tsv"),
    key=lambda f: (num(r"ses-(\d+)", f.name), num(r"run-(\d+)", f.name)),
)
print(f"{len(all_events)} events files total, {len(files)} main-task (5000scenes)")
other = sorted({re.search(r"task-([^_]+)", f.name).group(1) for f in all_events} - {"5000scenes"})
if other:
    print("other tasks skipped:", other)

frames = []
for f in files:
    df = pd.read_csv(f, sep="\t")
    df = df.sort_values("onset").reset_index(drop=True)
    df["session"] = num(r"ses-(\d+)", f.name)
    df["run"] = num(r"run-(\d+)", f.name)
    df["trial_in_run"] = range(1, len(df) + 1)
    frames.append(df)
ev = pd.concat(frames, ignore_index=True)

ev["image"] = ev["ImgName"].astype(str).str.strip().str.replace(r"^rep_", "", regex=True)
ev["trial_global"] = range(1, len(ev) + 1)
ev["trial_in_session"] = ev.groupby("session").cumcount() + 1
ev["n_presentations"] = ev.groupby("image")["image"].transform("size")

# link to image files
imgs = {p.name: p for p in STIM.rglob("Presented_Stimuli/*/*") if p.is_file()}
ev["image_path"] = ev["image"].map(lambda n: str(imgs.get(n, "")))
missing = sorted(set(ev.loc[ev["image_path"] == "", "image"]))

# cross check each scanned run against the presentation lists
def read_list(f):
    return tuple(l.strip().removeprefix("rep_") for l in f.read_text().splitlines() if l.strip())

lists = {}
for f in (STIM / "Stimuli_Presentation_Lists" / SUBJ).rglob("*_run*.txt"):
    lists[(num(r"sess(\d+)", f.name), num(r"run(\d+)", f.name))] = read_list(f)
by_content = {seq: key for key, seq in lists.items()}

same, moved, unmatched = 0, [], []
for (s, r), grp in ev.groupby(["session", "run"]):
    seq = tuple(grp["image"])
    if lists.get((s, r)) == seq:
        same += 1
    elif seq in by_content:
        moved.append(((s, r), by_content[seq]))
    else:
        unmatched.append((s, r))

# do the Sess/Run columns inside the events files agree with the file names?
col_note = ""
if {"Sess", "Run"} <= set(ev.columns):
    diff = ev[(ev["Sess"] != ev["session"]) | (ev["Run"] != ev["run"])]
    col_note = f"{diff[['session', 'run']].drop_duplicates().shape[0]} runs where Sess/Run columns differ from file names"

cols = ["trial_global", "session", "run", "trial_in_run", "trial_in_session",
        "image", "ImgType", "n_presentations", "onset", "image_path"]
cols = [c for c in cols if c in ev.columns]
ev.insert(0, "subject", SUBJ)
ev[["subject"] + cols].to_csv(OUT, index=False)

# summary
print(f"\nWrote {OUT}")
print(f"{len(ev)} trials, {ev['image'].nunique()} unique images, {len(missing)} missing image files")
print("images shown more than once:", int((ev.groupby('image').size() > 1).sum()))
print("runs per session:", ev.groupby("session")["run"].nunique().to_dict())
print("trials per session:", ev.groupby("session").size().to_dict())
print("\nCross-check vs presentation lists:")
print(f"  {same} runs identical (same session/run number, same images, same order)")
print(f"  {len(moved)} runs found under a different session/run number in the lists")
for scan, lst in moved[:10]:
    print(f"     scanned ses {scan[0]} run {scan[1]}  <-  list sess {lst[0]} run {lst[1]}")
print(f"  {len(unmatched)} runs with no exact match in the lists: {unmatched[:10]}")
if col_note:
    print(" ", col_note)
if missing:
    print("missing examples:", missing[:5])
