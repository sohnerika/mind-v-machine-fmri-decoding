"""Extract ResNet-50 features for the BOLD5000 images, in the same row order as y.

Usage (from the repo root):
    python3 src/extract_features.py              # CSI1, 4x4 spatial grid
    python3 src/extract_features.py --grid 2     # smaller files
    python3 src/extract_features.py --test 256   # timing test, saves nothing

Inputs:
    results/y/<SUBJ>_images.csv     row order shared with every y matrix
    data/stimuli/BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli/

Outputs in results/features/resnet50/:
    <SUBJ>_<layer>.npy   (n_images, n_features) float32, one file per stage
    <SUBJ>_info.json     layers, shapes, preprocessing, settings

ResNet-50 is built in four stages. We take the output of each one:
    layer1  early         256 channels on a 56x56 map
    layer2  early-middle  512 channels on 28x28
    layer3  middle       1024 channels on 14x14
    layer4  deep         2048 channels on 7x7
Each channel's map is averaged down to a grid x grid map, which keeps coarse
information about where things are in the image.

The CNN is pretrained on ImageNet and used as is (no training here). The first run
downloads the pretrained weights (about 100 MB) to ~/.cache/torch.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as tnf
from PIL import Image
from torchvision import models, transforms

parser = argparse.ArgumentParser()
parser.add_argument("--subject", default="CSI1")
parser.add_argument("--grid", type=int, default=4,
                    help="each channel's map is average-pooled to grid x grid")
parser.add_argument("--batch", type=int, default=32)
parser.add_argument("--test", type=int, default=0,
                    help="timing test: process only this many images, estimate the full run, save nothing")
args = parser.parse_args()

REPO = Path(__file__).resolve().parent.parent
ORDER = REPO / "results" / "y" / f"{args.subject}_images.csv"
STIM = REPO / "data" / "stimuli" / "BOLD5000_Stimuli" / "Scene_Stimuli" / "Presented_Stimuli"
OUT = REPO / "results" / "features" / "resnet50"
OUT.mkdir(parents=True, exist_ok=True)

# ---- 1. Images, in y's row order ------------------------------------------------
order = pd.read_csv(ORDER)
paths = {p.name: p for p in STIM.rglob("*") if p.is_file() and not p.name.startswith(".")}
files = [paths[n] for n in order["image"]]
n_total = len(files)
if args.test:
    files = files[:args.test]
print(f"{n_total} images, in the row order of {ORDER.name}"
      + (f"  (timing test on the first {len(files)})" if args.test else ""))

# ---- 2. Pretrained ResNet-50 ----------------------------------------------------
# We do NOT build or train the CNN. We load ResNet-50 already trained by others
# on ImageNet (ILSVRC-2012: about 1.28 million images in 1,000 object categories)
# and use it frozen, only reading out its layer activations.
#   Model and weights: torchvision ResNet-50, IMAGENET1K_V1
#     https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html
#   Architecture: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning
#     for Image Recognition. CVPR. https://arxiv.org/abs/1512.03385
if torch.backends.mps.is_available():
    device = "mps"          # Apple-silicon GPU
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
weights = models.ResNet50_Weights.IMAGENET1K_V1      # weights trained on ImageNet
model = models.resnet50(weights=weights).eval().to(device)   # eval(): inference only, no training
print(f"ResNet-50 loaded on {device}")

LAYERS = ["layer1", "layer2", "layer3", "layer4"]


def layer_outputs(x):
    """Run images through ResNet-50 stage by stage and keep each stage's output.

    x: a batch of images, shape (batch, 3, 224, 224)
    returns: {"layer1": ..., "layer4": ...}, each of shape (batch, channels, height, width)
    """
    x = model.maxpool(model.relu(model.bn1(model.conv1(x))))   # stem: first conv + pooling
    outs = {}
    x = model.layer1(x); outs["layer1"] = x    # early:        (batch,  256, 56, 56)
    x = model.layer2(x); outs["layer2"] = x    # early-middle: (batch,  512, 28, 28)
    x = model.layer3(x); outs["layer3"] = x    # middle:       (batch, 1024, 14, 14)
    x = model.layer4(x); outs["layer4"] = x    # deep:         (batch, 2048,  7,  7)
    return outs


def to_features(fmap):
    """Shrink each channel's map to grid x grid and flatten to one row per image.

    The shrinking (average pooling) happens on the GPU whenever the map divides evenly
    into the grid, so only the small pooled result is copied to the CPU. For example,
    layer1's 56x56 maps become 4x4 on the GPU: about 200 times less data to move.
    """
    h, w = fmap.shape[-2:]
    if h % args.grid == 0 and w % args.grid == 0:
        # pool on the GPU: each grid cell is the average of an (h/grid) x (w/grid) block
        fmap = tnf.avg_pool2d(fmap, kernel_size=(h // args.grid, w // args.grid))
    else:
        # map doesn't divide evenly (e.g. 7x7 into 4x4): pool on the CPU instead
        fmap = tnf.adaptive_avg_pool2d(fmap.float().cpu(), args.grid)
    return fmap.float().cpu().flatten(1).numpy()            # (batch, channels * grid * grid)


# Participants saw the whole image, so resize it (no center crop), then apply
# the ImageNet normalization ResNet-50 was trained with.
prep = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---- 3. Run every image through the network --------------------------------------
store = {}
t0 = time.time()
with torch.no_grad():
    for start in range(0, len(files), args.batch):
        chunk = files[start:start + args.batch]
        batch = torch.stack([prep(Image.open(f).convert("RGB")) for f in chunk]).to(device)
        outs = layer_outputs(batch)
        for name in LAYERS:
            a = to_features(outs[name])
            if name not in store:
                store[name] = np.empty((len(files), a.shape[1]), dtype=np.float32)
            store[name][start:start + len(a)] = a
        done = start + len(chunk)
        if done % (args.batch * 25) < args.batch or done == len(files):
            print(f"  {done:5d}/{len(files)} images  ({time.time() - t0:.0f} s)")

if args.test:
    secs = time.time() - t0
    print(f"\nTiming test: {len(files)} images in {secs:.1f} s "
          f"-> full run of {n_total} images about {secs / len(files) * n_total / 60:.1f} min")
    raise SystemExit("(timing test only, nothing saved)")

# ---- 4. Save and check -----------------------------------------------------------
print(f"\n{'layer':7s} {'shape':>16s} {'% zeros':>8s}  ok")
info_layers = {}
for name, F in store.items():
    np.save(OUT / f"{args.subject}_{name}.npy", F)
    ok = bool(np.isfinite(F).all()) and bool((F.std(axis=0) > 0).any())
    zeros = 100 * float((F == 0).mean())
    print(f"{name:7s} {str(F.shape):>16s} {zeros:8.1f}  {'yes' if ok else 'CHECK'}")
    info_layers[name] = {"shape": list(F.shape), "pct_zeros": round(zeros, 2)}

info = {
    "model": "resnet50",
    "weights": "IMAGENET1K_V1 (torchvision)",
    "layers": info_layers,
    "spatial_pooling": f"adaptive average pool to {args.grid}x{args.grid} per channel",
    "preprocessing": "resize to 224x224 (no crop), ImageNet mean/std normalization",
    "row_order": str(ORDER.relative_to(REPO)),
    "n_images": len(files),
    "device": device,
    "torch": torch.__version__,
    "seconds": round(time.time() - t0, 1),
}
(OUT / f"{args.subject}_info.json").write_text(json.dumps(info, indent=2))
print(f"\nSaved {len(store)} layers to {OUT}")