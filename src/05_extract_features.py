"""
Written by: Erika Sohn

What this code does:
* Extracts features for CSI1's 4,916 BOLD5000 images from a ResNet-50
  pretrained on ImageNet, used frozen (no training or fine-tuning)
* Takes the output of each of ResNet-50's four stages:
      layer1  early          256 channels on a 56x56 map
      layer2  early-middle   512 channels on 28x28
      layer3  middle        1024 channels on 14x14
      layer4  deep          2048 channels on 7x7
* Average-pools each channel's map to a 4x4 grid (keeping coarse
  information about where things are) and flattens it, giving one
  feature vector per image per layer
* Resizes each image to 224x224 without cropping, since participants saw
  the whole image, then applies ImageNet normalization
* Saves rows in the same order as the response matrices in results/y/,
  so row i is the same image in the features and in Y

Inputs: results/y/CSI1_images.csv (row order)
        data/stimuli/BOLD5000_Stimuli/Scene_Stimuli/Presented_Stimuli/
Outputs: results/features/resnet50/ (CSI1_layer*.npy, CSI1_info.json)
Run from the repo root: python3 src/05_extract_features.py
The first run downloads the pretrained weights (about 100 MB).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as tnf
from PIL import Image
from torchvision import models, transforms

SUBJ = "CSI1"
GRID = 4 # pool each channel's map to 4x4
BATCH = 32 # images per batch

REPO = Path(__file__).resolve().parent.parent
ORDER = REPO / "results" / "y" / f"{SUBJ}_images.csv"
STIM = REPO / "data" / "stimuli" / "BOLD5000_Stimuli" / "Scene_Stimuli" / "Presented_Stimuli"
OUT = REPO / "results" / "features" / "resnet50"
OUT.mkdir(parents=True, exist_ok=True)

# == STEP 1: LOAD IMAGES == #

# find every image file and remember where it is by file name
paths = {}
for p in STIM.rglob("*"):
    if p.is_file() and not p.name.startswith("."):
        paths[p.name] = p

# list the files in the same order as the rows of Y
order = pd.read_csv(ORDER)
files = []
for name in order["image"]:
    files.append(paths[name])
print(f"{len(files)} images, in the row order of {ORDER.name}")

# == STEP 2: PRETRAINED RESNET-50 == #

# we do NOT build or train the CNN. we load ResNet-50 already trained by others
# on ImageNet (ILSVRC-2012: about 1.28 million images in 1,000 object categories)
# and use it frozen, only reading out its layer activations.
#   Model and weights: torchvision ResNet-50, IMAGENET1K_V1
#     https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html
#   Architecture: He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning
#     for Image Recognition. CVPR. https://arxiv.org/abs/1512.03385
if torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

weights = models.ResNet50_Weights.IMAGENET1K_V1
model = models.resnet50(weights=weights)
model = model.eval() # inference only, no training
model = model.to(device)

LAYERS = ["layer1", "layer2", "layer3", "layer4"]


def layer_outputs(x):
    """Run a batch of images through ResNet-50 and keep each stage's output."""
    # CNN's first convolution and pooling, before layer1
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    outs = {}
    x = model.layer1(x)
    outs["layer1"] = x # (batch,  256, 56, 56)
    x = model.layer2(x)
    outs["layer2"] = x # (batch,  512, 28, 28)
    x = model.layer3(x)
    outs["layer3"] = x # (batch, 1024, 14, 14)
    x = model.layer4(x)
    outs["layer4"] = x # (batch, 2048,  7,  7)
    return outs


def to_features(fmap):
    """Average each channel's map down to 4x4, then flatten to one row per image."""
    h, w = fmap.shape[-2:]
    if h % GRID == 0 and w % GRID == 0:
        fmap = tnf.avg_pool2d(fmap, kernel_size=(h // GRID, w // GRID))
    else:
        fmap = tnf.adaptive_avg_pool2d(fmap.float().cpu(), GRID)
    fmap = fmap.float().cpu()
    return fmap.flatten(1).numpy()


# participants saw the whole image so we need to resize it,
# then apply the ImageNet normalization ResNet-50 was trained with
prep = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# == STEP 3: RUN EVERY IMAGE THROUGH THE CNN == #

# collect each batch's features in a list per layer, then join them at the end
pieces = {}
for name in LAYERS:
    pieces[name] = []

with torch.no_grad():
    for start in range(0, len(files), BATCH):
        chunk = files[start:start + BATCH]

        # load and preprocess this batch of images
        images = []
        for f in chunk:
            img = Image.open(f).convert("RGB")
            images.append(prep(img))
        batch = torch.stack(images).to(device)

        # run the batch through the network and save each layer's features
        outs = layer_outputs(batch)
        for name in LAYERS:
            pieces[name].append(to_features(outs[name]))

# == STEP 4: SAVE == #

shapes = {}
for name in LAYERS:
    F = np.concatenate(pieces[name]).astype(np.float32) # (n_images, n_features)
    np.save(OUT / f"{SUBJ}_{name}.npy", F)
    shapes[name] = list(F.shape)
    print(f"{name}: {F.shape}")

# record the settings used
info = {
    "model": "resnet50",
    "weights": "IMAGENET1K_V1 (torchvision)",
    "layer_shapes": shapes,
    "spatial_pooling": f"average pool to {GRID}x{GRID} per channel",
    "preprocessing": "resize to 224x224 (no crop), ImageNet mean/std normalization",
    "row_order": str(ORDER.relative_to(REPO)),
}
(OUT / f"{SUBJ}_info.json").write_text(json.dumps(info, indent=2))
print(f"Saved to {OUT}")