# Mind vs. Machine: fMRI Decoding with CNNs and Bayesian Inference
 
<p align="center">
  <img src="assets/brain_scan.gif" alt="Animated brain scan with visual cortex activation" width="320">
</p>
Do artificial and biological vision systems process images in similar ways?
 
The human visual system processes what we see in a hierarchy: early regions respond to edges and colors, while later regions respond to whole objects and scenes. CNNs were inspired by this design and learn a similar layered sequence. Drawing on the efficient coding hypothesis, which builds on Claude Shannon's information theory, many researchers argue this is no coincidence: any system that efficiently encodes natural images may converge on a similar solution.
 
This project puts that idea to the test. Using fMRI scans from people viewing ~5,000 natural images (BOLD5000), we build **encoding models** that predict voxel-level brain activity from CNN features, then flip them into a **Bayesian decoder** that identifies which image a person was viewing from their brain activity alone.
 
---
 
## Research Questions
 
| # | Question | Approach |
|---|----------|----------|
| **RQ1** | Do early, middle, and deep CNN layers strictly correspond to early, middle, and late visual processing regions in the human brain? | Compare encoding accuracy of each CNN layer across each visual region of interest or ROI |
| **RQ2** | How much does adding an image prior help the Bayesian model pick the right image, compared to relying only on the predicted brain response? | Compare identification accuracy of likelihood-only vs. full posterior (likelihood × prior) decoding |
| **RQ3** | Which copula family best captures the non-linear dependency structure and tail behavior between CNN feature representations and fMRI voxel activations? | Fit copulas to feature-voxel pairs and compare by AIC and tail dependence |
 
---
 
## Pipeline Overview
 
```mermaid
flowchart LR
    A[Images] --> B[Pretrained CNN<br/>ResNet]
    B --> C[Features from 4 layers<br/>early to deep]
    C --> D[Regression model]
    E[fMRI voxel data] --> D
    D --> F[Predicted brain response]
    F --> G[Bayesian decoder]
    G --> H[Guess which image<br/>they were viewing]
```
 
**Step 1: Get CNN features.** We run each image through a pretrained ResNet-50 and extract features from all four of its layers, from early to deep. Each layer slides small learned filters, called kernels, across the image to produce feature maps. Early kernels detect edges and colors, while deeper ones combine these to detect textures, object parts, and whole objects. We average each layer's feature maps down to a 4x4 grid, which keeps coarse information about where things are in the image, and flatten it into one feature vector per image.
 
<p align="center">
  <img src="assets/resnet_architecture.svg" alt="ResNet-50 architecture with the early, middle, and deep layers used for feature extraction" width="700">
</p>
**Step 2: Predict brain activity.** A regression model learns how the CNN features relate to fMRI voxel activity in each brain region. We check how well it predicts brain responses for images it hasn't seen before.
 
**Step 3: Decode the image with Bayes.** Now we flip it. Given a brain scan, we score every candidate image by its likelihood (how well its predicted brain response matches the real scan) and its prior (how likely that image was to begin with), then pick the image with the best combined score.
 
**Step 4: Compare layers to brain regions.** We check which CNN layer best predicts each brain region. If CNNs really mirror the brain, early layers should match early regions and deep layers should match later regions. The figure shows the brain from two angles because some regions sit on the outside of the brain and others on the inside. The **lateral** view is the outer surface seen from the side, and the **medial** view is the inner surface you'd see if you split the brain down the middle. LOC and OPA appear mainly in the lateral view because they sit on the outer surface, while RSC and PPA appear in the medial view because they sit on the inner surface and underside. Early visual cortex shows up in both because it wraps around the back of the brain.
 
<p align="center">
  <img src="assets/cnn_layers_brain_regions.svg" alt="CNN layers mapped to expected brain regions: early to EVC, middle to LOC, deep to PPA, OPA, and RSC" width="900">
</p>
**Step 5: Copulas.** Regression captures the average relationship between predicted and actual brain activity, but not whether that relationship changes at the extremes. Using unseen images, we pair each region's predicted response with its actual response and fit several copula families to these pairs. A Gaussian copula treats the relationship as symmetric, Clayton captures stronger agreement when both values are low, and Gumbel captures stronger agreement when both values are high. We pick the best-fitting family using AIC.
 
---
 
## Data
 
We use the **BOLD5000** dataset, where 4 participants viewed around 5,000 images (from SUN, COCO, and ImageNet) inside an fMRI scanner.
 
We focus on participant **CSI1** (15 sessions, 5,254 trials, 4,916 unique images) and use the Release 2.0 single-trial brain responses, estimated with GLMsingle, for five visual regions: early visual cortex (EarlyVis, labeled EVC in the figures), LOC, OPA, PPA, and RSC.
 
| Source | What's there |
|--------|--------------|
| [BOLD5000 website](https://bold5000-dataset.github.io/website/) | Overview, download links, and stimulus images |
| [Figshare (Release 2.0)](https://figshare.com/articles/dataset/BOLD5000_Release_2_0/14456124) | Updated, less noisy fMRI estimates (used in this project) |
| [Figshare (Release 1.0)](https://figshare.com/articles/BOLD5000/6459449) | Original release: ROI data, image labels, behavioral data |
| [OpenNeuro](https://openneuro.org/datasets/ds001499) | Raw fMRI data, event files (trial order), and ROI masks |
| [BOLD5000 paper](https://arxiv.org/abs/1809.01281) | Chang et al. (2019) |
 
### Getting the data
1. Download the stimulus images (`BOLD5000_Stimuli.zip`) from the [BOLD5000 download page](https://bold5000-dataset.github.io/website/download.html) and unzip them into `data/stimuli/`
2. Download `BOLD5000_GLMsingle_ROI_betas.zip` from Figshare Release 2.0 and unzip it into `data/fmri/roi_betas/`
3. Download CSI1's event files from OpenNeuro into `data/fmri/ds001499/`. These record which image was shown on each trial, in scan order:
```bash
aws s3 cp --no-sign-request s3://openneuro.org/ds001499/ data/fmri/ds001499 --recursive --exclude "*" --include "sub-CSI1/*events.tsv"
```
The data files are too big for GitHub, so they're not included in this repo.
 
### Running the pipeline
Run the scripts in `src/` in order, from the repo root:
 
| Step | Script | What it does |
|------|--------|--------------|
| 1 | `01_download_csi1_typed.sh` | Downloads the data |
| 2 | `02_build_trial_table_events.py` | Builds the trial table that maps each trial to its image. Its output, `CSI1_trials.csv`, needs to be in `data/` for the next step |
| 3 | `03_build_y.py` | Builds the brain response matrices. Combines hemispheres, z-scores each voxel within each session, and averages repeated images |
| 4 | `04_make_split.py` | Splits images into train (80%) and test (20%), stratified by image source, with 5 CV folds |
| 5 | `05_extract_features.py` | Extracts ResNet-50 features from all four layers |
| 6 | `06_fit_encoding.py` | Fits the encoding model (PCA + ridge regression per voxel) and scores it against each voxel's noise ceiling |
| 7 | `07_table2_bootstrap.py` | Bootstrap confidence intervals for the layer comparisons |
 
`check_outputs.py` confirms the split, brain responses, and encoding results match our report. `inspect_roi_betas.py` is an optional look at the raw ROI data.
 
---
 
## Method 1 Findings: Issues and Next Steps
 
### Results so far
- Deeper layers predict the higher visual regions much better. Layer 4 beats Layer 1 in LOC, OPA, PPA, and RSC by 0.17-0.22 in r / noise ceiling
- Early visual cortex shows no difference between Layer 1 and Layer 4, so the expected early-layer match doesn't appear. Layer 3 predicts it best
- The best layer for each region reaches 71-91% of the correlation a perfect model could reach, given fMRI noise
Running Method 1 also surfaced a few weaknesses in our analysis. Some are already fixed and the rest are planned for the next stage.
 
### Fixed
 
| Issue | What we found | Fix |
|-------|---------------|-----|
| Confidence intervals were too narrow | Our first CIs resampled voxels, but neighboring voxels share signal and noise, so they aren't independent. This made the intervals 2-5x too narrow | Resample test images instead (`07_table2_bootstrap.py`). This showed PPA's Layer 3 advantage over Layer 4 is not reliable, so it's now treated as a tie |
| Results might depend on the noise ceiling cutoff | We drop voxels with a noise ceiling below 0.1, which was an arbitrary choice | Reran with cutoffs from 0.05 to 0.3. The best layer per region holds from 0.05 to 0.2, and only OPA's tie flips at 0.3 |
 
### Next steps
 
| Issue | Why it matters | Plan |
|-------|----------------|------|
| No baseline | We can't tell if the layer pattern comes from ImageNet training or just the CNN's architecture | Repeat Method 1 with an untrained, random-weights ResNet-50 |
| Coarse feature pooling | Averaging each feature map to a 4x4 grid blurs fine spatial detail, which may hurt layer1 for early visual cortex | Try finer pooling for layer1 |
| PCA keeps different amounts of each layer | 1,000 components keep 97.6% of layer1's variance but only 74.4% of layer3's | Rerun with other numbers of components |
| Small leak in model selection | Standardization and PCA are fit on all training images, including the CV folds used to pick lambda | Fit them inside each CV fold. The test set is not affected |
| Noise ceiling uncertainty | The noise ceiling comes from only 113 repeated images, so it could be underestimated, which would inflate scores like RSC's 0.906 | Resample the repeated images to see how stable the ceiling is |
| Linear readout only | Ridge regression can only find linear relationships between features and voxels | Train a small neural network (MLP) on the same features and compare |
| Image prior for Method 2 | A prior based on image source gives every test image the same probability, so it can't change identification results | Choose a prior that differs across individual images |
 
---
 
## Repo Structure
 
```
mind-vs-machine-fmri-decoding/
├── assets/          # Images and diagrams used in this README
├── data/            # BOLD5000 images and fMRI data (not tracked)
├── notebooks/       # Analysis notebooks
├── src/             # Feature extraction, models, and decoding code
├── results/         # Outputs (only the small result tables are tracked)
├── LICENSE
└── README.md
```
 
---
 
## Skills
Python, PyTorch, CNNs, regression, Bayesian inference, copulas, fMRI analysis, data visualization
 
## References
Chang, N., Pyles, J. A., Marcus, A., Gupta, A., Tarr, M. J., & Aminoff, E. M. (2019). BOLD5000, a public fMRI dataset while viewing 5000 visual images. *Scientific Data, 6*, 49.
 
He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 770-778.
 
Naselaris, T., Prenger, R. J., Kay, K. N., Oliver, M., & Gallant, J. L. (2009). Bayesian reconstruction of natural images from human brain activity. *Neuron, 63*(6), 902-915.
 
Prince, J. S., Charest, I., Kurzawski, J. W., Pyles, J. A., Tarr, M. J., & Kay, K. N. (2022). Improving the accuracy of single-trial fMRI response estimates using GLMsingle. *eLife, 11*, e77599.
