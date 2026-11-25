## Method 1 Findings: Issues and Next Steps

Running Method 1 surfaced a few weaknesses in our analysis. Some are already fixed and the rest are planned for the next stage.

### Fixed

| Issue | What we found | Fix |
|-------|---------------|-----|
| Confidence intervals were too narrow | Our first CIs resampled voxels, but neighboring voxels share signal and noise, so they aren't independent. This made the intervals 2-4x too narrow | Resample test images instead (`07_table2_bootstrap.py`). This showed PPA's Layer 3 advantage over Layer 4 is not reliable, so it's now treated as a tie |
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
