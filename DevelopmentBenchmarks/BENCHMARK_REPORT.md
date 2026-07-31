# Initial label-free cell benchmark

- Date: 2026-07-30
- Machine: Intel MacBook Pro, CPU inference
- Fixtures: 10 LIVECell phase-contrast JPEGs (all eight cell types; no ground-truth masks)

## Cellpose 3 results

| Model | Weight | Seconds/image | Counts on the 10 fixtures | Assessment |
|---|---:|---:|---|---|
| `livecell_cp3` | 25 MB | 9.36 | 60, 53, 37, 64, 31, 72, 4, 131, 198, 520 | Best task match; visually plausible |
| `cyto3` | 25 MB | 11.53 | 64, 82, 65, 69, 35, 92, 10, 173, 215, 532 | Strong general fallback; tends to find more objects |
| `cyto2_cp3` | 25 MB | 15.61 | 43, 54, 34, 1, 43, 57, 2, 87, 188, 459 | Unstable across cell types |
| `cyto2` | 25 MB | 11.73 | 58, 54, 37, 2, 42, 64, 2, 74, 193, 446 | Unstable across cell types |
| `cyto` | 25 MB | 17.61 | 38, 49, 42, 30, 24, 66, 21, 82, 199, 444 | Older and slowest legacy candidate |
| `nuclei` | 25 MB | 6.77 | 36, 15, 19, 3, 17, 41, 3, 3, 138, 108 | Wrong target for whole-cell phase contrast |
| `tissuenet_cp3` | 25 MB | 10.76 | 1, 0, 0, 0, 4, 0, 0, 0, 0, 7 | Clear domain mismatch |

Counts are descriptive, not accuracy scores. Runtime includes model initialization. The exact
per-image label masks and previews are generated under the ignored `results/` directory.

## Current decision

`livecell_cp3` is the leading pretrained desktop reference because it was trained for the same
phase-contrast domain and is the most consistent candidate visually. `cyto3` should remain the
general-purpose comparison model. A defensible final selection still requires the official
LIVECell instance annotations and quantitative AP/IoU evaluation.

Cellpose 4 / Cellpose-SAM was evaluated separately. Its current `cpsam_v2` weight is about 1.15 GB
and it found 69 instances on the first A172 fixture. The cold run took 614 seconds, including the
weight download; inference alone still took several minutes on this Intel CPU. Before conversion
or runtime memory, that makes it a desktop reference rather than a model to bundle in an iPhone
app.

MobileNet and YOLO describe architectures, not ready-to-use cell segmenters. The paper's trained
MobileNet/VGG/DenseNet weights are not publicly downloadable, and generic ImageNet/COCO weights do
not contain a cell segmentation class. A mobile model must therefore be trained or fine-tuned on
licensed cell masks and converted to Core ML before it can honestly be offered in Cellect.
