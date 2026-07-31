# Cellect development benchmarks

This directory contains reproducible, development-only benchmarks for cell segmentation and
counting. Downloaded datasets, model weights, Python environments, and generated results are
git-ignored and are not included in the Cellect application bundle.

## Dataset policy

The initial smoke test uses ten curated LIVECell phase-contrast images, covering all eight cell
types in the dataset. They were retrieved through the `mserna/livecell-hf` Hugging Face mirror:

- Source: https://sartorius-research.github.io/LIVECell/
- Mirror: https://huggingface.co/datasets/mserna/livecell-hf
- License: CC BY-NC 4.0
- Citation: Edlund et al., *LIVECell — A large-scale dataset for label-free live cell
  segmentation*, Nature Methods (2021).

This small fixture set does **not** contain ground-truth masks. Its results can reveal runtime,
gross failures, and qualitative differences, but cannot establish segmentation accuracy. Final
accuracy measurements require the official LIVECell images and COCO annotations.

These assets are for non-commercial development evaluation. Do not copy them into the shipping
application without reviewing the license and intended distribution.

## Model policy

Only models with traceable public weights and licenses are benchmarked. A natural-image backbone
such as MobileNet or YOLO is not treated as a cell-segmentation model unless it has
microscopy-specific trained weights.

The first comparison uses the Cellpose 3 microscopy model zoo: `cyto3`, `livecell_cp3`,
`tissuenet_cp3`, `cyto2_cp3`, `cyto2`, `cyto`, and `nuclei`. Cellpose-SAM is evaluated in a
separate Cellpose 4 environment because it replaced the legacy API. Generic ImageNet MobileNet
and COCO YOLO weights are intentionally excluded: they have not learned cell instances and their
output labels do not include cells.
