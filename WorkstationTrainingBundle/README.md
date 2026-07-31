# Cellect workstation training bundle v3

This bundle prepares manually annotated transmitted-light microscopy data, trains semantic cell
segmentation candidates and Cellpose foundation models, calibrates deployable post-processing,
searches semantic ensembles, exports TorchScript and ONNX artifacts, and creates one result ZIP for
the Mac.

The accuracy workflow is deliberately staged. Do not start the long run with `./run.sh best`
directly. A fingerprint-matched `best-smoke` PASS is required first.

## Recommended workstation

- Linux with a recent NVIDIA driver
- NVIDIA RTX 3090 or a comparable CUDA GPU
- Python 3.11 or 3.12
- At least 100 GB free disk space for the expanded data and checkpoints
- Several days available for the full `best` run

The scripts create and manage `.venv` automatically. CUDA training is required; the pipeline exits
instead of silently falling back to CPU.

## Exact v3 workflow

Open a terminal in `WorkstationTrainingBundle`, point the bundle at the complete DeepSea directory,
and run the fingerprinted smoke gate:

```bash
cd /path/to/WorkstationTrainingBundle
export CELLECT_DEEPSEA_ROOT=/mnt/firecuda2/deepsea
chmod +x run.sh
./run.sh best-smoke
```

The environment variable must point to the directory containing the complete `track`, `segment`,
and `final` folders. The aliases `tracking`, `segmentation`, `tracking_dataset`,
`segmentation_dataset`, and `final_dataset` are also recognized.

`best-smoke` first performs the full-data preflight described below. It does not preflight only a
small sample: every development image/mask pair is decoded and checked, while final-test labels are
cryptographically sealed without being parsed. Every asset is added to the data fingerprint. After
that full preflight passes, the command runs a limited one-epoch path through the models, validation
roles, export, and workstation parity checks.

The command writes this marker only after the complete smoke path succeeds:

```text
output/best_smoke_pass_v3.json
```

Confirm that it contains `"status": "PASS"`. Only then start the long run:

```bash
./run.sh best
```

`best` verifies that the PASS marker has the same bundle version, dataset fingerprint, and stage
fingerprint as the current code and data. If any Python file, `requirements.txt`,
`run.sh`, or fingerprinted dataset input changes, the old marker is rejected and `best-smoke` must
be rerun. Smoke and full outputs live in separate fingerprint-specific directories, so a limited
run cannot be resumed as if it were a full run.

For troubleshooting, the preflight can be run by itself:

```bash
./run.sh preflight
```

This does not replace `best-smoke` and does not create its PASS marker.

The older `smoke` and `full` modes are retained for the original LIVECell-only development
comparison. They are not substitutes for the v3 `best-smoke` followed by `best` workflow.

## Full-data preflight and experiment identity

Before expensive training, the v3 preflight:

- decodes every development image and mask and checks dimensions, finite pixels, annotations, and
  non-empty instances;
- decode-checks final-test images for transport integrity, but only existence-checks, sizes, and
  SHA-256 seals final-test labels without decoding them or exposing instance statistics;
- checks physical paths and byte-identical image hashes for cross-split leakage;
- validates every required DeepSea image/mask/contact-map triplet;
- combines upstream LIVECell train+validation acquisitions, then assigns and records four disjoint
  development roles;
- checks available disk space;
- writes `output/splits_v3.jsonl` and `output/preflight_v3.json`;
- computes a dataset fingerprint used by all v3 experiment directories and checkpoints.

For development samples, the split manifest records paths, hashes, upstream split, scientific role,
dimensions, and instance counts. Final-test rows deliberately omit label-derived dimensions and
instance counts. No model is run against final-test images, and their performance is not measured
or used. Fitting, checkpoint selection, post-processing calibration, model selection, and ensemble
selection do not parse the final-test masks.

Checkpoints also contain the stage fingerprint. A checkpoint from different code, dependencies,
data, or split metadata is refused rather than resumed.

Scientific role hashes use dataset-relative acquisition identities rather than absolute workstation
paths. Moving an unchanged bundle or dataset to another mount point therefore cannot silently
change the split.

## Scientific data roles

The official final-test split is retained and sealed. Upstream train and validation data are
combined into one development pool and divided deterministically by acquisition group into four
roles, with a 70/10/10/10 target allocation:

1. `train` — parameter fitting only.
2. `checkpoint` — validation loss, scheduler decisions, early stopping, and checkpoint selection.
3. `calibration` — frozen single-model reconstruction thresholds, object filters, and Cellpose
   inference settings.
4. `ensemble_selection` — comparison of frozen single models and joint selection of semantic
   ensemble membership and its fixed operating settings.

The role assignment is written to `splits_v3.jsonl`. All time points and crops from one LIVECell
acquisition remain in one role even when the source files came from different upstream development
JSONs. The official final-test split is not subdivided and is not evaluated by `best`.

At the end of `best`, the chosen single semantic model and its artifact hashes, chosen semantic
ensemble membership, fixed settings, algorithm versions, split-manifest hash, and selection roles
are frozen in:

```text
deployment_lock_v3.json
```

That lock is produced before any final-test evaluation. The returned ZIP intentionally contains no
final-test accuracy report. A future final evaluation must use only preregistered artifacts from the
lock, after the Mac Core ML/Swift gate passes. Changing a model, preprocessing rule, fusion rule, or
postprocessor after seeing a final result requires a new experiment and a new untouched test set.

## DeepSea mask and contact-map semantics

DeepSea's segmentation `masks` are binary cell-foreground masks; their pixel values are not
instance IDs. The sibling `unetwmaps` files provide the official touching-cell edge/contact signal.
The v3 adapter removes those contact pixels to seed separate connected interiors, reconstructs
instance labels, restores contiguous cell foreground, and retains the contact signal for the
boundary target.

The preflight fails closed if an image, binary mask, or matching `unetwmaps` file is missing. It does
not silently convert one connected binary region into one cell. The `segment` collection is the
canonical segmentation source. `track` and `final` are recognized for provenance but are not added
again because they contain overlapping frames.

DeepSea's official segmentation test folder remains sealed as final test. Because its training
folder has no separate checkpoint/calibration set, v3 deterministically reserves 15% of the
official training acquisitions as development validation. Filename variants such as
`A11_z003_c001` and `A11_z016_c001` stay together, and preflight fails if one source acquisition
crosses a training, validation, or final-test role.

Across all semantic datasets, the boundary head is trained for internal cell-cell contacts and
near-contact ridges, not the ordinary cell/background outline. DeepSea's explicit contact map is
kept only where it is consistent with two adjacent reconstructed cells.

## Models and training objective

The `best` profile trains nine semantic candidates from mobile to heavy architectures:

- MobileNetV3-Small U-Net
- MobileNetV3-Large DeepLabV3+
- EfficientNet-B0 U-Net
- ResNet18 U-Net
- EfficientNet-B3 U-Net++
- ResNet50 DeepLabV3+
- ResNet101 U-Net++
- SegFormer-B2
- SegFormer-B5

It also fine-tunes `cpsam_v2`, `cpdino`, and `cpdino-vitb`. Foundation models remain research-only
until each one can be converted and passes the same Core ML and physical-iPhone parity gates as an
app model.

Semantic training uses AdamW, separate encoder/decoder learning rates, EMA checkpoint weights,
gradient accumulation and clipping, ReduceLROnPlateau, mixed precision, and early stopping on the
checkpoint role. The foreground objective combines focal BCE, soft Dice, and asymmetric Tversky.
The boundary objective combines positive-weighted focal BCE and soft Dice against the internal
contact target. Background-dominated pixel accuracy is not used for selection.

Checkpoint loss, overlap, boundary, count, and sampled instance metrics are first averaged within
each microscopy dataset and then equally across datasets. The same macro-domain rule is used when
ranking frozen single models. This prevents LIVECell's larger image count from overwhelming smaller
transmitted-light domains; per-domain metrics and sample counts remain in the reports.

The current `best` ceilings and optimizer settings are:

| Setting | Value |
|---|---|
| Epoch ceiling | 300 |
| Early-stop patience / minimum improvement | 30 / `1e-4` composite score |
| Encoder / decoder learning rate | `3e-5` / `1e-4` |
| AdamW weight decay / explicit L1 | `1e-4` / `0` |
| Gradient accumulation / clipping | 8 micro-batches / L2 norm 1.0 |
| EMA | decay 0.999 |
| Scheduler | factor 0.5, patience 8, floor `1e-7` |
| Boundary positive weight | 5.0 |
| Random seed | 1337, deterministic accuracy mode |

The effective micro-batch may be reduced automatically by the GPU-memory probe. The effective
batch and accumulation settings are recorded with each run.

Cellpose foundation training starts independently from each built-in model. It uses learning rate
`1e-5`, weight decay `0.1`, batch size 1, 256-pixel SAM tiles or 384-pixel DINO tiles, and saved
snapshots. Snapshot choice uses the checkpoint role; `cellprob_threshold` and `flow_threshold` use
the separate calibration role. No final-test evaluation is produced.

## Calibration and ensemble selection

After a semantic checkpoint is frozen, the calibration role searches foreground/contact cutoffs,
scale-normalized minimum area, and versioned object-confidence filters. Reports retain the baseline,
all tested candidates, curve summaries, per-domain metrics, and the selected fixed configuration.
The deployable reconstruction is the versioned iPhone-compatible 3×3 morphology, 8-connected
interior-marker, raster-ordered FIFO expansion implemented by `deployment_runtime.py`. A separate
research reconstruction must not select settings shipped to the app.

Semantic ensemble selection uses only the `ensemble_selection` role. Candidate combinations receive
probability-map proxy scores; shortlisted combinations receive the full fixed reconstruction and
filter search. The selected members, equal-weight probability merge, and operating settings are
written to the deployment lock. Artifact integrity is retained in the model records and checksum
manifest. The report does not include held-out final-test metrics.

## Whole-frame deployment path versus tiled research path

Two inference paths must not be conflated:

- **Current iPhone deployment parity path:** the entire oriented source image is rendered to one
  fixed square model input, converted to grayscale, repeated across three float32 channels in
  `[0,1]`, and passed through the model once. Raw foreground/contact logits are converted to
  probabilities, followed by the iPhone-compatible FIFO instance reconstruction. This is the path
  used when calibrating settings intended for the current Swift app.
- **Native/aspect-preserving tiled research path:** the source keeps its native rectangular shape,
  is reflect-padded when needed, and is processed with overlapping model-sized tiles. Raw logits are
  joined with a Hann window before one sigmoid and one full-image reconstruction. This path is
  memory-bounded and includes a non-square seam/finite-output gate, but it is not yet the current
  iPhone implementation and has no app-performance claim.

The existence of tiled artifacts or a tiled seam PASS does not make tiling deployable. Tiling can be
enabled in Cellect only after the same algorithm is implemented in Swift and passes Core ML and
physical-device parity.

## Deployment parity gates

The workstation gate runs before packaging each semantic model. On real calibration images it
checks eager PyTorch against the exported TorchScript and ONNX logits, verifies output shape and
finite values, enforces numerical error limits, writes a golden input/logit fixture, and exercises
the non-square tiled seam path. The artifacts are:

```text
deployment_parity/deployment_parity_v4.json
deployment_parity/deployment_golden_v4.npz
```

A workstation PASS establishes only PyTorch/TorchScript/ONNX parity on Linux. It is necessary but
not sufficient for iPhone deployment.

After copying the result ZIP to the Mac, the later Mac gate must:

1. verify the returned SHA-256 manifest and source TorchScript hash;
2. convert the frozen artifact to Core ML;
3. compare Core ML logits with `deployment_golden_v4.npz`;
4. compare CoreGraphics grayscale/resizing with the frozen input tensor;
5. run the Swift fusion and FIFO postprocessor on golden probability fixtures;
6. verify the compiled model on a physical target iPhone.

Do not mark a model or ensemble as deployed, and do not open the final test for performance
evaluation, until that Mac/Core ML/Swift/device gate passes.

## Dataset scope

The v3 accuracy profile is restricted to manually annotated, transmitted-light eukaryotic-cell
images relevant to phone or microscope-camera counting. It includes LIVECell and available
brightfield, phase-contrast, DIC, and quantitative-phase sources. Yeast remains capped during
semantic and Cellpose training so its abundant round/oval morphology cannot dominate sampling.
Bacterial and fluorescence-only sources remain outside the primary model.

Dataset documentation estimates are not treated as run results. The authoritative discovered pair,
role, and instance counts come from `preflight_v3.json` and `splits_v3.jsonl` for that exact
fingerprint. Access and licensing notes are in `DATASET_ACCESS.md`.

Synthetic phone-through-eyepiece augmentation includes illumination variation, vignetting, tone
changes, shot/read noise, blur, downsampling, compression, sharpening, mild geometric distortion,
and sparse occlusion while retaining clean samples. Synthetic augmentation does not prove phone
generality. A separately locked, manually annotated real-phone test set is still required for a
publishable iPhone-generalization claim.

## Returned artifacts

After `best` completes, copy this file back to the Mac:

```text
output/cellect_workstation_results.zip
```

Place it in `cellect/WorkstationResults/inbox/`. The archive includes checksums, the v3 preflight and
split manifest, training histories, frozen checkpoints, calibration and ensemble-selection reports,
TorchScript/ONNX exports, workstation parity reports and golden fixtures, and
`deployment_lock_v3.json`. It intentionally does not contain downloaded training data or a
final-test performance evaluation.

Every reported metric in the archive is a development metric tied to one of the three recorded
validation roles. Do not present those values as held-out final-test results.
