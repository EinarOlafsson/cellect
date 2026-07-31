# Cellect v3 training audit

This file separates the earlier workstation result from the new v3 experiment. The executable
workflow and exact commands are in `README.md`.

## Earlier result is historical only

The previously returned workstation ZIP was a LIVECell-only `full` run. Its approximately 0.90
values were foreground Dice scores, not a hard-coded accuracy ceiling. That run did not establish
brightfield, DIC, quantitative-phase, microscope-to-microscope, or phone generality, and its
checkpoints, validation results, and post-processing choices must not be mixed into v3.

The older run used AdamW weight decay `1e-4` and no explicit L1 term. It selected checkpoints mainly
from foreground overlap, so touching-cell separation and count accuracy could remain poor despite a
high foreground Dice.

## v3 corrective controls

The replacement bundle adds the controls required before another long run:

- the reported synthetic self-check crash is isolated from any real DeepSea directory;
- DeepSea binary foreground masks are paired strictly with their `unetwmaps` contact maps and are
  reconstructed as instances instead of interpreting binary values as label IDs;
- DeepSea z/c variants from one annotated source are grouped together; 15% of official training
  acquisitions are reserved for development validation while the official test remains sealed;
- every discovered input is covered by a content-addressed preflight and split manifest before
  training;
- smoke, full, and best artifacts use fingerprint-specific directories and stale checkpoints are
  refused;
- upstream train+validation data is combined and divided by acquisition group into train,
  checkpoint, calibration, and ensemble-selection roles, so time points/crops cannot leak;
- official final-test masks receive size/SHA-256 integrity seals only, are never parsed for counts
  or dimensions, and are not scored by `best-smoke` or `best`;
- semantic training targets internal cell-cell contacts rather than ordinary outer cell contours;
- the deployable calibration reference mirrors the current iPhone 3x3 morphology, 8-connected
  markers, and FIFO expansion;
- full-size training micro-batches are probed with model, EMA, gradients, and optimizer state, with
  automatic micro-batch reduction on CUDA out-of-memory;
- checkpoint and frozen single-model metrics are averaged within dataset and then equally across
  datasets, with per-domain results retained, rather than being dominated by LIVECell's size;
- ensemble inference is cached one model at a time so all candidates are never resident on the GPU
  together;
- native, overlapping, aspect-preserving tiled inference is exercised as a separately labelled
  research path; it is not represented as current iPhone behavior;
- eager PyTorch, TorchScript, and ONNX logits are checked on real calibration images before results
  are packaged, and golden fixtures are retained for the later Core ML/Swift gate.

The required order is `./run.sh best-smoke`, confirm a fingerprint-matched PASS, and only then
`./run.sh best`.

## What v3 still cannot prove by itself

A successful workstation run does not prove Core ML conversion parity or physical-iPhone behavior.
The returned frozen artifacts must be converted and tested on the Mac and target iPhone before any
model is called deployed. Tiled inference likewise needs a matching Swift implementation before it
can become an app option.

Synthetic phone augmentation also cannot establish phone generality. A manually annotated,
preregistered set of real phone-through-microscope images remains necessary as a separate final
external test. Do not use that set for checkpoint choice, threshold calibration, ensemble selection,
or redesign after inspecting results.

For a paper, report the immutable split and experiment hashes, all optimizer and augmentation
settings, per-domain foreground and internal-boundary metrics, instance AP50/AP75/AP90, count error,
runtime, model size, calibration search, ensemble membership, export parity, and domain-specific
limitations. Preserve raw logs and history CSV files; do not collapse these measurements into one
generic “accuracy” number.
