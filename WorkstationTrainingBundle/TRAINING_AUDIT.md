# Cellect v4 training audit

This record distinguishes historical runs, executable v4 controls, and claims that still require
external evaluation. The exact commands are in `README.md`.

## Historical results are not v4 results

The previously returned LIVECell-only workstation ZIP reported values near 0.90 that were primarily
foreground Dice, not a fixed accuracy ceiling. It did not establish transmitted-light domain,
touching-cell, microscope, phone, tracking, Core ML, or physical-device generality. Its validation
metrics and postprocessing choices must not be relabelled as v4 evidence.

V4 may initialize exact compatible tensors from selected v3 semantic checkpoints. That transfer is
recorded by source SHA-256, loaded keys, tensor coverage, and phase-specific freezing. It does not
permit importing the earlier metrics or treating a v3 checkpoint as a v4 result.

The acquisition split salts are intentionally retained as
`cellect-acquisition-splits-v3-frozen-for-v4`. This preserves paired role membership for a fair
comparison. References to those frozen salts and to explicitly imported v3 backbone tensors are
intentional; stale v3 output filenames are not.

## V4 controls implemented before the long run

- `./run.sh best-smoke` performs the full-data preflight and a bounded real optimization/export
  path before `./run.sh best` is authorized.
- The PASS marker, output directories, checkpoints, and run contracts are fingerprinted by code,
  dependencies, input hashes, split manifest, configuration, and model sources. Mismatched or
  incomplete resumes fail closed.
- Cellpose foundation masks are converted to verified flow files one at a time and all foundation
  runs use file streaming. A RAM estimator rejects a single sample that would consume more than
  half of host memory. Stateful checkpoints preserve model/AdamW/LR/history and all RNG states;
  the bundled interruption test requires exact continuation.
- DeepSea binary foreground is combined strictly with its contact map to reconstruct instances;
  binary pixel values are never treated as instance IDs.
- DeepSea segmentation is raw-image-anchored: each admitted image requires its mask and contact
  map, while image-less publisher companions are excluded and sealed by relative path, size,
  SHA-256, and reason in the full-data preflight and dataset fingerprint.
- DeepSea tracking is anchored to published raw images: every admitted image requires its mask and
  label, while publisher-supplied companions beyond the last image are excluded and sealed by
  relative path, SHA-256, and reason in both the sequence fingerprint and preflight report.
- Entire acquisitions, time series, z/c variants, and crops are kept in one of `train`,
  `checkpoint`, `calibration`, or `ensemble_selection`.
- Final-test labels are existence/size/SHA-256 sealed but not decoded, counted, scored, or used by
  `best-smoke`, `best`, shape boundary search, or tracking development metrics.
- Dataset metrics are pooled within domain and macro-averaged equally across domains so larger
  sources cannot dominate silently.
- Every shape student independently supervises foreground, learned fused contact, image-context
  contact, center-flow contact, whole-shape contact, signed distance, center flow, and four local
  same-instance affinities.
- Compatible v3 backbones require at least 99% exact tensor coverage and are fine-tuned in
  auditable heads/decoder/full phases. New heads are
  zero-initialized where required to preserve the initial legacy logits, and the first gradient
  step is audited for frozen/trainable parameter violations.
- Cellpose teacher identity, revision, weights SHA-256, raw-output semantics, inference settings,
  and cache provenance are content-addressed. Teacher generation is serial; the GPU model is
  released before DataLoader workers. In-sample full-train teacher predictions are rejected.
- The Ceb-inspired teacher transfers no claimed Ceb checkpoint. Its candidates and input maps come
  from a hash-audited frozen v3 predictor (plus counterfactual cuts of those predictions); ground
  truth is consulted only after proposal construction to label keep/merge utility. It learns new
  graph weights on `train`, is selected on `checkpoint`, and is distilled into the deployable
  student. A contract-test-only fixture path that uses synthetic truth proposals is explicitly
  marked non-scientific and is forbidden in best-smoke/best.
- Frozen five-channel outputs and instance truth are disk-cached only for `calibration` and
  `ensemble_selection`. Search constructs configurations on calibration, freezes their exact IDs,
  then uses ensemble selection only for ranking.
- Legacy two-channel and extended five-channel TorchScript exports are shape/parity checked; ONNX
  is emitted where supported. Heavy export is moved to CPU to avoid holding duplicate GPU models.
- Every returned segmentation and CellectTrack candidate is converted on Linux under the exact
  PyTorch 2.7.1 environment with coremltools 9.0. Each package is bound to numeric golden inputs
  and exact PyTorch outputs by source, package-tree, and fixture hashes. Conversion-only mode never
  claims native Core ML parity, and a single failed/missing candidate fails `best-smoke`/`best`.
  Float16 candidates also carry a Float32 fallback. Native validation gates raw tensors,
  probabilities, and frozen-threshold foreground/boundary/link/event decisions rather than relying
  on an average logit error that could conceal deployment-changing threshold flips.

## Exact boundary semantics and calibration audit

V4 has one foreground head and three interpretable division heads:

- context: local/global image appearance suggesting two adjacent cells;
- flow: discontinuity in center-directed Cellpose-style y/x geometry;
- shape: signed-distance and same-instance-affinity evidence for distinct cell shapes.

A fourth boundary channel is a learned per-pixel fusion of those three plus a residual. The four
head-calibration thresholds are foreground, context, flow, and shape. The post-fusion boundary
cutoff is a separate reconstruction operating point. Lower cutoffs usually remove more contact
pixels and create more splits; higher cutoffs usually produce fewer splits and more merges.

Calibration compares learned fusion, each branch, any/majority/all voting, and normalized weighted
branch combinations. Its bounded three-stage search independently shortlists branch thresholds,
screens every fusion family/weight over their Cartesian product, and tunes foreground/minimum area
for family finalists on at most 24 deterministic records per role/domain. It retains the full
boundary-cutoff curve at every stage. Its reported normalized boundary-cutoff AUC is trapezoidal area under *labelled
calibration metrics* across cutoffs. It measures operating-point robustness; it cannot be computed
as accuracy for an unlabeled phone image. The app's unlabeled adaptive stability sweep is a
different heuristic and must not be described as validation AUC.

The frozen boundary-search objective is 0.25 foreground Dice + 0.15 reconstructed contact-boundary
Dice + 0.35 instance AP50 + 0.25 instance AP75. No background-dominated pixel accuracy is used as
the sole selection target.

## Knowledge-transfer claims that are and are not supported

Supported if the corresponding manifests are present:

- compatible v3 semantic tensor initialization;
- soft distillation of Cellpose foreground probability and raw center flow;
- Ceb-inspired, independently trained keep/merge boundary supervision;
- retained supervised context, flow, shape, and learned-fusion behavior in a new student graph.

Not supported:

- literal transplantation of a Cellpose-SAM decoder/head into Cellect;
- transfer of an official Ceb boundary checkpoint (none is published in the referenced repository);
- a claim that teacher knowledge was retained merely because code paths exist;
- superiority to Cellpose, Ceb, or another model without a locked external comparison.

For publication, report teacher/base checkpoint hashes, distillation ablations, phase schedules,
all loss weights, per-domain results, variance across preregistered seeds, model/runtime/memory size,
and confidence intervals. Compare v3 initialization alone, each teacher alone, all teachers, and
the final learned fusion under the same frozen data roles.

## CellectTrack v4 controls

The tracking code defines a new tracking-by-detection transformer with fixed cell tokens,
spatial/temporal relative attention, a relative-motion association scorer, and division/birth/death/
uncertainty heads. It consumes strict DeepSea-train and CTC training gold tracking annotations,
the CC BY 4.0 LiveCellTrack public preview's human MOT identity boxes, CTMC-v1 official TRAIN, and
ALFI Task-1 MI01–MI08 DTLTruth boxes/parents. Official CTC/CTMC test/result and DeepSea test paths
are not searched; LiveCellTrack `annotations/train.json`, ALFI Task 2, and ALFI semantic masks as
whole-cell instances are ignored.
LiveCellTrack boxes provide identity centers and explicitly labelled elliptical size/shape proxies;
they are not dense manual masks. Since its MOT rows contain no parent field, the adapter creates no
parent links or division targets from that source. Feature normalization is fit on `train`, and
acquisitions retain the same four scientific roles.
Duplicate same-frame occurrences are annotation ambiguity. LiveCellTrack's affected identity is
quarantined under its source contract; ALFI's 9 duplicate keys/18 affected rows are quarantined
event-by-event while 16,627 rows, 16,618 unique frame/ID keys, 16,564 published annotations, 796
frames, and 331 tracks remain separately recorded. CTMC-v1 is accepted only at 47 acquisitions,
80,389 frames, 1,616 tracks, and 1,097,223 boxes; runs of one cell-line prefix cannot cross roles.
Its publisher supplies no checksum, so a stable immutable first-byte SHA marker is required. CTMC
licensing is unknown/research-permission/no-redistribution. ALFI's Figshare-CC0/README-CC-BY
conflict is treated operationally as CC BY with attribution.

Transparent development metrics include detection, association, selected temporal links, exact
division events, parent/child edges, IDF1/switches/fragmentation, count consistency, Brier score,
and ECE with per-domain and macro-domain summaries. They are not official CTC DET/SEG/TRA measures.
An official score requires a separately versioned official evaluator and saved provenance.

The heavy and mobile models each train heads/adapters/full stages; mobile additionally distills the
frozen heavy model. Every stage has a code/data/model/teacher/phase fingerprint, SHA-256 checkpoint
sidecar, real-optimizer-update assertion, trainable-gradient audit, and checkpoint-role-only early
stopping. Fixed-step sampling is balanced modality → dataset → acquisition → window; checkpoint
loss is macro-domain/acquisition aggregated; exact resume restores Python/NumPy/Torch/CUDA,
GradScaler, optimizer/scheduler, and sampler state under a policy fingerprint.

Train-only robustness uses actual image/mask-derived one-pixel erosion/dilation, boundary/centroid
jitter, merge and split proposals, background false positives, and missed detections. Synthetic
merge/split/false-positive identities are context-only and every dependent association/event/
teacher label is masked. Clean and deterministic-perturbed checkpoint losses are reported
separately. This is a controlled fallback, not a claim that ground-truth tokens equal Cellect
output. A role-safe, frozen, out-of-fold Cellect-proposal cache is preferred and has a manifest
contract, but cannot be claimed until segmentation folds actually produce it.

No admitted source explicitly labels positive biological birth/death. Endpoints are censored,
those gates are disabled at threshold 1.0, and negative-only heads cannot block a link. Calibration
freezes association/division thresholds and compares greedy with deterministic per-frame Hungarian
decoding. The latter globally maximizes threshold-relative log-odds with explicit unmatched
assignments/division slots, then applies gap-2 only to still-unmatched endpoints.
`ensemble_selection` ranks heavy, mobile, and their probability mean by a macro-domain combination
of temporal-link F1, exact-division F1, division-edge F1, IDF1, and mostly-tracked fraction. It does
not retune thresholds. Each candidate keeps its own thresholds/decoder. Both neural models export
fixed-shape state, TorchScript, and ONNX with hashed numerical parity manifests; the mean ensemble
exports an explicit two-model probability-fusion/decode-once runtime manifest.

Trackastra 0.3.0 `general_2d` is workstation-only. Its archive is pinned and verified; its
association output supervises adjacent-only 64-frame training chunks with one-frame overlap, while
the mobile student receives internal
heavy-model distillation. Neither Trackastra code nor its runtime is part of the exported Cellect
model. The Ker data is not represented as direct CellectTrack ground truth.

The outer `best-smoke` now runs one bounded real optimizer update in every tracking phase, freezes
threshold/model selection, and exercises exports before writing its fingerprinted marker. `best`
requires the matching tracking marker. These controls establish experiment integrity, not tracking
performance; this audit claims no final-test or official CTC result.

## What a successful workstation run still cannot prove

A PASS establishes Linux conversion and the portable golden contract, but cannot establish native
Core ML numerical parity, Swift preprocessing/postprocessing parity, on-device memory/runtime, or
physical-iPhone behavior. Every preconverted package must pass the PyTorch-free Intel-Mac golden
fixture check, followed by Swift fusion/FIFO reconstruction and physical target-device gates.
Native tiled inference likewise remains research-only until Swift implements the identical path.

Synthetic phone augmentation cannot establish real phone generality. A preregistered,
acquisition-separated, manually annotated phone-through-microscope test set is still required. It
must not be used for checkpoint choice, threshold calibration, ensemble selection, or redesign.

For a paper, preserve the immutable code/data/stage/split hashes, JSONL histories, checkpoint and
export hashes, optimizer and augmentation settings, calibration curves, per-domain foreground,
contact, instance, count, and tracking metrics, boundary-fusion ablations, runtime and model size,
and all domain-specific limitations. Never collapse them into an undefined generic “accuracy.”
