# Cellect workstation training bundle v4

This bundle trains and audits Cellect's transmitted-light cell segmentation models. The v4 path
retains compatible v3 foreground/contact backbones, adds independent context-, flow-, and
shape-aware boundary mechanisms, learns a per-pixel fusion, searches deployable reconstruction
settings, and exports legacy two-channel and extended five-channel models. The repository also
trains the CellectTrack v4 tracking-by-detection heavy/mobile models, and packages their strict
transmitted-light data, selection, export, and metric contracts.

All reported values produced by the normal workflow are development values. The official final
test remains sealed and is not evaluated by `best-smoke` or `best`.

## Exact workstation workflow

Use Linux, Python 3.11 or 3.12, a recent NVIDIA driver, an RTX 3090-class CUDA GPU, and at least
300 GB of free disk. Several days may be required for the full run. From the transferred bundle:

```bash
cd /path/to/WorkstationTrainingBundle
export CELLECT_DEEPSEA_ROOT=/mnt/firecuda2/deepsea
chmod +x run.sh
./run.sh best-smoke
```

`CELLECT_DEEPSEA_ROOT` must contain the complete `track`, `segment`, and `final` collections. The
aliases `tracking`, `segmentation`, `tracking_dataset`, `segmentation_dataset`, and `final_dataset`
are accepted, including the split-Google-Drive layout
`track/tracking_dataset`, `segment/segmentation_dataset`, and `final/final_dataset`. The script
uses `CELLECT_DEEPSEA_ROOT` authoritatively: an invalid configured path fails clearly instead of
silently selecting a different nearby `deepsea` directory. It then creates `.venv`, installs the
pinned requirements, runs the bundle
self-check, performs the full segmentation and tracking preflight, downloads and verifies the
pinned workstation teachers, public LiveCellTrack preview, CTMC-v1 TRAIN, and pinned ALFI archive,
and then executes a bounded but
genuine pass through segmentation, tracking, validation-role separation, boundary search, export,
and parity checks. It also converts every returned segmentation and CellectTrack candidate with
`coremltools==9.0` beside PyTorch 2.7.1. This conversion-only Linux step never calls
`MLModel.predict`; any missing or failed candidate prevents the smoke PASS marker. The
public segmentation downloader retries transient DNS, timeout, and connection failures with
bounded exponential backoff; completed archives are reused and `.partial` bytes remain resumable.
LiveCellTrack archive resumes through a `.partial` file and is accepted only
at exactly 173,935,248 bytes with SHA-256
`c3c824e3cb9db0673d84245ffcc9a5a85f9b1a8aafcf81d529195105a7cf3d7f`.
Large local sources may instead be supplied as extracted directories or ZIPs with
`CELLECT_CTMC_ROOT` and `CELLECT_ALFI_ROOT`. CTMC-v1 has no publisher checksum, so its first
successful bytes/SHA are immutably recorded; ALFI is fixed to 8,423,073,056 bytes and MD5
`fe3326323c10b1748302e962eae26150`.

When a tracking publisher is unreachable, a source may be omitted deliberately with
`CELLECT_OMIT_TRACKING_SOURCES`, a comma-separated list drawn from `ctmc_v1`, `alfi_task1`, and
`livecelltrack_preview`; any other name is rejected, because DeepSea and the five CTC training sets
carry the dense masks and lineage CellectTrack is built on. An omission is recorded in
`tracking_preflight_v4.json` under `omitted_sources` and folded into the tracking dataset
fingerprint, so a reduced run has a different experiment identity, its marker cannot authorize a
full-contract `best` run, and its metrics must not be compared with full-contract numbers.

The smoke command succeeds only after writing:

```text
output/best_smoke_pass_v4.json
output/cellect_track_v4/tracking_best_smoke_pass_v4.json
```

Confirm that the marker says `"status": "PASS"`, then run:

```bash
./run.sh best
```

`best` refuses to start unless the marker's bundle version, dataset fingerprint, and stage
fingerprint match the current code and data. Editing a fingerprinted Python file, `run.sh`,
`requirements.txt`, or an input invalidates the marker; rerun `./run.sh best-smoke`. Smoke and full
checkpoints live in separate fingerprinted directories and cannot be mixed.

For diagnosis only:

```bash
./run.sh preflight
```

This does not create the smoke PASS marker. The older `smoke` and `full` modes remain lightweight
LIVECell development paths; they are not substitutes for `best-smoke` followed by `best`. CUDA is
required and the pipeline exits instead of silently falling back to CPU.

## Preflight, split identity, and sealed test

The v4 preflight decodes every development image/mask pair, validates dimensions, finite values,
annotations, DeepSea triplets, physical paths, and content hashes, and rejects acquisition or byte
identity leakage. It checks final-test image transport but only size/SHA-256-seals final-test labels;
it does not decode them or derive their dimensions, object counts, or metrics.

Two upstream realities are handled without weakening those gates. LIVECell repeats 96 image
listings — 66 inside a single JSON and 30 across the train and validation files — and train and
validation are one acquisition-grouped development pool, so each such image enters the manifest,
the role tallies, and the fingerprint exactly once; the repeated listings are named in
`combined_train_val.upstream_duplicate_listings`.
Some published archives also contain frames whose masks annotate no cell at all: those frames are
intact but carry no supervision, so they are withheld from the manifest and from training and
recorded with their hashes in `withheld_development_samples`. Any overlap with the sealed test
split, by physical path or by content, still fails the run.

Tracking preflight separately validates every DeepSea-train, CTC-training, LiveCellTrack
public-preview, CTMC-v1 TRAIN, and ALFI Task-1 acquisition, frame, identity, lifetime, ambiguity,
and available parent link before either
smoke or full optimization. Its fingerprint is combined with the segmentation fingerprint, so
changing either invalidates the outer smoke gate.

The principal records are:

```text
output/preflight_v4.json
output/splits_v4.jsonl
output/tracking_preflight_v4.json
output/tracking_splits_v4.jsonl
```

The four acquisition-separated development roles are:

1. `train`: parameter fitting and train-only feature normalization.
2. `checkpoint`: scheduler, early stopping, and weight selection.
3. `calibration`: thresholds, filters, fusion candidates, and operating-point selection.
4. `ensemble_selection`: ranking exact configurations frozen on calibration.

Metrics are averaged within microscopy domains and then macro-averaged equally across domains.
The official final-test split is neither subdivided nor read by the model-selection code.

`scientific_splits.py` deliberately retains the v3 hash salts and records
`cellect-acquisition-splits-v3-frozen-for-v4`. This is not a stale filename: keeping each
acquisition in its prior role permits paired v3/v4 comparisons without a favorable reshuffle.

## DeepSea masks and boundaries

DeepSea segmentation `masks` are binary foreground, not instance identifiers. Their matching
`unetwmaps` encode touching-cell contacts. The adapter removes valid contact pixels to seed
separate interiors, reconstructs instances, restores contiguous foreground, and keeps only contact
evidence consistent with adjacent cells. Every raw image must have a mask and contact map or the
run fails closed. Publisher-supplied masks/contact maps beyond the last available image cannot form
supervised samples; they are excluded with relative paths, sizes, SHA-256 hashes, and reasons in
the full-data preflight and dataset fingerprint.

`segment` is the canonical semantic source. Overlapping `track` and `final` frames are recognized
for provenance rather than duplicated. The official segmentation test remains sealed. Variants
from one source acquisition remain together when development roles are assigned.

A locally assembled DeepSea collection is normalized before anything is admitted. Files carrying a
browser/Drive copy suffix such as `frame(1).png` that are byte-identical to the canonical file are
duplicate downloads, not second acquisitions, and are excluded so one frame cannot be sampled
twice or land on both sides of the deterministic train/validation hash. DeepSea also distributes
several z/c variants of acquisitions `A11`–`A19` across its own train and test folders; the
official test folder is kept intact so the published benchmark stays comparable, and the
training-side copies of those acquisitions are withheld instead. Both kinds of exclusion carry
relative paths, sizes, SHA-256 hashes, and reasons in the preflight provenance, and the withheld
frames are added back when the collection is checked against the known complete 3,624/3,686-triplet
layouts. A touching-edge map that borders only one reconstructed cell is annotation noise rather
than corruption: that frame is trained without an explicit contact target and is listed in
`deepsea_touching_edge_without_contact`.

All Cellect boundary heads target an *internal division between cells*, not the ordinary outer
cell/background contour.

## Baseline semantic models and retained knowledge

Before v4 shape training, `best` trains the compatible semantic candidates from mobile through
heavy backbones: MobileNetV3-Small U-Net, MobileNetV3-Large DeepLabV3+, EfficientNet-B0 U-Net,
ResNet18 U-Net, EfficientNet-B3 U-Net++, ResNet50 DeepLabV3+, ResNet101 U-Net++, SegFormer-B2, and
SegFormer-B5. Their foreground/contact checkpoints provide auditable v3-compatible initialization
for the two v4 students:

| v4 student | retained backbone | input | new refinement |
|---|---|---:|---|
| `shape_mobile_mobilenetv3_v4` | MobileNetV3-Small U-Net | 512 | 48 channels, 4 depthwise shape blocks |
| `shape_context_segformer_b5_v4` | SegFormer-B5 | 768 | 128 channels, 6 shape blocks, 12×12 pooled transformer with 2 layers/8 heads |

The three Cellpose foundation fine-tunes (`cpsam_v2`, `cpdino`, and `cpdino-vitb`) do not load the
combined corpus into RAM. Instance masks are converted serially into content-verified flow TIFFs
in one shared fingerprinted cache, and training/validation read one image/flow pair at a time.
`ram_preflight.json` records a conservative per-sample host-memory estimate. A state checkpoint
every five epochs contains the model, AdamW moments, learning-rate position, complete loss arrays,
and Python/NumPy/Torch/CUDA RNG states; interruption deterministically replays no more than four
epochs. Training draws use inverse-square-root dataset-domain frequency (the capped yeast sources
share one domain), so small transmitted-light sources are not erased by corpus size.
Twenty-five-epoch weight snapshots remain available for checkpoint-role AP selection.

Only exact shape-compatible backbone tensors are loaded, production requires at least 99% of the
semantic backbone's parameter/buffer elements, and coverage plus SHA-256 are recorded,
and the new residual heads start at zero so the initial legacy output preserves the imported
semantic logits. Retained-v3 transfer is therefore exact initialization followed by an audited
freeze/unfreeze fine-tuning schedule; it is not described as distillation.

The workstation-only teachers are:

- **Cellpose-SAM `cpsam_v2`:** supplies raw foreground logits and center-directed y/x flows.
  Foundation weights, revision, byte size, SHA-256, inference settings, and cache provenance are
  pinned. Teacher outputs are generated serially, cached losslessly, then the model is released
  before multi-worker training. A fine-tuned teacher may supervise train only through an explicitly
  out-of-fold checkpoint; full-train teacher predictions on their own fitting samples are rejected.
- **Ceb-inspired boundary graph teacher:** Ceb publishes a useful keep/merge mechanism but no
  official pretrained boundary-classifier checkpoint. Cellect therefore transfers no Ceb weights.
  Its independent 128-pixel candidate teacher uses image, frozen-v3 semantic probability,
  Cellpose flow, candidate-line, two-region shape/distance, contour, uncertainty, and geometry
  features; graph attention reasons jointly over boundaries sharing cells. Candidate regions are
  reconstructed from the frozen v3 prediction, with deterministic counterfactual cuts of those
  predictions to expose merge errors. Ground-truth instances enter only afterward to score the
  keep-versus-merge utility; no ground-truth mask, contact map, or centroid flow is rendered into
  the teacher input. It is fit on `train`, selected on `checkpoint`, and its proposal-line
  corrections are distilled into the shape branch.

The deployable student does not contain Cellpose, Ceb, Python, or either teacher graph.

## Five-channel output and the four head thresholds

The extended v4 output order is fixed:

1. foreground logit;
2. learned fused-boundary logit;
3. context-boundary logit;
4. flow-boundary logit;
5. shape-boundary logit.

The three independent boundary meanings are exact:

- **Context:** image/backbone appearance evidence that adjacent foreground belongs to different
  cells.
- **Flow:** discontinuity in learned center-directed, Cellpose-style y/x flow geometry.
- **Shape:** signed-distance and four local same-instance affinities (right, down, down-right, and
  down-left) indicating distinct whole-cell shapes.

The four *head calibration thresholds* are foreground, context, flow, and shape. Foreground decides
which pixels are candidate cell material. Before any manual branch fusion, each context, flow, or
shape probability is monotonically recentered so that its own selected raw threshold maps to 0.5:
values below the threshold map linearly into 0–0.5 and values above it map linearly into 0.5–1. This
preserves confidence margins instead of reducing the branch to a binary vote. It does not alter the
learned-fusion channel. Lower branch thresholds admit weaker split evidence, while higher values
require stronger evidence.

The learned fused channel is a separately supervised per-pixel softmax-weighted combination of
context, flow, and shape plus a learned residual. The boundary search compares it with each single
branch, continuous any/majority/all votes, and normalized non-negative weighted combinations. A
legacy two-channel export contains foreground plus this learned fused boundary; the extended export
contains all five channels.

The iOS resolver uses the same continuous semantics: single branch selects one recentered map,
any/majority/all take its pixelwise maximum/median/minimum, and weighted fusion takes a normalized
non-negative context/flow/shape mean. It then applies the separate post-fusion boundary cutoff.
This keeps workstation calibration, model-comparison sweeps, and on-device manual controls in
deployment parity. Older two-channel exports safely fall back to their learned fused boundary.

After fusion, the distinct **boundary cutoff** determines which fused boundary pixels are removed
before interior markers expand into instances. Lower cutoffs normally create more divisions;
higher cutoffs normally retain more interior and merge more cells. Minimum-area filtering is also
selected on calibration. Do not confuse the post-fusion boundary cutoff with the three branch
calibration thresholds.

## v4 shape phases, optimizer, and losses

The full shape-student schedule is:

| Phase | Trainable tensors | Epoch ceiling | AdamW learning rate |
|---|---|---:|---:|
| `heads` | new refinement and geometry/fusion heads | 15 | `1e-3` |
| `decoder` | above plus retained decoder/segmentation head | 25 | `3e-4` |
| `full` | complete network | 160 | `5e-5` |

AdamW weight decay is `1e-4`; there is no explicit L1 penalty. Gradients are clipped to L2 norm
1.0. Mobile uses batch 4 with accumulation 4; heavy uses batch 1 with accumulation 8. Full-mode
early stopping is phase-local, cannot begin before 40 full-phase epochs, has patience 30, and uses
minimum improvement `1e-4`. The heads/decoder phases cannot consume full-phase patience. The
ReduceLROnPlateau floor is `1e-7`. The random seed is 1701. The Ceb-inspired teacher has a 40-epoch ceiling, patience 8,
AdamW learning rate `3e-4`, weight decay `1e-4`, and at most 16 candidates per image.

The multitask loss is the sum of these recorded components:

| Component | Form | Weight |
|---|---|---:|
| foreground | BCE + soft Dice | `1.0 + 1.0` |
| fused contact boundary | positive-weighted BCE + soft Dice, positive weight 5 | `1.0` |
| context / flow / shape contact branches | same boundary loss | `0.35` each |
| signed distance | Smooth L1 | `0.45` |
| center-flow regression / direction | Smooth L1 / cosine | `0.55` / `0.25` |
| four same-instance affinities | BCE | `0.40` |
| predictive uncertainty / retained base semantic | heteroscedastic term / BCE | `0.05` / `0.10` |
| Cellpose foreground / raw-flow distillation | soft BCE / Smooth L1+cosine | `0.20` / `0.40` |
| context / flow / shape / fused teacher terms | soft BCE when supplied | `0.20` / `0.25` / `0.35` / `0.20` |

All foreground, internal-contact, signed-distance, centroid-flow, affinity, and validity targets are
generated after the same deterministic geometric augmentation. The v4 student adds deterministic
image-only gamma, exposure, and sensor-noise variation; its retained semantic backbones were also
fit with the broader phone-through-eyepiece augmentation documented in their fingerprinted v4
baseline histories.

## Boundary calibration and AUC semantics

Checkpoint selection uses macro-domain foreground and four boundary-branch overlap components.
Once the best weights are frozen, five-channel probabilities and resized instance truth are written
to content-verified disk caches for `calibration` and `ensemble_selection` only.

Calibration searches foreground threshold, three branch thresholds, boundary cutoff, minimum area,
and fusion family with a bounded three-stage design: each head's threshold is screened first, the
best two per head enter every fusion family and configured weight, then foreground/minimum-area
settings are tuned for each family finalist. All stages retain the complete nine-point boundary
cutoff curve. Up to 24 deterministic SHA-256-selected records per role/domain are used so large
datasets cannot turn calibration into an unbounded watershed job; domains are still macro-weighted
equally. Every frozen candidate is reconstructed by the versioned iPhone-compatible
3×3/8-connected/FIFO postprocessor. The boundary-search score is:

```text
0.25 foreground Dice + 0.15 reconstructed contact-boundary Dice
+ 0.35 instance AP50 + 0.25 instance AP75
```

For each otherwise fixed configuration, the report integrates labelled validation metrics over the
boundary-cutoff curve using normalized trapezoidal area. This **boundary-cutoff AUC measures
calibration robustness**. It requires ground-truth masks and is not per-image confidence or an
accuracy value available from an unlabeled phone image. Configuration construction stops after
`calibration`; `ensemble_selection` ranks only exact frozen candidate IDs.

## CellectTrack v4 architecture and evaluation contract

CellectTrack is an independent tracking-by-detection transformer, not copied Trackastra source.
Each segmented cell becomes a fixed padded token containing time/y/x plus 24 intensity, gradient,
texture, area/perimeter, shape, signed-distance, flow, density, neighborhood, and contact features.
Feature normalization is fit on `train` only. Windows contain three overlapping frames and preserve
whole-acquisition 70/10/10/10 roles.

Alternating same-frame spatial and cross-frame temporal attention uses learned relative time,
displacement, and distance bias. A relative-motion pair scorer predicts directed links; node heads
predict division, birth, death, and uncertainty. The heavy model uses dimension 192, 8 heads,
6 layers, pair width 256, audited one/two-frame links only, and dropout 0.1. The mobile/Core-ML
candidate uses dimension 64, 4 heads, 2 layers, pair width 96, the same two-frame maximum, and no dropout. Its fixed
deployment adapter returns association logits/mask plus division, birth, death, and uncertainty.

Strict tracking inputs are DeepSea `tracking_dataset/train` and the *training* gold tracking
annotations for CTC BF-C2DL-HSC, BF-C2DL-MuSC, DIC-C2DH-HeLa, PhC-C2DH-U373, and PhC-C2DL-PSC.
DeepSea sequences are anchored to their raw image frames. Every image must have its official mask
and label companion; publisher-supplied mask/label files beyond the last available image are not
trainable frames and are excluded with relative paths, SHA-256 hashes, and reasons recorded in the
sequence fingerprint and preflight report.
They also include the CC BY 4.0 LiveCellTrack preview (DOI `10.17632/cgwcpz34mr.1`): ten
scratch-wound acquisitions of 100 frames and ten HeLa acquisitions of 50 frames. Only each
acquisition's human MOT `gt/gt.txt` is read. Box centers supervise identity/motion, and deterministic
inscribed-ellipse proxies provide coarse size/shape features; these proxies are not described as
manual segmentation. The preview has no explicit parent lineage, so it contributes continuation,
but **no invented division or positive birth/death targets**. Its `annotations/train.json` is
not used. CTC `*_ST`, `*_RES`, and official test labels and DeepSea test are not searched. The
optional Ker phase-contrast data is not used as direct CellectTrack supervision; it may be inherited
only through the Trackastra teacher.

Two additional audited sources are accepted. CTMC-v1 reads only official TRAIN
`seqinfo.ini`/`img1`/10-column MOT `gt.txt`/4-column TRA lineage and refuses training unless the
current 47-acquisition, 80,389-frame, 1,616-track, 1,097,223-box totals match. Because no checksum
is published, its first successful archive bytes/SHA are immutably pinned; its license is unknown,
research permission is required, and raw redistribution is prohibited. ALFI reads Task-1 MI01–MI08
images and DTLTruth boxes/parents only. It records the published 16,564 annotation count separately
from the current archive's 16,627 rows and 16,618 unique frame/ID keys, quarantines duplicate events,
and treats the Figshare-CC0/README-CC-BY conflict operationally as CC BY with attribution. ALFI Task
2 and its heterogeneous semantic masks never become whole-cell instance truth.

If one source ID has multiple human boxes in the same frame, that entire source identity is
quarantined and recorded in the split/preflight artifacts; Cellect never guesses which box or
association was intended. This handles the pinned preview's small annotation ambiguity without
silently fabricating track labels.

Both models run heads 6 epochs at `3e-4`, adapters 12 at `1e-4`, and full 120 at `3e-5`; mobile then
runs a 40-epoch heavy-model distillation phase at `2e-5`. AdamW weight decay is `1e-4`, clipping is
1.0, batch size is 2, checkpoint patience is 18, and seed is 20260731. Each epoch has a recorded,
fixed optimizer/checkpoint step count and a deterministic modality → dataset → acquisition →
window sampler. Checkpoint loss is macro-averaged by acquisition and domain. Resume checkpoints
restore Python, NumPy, Torch CPU/CUDA, GradScaler, optimizer, scheduler, and sampler state under a
policy fingerprint. Supervision combines
positive-weighted link BCE with masked Sinkhorn assignment, division classification/link-count
consistency, censored/non-birth and non-death BCE, and uncertainty BCE. No current source explicitly
labels positive biological birth/death, so first/last observations are censored and both decoder
gates remain disabled at threshold 1.0. Mobile distillation transfers association,
event, and uncertainty outputs from the frozen heavy model; optional Trackastra association targets
have weight 0.35.

Training mixes clean tokens with deployment-like, image-derived detector errors: actual one-pixel
mask erosion/dilation, centroid/boundary jitter, merged detections, split detections, background
false positives, and missed detections. Merge/split/false-positive identities are never guessed;
they influence transformer context while all dependent link/event/teacher labels are masked.
Checkpoint diagnostics separately report clean versus deterministically perturbed macro-domain/
acquisition loss. A role-safe out-of-fold Cellect proposal cache remains the preferred next step;
the manifest contract is frozen, but no such cache is claimed before segmentation folds produce it.

Calibration separately chooses association/division thresholds and compares greedy versus global
Hungarian decoding; birth/death remain disabled. The global decoder maximizes threshold-relative
association log-odds per adjacent frame pair with explicit unmatched assignments and division
slots, then applies gap-2 recovery only to still-unmatched endpoints. With candidate-specific
thresholds and decoder frozen, `ensemble_selection` compares heavy, mobile, and their arithmetic probability
mean using a macro-domain diagnostic weighted 0.35 temporal-link F1, 0.20 exact-division F1,
0.10 division-edge F1, 0.25 IDF1, and 0.10 mostly-tracked fraction.
Division metrics omit identity-only acquisitions whose annotation format has no parent lineage;
missing parents are never scored as negative divisions.

Development reports must include detection, association-pair, temporal-link, division-event and
parent/child PRF; ID precision/recall/IDF1, switches, fragmentation, mostly tracked/lost; count
consistency; link Brier score and ECE; per-domain values and equal-domain macro means. These are
transparent Cellect diagnostics, not official CTC DET/SEG/TRA claims. Official scores may be
reported only through a separately versioned official-evaluator adapter and artifact record.

The outer `best-smoke` invokes this complete tracking path with one real optimizer epoch/update in
every phase, verifies TorchScript/ONNX parity, and writes the fingerprinted tracking smoke marker.
`best` then requires the same teacher/data/model/feature fingerprint before long training. A PASS
marker proves execution and contract integrity, not tracking accuracy or superiority.

## Teacher provenance and distribution status

- Cellpose code and declared model weights are BSD-3-Clause, but Cellpose states its models were
  trained on CC-BY-NC data. `cpsam_v2` also derives from Apache-2.0 Segment Anything; CPDINO uses
  the separate DINOv3 license. Teacher-derived outputs/weights remain
  `research_only_pending_legal_review` for distribution.
- Ceb code is MIT. Cellect reimplements the published idea and trains new weights; no official Ceb
  classifier checkpoint exists or is represented as transferred.
- Trackastra is pinned to package/model v0.3.0 `general_2d`, SHA-256
  `35cefd8634860d1dd43bcdcbdef7ae0caa24445f19bcfd35ec6b19039f2cd876`, and BSD-3-Clause. It is a
  workstation-only teacher, loaded lazily from a verified artifact, and is absent from the mobile
  CellectTrack graph. The public tracking API permits no teacher, but the standard `run.sh`
  accuracy workflow downloads and verifies this pinned teacher before training.

Dataset restrictions still apply independently. LIVECell and Revvity-25 are non-commercial, so
the research build is not commercially cleared. Preserve notices and obtain legal review before
shipping teacher-derived or dataset-derived weights.

## Deployment parity and returned artifacts

Each v4 shape student produces hashed run contracts, JSONL history, last/best checkpoints,
calibration and ensemble-selection reports, the five-channel boundary-search report/caches, and:

```text
*_legacy2.torchscript.pt
*_extended5.torchscript.pt
*.onnx                         # where the installed exporter supports the graph
```

CellectTrack adds `tracking_preflight_v4.json`, train-only feature normalization, hash-sidecar
stage checkpoints, clean/perturbed detector-error ablation, calibration/selection reports,
`deployment_lock_tracking_v4.json`, heavy and mobile state/TorchScript/ONNX artifacts, and
candidate-specific fixed-input parity manifests. A third `mean_heavy_mobile/manifest.json`
explicitly defines the two-artifact probability-fusion runtime, its own thresholds/decoder, and
decode-once rule rather than pretending the ensemble is a standalone neural-network file. The tracker input is
`[1,128,3]` time/y/x, `[1,128,24]` ordered features, and `[1,128]` padding mask; outputs are
association logits/mask plus division, birth, death, and uncertainty probabilities.

The outer workflow writes `deployment_lock_v4.json`, workstation PyTorch/TorchScript/ONNX parity
reports, and a self-contained `coreml_artifacts/` tree. For every segmentation and tracking
candidate that tree contains a preconverted `.mlpackage`, deterministic numeric `.npz` inputs and
exact PyTorch outputs, and a JSON manifest binding candidate identity, source SHA-256, package-tree
SHA-256, shapes, names, precision, versions, and parity tolerances. It also contains the generated
segmentation catalog and conversion report. Float16 candidates include a separately hashed Float32
safety fallback, so native parity can reject unsafe quantization without another workstation trip.
The catalog embeds each model's frozen foreground/boundary thresholds, branch fusion mode/weights,
confidence filters, and area fraction; exactly one model is recommended globally, preferring the
selected v4 candidate over the legacy semantic candidate. All files are covered by the outer `SHA256SUMS.txt`;
downloaded training data are not returned.

Linux conversion is necessary but cannot execute Apple's native runtime. On the Mac, import the
archive, create a clean **PyTorch-free** validation environment, and verify every preconverted
package against its golden fixture:

```bash
python3 WorkstationResults/import_results.py
/usr/bin/python3 -m venv /tmp/cellect-coreml-runtime
/tmp/cellect-coreml-runtime/bin/pip install -r \
  WorkstationResults/coreml_runtime_requirements.txt
/tmp/cellect-coreml-runtime/bin/python WorkstationResults/convert_coreml.py
```

The normal Mac path validates source and package hashes first, calls Core ML prediction for every
candidate, and gates raw tensors, sigmoid probabilities, and decisions at the frozen thresholds.
Raw mean/max error must be <= 0.05/0.5; segmentation probability mean/max <= 0.005/0.05;
tracking probability mean/max <= 0.002/0.02; and no decision map may disagree on more than 0.01%
of entries. An unsafe Float16 package is replaced by its independently verified Float32 fallback.
Only then does the bridge atomically install verified packages and merge the segmentation catalog.
It never imports or
loads the PyTorch 2.7 TorchScript. CellectTrack packages remain separate from the segmentation
catalog. `coreml_requirements.txt` is an optional PyTorch-2.2 legacy fallback for older archives;
newer TorchScript fails fast with instructions to return a preconverted archive. After native
parity, test Swift fusion/FIFO reconstruction and run the selected models on the physical iPhone.

After `best` completes, copy:

```text
output/cellect_workstation_results.zip
```

to `cellect/WorkstationResults/inbox/` on the Mac. The archive intentionally excludes downloaded
training data and any final-test performance report. Do not call any included development metric a
held-out final-test result.
