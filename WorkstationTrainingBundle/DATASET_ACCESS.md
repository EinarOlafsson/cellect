# Cellect dataset scope and access record

Reviewed 2026-07-31. The user confirmed on that date that they hold permission for all sources
previously identified as permission-controlled. Preserve the actual permission evidence outside
this repository. This is a reproducibility record, not legal advice.

## Segmentation data used by `./run.sh best`

| Dataset | Why it is in scope | Published terms | Practical status |
|---|---|---|---|
| LIVECell | phase-contrast, eight adherent mammalian lines | CC BY-NC 4.0 | suitable for attributed non-commercial research; obtain separate permission before commercial use |
| Yeast in Microstructures | brightfield, manually reviewed instances | MIT | usable with required notices; sampled within a 10% yeast ceiling |
| Revvity-25 | brightfield human cancer cells, expert-validated instances | CC BY-NC 4.0 | suitable for attributed non-commercial research; obtain separate permission before commercial use |
| YeaZ | phase and multi-exposure brightfield yeast | user-confirmed permission | downloaded automatically; all yeast sources share a 10% sampling ceiling |
| DeepSea | phase-contrast mammalian cells | user-confirmed permission | a local `deepsea/{track,segment,final}` collection is detected first; otherwise `deepsea_full.zip` or the verified 100-pair sample is used |
| QPI Adherent Cells | label-free quantitative phase, five adherent lines | CC BY 4.0 | usable with attribution and license notice |
| BBBC009 | DIC red blood cells with manual outlines | copyright waived/CC0 | usable; preserve the recommended scientific citation |
| CTC BF-HSC/BF-MuSC/DIC-HeLa/PhC-U373/PhC-PSC | transmitted-light human/mouse cells | user-confirmed organizer/provider permission | downloaded automatically; only sparse human-curated gold masks train the semantic models, with unlabelled pixels ignored |

The research build is therefore not commercially cleared: LIVECell and Revvity-25 are
non-commercial. Whether trained weights are legally a derivative of training images is not settled
by these dataset pages. Ask the licensors explicitly before putting these weights in a paid app,
commercial service, or commercially sponsored distribution.

The v4 preflight is authoritative for what a particular run actually found. Dataset web-page
counts are planning estimates, not experimental sample counts. Use `preflight_v4.json` and
`splits_v4.jsonl` from the returned fingerprinted run when reporting data.

DeepSea segmentation is image-anchored. Every admitted raw image must have its official binary
mask and touching-cell weight map. Official companion files with no corresponding raw image are
not training samples; their relative paths, byte sizes, SHA-256 hashes, and exclusion reasons are
retained in the preflight report and dataset fingerprint. The same record holds two further
exclusions from a locally assembled collection: byte-identical `name(1)` copies of a canonical
file, and the training-side copies of acquisitions `A11`–`A19`, which DeepSea also ships inside its
official test folder. The published test folder is never trimmed.

Some frames in published archives annotate no cell at all — 65 Yeast-in-Microstructures fields
contain only the microstructure, and one YeaZ phase-contrast crop has an empty gold mask. They are
withheld from training rather than treated as corruption, and appear with their hashes under
`withheld_development_samples`.

## CellectTrack v4 data contract

The tracking adapters accept only complete development annotations from these sources:

| Source | Accepted annotation | Modality/cells | Excluded paths |
|---|---|---|---|
| DeepSea tracking | `tracking_dataset/train/<set>/{images,masks,labels}` with official masked-area images and position/lineage labels | phase-contrast mammalian cells | the entire DeepSea test partition |
| CTC BF-C2DL-HSC | training `NN_GT/TRA/man_track*.tif` + `man_track.txt` | brightfield mouse hematopoietic stem cells | `*_ST`, `*_RES`, and official test |
| CTC BF-C2DL-MuSC | training gold tracking annotations | brightfield mouse muscle stem cells | same exclusions |
| CTC DIC-C2DH-HeLa | training gold tracking annotations | DIC human HeLa | same exclusions |
| CTC PhC-C2DH-U373 | training gold tracking annotations | phase-contrast human U373 | same exclusions |
| CTC PhC-C2DL-PSC | training gold tracking annotations | phase-contrast pancreatic stem cells | same exclusions |
| LiveCellTrack public preview v1 | human MOT `gt/gt.txt` identity boxes from 10 scratch-wound (100-frame) and 10 HeLa (50-frame) acquisitions | transmitted-light live-cell microscopy | `annotations/train.json`, detector output, invented parent/division labels, and any data outside the pinned preview |
| CTMC-v1 | official `train/<sequence>/seqinfo.ini`, `img1`, 10-column `gt/gt.txt`, and 4-column `TRA/man_track.txt` | transmitted-light cultured eukaryotic cells | the entire official test partition and all non-train paths |
| ALFI Task 1 | MI01–MI08 PNG images plus `MIxx_DTLTruth.csv` identity boxes and explicit adjacent parent IDs | DIC U2OS, HeLa, and hTERT RPE-1 cells | Task 2, semantic masks as whole-cell instances, and all ambiguous duplicate frame/ID events |

All frames and lineage edges from one sequence share one acquisition-level
train/checkpoint/calibration/ensemble-selection role. CTC gold tracking masks are intentionally
sparse; the adapter does not substitute silver masks or result folders. If the tracking
orchestrator is enabled, extracted roots can be supplied with `CELLECT_CTC_BF_HSC_ROOT`,
`CELLECT_CTC_BF_MUSC_ROOT`, `CELLECT_CTC_DIC_HELA_ROOT`, `CELLECT_CTC_PHC_U373_ROOT`, and
`CELLECT_CTC_PHC_PSC_ROOT`.

DeepSea tracking is image-anchored, matching the published BasicTrackerDataset iteration rule.
Each selected image must have a mask and label. A small number of official acquisitions contain
mask/label companions beyond their last raw image; those files are never treated as frames, but
their relative paths, hashes, and exclusion reason are retained in the preflight provenance.

DeepSea's tracking export is read on its own terms rather than an idealized one, verified against
all 40 published training acquisitions:

- Frames are addressed by published order, not by the number in the filename. Three numbering
  styles occur — a terminal counter, a mid-name slice counter (`A11_z003_c001.png`, whose terminal
  `c` field never varies), and a time index sampled every fourth frame — and only the rightmost
  numeric field that is unique across an acquisition identifies a frame. One acquisition
  (`set_6_MC2C12`) is sampled unevenly, so adjacent frames would span different amounts of time;
  it is withheld and named in `sources.deepsea.irregular_sequences`.
- `masks` segment every visible cell while `labels` name only the tracked ones, so a mask
  component without an identity is neither a detection nor background. Those components are
  withheld from tokenization and counted in `unlabeled_mask_regions`; 84% of them touch the image
  border, where a cell is entering or leaving the field.
- A position label is the rounded centroid of its cell's mask component, so labels are matched to
  components by a one-to-one centroid assignment rather than a point-in-component test. The worst
  assignment across the published training frames is 2.2px; a match beyond 4px fails the run.
- Identities are annotated on a staggered schedule, so a track may be unobserved for a stretch and
  return. The gap is kept as published: training builds no continuation edge across it and
  quarantines its endpoints from birth/death supervision, so no link is invented.
- A division is written into the label names (`P` becomes `P_1` and `P_2`) and is resolved across
  the whole acquisition. A division with only one annotated daughter, one written with a hyphen
  separator, or one whose parent is still observed after a daughter starts is quarantined in
  `division_quarantined_parent_track_ids` rather than guessed.
- Three frames give one identity two markers. Both cells stay as detections and that frame leaves
  identity supervision, recorded in `annotation_exclusions`, matching the ALFI and LiveCellTrack
  contract.

LiveCellTrack is CC BY 4.0, DOI `10.17632/cgwcpz34mr.1`. The standard pipeline downloads the
173,935,248-byte archive into `data/external`, resumes an interrupted `.partial` transfer, verifies
SHA-256 `c3c824e3cb9db0673d84245ffcc9a5a85f9b1a8aafcf81d529195105a7cf3d7f`, and extracts it through
a traversal/link-safe atomic staging directory. An already extracted copy may instead be supplied
with `CELLECT_LIVECELLTRACK_ROOT`. The adapter treats boxes as identity/center annotations and makes
explicitly named elliptical feature proxies; it does not convert them into claimed human instance
masks. Because no parent IDs are present, this source teaches continuation associations but not
division lineage. No admitted source explicitly labels biological birth/death: first/last track
observations are censored, both decoder gates are frozen off at threshold 1.0, and no endpoint is
invented as a positive event.
Any source ID appearing more than once in one frame is quarantined in full and reported with its
observation count and reason. The adapter does not split, merge, or spatially reassociate such an
ambiguous identity.

CTMC-v1 uses official **TRAIN only**. The current strict source contract is 47 acquisitions,
80,389 frames, 1,616 tracks, and 1,097,223 boxes. The upstream archive publishes no checksum, so
the first complete TLS/user-supplied archive is recorded by byte count and SHA-256 in a stable,
immutable path-specific marker; changed bytes at that path are rejected. ZIP paths/types/CRC are
validated and only `train/` is extracted. All runs with the same cell-line prefix are assigned to
one scientific role. Configure an extracted root or ZIP with `CELLECT_CTMC_ROOT` (or a download
archive with `CELLECT_CTMC_ARCHIVE`). Its license remains **unknown** in this audit: research
permission is required and neither raw files nor the archive may be redistributed.

ALFI is pinned to Figshare file ID `41740227`, DOI `10.6084/m9.figshare.23798451.v1`, URL
`https://ndownloader.figshare.com/files/41740227`, exactly 8,423,073,056 bytes, and MD5
`fe3326323c10b1748302e962eae26150`; its verified SHA-256 is additionally recorded after download.
The source reports 8 sequences, 796 frames, 16,564 published cell annotations, and 331 tracks.
The current archive actually contains 16,627 DTL rows and 16,618 unique frame/identity keys; those
three values are preserved separately. The nine duplicate keys (18 affected rows) are retained as
visual boxes but excluded event-by-event from identity supervision. Non-binary/unresolved lineage
is also quarantined. Figshare's API reports CC0 1.0 while the updated official README says CC-BY;
Cellect therefore operates under the more restrictive **CC BY with attribution** and records the
conflict. Configure an extracted root or pinned ZIP with `CELLECT_ALFI_ROOT`, or a download archive
with `CELLECT_ALFI_ARCHIVE`.

The Ker et al. phase-contrast tracking dataset is not direct supervised CellectTrack training data
in v4. Its influence is allowed only through the optional, separately identified Trackastra
teacher, and its availability/permission must be recorded with that teacher experiment.

## Permission confirmed, but a local archive is still required

| Dataset | Can it be downloaded? | What the terms currently say | Cellect decision |
|---|---|---|---|
| Cellpose training set | yes, after its form | user confirmed permission | the official form returned HTTP 500 on 2026-07-31; once obtained, place a manually curated transmitted-light subset at `permissioned_datasets/inbox/cellpose_transmitted_light.zip`; do not ingest the broad mixed-domain set blindly |
| Das et al. (2025) | not anonymously; authors provide it directly | user confirmed permission | place the author-provided archive at `permissioned_datasets/inbox/das_2025.zip` |

The input archives and raw datasets must not be placed in the repository or workstation-results
ZIP. Record source versions/checksums and retain the permission messages with the paper files.

## Scientifically excluded from the primary model

DeepBacs and the bacterial portions of Omnipose are legally downloadable under CC BY or CC BY-NC,
but are not included in the general eukaryotic model. Bacteria are routinely counted and imaged by
brightfield, phase contrast, DIC, and fluorescence microscopy, often with high-magnification/high-NA
optics. Their scale, density, and morphology are different enough that they should train a separate
bacterial specialist and be evaluated on a separate test set. Omnipose fluorescence is also a
different intensity domain from unstained phone-through-eyepiece images.

TissueNet and DynamicNuclearNet are also excluded even with access: they are fluorescence-centric,
and DynamicNuclearNet targets nuclei/tracking rather than transmitted-light whole-cell boundaries.
TissueNet may be useful later for a separate fluorescence whole-cell model.

These segmentation exclusions do not authorize silently using the same source for tracking. A new
tracking source needs complete identity/lineage semantics, acquisition-level grouping, explicit
terms, and an untouched evaluation plan.

## Teacher provenance and licenses

Teachers are software/model artifacts rather than raw segmentation datasets, but their obligations
are part of the experiment record:

| Teacher/mechanism | Pinned use | License/provenance status | Distribution decision |
|---|---|---|---|
| Cellpose-SAM `cpsam_v2` | raw foreground logit and center-directed y/x flow; revision, size, SHA-256, settings, and cache key recorded | Cellpose code and declared weights: BSD-3-Clause; Cellpose states models were trained on CC-BY-NC data; SAM dependency: Apache-2.0 | teacher-derived Cellect weights remain research-only pending combined code/model/data review |
| CPDINO variants | optional Cellpose foundation comparisons | Cellpose terms plus the separate DINOv3 license and notice/redistribution obligations | not commercially or App-Store cleared by this record |
| Ceb-inspired graph teacher | independently trained keep/merge mechanism | Ceb repository: MIT; no official pretrained boundary-classifier checkpoint is published | cite Ceb, retain MIT notice where required, and do not claim checkpoint transfer |
| Trackastra `general_2d` v0.3.0 | optional workstation association teacher only | BSD-3-Clause; archive SHA-256 `35cefd8634860d1dd43bcdcbdef7ae0caa24445f19bcfd35ec6b19039f2cd876` | absent from the mobile model; preserve source/version/hash/license in any teacher experiment |

Teacher distillation does not erase the obligations of the teacher's own training data. Preserve
the generated provenance manifests and obtain legal review before distributing derived weights.

## Permission request template

> I am developing Cellect, an on-device cell-segmentation research app, and plan to publish the
> training and evaluation study. May I use [dataset] to train and evaluate segmentation models?
> Please confirm whether permission covers: (1) non-commercial research and publication,
> (2) publication of aggregate metrics and example figures, (3) distribution of trained model
> weights in an open/free iPhone app, (4) possible later commercial use, and (5) whether raw images
> or annotations may be redistributed. I will cite the requested papers and dataset source. I do
> not plan to redistribute raw data unless explicitly allowed.

Save each response with the experiment record. Do not place gated/raw datasets in the return ZIP or
the Git repository; preserve source URLs, versions, checksums, license text, and citations instead.

## Primary source links

- LIVECell: https://sartorius-research.github.io/LIVECell/
- Yeast in Microstructures: https://christophreich1996.github.io/yeast_in_microstructures_dataset/
- Revvity-25: https://huggingface.co/datasets/YaroslavPrytula/Revvity-25
- QPI Adherent Cells: https://zenodo.org/records/5153251
- BBBC009: https://bbbc.broadinstitute.org/BBBC009
- Cell Tracking Challenge conditions: https://celltrackingchallenge.net/datasets/
- Cellpose terms: https://www.cellpose.org/dataset
- TissueNet: https://deepcell.readthedocs.io/en/latest/data-gallery/tissuenet.html
- DynamicNuclearNet: https://deepcell.readthedocs.io/en/latest/data-gallery/dynamicnuclearnet.html
- YeaZ: https://www.epfl.ch/labs/lpbs/data-and-software/
- DeepSea: https://deepseas.org/datasets/
- LiveCellTrack preview: https://doi.org/10.17632/cgwcpz34mr.1
- Das et al.: https://arxiv.org/html/2508.14106v1
- Ceb source/license: https://github.com/pxliang/Ceb
- Trackastra source: https://github.com/weigertlab/trackastra/tree/0.3.0
- Trackastra model release: https://github.com/weigertlab/trackastra-models/releases/tag/v0.3.0
- Cellpose-SAM model registry: https://huggingface.co/mouseland/cellpose-sam
