# Cellect dataset scope and access record

Reviewed 2026-07-31. The user confirmed on that date that they hold permission for all sources
previously identified as permission-controlled. Preserve the actual permission evidence outside
this repository. This is a reproducibility record, not legal advice.

## Automatically used by `./run.sh best`

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
- Das et al.: https://arxiv.org/html/2508.14106v1
