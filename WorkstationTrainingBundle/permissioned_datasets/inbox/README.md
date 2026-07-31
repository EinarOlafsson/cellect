# Permissioned dataset inbox

The user confirmed permission for these sources on 2026-07-31. Their websites do not expose an
anonymous archive URL, so place either or both files here before running `./run.sh best`:

- `das_2025.zip` — the author-provided 783 brightfield/85 phase-contrast dataset. Images and masks
  should have matching names ending in `_img`/`_mask`, `_im`/`_masks`, or `_image`/`_labels`.
- `deepsea_full.zip` — an archive produced by downloading the full 3,686-pair `segmentation_dataset`
  folder from Google Drive. Without it, the script uses the verifiable 100-pair public sample that
  anonymous Drive listing exposes. A complete unzipped `deepsea/{track,segment,final}` collection
  beside `run.sh` takes precedence over this archive and is used in place without copying.
- `cellpose_transmitted_light.zip` — only the manually curated transmitted-light cellular subset
  from the Cellpose training set. Use Cellpose pairs such as `field.tif` and `field_masks.tif`.

Do not put the complete broad Cellpose archive here without curation: it includes microscopy
domains that do not match Cellect's phone/label-free goal. The importer intentionally does not
guess whether an image is transmitted-light or fluorescence.

The input archives and extracted raw data are never included in the workstation results ZIP.
