#!/usr/bin/env python3
"""Deterministic, acquisition-group-aware roles for Cellect model development.

LIVECell's upstream train and validation splits are treated as one development pool.  Every
timepoint/crop from one acquisition is assigned to exactly one of training, checkpoint selection,
post-processing calibration, or ensemble selection.  The upstream test split is deliberately
absent from this module so its labels can remain sealed until the final evaluation.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


BUNDLE_VERSION = "3.1.0"
VALIDATION_ROLES = ("checkpoint", "calibration", "ensemble_selection")
LIVECELL_ROLES = ("train", *VALIDATION_ROLES)

_STRUCTURAL_DIRECTORIES = {
    "annotation",
    "annotations",
    "gt",
    "image",
    "images",
    "label",
    "labels",
    "mask",
    "masks",
    "seg",
    "test",
    "testing",
    "train",
    "training",
    "val",
    "valid",
    "validation",
}


def acquisition_group(value: str | Path) -> str:
    """Remove crop/frame suffixes and retain one path-independent source directory."""
    path = Path(value)
    stem = re.sub(
        r"_(?:im|img|image|mask|masks|label|labels)$",
        "",
        path.stem,
        flags=re.I,
    )
    stem = re.sub(r"_(?:crop|patch|tile)_?\d+.*$", "", stem, flags=re.I)
    stem = re.sub(r"^(?:t|time|frame)_?\d+$", "sequence", stem, flags=re.I)
    stem = re.sub(r"(?:[_-](?:t|time|frame)_?\d+)$", "", stem, flags=re.I)
    # Absolute workstation paths must not change the scientific split.  The nearest
    # non-structural directory usually identifies a sequence, plate, or dataset release;
    # train/val/test and images/masks containers are deliberately ignored.
    parent_token = "root"
    for part in reversed(path.parent.parts):
        normalized = part.casefold()
        if not normalized or part == path.anchor:
            continue
        if normalized in _STRUCTURAL_DIRECTORIES:
            continue
        parent_token = part
        break
    return f"{parent_token}/{stem}"


def scientific_group(dataset: str, value: str | Path) -> str:
    """Return the dataset-aware acquisition group used for all v3 role checks."""
    normalized_dataset = dataset.casefold()
    if normalized_dataset == "livecell":
        # LIVECell names encode one acquisition as
        # ``CELL_LINE_MODALITY_WELL_SITE_DDdHHhMMm_CROP``.  Neither the timepoint nor the
        # exported crop may cross scientific roles.  Use the basename so the same acquisition
        # has one identity whether it came from the upstream train or validation JSON.
        stem = Path(value).stem
        stem = re.sub(
            r"_\d{1,3}d\d{1,2}h\d{1,2}m(?:_\d+)*$",
            "",
            stem,
            flags=re.I,
        )
        return f"livecell/{stem}"

    if normalized_dataset == "deepsea_phase":
        # DeepSea's segmentation exports use names such as A11_z003_c001.  The z/c
        # variants are derived from one annotated source field and must never be split
        # independently.  Ignore the upstream train/test directory here as well so the
        # preflight can detect an acquisition that accidentally appears on both sides.
        stem = Path(value).stem
        stem = re.sub(r"_z\d+(?:_c\d+)?$", "", stem, flags=re.I)
        stem = re.sub(r"_c\d+$", "", stem, flags=re.I)
        return f"deepsea_phase/{stem}"

    group = acquisition_group(value)
    if normalized_dataset == "yeaz_brightfield":
        # Six exposure variants of one field must remain in the same scientific role.
        group = re.sub(
            r"BF_(?:1(?:\.5)?|2|5|10|20)$",
            "BF",
            group,
            flags=re.I,
        )
    return group


def livecell_role(value: str | Path) -> str:
    """Assign a LIVECell acquisition to the 70/10/10/10 development partition."""
    group = scientific_group("livecell", value)
    digest = hashlib.sha256(f"cellect-v3:livecell-role:{group}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 70:
        return "train"
    if bucket < 80:
        return "checkpoint"
    if bucket < 90:
        return "calibration"
    return "ensemble_selection"


def validation_role(dataset: str, value: str | Path) -> str:
    """Assign one stable role without Python's process-randomized ``hash``."""
    group = scientific_group(dataset, value)
    digest = hashlib.sha256(f"cellect-v3:{dataset}:{group}".encode()).digest()
    return VALIDATION_ROLES[int.from_bytes(digest[:4], "big") % len(VALIDATION_ROLES)]


def belongs_to_validation_role(
    dataset: str,
    value: str | Path,
    role: str | None,
) -> bool:
    if role is None:
        return True
    if role not in VALIDATION_ROLES:
        raise ValueError(f"Unknown validation role: {role}")
    if dataset.casefold() == "livecell":
        return livecell_role(value) == role
    return validation_role(dataset, value) == role
