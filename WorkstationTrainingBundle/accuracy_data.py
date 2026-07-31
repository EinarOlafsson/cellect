#!/usr/bin/env python3
"""Download and adapt manually annotated transmitted-light eukaryotic-cell datasets.

The primary accuracy profile is deliberately aligned with Cellect's intended deployment domain:
whole eukaryotic cells photographed from brightfield, phase-contrast, DIC, or quantitative-phase
microscopes. Yeast is retained as a capped morphology domain. Bacteria and fluorescence-only data
are documented as optional specialist sources, but are never downloaded or mixed into the primary
model automatically.

Every source keeps its official validation/test split where one exists. Sources without one are
split deterministically by acquisition group, not by random crops from the same field of view.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import albumentations as A
import cv2
import gdown
import numpy as np
import torch
from pycocotools.coco import COCO
from scipy import ndimage
from skimage.segmentation import watershed
from torch.utils.data import Dataset
from tqdm import tqdm

from mask_targets import internal_contact_boundary
from scientific_splits import (
    belongs_to_validation_role,
    scientific_group,
    validation_role,
)


YIM_URL = (
    "https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/"
    "tudatalib/3799/yeast_cell_in_microstructures_dataset.zip"
)
DEEPBACS_URL = (
    "https://zenodo.org/api/records/5550935/files/"
    "DeepBacs_Data_Segmentation_E.coli_Brightfield_dataset.zip/content"
)
REVVITY_REPOSITORY = "https://huggingface.co/datasets/YaroslavPrytula/Revvity-25"
REVVITY_REVISION = "ef6234dfe19905cb7c71fa12e61712a9090836d4"
YEAZ_ARCHIVES = {
    "yeaz_phase": (
        "https://drive.usercontent.google.com/download?"
        "id=14MUIN26ou0L12UC9UV_AC2S3isj1qBMY&export=download&confirm=t"
    ),
    "yeaz_brightfield": (
        "https://drive.usercontent.google.com/download?"
        "id=1Sot3bau0F0dsBjRxoQzdGOeUy_wMezal&export=download&confirm=t"
    ),
}
QPI_URL = "https://zenodo.org/api/records/5153251/files/labelled.zip/content"
OMNIPOSE_API_ROOTS = {
    "omnipose_bact_phase": (
        "https://api.osf.io/v2/nodes/xmury/files/osfstorage/"
        "62f5813e0beb5f0558b0ba6c/"
    ),
    "omnipose_bact_fluor": (
        "https://api.osf.io/v2/nodes/xmury/files/osfstorage/"
        "62f581630beb5f0558b0baa2/"
    ),
}
DEEPSEA_FOLDER_ID = "18odgkzafW8stHkzME_s7Es-ue7odVAc5"
BBBC009_URLS = {
    "images": "https://data.broadinstitute.org/bbbc/BBBC009/BBBC009_v1_images.zip",
    "outlines": "https://data.broadinstitute.org/bbbc/BBBC009/BBBC009_v1_outlines.zip",
}
CTC_DATASETS = {
    "ctc_bf_hsc": {
        "archive": "BF-C2DL-HSC",
        "gold_frames": 57,
        "modality": "brightfield",
        "organism": "mouse hematopoietic stem cells",
    },
    "ctc_bf_musc": {
        "archive": "BF-C2DL-MuSC",
        "gold_frames": 100,
        "modality": "brightfield",
        "organism": "mouse muscle stem cells",
    },
    "ctc_dic_hela": {
        "archive": "DIC-C2DH-HeLa",
        "gold_frames": 18,
        "modality": "DIC",
        "organism": "human HeLa cells",
    },
    "ctc_phc_u373": {
        "archive": "PhC-C2DH-U373",
        "gold_frames": 34,
        "modality": "phase contrast",
        "organism": "human glioblastoma-astrocytoma U373 cells",
    },
    "ctc_phc_psc": {
        "archive": "PhC-C2DL-PSC",
        "gold_frames": 4,
        "modality": "phase contrast",
        "organism": "pancreatic stem cells",
    },
}

DATASET_MANIFEST = {
    "yeast_microstructures": {
        "modality": "brightfield",
        "organism": "Saccharomyces cerevisiae",
        "annotation": "dense manually reviewed instance masks",
        "license": "MIT",
        "citation": "Reich et al., EMBC 2023",
        "source": "https://christophreich1996.github.io/yeast_in_microstructures_dataset/",
    },
    "revvity_25": {
        "modality": "brightfield",
        "organism": "human cancer cells on endothelial monolayers",
        "annotation": "110 manually labelled, expert-validated instance-mask images",
        "license": "CC BY-NC 4.0",
        "citation": "Prytula et al., CVPR Workshops 2025",
        "source": "https://huggingface.co/datasets/YaroslavPrytula/Revvity-25",
        "revision": REVVITY_REVISION,
    },
    "yeaz_phase": {
        "modality": "phase contrast",
        "organism": "Saccharomyces cerevisiae (diverse mutants and conditions)",
        "annotation": "manual instance masks; 10,422 annotated cells",
        "access_basis": "user confirmed permission on 2026-07-31",
        "citation": "Dietler et al., Nature Communications 2020",
        "source": "https://www.epfl.ch/labs/lpbs/data-and-software/",
    },
    "yeaz_brightfield": {
        "modality": "brightfield (six exposure levels)",
        "organism": "Saccharomyces cerevisiae",
        "annotation": "manual instance masks; 3,841 cells at six exposures",
        "access_basis": "user confirmed permission on 2026-07-31",
        "citation": "Dietler et al., Nature Communications 2020",
        "source": "https://www.epfl.ch/labs/lpbs/data-and-software/",
    },
    "deepsea_phase": {
        "modality": "phase contrast time-lapse",
        "organism": "mouse embryonic stem, bronchial epithelial, and C2C12 cells",
        "annotation": (
            "100-pair public train/test sample downloaded automatically; "
            "3,686-pair full archive used when supplied"
        ),
        "access_basis": "user confirmed permission on 2026-07-31",
        "citation": "Zargari et al., Cell Reports Methods 2023",
        "source": "https://deepseas.org/datasets/",
    },
    "qpi_adherent": {
        "modality": "quantitative phase microscopy",
        "organism": "PC-3, PNT1A, G361, A2050, and HOB adherent cells",
        "annotation": "524 manually annotated instance-mask images",
        "license": "CC BY 4.0",
        "citation": "Vicar et al., 2021",
        "source": "https://doi.org/10.5281/zenodo.5153251",
    },
    "bbbc009_dic": {
        "modality": "differential interference contrast",
        "organism": "human red blood cells",
        "annotation": "five manually drawn outline images",
        "license": "CC0",
        "citation": "Ljosa et al., Nature Methods 2012",
        "source": "https://bbbc.broadinstitute.org/BBBC009",
    },
    **{
        name: {
            "modality": details["modality"],
            "organism": details["organism"],
            "annotation": (
                f"{details['gold_frames']} human-made gold segmentation frames; "
                "unlabelled pixels ignored in loss"
            ),
            "access_basis": "user confirmed CTC permission on 2026-07-31",
            "citation": "Maška et al., Nature Methods 2023; Cell Tracking Challenge",
            "source": (
                "https://celltrackingchallenge.net/2d-datasets/"
                f"#{details['archive']}"
            ),
        }
        for name, details in CTC_DATASETS.items()
    },
}

# These sources are potentially valuable for separate task-specific models. Keeping their metadata
# here makes the exclusion explicit and reproducible; prepare_external_datasets never downloads
# them for the primary eukaryotic model.
OPTIONAL_SPECIALIST_DATASET_MANIFEST = {
    "deepbacs_ecoli": {
        "recommended_role": "bacterial specialist only",
        "modality": "brightfield",
        "organism": "Escherichia coli",
        "annotation": "manual instance masks (ROI maps)",
        "license": "CC BY 4.0",
        "citation": "Spahn et al., Communications Biology 2022",
        "source": "https://doi.org/10.5281/zenodo.5550935",
    },
    "omnipose_bact_phase": {
        "recommended_role": "bacterial specialist only",
        "modality": "phase contrast",
        "organism": "diverse bacterial species and morphologies",
        "annotation": "manually curated instance masks",
        "license": "CC BY-NC 3.0",
        "citation": "Cutler et al., Nature Methods 2022",
        "source": "https://osf.io/xmury/",
    },
    "omnipose_bact_fluor": {
        "recommended_role": "fluorescence bacterial specialist only",
        "modality": "widefield fluorescence (cytosol or membrane)",
        "organism": "diverse bacterial species",
        "annotation": "manually curated instance masks",
        "license": "CC BY-NC 3.0",
        "citation": "Cutler et al., Nature Methods 2022",
        "source": "https://osf.io/xmury/",
    },
}

GATED_DATASET_MANIFEST = {
    "cellpose_transmitted_light": {
        "status": "permission confirmed; local curated archive still required",
        "expected_path": "permissioned_datasets/inbox/cellpose_transmitted_light.zip",
        "note": "do not ingest fluorescence or non-cell images from the broad Cellpose set",
    },
    "das_2025": {
        "status": "permission confirmed; author-provided archive still required",
        "expected_path": "permissioned_datasets/inbox/das_2025.zip",
        "note": "783 brightfield and 85 phase-contrast images reported in the paper",
    },
}


@dataclass(frozen=True)
class ExternalSample:
    dataset: str
    split: str
    image_path: Path
    instance_path: Path
    class_path: Path | None = None
    boundary_path: Path | None = None


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1024:
        print(f"Reusing {destination}")
        return
    partial = destination.with_suffix(destination.suffix + ".partial")
    existing = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url)
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    print(f"Downloading {url}")
    with urllib.request.urlopen(request) as response:
        resumed = existing > 0 and getattr(response, "status", None) == 206
        mode = "ab" if resumed else "wb"
        if existing and not resumed:
            existing = 0
        remaining_header = response.headers.get("Content-Length")
        remaining = int(remaining_header) if remaining_header else None
        with partial.open(mode) as target:
            with tqdm(
                total=(existing + remaining) if remaining else None,
                initial=existing,
                unit="B",
                unit_scale=True,
                desc=destination.name,
            ) as progress:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
                    progress.update(len(chunk))
    partial.replace(destination)


def safe_extract(archive_path: Path, destination: Path) -> None:
    marker = destination / ".extracted"
    if marker.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                target = (destination / member.filename).resolve()
                if destination_root not in target.parents and target != destination_root:
                    raise RuntimeError(f"Unsafe ZIP member: {member.filename}")
            archive.extractall(destination)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as archive:
            for member in archive.getmembers():
                target = (destination / member.name).resolve()
                if destination_root not in target.parents and target != destination_root:
                    raise RuntimeError(f"Unsafe TAR member: {member.name}")
                if member.issym() or member.islnk():
                    raise RuntimeError(f"Refusing archive link: {member.name}")
            archive.extractall(destination)
    else:
        raise RuntimeError(f"Unsupported archive: {archive_path}")
    marker.touch()


def stable_split(dataset: str, group: str) -> str:
    """Create a reproducible 70/15/15 split without Python's salted hash."""
    value = int(hashlib.sha256(f"{dataset}:{group}".encode()).hexdigest()[:8], 16) % 100
    if value < 15:
        return "val"
    if value < 30:
        return "test"
    return "train"


def source_group(path: Path) -> str:
    stem = re.sub(r"_(?:im|img|image|mask|masks|label|labels)$", "", path.stem, flags=re.I)
    stem = re.sub(r"_crop_?\d+.*$", "", stem, flags=re.I)
    return f"{path.parent.name}/{stem}"


def source_split(dataset: str, path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if parts & {"test", "testing", "test_sorted"}:
        return "test"
    if parts & {"val", "valid", "validation"}:
        return "val"
    if dataset == "deepsea_phase" and "train" in parts:
        # DeepSea publishes train and final-test folders, but no independent checkpoint/
        # calibration pool.  Reserve 15% of *source acquisitions* from the official training
        # folder as development validation.  All z/c variants from one source stay together.
        group = scientific_group(dataset, path)
        value = int(
            hashlib.sha256(f"{dataset}:official-train:{group}".encode()).hexdigest()[:8],
            16,
        ) % 100
        return "val" if value < 15 else "train"
    # Official training directories still need a source-level validation subset.
    if dataset == "deepsea_phase":
        group = scientific_group(dataset, path)
    else:
        group = source_group(path)
        if dataset == "yeaz_brightfield":
            # The same yeast field appears at six exposure levels. Keep all exposures and crops
            # together so near-duplicate cells never cross train/validation/test boundaries.
            group = re.sub(r"BF_(?:1(?:\.5)?|2|5|10|20)$", "BF", group, flags=re.I)
    return stable_split(dataset, group)


def discover_suffix_pairs(root: Path, dataset: str) -> list[ExternalSample]:
    """Discover common ``*_im``/``*_mask`` instance-label conventions."""
    extensions = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
    samples: list[ExternalSample] = []
    for mask_path in sorted(root.rglob("*"), key=natural_key):
        if not mask_path.is_file() or mask_path.suffix.lower() not in extensions:
            continue
        if not re.search(r"_(?:mask|masks|label|labels)$", mask_path.stem, flags=re.I):
            continue
        image_path = None
        for image_suffix in ("_im", "_img", "_image", ""):
            candidate_stem = re.sub(
                r"_(?:mask|masks|label|labels)$",
                image_suffix,
                mask_path.stem,
                flags=re.I,
            )
            for extension in extensions:
                candidate = mask_path.with_name(candidate_stem + extension)
                if candidate.is_file():
                    image_path = candidate
                    break
            if image_path is not None:
                break
        if image_path is None:
            continue
        samples.append(
            ExternalSample(
                dataset=dataset,
                split=source_split(dataset, image_path),
                image_path=image_path,
                instance_path=mask_path,
            )
        )
    if not samples:
        raise RuntimeError(f"Could not find image/mask suffix pairs under {root}")
    return unique_samples(samples)


def discover_cellpose_pairs(root: Path, dataset: str) -> list[ExternalSample]:
    samples: list[ExternalSample] = []
    extensions = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
    for mask_path in sorted(root.rglob("*"), key=natural_key):
        if (
            not mask_path.is_file()
            or mask_path.suffix.lower() not in extensions
            or not mask_path.stem.lower().endswith("_masks")
        ):
            continue
        base = re.sub(r"_masks$", "", mask_path.stem, flags=re.I)
        image_path = next(
            (
                candidate
                for stem in (base, f"{base}_img", f"{base}_image")
                for extension in extensions
                if (candidate := mask_path.with_name(stem + extension)).is_file()
            ),
            None,
        )
        if image_path is None:
            continue
        samples.append(
            ExternalSample(
                dataset=dataset,
                split=source_split(dataset, image_path),
                image_path=image_path,
                instance_path=mask_path,
            )
        )
    if not samples:
        raise RuntimeError(f"Could not find Cellpose image/mask pairs under {root}")
    return unique_samples(samples)


def download_osf_tree(api_root: str, destination: Path) -> None:
    marker = destination / ".downloaded"
    if marker.is_file():
        return
    destination.mkdir(parents=True, exist_ok=True)

    def visit(api_url: str, local_root: Path) -> None:
        next_url: str | None = api_url
        while next_url:
            request_url = next_url + ("&" if "?" in next_url else "?") + "page[size]=100"
            with urllib.request.urlopen(request_url) as response:
                payload = json.load(response)
            for entry in payload["data"]:
                name = entry["attributes"]["name"]
                if name.startswith("."):
                    continue
                if entry["attributes"]["kind"] == "folder":
                    related = entry["relationships"]["files"]["links"]["related"]["href"]
                    visit(related, local_root / name)
                elif not re.search(r"_(?:flows)\.tiff?$", name, flags=re.I):
                    download(entry["links"]["download"], local_root / name)
            next_url = payload.get("links", {}).get("next")

    visit(api_root, destination)
    marker.touch()


DEEPSEA_WMAP_DIRECTORY_NAMES = (
    "unetwmaps",
    "unet_wmaps",
    "wmaps",
    "weight_maps",
    "weightmaps",
)


def _stable_path_key(path: Path) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", path.as_posix())
    )


def deepsea_subfolders(root: Path) -> dict[str, Path]:
    if not root.is_dir():
        return {}
    aliases = {
        "track": {"track", "tracking", "tracking_dataset"},
        "segment": {"segment", "segmentation", "segmentation_dataset"},
        "final": {"final", "final_dataset", "original", "original_annotated_dataset"},
    }
    children = sorted(
        (
            child
            for child in root.iterdir()
            if child.is_dir() and not child.name.startswith(".")
        ),
        key=_stable_path_key,
    )
    matches: dict[str, Path] = {}
    for role, names in aliases.items():
        candidates = [child for child in children if child.name.casefold() in names]
        if len(candidates) > 1:
            raise RuntimeError(
                f"Ambiguous DeepSea {role} directories under {root}: "
                + ", ".join(str(path) for path in candidates)
            )
        if candidates:
            matches[role] = candidates[0]
    return matches


def find_local_deepsea_root(
    search_roots: tuple[Path, ...] | None = None,
) -> tuple[Path, dict[str, Path]] | None:
    bundle_root = Path(__file__).resolve().parent
    if search_roots is None:
        environment_root = os.environ.get("CELLECT_DEEPSEA_ROOT")
        candidates = [
            bundle_root,
            bundle_root / "deepsea",
            bundle_root / "DeepSea",
            bundle_root.parent / "deepsea",
            bundle_root.parent / "DeepSea",
            Path.cwd(),
            Path.cwd() / "deepsea",
            Path.cwd() / "DeepSea",
        ]
        if environment_root:
            candidates.insert(0, Path(environment_root).expanduser())
        for ancestor in list(bundle_root.parents)[:4]:
            candidates.extend((ancestor / "deepsea", ancestor / "DeepSea"))
    else:
        candidates = []
        for root in search_roots:
            candidates.extend((root, root / "deepsea", root / "DeepSea"))
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        folders = deepsea_subfolders(resolved)
        if {"track", "segment", "final"} <= set(folders):
            print(f"DeepSea root selected: {resolved}")
            return resolved, folders
    if search_roots is None and environment_root:
        raise RuntimeError(
            "CELLECT_DEEPSEA_ROOT was set but does not contain complete "
            f"track/segment/final folders: {environment_root}"
        )
    return None


def _deepsea_role_directories(root: Path) -> dict[Path, dict[str, list[Path]]]:
    """Group every canonical DeepSea image/mask/wmap directory by its parent."""
    groups: dict[Path, dict[str, list[Path]]] = {}
    for directory in sorted(
        (path for path in root.rglob("*") if path.is_dir()),
        key=_stable_path_key,
    ):
        name = directory.name.casefold()
        if name == "images":
            role = "images"
        elif name == "masks":
            role = "masks"
        elif name in DEEPSEA_WMAP_DIRECTORY_NAMES:
            role = "wmaps"
        else:
            continue
        roles = groups.setdefault(
            directory.parent,
            {"images": [], "masks": [], "wmaps": []},
        )
        roles[role].append(directory)
    return groups


def _deepsea_file_index(directory: Path, role: str) -> dict[str, Path]:
    """Index one DeepSea role by case-insensitive stem and reject ambiguous files."""
    paths = image_files(directory)
    if not paths:
        raise RuntimeError(f"DeepSea {role} directory contains no images: {directory}")
    indexed: dict[str, Path] = {}
    for path in paths:
        stem = path.stem.casefold()
        previous = indexed.get(stem)
        if previous is not None:
            raise RuntimeError(
                f"Ambiguous DeepSea {role} stem {path.stem!r} under {directory}: "
                f"{previous.name}, {path.name}"
            )
        indexed[stem] = path
    return indexed


def deepsea_boundary_path(mask_directory: Path, mask_path: Path) -> Path | None:
    """Find one unambiguous DeepSea touching-cell map paired by filename stem."""
    directories = sorted(
        (
            directory
            for directory in mask_directory.parent.iterdir()
            if directory.is_dir()
            and directory.name.casefold() in DEEPSEA_WMAP_DIRECTORY_NAMES
        ),
        key=_stable_path_key,
    )
    if len(directories) > 1:
        raise RuntimeError(
            f"Ambiguous DeepSea wmap directories under {mask_directory.parent}: "
            + ", ".join(str(path) for path in directories)
        )
    if not directories:
        return None
    return _deepsea_file_index(directories[0], "wmap").get(mask_path.stem.casefold())


def download_deepsea(destination: Path) -> Path:
    local = find_local_deepsea_root()
    if local is not None:
        root, folders = local
        print(
            "Using local DeepSea collection without copying it: "
            f"{root} (track={folders['track'].name}, "
            f"segment={folders['segment'].name}, final={folders['final'].name})"
        )
        return root
    inbox_archive = (
        Path(__file__).resolve().parent
        / "permissioned_datasets"
        / "inbox"
        / "deepsea_full.zip"
    )
    if inbox_archive.is_file():
        safe_extract(inbox_archive, destination)
        (destination / ".full_permissioned_archive").touch()
        return destination
    marker = destination / ".downloaded"
    if marker.is_file():
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    result = gdown.download_folder(
        id=DEEPSEA_FOLDER_ID,
        output=str(destination),
        quiet=False,
        remaining_ok=True,
    )
    if not result:
        raise RuntimeError("DeepSea Google Drive folder download produced no files.")
    if len(result) != 300:
        raise RuntimeError(
            "Expected the verifiable 100-pair DeepSea public sample "
            f"(300 image/mask/weight files), downloaded {len(result)} files."
        )
    marker.touch()
    return destination


def discover_deepsea(root: Path) -> list[ExternalSample]:
    local_folders = deepsea_subfolders(root)
    if {"track", "segment", "final"} <= set(local_folders):
        # The official segmentation download is the canonical 3,686 image/cell-mask pairing.
        # `final` contains the corresponding originals plus cell/nucleus masks, and `track`
        # contains the same 47 time-lapse sets. Adding all three would triple-count frames and
        # leak adjacent duplicates across the study. Prefer `segment`; keep the other two present
        # for provenance and future temporal evaluation.
        samples = discover_deepsea(local_folders["segment"])
        print(
            "DeepSea local collection: using "
            f"{len(samples)} segmentation pairs; final/original and tracking folders "
            "are recognized but not duplicated into segmentation training."
        )
        return samples
    if local_folders:
        missing_roles = sorted({"track", "segment", "final"} - set(local_folders))
        raise RuntimeError(
            f"Incomplete DeepSea collection under {root}; missing top-level roles: "
            + ", ".join(missing_roles)
        )

    groups = _deepsea_role_directories(root)
    if not groups:
        raise RuntimeError(
            "DeepSea contains no canonical images/masks/wmaps directory groups under "
            f"{root}"
        )

    samples: list[ExternalSample] = []
    for parent in sorted(groups, key=_stable_path_key):
        roles = groups[parent]
        missing_directories = [role for role, paths in roles.items() if not paths]
        ambiguous_directories = {
            role: paths for role, paths in roles.items() if len(paths) > 1
        }
        if missing_directories or ambiguous_directories:
            ambiguous_text = {
                role: [path.name for path in paths]
                for role, paths in ambiguous_directories.items()
            }
            raise RuntimeError(
                f"Invalid DeepSea triplet directories under {parent}: "
                f"missing={missing_directories}, ambiguous={ambiguous_text}"
            )

        image_directory = roles["images"][0]
        mask_directory = roles["masks"][0]
        wmap_directory = roles["wmaps"][0]
        images = _deepsea_file_index(image_directory, "image")
        masks = _deepsea_file_index(mask_directory, "mask")
        wmaps = _deepsea_file_index(wmap_directory, "wmap")
        image_stems = set(images)
        mask_stems = set(masks)
        wmap_stems = set(wmaps)
        if not (image_stems == mask_stems == wmap_stems):
            missing_images = sorted((mask_stems | wmap_stems) - image_stems)
            missing_masks = sorted((image_stems | wmap_stems) - mask_stems)
            missing_wmaps = sorted((image_stems | mask_stems) - wmap_stems)
            orphan_images = sorted(image_stems - (mask_stems & wmap_stems))
            orphan_masks = sorted(mask_stems - (image_stems & wmap_stems))
            orphan_wmaps = sorted(wmap_stems - (image_stems & mask_stems))
            raise RuntimeError(
                f"DeepSea requires one image/mask/wmap per stem under {parent}: "
                f"missing images={missing_images[:5]}, missing masks={missing_masks[:5]}, "
                f"missing wmaps={missing_wmaps[:5]}, orphan images={orphan_images[:5]}, "
                f"orphan masks={orphan_masks[:5]}, orphan wmaps={orphan_wmaps[:5]}"
            )

        for stem in sorted(image_stems, key=lambda value: _stable_path_key(Path(value))):
            image_path = images[stem]
            samples.append(
                ExternalSample(
                    "deepsea_phase",
                    source_split("deepsea_phase", image_path),
                    image_path,
                    masks[stem],
                    boundary_path=wmaps[stem],
                )
            )

    if not samples:
        raise RuntimeError(f"DeepSea contains no complete image/mask/wmap triplets under {root}")
    return unique_samples(samples)


def discover_suffix_pairs_if_available(root: Path, dataset: str) -> list[ExternalSample]:
    try:
        return discover_suffix_pairs(root, dataset)
    except RuntimeError:
        return []


def prepare_bbbc009(root: Path, downloads: Path) -> list[ExternalSample]:
    image_archive = downloads / "bbbc009_images.zip"
    outline_archive = downloads / "bbbc009_outlines.zip"
    download(BBBC009_URLS["images"], image_archive)
    download(BBBC009_URLS["outlines"], outline_archive)
    safe_extract(image_archive, root / "images")
    safe_extract(outline_archive, root / "outlines")
    images = {
        path.name: path
        for path in (root / "images").rglob("*.tif")
        if not path.name.startswith(".")
    }
    outlines = {
        path.name: path
        for path in (root / "outlines").rglob("*.tif")
        if not path.name.startswith(".")
    }
    names = sorted(images.keys() & outlines.keys())
    if len(names) != 5:
        raise RuntimeError(f"Expected five BBBC009 DIC pairs, found {len(names)}")
    splits = ("train", "train", "train", "val", "test")
    return [
        ExternalSample("bbbc009_dic", split, images[name], outlines[name])
        for name, split in zip(names, splits, strict=True)
    ]


def prepare_ctc(external_root: Path, downloads: Path) -> list[ExternalSample]:
    """Download transmitted-light CTC training sets and expose human gold masks only."""
    samples: list[ExternalSample] = []
    for dataset, details in CTC_DATASETS.items():
        archive_name = details["archive"]
        archive_path = downloads / f"ctc_{archive_name}.zip"
        root = external_root / dataset
        download(
            f"https://data.celltrackingchallenge.net/training-datasets/{archive_name}.zip",
            archive_path,
        )
        safe_extract(archive_path, root)
        dataset_samples = discover_ctc_gold(root, dataset)
        expected = int(details["gold_frames"])
        if len(dataset_samples) != expected:
            raise RuntimeError(
                f"Expected {expected} gold-mask frames in {archive_name}, "
                f"found {len(dataset_samples)}"
            )
        samples.extend(dataset_samples)
        print(f"CTC {archive_name}: {len(dataset_samples)} gold-mask frames")
    return samples


def discover_ctc_gold(root: Path, dataset: str) -> list[ExternalSample]:
    samples: list[ExternalSample] = []
    for mask_path in sorted(root.rglob("man_seg*.tif"), key=natural_key):
        if mask_path.parent.name != "SEG" or not mask_path.parent.parent.name.endswith("_GT"):
            continue
        match = re.fullmatch(r"man_seg(\d+)", mask_path.stem)
        if match is None:
            continue
        sequence = mask_path.parent.parent.name.removesuffix("_GT")
        image_path = mask_path.parent.parent.parent / sequence / f"t{match.group(1)}.tif"
        if not image_path.is_file():
            raise RuntimeError(f"CTC gold mask has no matching image: {mask_path}")
        # Real 2-D CTC gold masks intentionally annotate only selected cells/frames. They are
        # useful for training only when unlabelled pixels are ignored, so keep them out of val/test.
        samples.append(ExternalSample(dataset, "train", image_path, mask_path))
    if not samples:
        raise RuntimeError(f"No human gold segmentation masks found under {root}")
    return unique_samples(samples)


def prepare_gated_datasets(external_root: Path) -> list[ExternalSample]:
    """Ingest permissioned archives that have no anonymous public download URL."""
    inbox = Path(__file__).resolve().parent / "permissioned_datasets" / "inbox"
    samples: list[ExternalSample] = []
    importers = {
        "cellpose_transmitted_light": discover_cellpose_pairs,
        "das_2025": discover_suffix_pairs,
    }
    for dataset, importer in importers.items():
        archive = inbox / f"{dataset}.zip"
        if not archive.is_file():
            print(f"Optional permissioned archive not present: {archive}")
            continue
        root = external_root / dataset
        safe_extract(archive, root)
        samples.extend(importer(root, dataset))
    return samples


def prepare_external_datasets(data_root: Path) -> list[ExternalSample]:
    external_root = data_root / "external"
    downloads = external_root / "downloads"

    yim_archive = downloads / "yeast_microstructures.zip"
    yim_root = external_root / "yeast_microstructures"
    download(YIM_URL, yim_archive)
    safe_extract(yim_archive, yim_root)

    revvity_root = external_root / "revvity_25"
    revvity_samples = prepare_revvity(revvity_root)

    yeaz_samples: list[ExternalSample] = []
    for dataset, url in YEAZ_ARCHIVES.items():
        archive = downloads / f"{dataset}.tar.gz"
        root = external_root / dataset
        download(url, archive)
        safe_extract(archive, root)
        yeaz_samples.extend(discover_suffix_pairs(root, dataset))

    qpi_archive = downloads / "qpi_adherent_labelled.zip"
    qpi_root = external_root / "qpi_adherent"
    download(QPI_URL, qpi_archive)
    safe_extract(qpi_archive, qpi_root)
    qpi_samples = discover_suffix_pairs(qpi_root, "qpi_adherent")

    deepsea_root = external_root / "deepsea_phase"
    deepsea_source = download_deepsea(deepsea_root)
    deepsea_samples = discover_deepsea(deepsea_source)

    bbbc_samples = prepare_bbbc009(external_root / "bbbc009_dic", downloads)
    ctc_samples = prepare_ctc(external_root, downloads)
    gated_samples = prepare_gated_datasets(external_root)

    samples = (
        discover_yim(yim_root)
        + revvity_samples
        + yeaz_samples
        + qpi_samples
        + deepsea_samples
        + bbbc_samples
        + ctc_samples
        + gated_samples
    )
    if not samples:
        raise RuntimeError("No external image/mask pairs were discovered.")
    dataset_counts: dict[str, int] = {}
    for sample in samples:
        dataset_counts[sample.dataset] = dataset_counts.get(sample.dataset, 0) + 1
    print("External manually annotated samples:", dataset_counts)
    return samples


def prepare_revvity(root: Path) -> list[ExternalSample]:
    """Download the pinned public COCO release and materialize polygons as label TIFFs."""
    annotation_root = root / "annotations"
    image_root = root / "images"
    mask_root = root / "instance_masks"
    samples: list[ExternalSample] = []
    for source_split, split in (("train", "train"), ("valid", "val")):
        annotation_path = annotation_root / f"{source_split}.json"
        annotation_url = (
            f"{REVVITY_REPOSITORY}/resolve/{REVVITY_REVISION}/"
            f"annotations/{source_split}.json"
        )
        download(annotation_url, annotation_path)
        coco = COCO(str(annotation_path))
        split_mask_root = mask_root / source_split
        split_mask_root.mkdir(parents=True, exist_ok=True)
        for image_id in tqdm(
            sorted(coco.getImgIds()),
            desc=f"prepare Revvity-25 {source_split}",
        ):
            metadata = coco.loadImgs([image_id])[0]
            file_name = Path(metadata["file_name"]).name
            image_path = image_root / file_name
            image_url = (
                f"{REVVITY_REPOSITORY}/resolve/{REVVITY_REVISION}/images/"
                f"{quote(file_name)}"
            )
            download(image_url, image_path)
            mask_path = split_mask_root / f"{Path(file_name).stem}.tif"
            if not mask_path.is_file():
                labels = np.zeros(
                    (int(metadata["height"]), int(metadata["width"])),
                    dtype=np.uint16,
                )
                annotations = coco.loadAnns(
                    coco.getAnnIds(imgIds=[image_id], iscrowd=None)
                )
                for label, annotation in enumerate(annotations, start=1):
                    labels[coco.annToMask(annotation).astype(bool)] = label
                if not cv2.imwrite(str(mask_path), labels):
                    raise RuntimeError(f"Could not write {mask_path}")
            samples.append(
                ExternalSample(
                    dataset="revvity_25",
                    split=split,
                    image_path=image_path,
                    instance_path=mask_path,
                )
            )
    if len(samples) != 110:
        raise RuntimeError(
            f"Expected 110 Revvity-25 image/mask pairs, discovered {len(samples)}."
        )
    return samples


def discover_yim(root: Path) -> list[ExternalSample]:
    samples: list[ExternalSample] = []
    for split in ("train", "val", "test"):
        input_directories = [
            directory
            for directory in root.rglob("inputs")
            if directory.parent.name.lower() == split
        ]
        for input_directory in input_directories:
            split_root = input_directory.parent
            instance_directory = split_root / "instances"
            class_directory = split_root / "classes"
            if not instance_directory.is_dir() or not class_directory.is_dir():
                continue
            for image_path in sorted(input_directory.glob("*.pt"), key=natural_key):
                instance_path = instance_directory / image_path.name
                class_path = class_directory / image_path.name
                if instance_path.is_file() and class_path.is_file():
                    samples.append(
                        ExternalSample(
                            "yeast_microstructures",
                            split,
                            image_path,
                            instance_path,
                            class_path,
                        )
                    )
    if not samples:
        raise RuntimeError(f"Could not find Yeast in Microstructures samples under {root}")
    return samples


def discover_deepbacs(root: Path) -> list[ExternalSample]:
    samples: list[ExternalSample] = []
    for official_split in ("train", "test"):
        split_roots = [
            directory
            for directory in root.rglob("*")
            if directory.is_dir() and directory.name.lower() == official_split
        ]
        for split_root in split_roots:
            image_directories = [
                directory
                for directory in split_root.rglob("*")
                if directory.is_dir() and directory.name.lower() == "brightfield"
            ]
            mask_directories = [
                directory
                for directory in split_root.rglob("*")
                if directory.is_dir()
                and directory.name.lower() in {"instance_segmentation_gt", "roimaps"}
            ]
            for image_directory in image_directories:
                mask_directory = closest_directory(image_directory, mask_directories)
                if mask_directory is None:
                    continue
                image_paths = sorted(image_files(image_directory), key=natural_key)
                mask_paths = sorted(image_files(mask_directory), key=natural_key)
                if not image_paths or len(image_paths) != len(mask_paths):
                    continue
                for pair_index, (image_path, mask_path) in enumerate(
                    zip(image_paths, mask_paths, strict=True)
                ):
                    if official_split == "test":
                        split = "test"
                    else:
                        # Keep a stable, source-level validation split without touching test data.
                        split = "val" if pair_index % 5 == 0 else "train"
                    samples.append(
                        ExternalSample(
                            "deepbacs_ecoli",
                            split,
                            image_path,
                            mask_path,
                        )
                    )
    if not any(sample.dataset == "deepbacs_ecoli" for sample in samples):
        raise RuntimeError(f"Could not find DeepBacs image/mask pairs under {root}")
    return unique_samples(samples)


def closest_directory(source: Path, candidates: list[Path]) -> Path | None:
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda candidate: len(set(source.parts) ^ set(candidate.parts)),
    )


def image_files(directory: Path) -> list[Path]:
    extensions = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in extensions
        ),
        key=_stable_path_key,
    )


def unique_samples(samples: list[ExternalSample]) -> list[ExternalSample]:
    unique: dict[tuple[str, str], ExternalSample] = {}
    for sample in sorted(
        samples,
        key=lambda item: (
            _stable_path_key(item.image_path),
            _stable_path_key(item.instance_path),
            _stable_path_key(item.boundary_path) if item.boundary_path else (),
        ),
    ):
        unique.setdefault(
            (str(sample.image_path), str(sample.instance_path)),
            sample,
        )
    return list(unique.values())


def natural_key(path: Path) -> list[object]:
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def deepsea_instance_labels(
    binary_mask: np.ndarray,
    touching_edge_map: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Recover DeepSea instances without interpreting its binary mask as label IDs.

    DeepSea supplies a binary cell-body mask plus a separate U-Net weight/edge map.  The official
    loader removes the positive edge-map pixels from the mask.  We use those same pixels as
    instance-separating seeds, then assign the removed pixels back to the nearest seeded object so
    Cellect receives a contiguous foreground target and an explicit contact-boundary target.
    """
    foreground = np.asarray(binary_mask) > 0
    if touching_edge_map is None:
        components, labels = cv2.connectedComponents(
            foreground.astype(np.uint8), connectivity=8
        )
        return labels.astype(np.int32), None
    touching = np.asarray(touching_edge_map) > 0
    if touching.shape != foreground.shape:
        raise RuntimeError(
            "DeepSea mask and touching-edge map have different dimensions: "
            f"{foreground.shape} versus {touching.shape}"
        )
    interiors = foreground & ~touching
    _, markers = cv2.connectedComponents(interiors.astype(np.uint8), connectivity=8)
    if int(markers.max()) == 0:
        _, markers = cv2.connectedComponents(foreground.astype(np.uint8), connectivity=8)
    if int(markers.max()) > 0:
        distance = ndimage.distance_transform_edt(foreground)
        labels = watershed(-distance, markers, mask=foreground).astype(np.int32)
    else:
        labels = np.zeros(foreground.shape, dtype=np.int32)
    return labels, touching


def load_external_record(
    sample: ExternalSample,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    if sample.dataset == "yeast_microstructures":
        image_tensor = torch.load(
            sample.image_path,
            map_location="cpu",
            weights_only=True,
        )
        instances = torch.load(
            sample.instance_path,
            map_location="cpu",
            weights_only=True,
        )
        classes = torch.load(
            sample.class_path,
            map_location="cpu",
            weights_only=True,
        )
        image = image_tensor.detach().cpu().numpy().squeeze()
        if not np.isfinite(image).all():
            raise RuntimeError(f"Image tensor contains NaN or infinity: {sample.image_path}")
        image = normalize_uint8(image)
        instance_array = instances.detach().cpu().numpy().astype(bool)
        class_array = classes.detach().cpu().numpy()
        if class_array.ndim > 1:
            class_array = class_array.argmax(axis=-1)
        labels = np.zeros(image.shape[-2:], dtype=np.int32)
        next_label = 0
        for index in range(instance_array.shape[0]):
            if int(class_array[index]) != 1:
                continue
            next_label += 1
            labels[instance_array[index]] = next_label
        return image, labels, None

    image = cv2.imread(str(sample.image_path), cv2.IMREAD_UNCHANGED)
    labels = cv2.imread(str(sample.instance_path), cv2.IMREAD_UNCHANGED)
    if image is None or labels is None:
        raise RuntimeError(f"Could not decode {sample.image_path} / {sample.instance_path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if labels.ndim == 3:
        labels = labels[..., 0]
    if not np.isfinite(image).all():
        raise RuntimeError(f"Image contains NaN or infinity: {sample.image_path}")
    image = normalize_uint8(image)
    explicit_boundary: np.ndarray | None = None
    if sample.dataset == "bbbc009_dic":
        labels = outlines_to_instances(labels)
    elif sample.dataset == "deepsea_phase":
        touching = None
        if sample.boundary_path is not None:
            touching = cv2.imread(str(sample.boundary_path), cv2.IMREAD_UNCHANGED)
            if touching is None:
                raise RuntimeError(f"Could not decode {sample.boundary_path}")
            if touching.ndim == 3:
                touching = touching[..., 0]
        labels, explicit_boundary = deepsea_instance_labels(labels, touching)
    else:
        labels = ensure_instance_labels(labels)
    return image, labels, explicit_boundary


def load_external_sample(sample: ExternalSample) -> tuple[np.ndarray, np.ndarray]:
    image, labels, _ = load_external_record(sample)
    return image, labels


def outlines_to_instances(outlines: np.ndarray) -> np.ndarray:
    outline = outlines > 0
    # Some TIFF readers invert 1-bit images. Cell outlines occupy the minority.
    if outline.mean() > 0.5:
        outline = ~outline
    contours, hierarchy = cv2.findContours(
        outline.astype(np.uint8),
        cv2.RETR_CCOMP,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    labels = np.zeros(outline.shape, dtype=np.int32)
    next_label = 0
    hierarchy_row = hierarchy[0] if hierarchy is not None else []
    for index, contour in enumerate(contours):
        if len(hierarchy_row) and hierarchy_row[index][3] != -1:
            continue
        if cv2.contourArea(contour) < 4:
            continue
        next_label += 1
        cv2.drawContours(labels, [contour], -1, next_label, thickness=cv2.FILLED)
    return labels


def ensure_instance_labels(labels: np.ndarray) -> np.ndarray:
    labels = labels.astype(np.int32)
    values = np.unique(labels)
    foreground_values = values[values != 0]
    if len(foreground_values) <= 1:
        _, labels = cv2.connectedComponents((labels != 0).astype(np.uint8), connectivity=8)
    return labels.astype(np.int32)


def normalize_uint8(image: np.ndarray) -> np.ndarray:
    # Native 8-bit microscope and phone-like images already match the app's raw `/255` contract.
    # Percentile-normalizing them here would give external validation data preprocessing that the
    # deployed iPhone pipeline does not perform.
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    image = image.astype(np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.uint8)
    lower, upper = np.percentile(image[finite], (0.5, 99.5))
    if upper <= lower:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip((image - lower) * 255.0 / (upper - lower), 0, 255).astype(np.uint8)


class ExternalCellDataset(Dataset):
    def __init__(
        self,
        samples: list[ExternalSample],
        split: str,
        image_size: int,
        train: bool,
        transform: A.Compose | None,
        limit: int | None = None,
        datasets: set[str] | None = None,
        validation_role: str | None = None,
    ) -> None:
        self.samples = [
            sample
            for sample in samples
            if sample.split == split
            and (datasets is None or sample.dataset in datasets)
            and (
                split != "val"
                or belongs_to_validation_role(
                    sample.dataset,
                    sample.image_path,
                    validation_role,
                )
            )
        ]
        if limit:
            # Smoke limits are stratified so every microscopy domain is exercised.
            retained: list[ExternalSample] = []
            per_dataset: dict[str, int] = {}
            for sample in self.samples:
                count = per_dataset.get(sample.dataset, 0)
                if count >= limit:
                    continue
                retained.append(sample)
                per_dataset[sample.dataset] = count + 1
            self.samples = retained
        self.image_size = image_size
        self.train = train
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        gray, labels, explicit_boundary = load_external_record(sample)
        if sample.dataset.startswith("ctc_"):
            # Human gold masks are sparse. Supervise the annotated cells and a narrow exterior halo,
            # while ignoring other visible but unlabelled cells in the same frame.
            iterations = max(3, int(round(min(labels.shape) / 256 * 6)))
            valid = cv2.dilate(
                (labels > 0).astype(np.uint8),
                np.ones((3, 3), dtype=np.uint8),
                iterations=iterations,
            )
        else:
            valid = np.ones(labels.shape, dtype=np.uint8)
        image = np.repeat(gray[..., None], 3, axis=2)
        if self.transform is not None:
            mask_channels = [labels, valid]
            if explicit_boundary is not None:
                mask_channels.append(explicit_boundary.astype(np.uint8))
            combined_mask = np.stack(mask_channels, axis=-1)
            transformed = self.transform(image=image, mask=combined_mask)
            image = transformed["image"]
            labels = transformed["mask"][..., 0]
            valid = transformed["mask"][..., 1]
            if explicit_boundary is not None:
                explicit_boundary = transformed["mask"][..., 2] > 0
        else:
            image = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA,
            )
            labels = cv2.resize(
                labels,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            )
            valid = cv2.resize(
                valid,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            )
            if explicit_boundary is not None:
                explicit_boundary = cv2.resize(
                    explicit_boundary.astype(np.uint8),
                    (self.image_size, self.image_size),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
        targets, count = labels_to_targets(labels, valid, explicit_boundary)
        return {
            "image": torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0,
            "mask": torch.from_numpy(targets.transpose(2, 0, 1).astype(np.float32)),
            "instances": (
                torch.from_numpy(labels.astype(np.int32))
                if not self.train
                else torch.empty(0, dtype=torch.int32)
            ),
            "count": count,
            "name": f"{sample.dataset}_{sample.image_path.stem}",
            "dataset": sample.dataset,
        }


def labels_to_targets(
    labels: np.ndarray,
    valid: np.ndarray | None = None,
    explicit_boundary: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    labels = labels.astype(np.int64)
    foreground = labels > 0
    boundary = internal_contact_boundary(labels, explicit_boundary)
    count = int(np.unique(labels[foreground]).size)
    if valid is None:
        valid = np.ones(labels.shape, dtype=np.uint8)
    return np.stack([foreground, boundary, valid > 0], axis=-1).astype(np.uint8), count


def materialize_cellpose_pairs(
    samples: list[ExternalSample],
    destination: Path,
    limit_per_dataset_role: int | None = None,
) -> dict[str, int]:
    yeast_datasets = {"yeast_microstructures", "yeaz_phase", "yeaz_brightfield"}
    ctc_samples = [sample for sample in samples if sample.dataset.startswith("ctc_")]
    supported_samples = [
        sample for sample in samples if not sample.dataset.startswith("ctc_")
    ]
    train_samples = [sample for sample in supported_samples if sample.split == "train"]
    non_yeast_train = [
        sample for sample in train_samples if sample.dataset not in yeast_datasets
    ]
    yeast_by_dataset: dict[str, list[ExternalSample]] = {}
    for sample in train_samples:
        if sample.dataset in yeast_datasets:
            yeast_by_dataset.setdefault(sample.dataset, []).append(sample)
    for dataset_samples in yeast_by_dataset.values():
        dataset_samples.sort(
            key=lambda sample: hashlib.sha256(
                str(sample.image_path).encode()
            ).hexdigest()
        )

    # Cellpose's CLI samples materialized images uniformly and has no source-balanced CLI option.
    # Interleave the yeast sources, then retain only enough for yeast to be <=10% of the combined
    # LIVECell + external training images. Validation remains uncapped and per-domain.
    interleaved_yeast: list[ExternalSample] = []
    ordered_datasets = sorted(yeast_by_dataset)
    next_index = 0
    while ordered_datasets:
        remaining = []
        for dataset in ordered_datasets:
            dataset_samples = yeast_by_dataset[dataset]
            if next_index < len(dataset_samples):
                interleaved_yeast.append(dataset_samples[next_index])
            if next_index + 1 < len(dataset_samples):
                remaining.append(dataset)
        ordered_datasets = remaining
        next_index += 1
    livecell_train_count = len(
        list((destination / "train").glob("livecell_train_*_masks.tif"))
    )
    non_yeast_count = livecell_train_count + len(non_yeast_train)
    maximum_yeast_train = int(non_yeast_count * 0.10 / 0.90)
    selected_yeast = set(interleaved_yeast[:maximum_yeast_train])

    selected_samples = [
        sample
        for sample in supported_samples
        if sample.split != "train"
        or sample.dataset not in yeast_datasets
        or sample in selected_yeast
    ]
    counts = {
        "train": 0,
        "checkpoint": 0,
        "calibration": 0,
        "ensemble_selection": 0,
        "final_test_excluded": 0,
        "omitted_yeast_train": len(interleaved_yeast) - len(selected_yeast),
        "omitted_ctc_sparse_gold": len(ctc_samples),
        "yeast_train_probability_ceiling_percent": 10,
    }
    materialized_per_dataset_role: dict[tuple[str, str], int] = {}
    for sample in tqdm(selected_samples, desc="materialize external Cellpose data"):
        if sample.split == "test":
            # Keep every source's official test data untouched.
            counts["final_test_excluded"] += 1
            continue
        output_split = (
            "train"
            if sample.split == "train"
            else validation_role(sample.dataset, sample.image_path)
        )
        count_key = (sample.dataset, output_split)
        if (
            limit_per_dataset_role is not None
            and materialized_per_dataset_role.get(count_key, 0)
            >= limit_per_dataset_role
        ):
            continue
        output_directory = destination / output_split
        output_directory.mkdir(parents=True, exist_ok=True)
        gray, labels = load_external_sample(sample)
        source_digest = hashlib.sha256(str(sample.image_path).encode()).hexdigest()[:10]
        stem = (
            f"{sample.dataset}_{sample.split}_{sample.image_path.stem}_{source_digest}"
        )
        cv2.imwrite(str(output_directory / f"{stem}_img.tif"), gray)
        cv2.imwrite(
            str(output_directory / f"{stem}_masks.tif"),
            labels.astype(np.uint16),
        )
        counts[output_split] += 1
        materialized_per_dataset_role[count_key] = (
            materialized_per_dataset_role.get(count_key, 0) + 1
        )
    return counts
