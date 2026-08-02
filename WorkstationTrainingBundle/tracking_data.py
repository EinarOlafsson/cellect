#!/usr/bin/env python3
"""Strict, acquisition-grouped transmitted-light cell-tracking data adapters.

The public API returns immutable :class:`TrackingSequence` objects.  Each sequence owns one
scientific role, a tuple of :class:`TrackingFrame` records linking raw images to dense instance
annotations or explicitly identified box-derived feature proxies,
per-frame :class:`FrameInstance` records, complete track lifetimes, and explicit parent/child
links.  Paths are retained for training, while SHA-256 digests and a path-independent sequence
fingerprint make the scientific input auditable.

Supported sources
-----------------

* Cell Tracking Challenge *training* archives BF-C2DL-HSC, BF-C2DL-MuSC, DIC-C2DH-HeLa,
  PhC-C2DH-U373, and PhC-C2DL-PSC.  Only ``NN_GT/TRA/man_track*.tif`` and the corresponding
  ``man_track.txt`` are parsed.  ``*_ST``, ``*_RES``, and any official test labels are never
  searched or opened.
* DeepSea's published tracker layout
  ``tracking_dataset/train/<set>/images|masks|labels``.  The adapter implements the exact
  ``*_cell_area_masked.png`` and ``*_cell_pos_labels.txt`` convention used by DeepSea's official
  ``BasicTrackerDataset``.  DeepSea's ``test`` directory is intentionally ignored.
* The public LiveCellTrack preview (Mendeley Data DOI ``10.17632/cgwcpz34mr.1``), using only
  ``Cell_imaging/{scratch_wound,HeLa}/data/<acquisition>/img1`` and the corresponding MOT
  ``gt/gt.txt``.  Bounding boxes provide identity, centers, and a deterministic elliptical shape
  proxy; they are not misrepresented as manually segmented masks.  No division lineage is
  inferred because the preview annotations do not contain parent IDs.
* CTMC-v1's official ``train`` partition (MOT boxes plus ``TRA/man_track.txt`` lineage).  Its
  hidden-label ``test`` partition is neither extracted nor parsed.  Box-derived ellipses are
  generated in memory when features are requested; the adapter never writes 80,389 proxy masks.
* ALFI Task 1 only (MI01--MI08 ``*_DTLTruth.csv`` and DIC PNG frames).  Task 2, phenotype CSVs,
  semantic masks, and ND2 videos are deliberately outside the tracking contract.

No permissive filename guessing is performed.  A different DeepSea export must be converted to
the documented layout or handled by a new, explicitly tested adapter.
"""

from __future__ import annotations

import argparse
import configparser
import csv
import hashlib
import io
import json
import os
import re
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np
import tifffile
from scipy.optimize import linear_sum_assignment


TRACKING_SCHEMA_VERSION = 4
TRACKING_ROLE_VERSION = "cellect-v4-tracking-acquisition-role-v1"
TRACKING_ROLES = ("train", "checkpoint", "calibration", "ensemble_selection")
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}

LIVECELLTRACK_PREVIEW_URL = (
    "https://data.mendeley.com/public-files/datasets/cgwcpz34mr/files/"
    "aa861330-bd1a-4811-adf6-741ab0a33d39/file_downloaded"
)
LIVECELLTRACK_PREVIEW_DOI = "10.17632/cgwcpz34mr.1"
LIVECELLTRACK_PREVIEW_LICENSE = "CC BY 4.0"
LIVECELLTRACK_PREVIEW_ARCHIVE_BYTES = 173_935_248
LIVECELLTRACK_PREVIEW_ARCHIVE_SHA256 = (
    "c3c824e3cb9db0673d84245ffcc9a5a85f9b1a8aafcf81d529195105a7cf3d7f"
)
LIVECELLTRACK_PREVIEW_DIRECTORY = "livecelltrack_preview"
LIVECELLTRACK_PREVIEW_DOMAINS: Mapping[str, dict[str, str]] = {
    "scratch_wound": {
        "dataset": "livecelltrack_scratch_wound",
        "modality": "transmitted-light live-cell microscopy",
        "organism": "scratch-wound cultured cells",
    },
    "HeLa": {
        "dataset": "livecelltrack_hela",
        "modality": "transmitted-light live-cell microscopy",
        "organism": "human HeLa cells",
    },
}
LIVECELLTRACK_PREVIEW_EXPECTED: Mapping[str, tuple[int, int]] = {
    "scratch_wound": (10, 100),
    "HeLa": (10, 50),
}
LIVECELLTRACK_SOURCE_FORMAT = "livecelltrack_mot_bbox_proxy_v1"
LIVECELLTRACK_PROXY_DIRECTORY = ".cellect_bbox_proxy_v1"
LIVECELLTRACK_EXTRACTION_MARKER = ".cellect_livecelltrack_preview_v1.json"

# DeepSea and the five CTC training sets carry the dense masks and lineage CellectTrack is built
# on, so they are never omittable.  These three contribute identity boxes or extra domains, and a
# run may proceed without one when its publisher is unreachable — only when the operator names it,
# and the omission travels with the tracking fingerprint and the preflight record.
OMITTABLE_TRACKING_SOURCES = ("alfi_task1", "ctmc_v1", "livecelltrack_preview")
OMIT_TRACKING_SOURCES_ENVIRONMENT_NAME = "CELLECT_OMIT_TRACKING_SOURCES"


def resolve_omitted_tracking_sources() -> frozenset[str]:
    """Read and validate the explicit tracking-source opt-out list."""
    configured = os.environ.get(OMIT_TRACKING_SOURCES_ENVIRONMENT_NAME, "")
    requested = {
        value.strip().casefold() for value in configured.split(",") if value.strip()
    }
    unknown = sorted(requested - set(OMITTABLE_TRACKING_SOURCES))
    if unknown:
        raise ValueError(
            f"{OMIT_TRACKING_SOURCES_ENVIRONMENT_NAME} lists sources that cannot be omitted: "
            + ", ".join(unknown)
            + f"; omittable sources are {', '.join(OMITTABLE_TRACKING_SOURCES)}"
        )
    return frozenset(requested)


CTMC_V1_URL = "https://motchallenge.net/data/CTMCV1.zip"
CTMC_V1_DIRECTORY = "ctmc_v1"
CTMC_V1_SOURCE_FORMAT = "ctmc_v1_mot_bbox_proxy_v1"
CTMC_V1_ARCHIVE_MARKER = ".cellect_ctmc_v1_archive"
CTMC_V1_EXTRACTION_MARKER = ".cellect_ctmc_v1_train.json"
CTMC_V1_LICENSE = "unknown; research permission required; do not redistribute"
CTMC_V1_EXPECTED = {
    "acquisitions": 47,
    "frames": 80_389,
    "tracks": 1_616,
    "boxes": 1_097_223,
}

ALFI_URL = "https://ndownloader.figshare.com/files/41740227"
ALFI_DOI = "10.6084/m9.figshare.23798451.v1"
ALFI_FILE_ID = 41_740_227
ALFI_ARCHIVE_BYTES = 8_423_073_056
ALFI_ARCHIVE_MD5 = "fe3326323c10b1748302e962eae26150"
ALFI_LICENSE = (
    "operationally CC BY with attribution: Figshare API reports CC0 1.0, "
    "but the official updated README states CC-BY"
)
ALFI_DIRECTORY = "alfi_task1"
ALFI_SOURCE_FORMAT = "alfi_task1_dtl_bbox_proxy_v1"
ALFI_EXTRACTION_MARKER = ".cellect_alfi_task1_v1.json"
ALFI_SEQUENCES = tuple(f"MI{index:02d}" for index in range(1, 9))
ALFI_EXPECTED = {
    "acquisitions": 8,
    "frames": 796,
    "published_cell_annotations": 16_564,
    "archive_dtl_rows": 16_627,
    "archive_unique_frame_identity_keys": 16_618,
    "tracks": 331,
}

CTC_TRACKING_DATASETS: Mapping[str, dict[str, str]] = {
    "ctc_bf_hsc": {
        "archive": "BF-C2DL-HSC",
        "modality": "brightfield",
        "organism": "mouse hematopoietic stem cells",
    },
    "ctc_bf_musc": {
        "archive": "BF-C2DL-MuSC",
        "modality": "brightfield",
        "organism": "mouse muscle stem cells",
    },
    "ctc_dic_hela": {
        "archive": "DIC-C2DH-HeLa",
        "modality": "differential interference contrast",
        "organism": "human HeLa cells",
    },
    "ctc_phc_u373": {
        "archive": "PhC-C2DH-U373",
        "modality": "phase contrast",
        "organism": "human U373 glioblastoma-astrocytoma cells",
    },
    "ctc_phc_psc": {
        "archive": "PhC-C2DL-PSC",
        "modality": "phase contrast",
        "organism": "pancreatic stem cells",
    },
}

DEEPSEA_TRACKING_ALIASES = {"track", "tracking", "tracking_dataset"}
DEEPSEA_IMAGE_DIRECTORY = "images"
DEEPSEA_MASK_DIRECTORY = "masks"
DEEPSEA_LABEL_DIRECTORY = "labels"
DEEPSEA_MASK_SUFFIX = "_cell_area_masked.png"
DEEPSEA_LABEL_SUFFIX = "_cell_pos_labels.txt"
# DeepSea position labels are integer-rounded component centroids, so a correct marker sits within
# rounding distance of its cell's centroid.  Measured over the 3,033 published training frames the
# worst optimal assignment is 2.2px; 4px keeps that headroom while still rejecting a marker that
# belongs to a different cell.
DEEPSEA_CENTROID_TOLERANCE_PIXELS = 4.0


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def file_sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_md5(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    """Return the source-published ALFI integrity digest (not a security identity)."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file(path: Path, *, expected_bytes: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != expected_bytes:
        raise RuntimeError(
            f"Verified download size mismatch for {path}: expected {expected_bytes}, found {size}"
        )
    digest = file_sha256(path)
    if digest != expected_sha256:
        raise RuntimeError(
            f"Verified download SHA-256 mismatch for {path}: expected {expected_sha256}, "
            f"found {digest}. Remove the rejected file and retry."
        )


def _verify_md5_file(path: Path, *, expected_bytes: int, expected_md5: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    size = path.stat().st_size
    if size != expected_bytes:
        raise RuntimeError(
            f"Verified download size mismatch for {path}: expected {expected_bytes}, found {size}"
        )
    digest = file_md5(path)
    if digest != expected_md5:
        raise RuntimeError(
            f"Verified download MD5 mismatch for {path}: expected {expected_md5}, found "
            f"{digest}. Remove the rejected file and retry."
        )


def _download_verified_file(
    destination: Path,
    *,
    url: str,
    expected_bytes: int,
    expected_sha256: str,
    allow_network: bool,
    opener=urllib.request.urlopen,
) -> Path:
    """Download one exact artifact; ``opener`` exists only for the no-network self-test."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _verify_file(
            destination,
            expected_bytes=expected_bytes,
            expected_sha256=expected_sha256,
        )
        return destination
    if not allow_network:
        raise FileNotFoundError(
            f"Verified LiveCellTrack preview archive is absent: {destination}"
        )

    partial = destination.with_suffix(destination.suffix + ".partial")
    existing = partial.stat().st_size if partial.exists() else 0
    if existing > expected_bytes:
        raise RuntimeError(
            f"Partial LiveCellTrack download is larger than the pinned artifact: {partial}"
        )
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Cellect-v4-data-preflight/1"},
    )
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    print(
        f"Downloading verified LiveCellTrack preview to {destination} "
        f"({expected_bytes:,} bytes)"
    )
    with opener(request, timeout=120) as response:
        status = getattr(response, "status", response.getcode())
        resumed = existing > 0 and status == 206
        if resumed:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {existing}-"):
                raise RuntimeError(
                    "LiveCellTrack server returned an incompatible Content-Range while resuming: "
                    f"{content_range!r}"
                )
        elif existing:
            # A server may legally ignore Range. Restart this generated partial file in place.
            existing = 0
        mode = "ab" if resumed else "wb"
        with partial.open(mode) as target:
            while True:
                chunk = response.read(4 * 1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())

    _verify_file(
        partial,
        expected_bytes=expected_bytes,
        expected_sha256=expected_sha256,
    )
    os.replace(partial, destination)
    return destination


def download_livecelltrack_preview_archive(
    data_root: Path,
    *,
    allow_network: bool = True,
) -> Path:
    """Fetch the exact public preview ZIP with resumable, fail-closed verification."""
    destination = (
        data_root.expanduser().resolve()
        / "external"
        / "livecelltrack_preview_v1.zip"
    )
    return _download_verified_file(
        destination,
        url=LIVECELLTRACK_PREVIEW_URL,
        expected_bytes=LIVECELLTRACK_PREVIEW_ARCHIVE_BYTES,
        expected_sha256=LIVECELLTRACK_PREVIEW_ARCHIVE_SHA256,
        allow_network=allow_network,
    )


def _safe_extract_livecelltrack_archive(archive_path: Path, destination: Path) -> Path:
    """Extract one verified ZIP atomically, rejecting traversal, links, and special files."""
    marker = destination / LIVECELLTRACK_EXTRACTION_MARKER
    if destination.exists():
        if not marker.is_file():
            raise RuntimeError(
                f"LiveCellTrack extraction directory exists without its verified marker: "
                f"{destination}"
            )
        try:
            marker_payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Invalid LiveCellTrack extraction marker: {marker}") from error
        if (
            marker_payload.get("archive_sha256")
            != LIVECELLTRACK_PREVIEW_ARCHIVE_SHA256
            or marker_payload.get("archive_bytes")
            != LIVECELLTRACK_PREVIEW_ARCHIVE_BYTES
        ):
            raise RuntimeError(
                f"LiveCellTrack extraction marker does not match the pinned preview: {marker}"
            )
        expected_root = destination / "Cell_imaging"
        if not expected_root.is_dir():
            raise RuntimeError(
                f"Verified LiveCellTrack extraction lost Cell_imaging/: {destination}"
            )
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".livecelltrack-extract-", dir=destination.parent
    ) as temporary:
        staged = Path(temporary) / destination.name
        staged.mkdir()
        staged_root = staged.resolve()
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            if not members or len(members) > 100_000:
                raise RuntimeError(
                    f"Implausible LiveCellTrack ZIP member count: {len(members)}"
                )
            total_uncompressed = sum(member.file_size for member in members)
            if total_uncompressed > 20 * 1024**3:
                raise RuntimeError(
                    "Refusing implausibly large LiveCellTrack uncompressed archive"
                )
            seen: set[PurePosixPath] = set()
            for member in members:
                if member.flag_bits & 0x1:
                    raise RuntimeError(
                        f"Encrypted LiveCellTrack ZIP member is unsupported: {member.filename}"
                    )
                if "\\" in member.filename:
                    raise RuntimeError(
                        f"Unsafe backslash in LiveCellTrack ZIP member: {member.filename}"
                    )
                relative = PurePosixPath(member.filename)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not relative.parts
                    or relative.parts[0].endswith(":")
                ):
                    raise RuntimeError(
                        f"Unsafe LiveCellTrack ZIP member: {member.filename}"
                    )
                if relative in seen:
                    raise RuntimeError(
                        f"Duplicate LiveCellTrack ZIP member: {member.filename}"
                    )
                seen.add(relative)
                unix_mode = member.external_attr >> 16
                file_type = stat.S_IFMT(unix_mode)
                if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise RuntimeError(
                        f"Refusing LiveCellTrack archive link/special file: {member.filename}"
                    )
                target = (staged / Path(*relative.parts)).resolve()
                if target != staged_root and staged_root not in target.parents:
                    raise RuntimeError(
                        f"LiveCellTrack ZIP member escapes extraction root: {member.filename}"
                    )
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member, "r") as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output, length=4 * 1024 * 1024)

        imaging_root = staged / "Cell_imaging"
        if not imaging_root.is_dir():
            raise RuntimeError(
                "Pinned LiveCellTrack preview does not contain its declared Cell_imaging/ root"
            )
        marker_payload = {
            "schema_version": 1,
            "archive_bytes": LIVECELLTRACK_PREVIEW_ARCHIVE_BYTES,
            "archive_sha256": LIVECELLTRACK_PREVIEW_ARCHIVE_SHA256,
            "doi": LIVECELLTRACK_PREVIEW_DOI,
            "license": LIVECELLTRACK_PREVIEW_LICENSE,
            "source_url": LIVECELLTRACK_PREVIEW_URL,
        }
        (staged / LIVECELLTRACK_EXTRACTION_MARKER).write_bytes(
            _canonical_json(marker_payload) + b"\n"
        )
        os.replace(staged, destination)
    return destination


def prepare_livecelltrack_preview(
    data_root: Path,
    *,
    allow_network: bool = True,
) -> Path:
    """Download, verify, and safely extract the pinned public preview under data/external."""
    resolved_data = data_root.expanduser().resolve()
    archive = download_livecelltrack_preview_archive(
        resolved_data, allow_network=allow_network
    )
    destination = resolved_data / "external" / LIVECELLTRACK_PREVIEW_DIRECTORY
    return _safe_extract_livecelltrack_archive(archive, destination)


def _download_verified_md5_file(
    destination: Path,
    *,
    url: str,
    expected_bytes: int,
    expected_md5: str,
    allow_network: bool,
    opener=urllib.request.urlopen,
) -> Path:
    """Resume one source-pinned size/MD5 artifact and reject all other bytes."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        _verify_md5_file(
            destination,
            expected_bytes=expected_bytes,
            expected_md5=expected_md5,
        )
        return destination
    if not allow_network:
        raise FileNotFoundError(f"Verified ALFI archive is absent: {destination}")
    partial = destination.with_suffix(destination.suffix + ".partial")
    existing = partial.stat().st_size if partial.exists() else 0
    if existing > expected_bytes:
        raise RuntimeError(f"Partial ALFI download exceeds the pinned artifact: {partial}")
    request = urllib.request.Request(
        url, headers={"User-Agent": "Cellect-v4-data-preflight/1"}
    )
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    print(f"Downloading verified ALFI Task-1 archive ({expected_bytes:,} bytes)")
    with opener(request, timeout=120) as response:
        final_url = str(getattr(response, "geturl", lambda: url)())
        if not final_url.casefold().startswith("https://"):
            raise RuntimeError(f"ALFI download left TLS: {final_url}")
        status = getattr(response, "status", response.getcode())
        resumed = existing > 0 and status == 206
        if resumed:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {existing}-"):
                raise RuntimeError(
                    f"ALFI resume returned incompatible Content-Range {content_range!r}"
                )
        elif existing:
            existing = 0
        with partial.open("ab" if resumed else "wb") as target:
            for chunk in iter(lambda: response.read(4 * 1024 * 1024), b""):
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
    _verify_md5_file(
        partial,
        expected_bytes=expected_bytes,
        expected_md5=expected_md5,
    )
    os.replace(partial, destination)
    return destination


def download_alfi_archive(data_root: Path, *, allow_network: bool = True) -> Path:
    override = os.environ.get("CELLECT_ALFI_ARCHIVE")
    destination = (
        Path(override).expanduser().resolve()
        if override
        else data_root.expanduser().resolve() / "external" / "ALFIdatasetFinal.zip"
    )
    return _download_verified_md5_file(
        destination,
        url=ALFI_URL,
        expected_bytes=ALFI_ARCHIVE_BYTES,
        expected_md5=ALFI_ARCHIVE_MD5,
        allow_network=allow_network,
    )


def _read_json_marker(path: Path, expected: Mapping[str, object], label: str) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid {label} marker: {path}") from error
    mismatches = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"{label} marker identity mismatch at {path}: {mismatches}")


def _write_immutable_json_marker(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _canonical_json(dict(payload)) + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"Refusing to replace an immutable data marker: {path}")
        return
    try:
        with path.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise RuntimeError(f"Concurrent immutable marker mismatch: {path}")


def _verify_or_record_ctmc_archive(
    archive: Path,
    *,
    provenance: str,
    marker_directory: Path | None = None,
    minimum_bytes: int = 100 * 1024 * 1024,
) -> dict[str, object]:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    size = archive.stat().st_size
    if size < minimum_bytes:
        raise RuntimeError(f"CTMC-v1 archive is implausibly small ({size:,} bytes): {archive}")
    digest = file_sha256(archive)
    marker_root = (
        marker_directory.expanduser().resolve()
        if marker_directory is not None
        else archive.parent.resolve()
    )
    path_digest = hashlib.sha256(str(archive.resolve()).encode("utf-8")).hexdigest()[:8]
    # The marker name is stable for one resolved source path.  Including the digest in the
    # filename would silently accept changed bytes by creating a second marker; a stable name
    # instead makes the first successful byte identity immutable and fail-closed.
    marker = marker_root / f"{CTMC_V1_ARCHIVE_MARKER}.{path_digest}.json"
    identity = {
        "schema_version": 1,
        "source_url": CTMC_V1_URL,
        "archive_name": archive.name,
        "archive_path": str(archive.resolve()),
        "archive_bytes": size,
        "archive_sha256": digest,
        "published_checksum_available": False,
        "license": CTMC_V1_LICENSE,
    }
    if marker.exists():
        _read_json_marker(marker, identity, "CTMC-v1 archive")
        payload = json.loads(marker.read_text(encoding="utf-8"))
    else:
        payload = {**identity, "provenance": provenance}
        try:
            _write_immutable_json_marker(marker, payload)
        except OSError as error:
            raise RuntimeError(
                f"Cannot write CTMC-v1 immutable provenance marker under {marker_root}; "
                "set CELLECT_CTMC_ROOT to an extracted directory or make the bundle cache "
                "writable"
            ) from error
    return payload


def download_ctmc_v1_archive(
    data_root: Path,
    *,
    allow_network: bool = True,
    opener=urllib.request.urlopen,
) -> Path:
    """Resume CTMC-v1 over TLS, then pin the first successful bytes by SHA-256."""
    override = os.environ.get("CELLECT_CTMC_ARCHIVE")
    destination = (
        Path(override).expanduser().resolve()
        if override
        else data_root.expanduser().resolve() / "external" / "CTMCV1.zip"
    )
    if destination.exists():
        _verify_or_record_ctmc_archive(
            destination,
            provenance="user_supplied_or_cached",
            marker_directory=data_root.expanduser().resolve() / "external" / ".ctmc_provenance",
        )
        return destination
    if not allow_network:
        raise FileNotFoundError(f"CTMC-v1 archive is absent: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".partial")
    existing = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(
        CTMC_V1_URL, headers={"User-Agent": "Cellect-v4-data-preflight/1"}
    )
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    print("Downloading CTMC-v1; the first complete TLS artifact will be pinned by SHA-256")
    with opener(request, timeout=120) as response:
        final_url = str(getattr(response, "geturl", lambda: CTMC_V1_URL)())
        if not final_url.casefold().startswith("https://"):
            raise RuntimeError(f"CTMC-v1 download left TLS: {final_url}")
        status = getattr(response, "status", response.getcode())
        resumed = existing > 0 and status == 206
        if resumed:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {existing}-"):
                raise RuntimeError(
                    f"CTMC-v1 resume returned incompatible Content-Range {content_range!r}"
                )
        elif existing:
            existing = 0
        with partial.open("ab" if resumed else "wb") as target:
            for chunk in iter(lambda: response.read(4 * 1024 * 1024), b""):
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
    if partial.stat().st_size < 100 * 1024 * 1024:
        raise RuntimeError(
            f"Downloaded CTMC-v1 archive is implausibly small: {partial.stat().st_size:,} bytes"
        )
    os.replace(partial, destination)
    _verify_or_record_ctmc_archive(
        destination,
        provenance="downloaded_over_tls",
        marker_directory=data_root.expanduser().resolve() / "external" / ".ctmc_provenance",
    )
    return destination


def _validated_zip_member(member: zipfile.ZipInfo, *, label: str) -> PurePosixPath:
    if member.flag_bits & 0x1:
        raise RuntimeError(f"Encrypted {label} ZIP member is unsupported: {member.filename}")
    if "\\" in member.filename:
        raise RuntimeError(f"Unsafe backslash in {label} ZIP member: {member.filename}")
    relative = PurePosixPath(member.filename)
    if (
        relative.is_absolute()
        or not relative.parts
        or ".." in relative.parts
        or relative.parts[0].endswith(":")
    ):
        raise RuntimeError(f"Unsafe {label} ZIP member: {member.filename}")
    unix_mode = member.external_attr >> 16
    if stat.S_IFMT(unix_mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise RuntimeError(f"Refusing {label} archive link/special file: {member.filename}")
    return relative


def _extract_selected_members(
    archive: zipfile.ZipFile,
    selected: Iterable[tuple[zipfile.ZipInfo, PurePosixPath]],
    staged: Path,
    *,
    label: str,
    maximum_uncompressed_bytes: int,
) -> int:
    selected_rows = list(selected)
    names = [relative for _, relative in selected_rows]
    if len(names) != len(set(names)):
        raise RuntimeError(f"{label} archive maps multiple members to one output path")
    total = sum(member.file_size for member, _ in selected_rows)
    if total <= 0 or total > maximum_uncompressed_bytes:
        raise RuntimeError(f"Implausible selected {label} payload: {total:,} bytes")
    staged_root = staged.resolve()
    for member, relative in selected_rows:
        target = (staged / Path(*relative.parts)).resolve()
        if target != staged_root and staged_root not in target.parents:
            raise RuntimeError(f"{label} member escapes extraction root: {member.filename}")
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member, "r") as source, target.open("xb") as output:
            # Reading each selected member to EOF makes zipfile verify its CRC.
            shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
    return total


def _safe_extract_ctmc_train(archive_path: Path, destination: Path) -> Path:
    archive_identity = _verify_or_record_ctmc_archive(
        archive_path,
        provenance="user_supplied_or_cached",
        marker_directory=destination.parent / ".ctmc_provenance",
    )
    marker = destination / CTMC_V1_EXTRACTION_MARKER
    expected_marker = {
        "schema_version": 1,
        "archive_bytes": archive_identity["archive_bytes"],
        "archive_sha256": archive_identity["archive_sha256"],
        "source_url": CTMC_V1_URL,
        "partition_extracted": "train_only",
        "official_test_labels_parsed": False,
        "license": CTMC_V1_LICENSE,
    }
    if destination.exists():
        if not marker.is_file():
            raise RuntimeError(f"CTMC-v1 extraction exists without marker: {destination}")
        _read_json_marker(marker, expected_marker, "CTMC-v1 extraction")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ctmc-v1-extract-", dir=destination.parent) as tmp:
        staged = Path(tmp) / destination.name
        staged.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            members: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
            seqinfo_prefixes: Counter[tuple[str, ...]] = Counter()
            validated: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
            for member in archive.infolist():
                relative = _validated_zip_member(member, label="CTMC-v1")
                validated.append((member, relative))
                parts = relative.parts
                if len(parts) >= 3 and parts[-1] == "seqinfo.ini" and "train" in parts:
                    train_index = parts.index("train")
                    if train_index + 2 == len(parts) - 1:
                        seqinfo_prefixes[tuple(parts[:train_index])] += 1
            valid_prefixes = [
                prefix
                for prefix, count in seqinfo_prefixes.items()
                if count == CTMC_V1_EXPECTED["acquisitions"]
            ]
            if len(valid_prefixes) != 1:
                raise RuntimeError(
                    "CTMC-v1 ZIP must expose exactly one train prefix with 47 seqinfo.ini "
                    f"files; found {dict(seqinfo_prefixes)}"
                )
            prefix = valid_prefixes[0]
            train_root = prefix + ("train",)
            for member, relative in validated:
                if relative.parts[: len(train_root)] != train_root:
                    continue
                stripped = PurePosixPath(*relative.parts[len(prefix) :])
                members.append((member, stripped))
            _extract_selected_members(
                archive,
                members,
                staged,
                label="CTMC-v1 train",
                maximum_uncompressed_bytes=100 * 1024**3,
            )
        if not (staged / "train").is_dir():
            raise RuntimeError("CTMC-v1 selective extraction lost train/")
        _write_immutable_json_marker(staged / CTMC_V1_EXTRACTION_MARKER, expected_marker)
        os.replace(staged, destination)
    return destination


def prepare_ctmc_v1(data_root: Path, *, allow_network: bool = True) -> Path:
    configured_root = os.environ.get("CELLECT_CTMC_ROOT")
    if configured_root:
        candidate = Path(configured_root).expanduser().resolve()
        if candidate.is_dir():
            return candidate
        if candidate.is_file() and candidate.suffix.casefold() == ".zip":
            archive = candidate
        else:
            raise FileNotFoundError(f"CELLECT_CTMC_ROOT is neither a directory nor ZIP: {candidate}")
    else:
        archive = download_ctmc_v1_archive(data_root, allow_network=allow_network)
    destination = data_root.expanduser().resolve() / "external" / CTMC_V1_DIRECTORY
    return _safe_extract_ctmc_train(archive, destination)


def _safe_extract_alfi_task1(archive_path: Path, destination: Path) -> Path:
    _verify_md5_file(
        archive_path,
        expected_bytes=ALFI_ARCHIVE_BYTES,
        expected_md5=ALFI_ARCHIVE_MD5,
    )
    archive_sha256 = file_sha256(archive_path)
    marker = destination / ALFI_EXTRACTION_MARKER
    expected_marker = {
        "schema_version": 1,
        "archive_bytes": ALFI_ARCHIVE_BYTES,
        "archive_md5": ALFI_ARCHIVE_MD5,
        "archive_sha256": archive_sha256,
        "file_id": ALFI_FILE_ID,
        "doi": ALFI_DOI,
        "license": ALFI_LICENSE,
        "partition_extracted": "Task1_MI01_MI08_images_and_DTLTruth_only",
        "task2_parsed": False,
        "semantic_masks_used_as_instances": False,
    }
    if destination.exists():
        if not marker.is_file():
            raise RuntimeError(f"ALFI extraction exists without marker: {destination}")
        _read_json_marker(marker, expected_marker, "ALFI extraction")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".alfi-task1-extract-", dir=destination.parent) as tmp:
        staged = Path(tmp) / destination.name
        staged.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            selected: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
            csv_names: set[str] = set()
            image_counts: Counter[str] = Counter()
            for member in archive.infolist():
                relative = _validated_zip_member(member, label="ALFI")
                parts = relative.parts
                if len(parts) < 3 or parts[0] != "Data&Annotations" or parts[1] not in ALFI_SEQUENCES:
                    continue
                sequence_id = parts[1]
                keep = False
                if len(parts) == 3 and parts[2] == f"{sequence_id}_DTLTruth.csv":
                    keep = True
                    csv_names.add(sequence_id)
                elif (
                    len(parts) == 4
                    and parts[2] == "Images"
                    and re.fullmatch(rf"I_{sequence_id}_\d{{4}}\.png", parts[3])
                ):
                    keep = True
                    image_counts[sequence_id] += 1
                if keep:
                    selected.append((member, relative))
            if csv_names != set(ALFI_SEQUENCES):
                raise RuntimeError(f"ALFI Task-1 ZIP is missing DTLTruth CSVs: {sorted(set(ALFI_SEQUENCES)-csv_names)}")
            if sum(image_counts.values()) != ALFI_EXPECTED["frames"]:
                raise RuntimeError(
                    f"ALFI Task-1 ZIP exposes {sum(image_counts.values())} strict PNG frames; "
                    f"expected {ALFI_EXPECTED['frames']}"
                )
            _extract_selected_members(
                archive,
                selected,
                staged,
                label="ALFI Task 1",
                maximum_uncompressed_bytes=20 * 1024**3,
            )
        _write_immutable_json_marker(staged / ALFI_EXTRACTION_MARKER, expected_marker)
        os.replace(staged, destination)
    return destination


def prepare_alfi_task1(data_root: Path, *, allow_network: bool = True) -> Path:
    configured_root = os.environ.get("CELLECT_ALFI_ROOT")
    if configured_root:
        candidate = Path(configured_root).expanduser().resolve()
        if candidate.is_dir():
            return candidate
        if candidate.is_file() and candidate.suffix.casefold() == ".zip":
            archive = candidate
        else:
            raise FileNotFoundError(f"CELLECT_ALFI_ROOT is neither a directory nor ZIP: {candidate}")
    else:
        archive = download_alfi_archive(data_root, allow_network=allow_network)
    destination = data_root.expanduser().resolve() / "external" / ALFI_DIRECTORY
    return _safe_extract_alfi_task1(archive, destination)


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _natural_key(value: str | Path) -> tuple[object, ...]:
    text = value.as_posix() if isinstance(value, Path) else value
    return tuple(
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", text)
    )


def acquisition_group(dataset: str, sequence_id: str) -> str:
    return f"{dataset.casefold()}/{sequence_id.casefold()}"


def acquisition_role(dataset: str, sequence_id: str) -> str:
    """Assign all frames in one acquisition to a stable 70/10/10/10 role."""
    group = acquisition_group(dataset, sequence_id)
    digest = hashlib.sha256(f"{TRACKING_ROLE_VERSION}:{group}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    if bucket < 70:
        return "train"
    if bucket < 80:
        return "checkpoint"
    if bucket < 90:
        return "calibration"
    return "ensemble_selection"


@dataclass(frozen=True)
class ParentLink:
    """A lineage edge from a parent track to a child first observed after the parent ends.

    A source that annotates every frame places the child in the very next frame.  A source that
    annotates on a staggered schedule can first observe the child later, and the edge is still the
    publisher's own lineage claim, so it is kept as recorded.  Training builds a division edge only
    when both endpoints are annotated and quarantines the event otherwise, so a later first
    observation never turns into an invented division.
    """

    parent_track_id: int
    child_track_id: int
    parent_end_frame: int
    child_start_frame: int

    def __post_init__(self) -> None:
        if self.parent_track_id <= 0 or self.child_track_id <= 0:
            raise ValueError("Parent and child track IDs must be positive")
        if self.parent_track_id == self.child_track_id:
            raise ValueError("A track cannot be its own parent")
        if self.child_start_frame <= self.parent_end_frame:
            raise ValueError("A lineage child must begin after its parent ends")


@dataclass(frozen=True)
class TrackLifetime:
    """One complete, contiguous track and its optional parent."""

    track_id: int
    source_label: str
    first_frame: int
    last_frame: int
    parent_track_id: int
    observed_frames: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.track_id <= 0 or self.parent_track_id < 0:
            raise ValueError("Track IDs must be positive and parent IDs non-negative")
        if self.track_id == self.parent_track_id:
            raise ValueError("A track cannot be its own parent")
        if self.first_frame > self.last_frame:
            raise ValueError("Track lifetime is reversed")
        if (
            not self.observed_frames
            or self.observed_frames[0] != self.first_frame
            or self.observed_frames[-1] != self.last_frame
            or tuple(sorted(set(self.observed_frames))) != self.observed_frames
        ):
            raise ValueError(
                f"Track {self.track_id} observed frames are not unique, ordered lifetime bounds"
            )
        if not self.source_label:
            raise ValueError("Track source label cannot be empty")


@dataclass(frozen=True)
class FrameInstance:
    """One tracked cell linked to a component in its source annotation or named proxy."""

    track_id: int
    source_label: str
    component_id: int
    parent_track_id: int
    centroid_x: float
    centroid_y: float
    area_pixels: int
    bbox_left: float | None = None
    bbox_top: float | None = None
    bbox_width: float | None = None
    bbox_height: float | None = None
    identity_supervision_available: bool = True

    def __post_init__(self) -> None:
        if self.track_id <= 0 or self.component_id <= 0:
            raise ValueError("Track and component IDs must be positive")
        if self.parent_track_id < 0 or self.parent_track_id == self.track_id:
            raise ValueError("Frame instance parent ID is invalid")
        if self.area_pixels <= 0:
            raise ValueError("Frame instance must occupy at least one pixel")
        if not np.isfinite(self.centroid_x) or not np.isfinite(self.centroid_y):
            raise ValueError("Frame instance centroid is not finite")
        box = (self.bbox_left, self.bbox_top, self.bbox_width, self.bbox_height)
        if any(value is not None for value in box):
            if not all(value is not None and np.isfinite(value) for value in box):
                raise ValueError("Frame instance bounding box must be fully specified and finite")
            if float(self.bbox_width) <= 0.0 or float(self.bbox_height) <= 0.0:
                raise ValueError("Frame instance bounding box dimensions must be positive")
        if not isinstance(self.identity_supervision_available, bool):
            raise ValueError("Identity supervision availability must be Boolean")


@dataclass(frozen=True)
class AnnotationExclusion:
    """Fail-closed quarantine of ambiguous identity events; no links are guessed."""

    source_label: str
    observation_count: int
    duplicate_frame_count: int
    reason: str

    def __post_init__(self) -> None:
        if (
            not self.source_label
            or self.observation_count <= 0
            or self.duplicate_frame_count <= 0
            or not self.reason
        ):
            raise ValueError("Tracking annotation exclusion is incomplete")


@dataclass(frozen=True)
class UnlabeledMaskRegion:
    """A mask component the publisher segmented but never gave a tracked identity.

    DeepSea's ``cell_area_masked`` frames cover every visible cell, while its position labels list
    only the cells the annotators tracked — a cell entering or leaving the field is commonly
    segmented without an identity.  Such a component is neither a detection nor background: it is
    withheld from tokenization and recorded here, so it can never become a phantom track or be
    scored as a miss.
    """

    frame_index: int
    component_id: int
    area_pixels: int
    touches_image_border: bool

    def __post_init__(self) -> None:
        if self.frame_index < 0 or self.component_id <= 0 or self.area_pixels <= 0:
            raise ValueError("Unlabeled mask region record is incomplete")


@dataclass(frozen=True)
class SourceFileExclusion:
    """Hash-bound source file intentionally excluded from this sequence's supervision."""

    relative_path: str
    sha256: str
    reason: str

    def __post_init__(self) -> None:
        if (
            not self.relative_path
            or Path(self.relative_path).is_absolute()
            or ".." in Path(self.relative_path).parts
            or re.fullmatch(r"[0-9a-f]{64}", self.sha256) is None
            or not self.reason
        ):
            raise ValueError("Tracking source-file exclusion is incomplete")


@dataclass(frozen=True)
class TrackingFrame:
    """One complete image/instance-feature pair in a tracking acquisition."""

    frame_index: int
    image_path: Path
    instance_path: Path
    label_path: Path
    image_sha256: str
    instance_sha256: str
    label_sha256: str
    height: int
    width: int
    image_dtype: str
    instance_dtype: str
    instances: tuple[FrameInstance, ...]
    # Components present in the mask that carry no tracked identity.  Consumers build detections
    # from ``instances`` only; these ids identify the pixels that must stay unscored.
    unlabeled_component_ids: tuple[int, ...] = ()

    @property
    def instance_count(self) -> int:
        return len(self.instances)

    @property
    def track_ids(self) -> tuple[int, ...]:
        return tuple(instance.track_id for instance in self.instances)


@dataclass(frozen=True)
class TrackingSequence:
    """Audited temporal acquisition; this is the primary ingestion return type."""

    schema_version: int
    dataset: str
    sequence_id: str
    acquisition_group: str
    role: str
    modality: str
    organism: str
    source_format: str
    source_partition: str
    root: Path
    frames: tuple[TrackingFrame, ...]
    tracks: tuple[TrackLifetime, ...]
    parent_links: tuple[ParentLink, ...]
    lineage_sha256: str
    sequence_fingerprint_sha256: str
    annotation_exclusions: tuple[AnnotationExclusion, ...] = ()
    source_file_exclusions: tuple[SourceFileExclusion, ...] = ()
    division_quarantined_parent_track_ids: tuple[int, ...] = ()
    event_quarantined_track_frames: tuple[tuple[int, int], ...] = ()
    unlabeled_mask_regions: tuple[UnlabeledMaskRegion, ...] = ()
    # Spacing between the publisher's own frame numbers.  Frames are addressed by position, so a
    # stride above one means the acquisition was published at a lower effective frame rate.
    frame_index_stride: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != TRACKING_SCHEMA_VERSION:
            raise ValueError("Tracking sequence schema version mismatch")
        if self.role not in TRACKING_ROLES:
            raise ValueError(f"Unknown tracking role: {self.role}")
        if self.acquisition_group != acquisition_group(self.dataset, self.sequence_id):
            raise ValueError("Tracking acquisition group is not canonical")
        if not self.frames or not self.tracks:
            raise ValueError("Tracking sequence contains no frames or tracks")
        frame_indices = tuple(frame.frame_index for frame in self.frames)
        expected = tuple(range(frame_indices[0], frame_indices[-1] + 1))
        if frame_indices != expected:
            raise ValueError("Tracking sequence frame indices are not contiguous")
        if any(frame.image_path == frame.instance_path for frame in self.frames):
            raise ValueError("Tracking image and instance paths must be distinct")
        track_ids = {track.track_id for track in self.tracks}
        if not set(self.division_quarantined_parent_track_ids) <= track_ids:
            raise ValueError("Division quarantine refers to an unknown track")
        if any(track_id not in track_ids for track_id, _ in self.event_quarantined_track_frames):
            raise ValueError("Event quarantine refers to an unknown track")
        exclusion_paths = [value.relative_path for value in self.source_file_exclusions]
        if len(exclusion_paths) != len(set(exclusion_paths)):
            raise ValueError("Tracking source-file exclusions contain duplicate paths")

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def annotated_instance_frames(self) -> int:
        return sum(frame.instance_count for frame in self.frames)


def _sequence_fingerprint_payload(
    dataset: str,
    sequence_id: str,
    role: str,
    source_format: str,
    frames: Iterable[TrackingFrame],
    tracks: Iterable[TrackLifetime],
    parent_links: Iterable[ParentLink],
    lineage_sha256: str,
    annotation_exclusions: Iterable[AnnotationExclusion] = (),
    source_file_exclusions: Iterable[SourceFileExclusion] = (),
    division_quarantined_parent_track_ids: Iterable[int] = (),
    event_quarantined_track_frames: Iterable[tuple[int, int]] = (),
    frame_index_stride: int = 1,
) -> dict[str, object]:
    return {
        "schema_version": TRACKING_SCHEMA_VERSION,
        # Only recorded for a sub-sampled acquisition, so evenly numbered sources keep the
        # identity they had before published-order addressing existed.
        **({"frame_index_stride": frame_index_stride} if frame_index_stride != 1 else {}),
        "role_version": TRACKING_ROLE_VERSION,
        "dataset": dataset,
        "sequence_id": sequence_id,
        "role": role,
        "source_format": source_format,
        "lineage_sha256": lineage_sha256,
        "frames": [
            {
                "frame_index": frame.frame_index,
                "image_name": frame.image_path.name,
                "instance_name": frame.instance_path.name,
                "label_name": frame.label_path.name,
                "image_sha256": frame.image_sha256,
                "instance_sha256": frame.instance_sha256,
                "label_sha256": frame.label_sha256,
                "height": frame.height,
                "width": frame.width,
                "instances": [asdict(instance) for instance in frame.instances],
                # Absent unless the publisher segmented an untracked cell, so sequences without
                # one keep the identity they had before this record existed.
                **(
                    {"unlabeled_component_ids": list(frame.unlabeled_component_ids)}
                    if frame.unlabeled_component_ids
                    else {}
                ),
            }
            for frame in frames
        ],
        "tracks": [asdict(track) for track in tracks],
        "parent_links": [asdict(link) for link in parent_links],
        "annotation_exclusions": [
            asdict(exclusion) for exclusion in annotation_exclusions
        ],
        "source_file_exclusions": [
            asdict(exclusion) for exclusion in source_file_exclusions
        ],
        "division_quarantined_parent_track_ids": sorted(
            division_quarantined_parent_track_ids
        ),
        "event_quarantined_track_frames": sorted(event_quarantined_track_frames),
    }


def _image_files(directory: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file()
            and not path.name.startswith(".")
            and path.suffix.casefold() in IMAGE_SUFFIXES
        ),
        key=_natural_key,
    )


def _index_numbered_files(
    paths: Iterable[Path],
    pattern: re.Pattern[str],
    label: str,
) -> dict[int, Path]:
    indexed: dict[int, Path] = {}
    rejected: list[str] = []
    for path in paths:
        match = pattern.fullmatch(path.name)
        if match is None:
            rejected.append(path.name)
            continue
        frame_index = int(match.group(1))
        previous = indexed.get(frame_index)
        if previous is not None:
            raise RuntimeError(
                f"Duplicate {label} frame {frame_index}: {previous.name}, {path.name}"
            )
        indexed[frame_index] = path
    if rejected:
        raise RuntimeError(
            f"Unexpected files in strict {label} directory: {rejected[:8]}"
        )
    if not indexed:
        raise RuntimeError(f"No strict {label} frames were found")
    indices = sorted(indexed)
    if indices != list(range(indices[0], indices[-1] + 1)):
        missing = sorted(set(range(indices[0], indices[-1] + 1)) - set(indices))
        raise RuntimeError(f"{label} frame sequence has gaps: {missing[:12]}")
    return indexed


def _read_ctc_tiff(path: Path, label: str) -> np.ndarray:
    try:
        array = np.asarray(tifffile.imread(path))
    except Exception as error:
        # LiveCellTrack's HeLa acquisitions are chroma-subsampled TIFFs that tifffile declines to
        # decode.  OpenCV reads them losslessly, and its channel order does not matter because a
        # multi-channel frame is only accepted below when every channel is identical.
        decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise RuntimeError(
                f"Could not decode CTC {label} TIFF {path}: {error}"
            ) from error
        array = np.asarray(decoded)
    if array.ndim == 3:
        # LiveCellTrack's preview stores its transmitted-light frames as RGB with all three
        # channels byte-identical.  Taking one channel is lossless there, so it is done rather
        # than rejecting the acquisition; genuinely coloured data would need a stated conversion
        # and still fails here.
        if array.shape[2] and all(
            np.array_equal(array[..., 0], array[..., channel])
            for channel in range(1, array.shape[2])
        ):
            array = array[..., 0]
        else:
            raise RuntimeError(
                f"CTC 2-D {label} is multi-channel with differing channels, which has no "
                f"declared grayscale reduction: {array.shape}: {path}"
            )
    if array.ndim != 2:
        raise RuntimeError(f"CTC 2-D {label} must have shape HxW, found {array.shape}: {path}")
    return array


def _read_deepsea_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim not in (2, 3):
        raise RuntimeError(f"Could not decode DeepSea image {path}")
    return image


def _read_deepsea_binary_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Could not decode DeepSea mask {path}")
    if mask.ndim == 3:
        if not all(np.array_equal(mask[..., 0], mask[..., channel]) for channel in range(1, mask.shape[2])):
            raise RuntimeError(
                f"DeepSea RGB mask channels differ; conversion is ambiguous: {path}"
            )
        mask = mask[..., 0]
    if mask.ndim != 2:
        raise RuntimeError(f"DeepSea mask must have shape HxW, found {mask.shape}: {path}")
    return mask > 0


def _component_geometry(labels: np.ndarray, component_id: int) -> tuple[float, float, int]:
    y, x = np.nonzero(labels == component_id)
    if not len(x):
        raise RuntimeError(f"Connected component {component_id} has zero pixels")
    return float(x.mean()), float(y.mean()), int(len(x))


def _bbox_proxy_labels(
    shape: tuple[int, int],
    boxes: Iterable[tuple[int, str, float, float, float, float]],
    *,
    context: str,
) -> np.ndarray:
    """Rasterize deterministic inscribed ellipses for box-only feature extraction."""
    height, width = shape
    labels = np.zeros((height, width), dtype=np.uint32)
    best_distance = np.full((height, width), np.inf, dtype=np.float32)
    box_rows = sorted(boxes, key=lambda row: row[0])
    for component_id, source_label, left, top, box_width, box_height in box_rows:
        centroid_x = left + 0.5 * box_width
        centroid_y = top + 0.5 * box_height
        if not (0.0 <= centroid_x < width and 0.0 <= centroid_y < height):
            raise RuntimeError(
                f"{context} box center for {source_label!r} is outside {width}x{height}"
            )
        x0 = max(0, int(np.floor(left)))
        y0 = max(0, int(np.floor(top)))
        x1 = min(width, int(np.ceil(left + box_width)))
        y1 = min(height, int(np.ceil(top + box_height)))
        if x0 >= x1 or y0 >= y1:
            raise RuntimeError(f"{context} box for {source_label!r} has no image overlap")
        xx = np.arange(x0, x1, dtype=np.float32) + 0.5
        yy = np.arange(y0, y1, dtype=np.float32) + 0.5
        dx = (xx - centroid_x) / max(0.5 * box_width, 1e-6)
        dy = (yy - centroid_y) / max(0.5 * box_height, 1e-6)
        distance = dy[:, None] ** 2 + dx[None, :] ** 2
        inside = distance <= 1.0
        region_best = best_distance[y0:y1, x0:x1]
        update = inside & (distance < region_best)
        region_best[update] = distance[update]
        labels[y0:y1, x0:x1][update] = component_id
    retained = {int(value) for value in np.unique(labels) if value > 0}
    missing = [row[1] for row in box_rows if row[0] not in retained]
    if missing:
        raise RuntimeError(f"{context} proxy retained no pixels for {missing[:12]}")
    return labels


def bbox_proxy_labels(frame: TrackingFrame) -> np.ndarray:
    """Generate a box-only frame proxy in memory; no derived mask is written to disk."""
    boxes: list[tuple[int, str, float, float, float, float]] = []
    for instance in frame.instances:
        if None in (
            instance.bbox_left,
            instance.bbox_top,
            instance.bbox_width,
            instance.bbox_height,
        ):
            raise RuntimeError(
                f"Frame {frame.frame_index} does not carry a complete box-proxy contract"
            )
        boxes.append(
            (
                instance.component_id,
                instance.source_label,
                float(instance.bbox_left),
                float(instance.bbox_top),
                float(instance.bbox_width),
                float(instance.bbox_height),
            )
        )
    return _bbox_proxy_labels(
        (frame.height, frame.width), boxes, context=str(frame.image_path)
    )


def _validate_single_component(mask: np.ndarray, track_id: int, context: str) -> None:
    component_count, _ = cv2.connectedComponents(
        np.ascontiguousarray(mask == track_id, dtype=np.uint8), connectivity=8
    )
    if component_count != 2:
        raise RuntimeError(
            f"Track {track_id} occupies {component_count - 1} disconnected components in {context}"
        )


def _parse_ctc_lineage(path: Path) -> dict[int, TrackLifetime]:
    records: dict[int, tuple[int, int, int]] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 4:
            raise RuntimeError(
                f"CTC lineage {path}:{line_number} must contain ID begin end parent"
            )
        try:
            track_id, begin, end, parent = (int(field) for field in fields)
        except ValueError as error:
            raise RuntimeError(
                f"CTC lineage {path}:{line_number} contains a non-integer"
            ) from error
        if track_id <= 0 or begin < 0 or end < begin or parent < 0:
            raise RuntimeError(f"CTC lineage {path}:{line_number} has invalid values")
        if track_id == parent or track_id in records:
            raise RuntimeError(f"CTC lineage {path}:{line_number} has duplicate/self lineage")
        records[track_id] = (begin, end, parent)
    if not records:
        raise RuntimeError(f"CTC lineage file is empty: {path}")
    for track_id, (begin, _, parent) in records.items():
        if parent == 0:
            continue
        if parent not in records:
            raise RuntimeError(f"CTC track {track_id} refers to missing parent {parent}")
        parent_end = records[parent][1]
        # CTC gold tracking annotates selected cells and frames, so a daughter is sometimes first
        # segmented a few frames after its parent's last annotated frame.  Ordering is what makes
        # the lineage coherent; immediate adjacency is not part of the published contract.
        # Training builds a division edge only when both endpoints are annotated, so a delayed
        # first observation cannot turn into an invented division.
        if begin <= parent_end:
            raise RuntimeError(
                f"CTC child {track_id} begins at {begin}, which is not after parent {parent} "
                f"ends at {parent_end}"
            )
    return {
        track_id: TrackLifetime(
            track_id=track_id,
            source_label=str(track_id),
            first_frame=begin,
            last_frame=end,
            parent_track_id=parent,
            observed_frames=tuple(range(begin, end + 1)),
        )
        for track_id, (begin, end, parent) in records.items()
    }


def _validate_track_graph(tracks: Mapping[int, TrackLifetime]) -> None:
    for track_id in tracks:
        visited: set[int] = set()
        cursor = track_id
        while cursor:
            if cursor in visited:
                raise RuntimeError(f"Tracking lineage contains a cycle at track {cursor}")
            visited.add(cursor)
            cursor = tracks[cursor].parent_track_id


def _locate_ctc_training_root(root: Path, dataset: str) -> Path:
    if dataset not in CTC_TRACKING_DATASETS:
        raise ValueError(f"Unsupported CTC tracking dataset: {dataset}")
    resolved = root.resolve()
    if resolved.is_file() and resolved.suffix.casefold() == ".zip":
        raise RuntimeError(
            f"CTC tracking requires an extracted training archive, not {resolved}. "
            "Use the existing safe_extract step, then pass its dataset root."
        )
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    archive_name = CTC_TRACKING_DATASETS[dataset]["archive"]
    candidates = []
    if resolved.name == archive_name:
        candidates.append(resolved)
    candidates.extend(
        path for path in resolved.rglob(archive_name) if path.is_dir()
    )
    unique = sorted(set(candidates), key=_natural_key)
    if len(unique) != 1:
        raise RuntimeError(
            f"Expected one extracted CTC training directory named {archive_name} under "
            f"{resolved}, found {[str(path) for path in unique[:8]]}. Official test archives "
            "are not accepted by this adapter."
        )
    return unique[0]


def _ctc_sequence_from_tra(
    archive_root: Path,
    dataset: str,
    tra_directory: Path,
) -> TrackingSequence:
    gt_directory = tra_directory.parent
    if tra_directory.name != "TRA" or not gt_directory.name.endswith("_GT"):
        raise RuntimeError(f"Not a strict CTC training TRA directory: {tra_directory}")
    sequence_id = gt_directory.name.removesuffix("_GT")
    if not re.fullmatch(r"\d+", sequence_id):
        raise RuntimeError(f"CTC sequence ID must be numeric: {gt_directory.name}")
    image_directory = archive_root / sequence_id
    if not image_directory.is_dir():
        raise RuntimeError(
            f"CTC tracking ground truth has no raw sequence directory: {image_directory}"
        )
    lineage_path = tra_directory / "man_track.txt"
    if not lineage_path.is_file():
        raise RuntimeError(f"CTC TRA directory has no man_track.txt: {tra_directory}")

    image_paths = _index_numbered_files(
        _image_files(image_directory),
        re.compile(r"t(\d+)\.tiff?", flags=re.I),
        f"CTC {dataset}/{sequence_id} image",
    )
    mask_paths = _index_numbered_files(
        sorted(tra_directory.glob("man_track*.tif*"), key=_natural_key),
        re.compile(r"man_track(\d+)\.tiff?", flags=re.I),
        f"CTC {dataset}/{sequence_id} tracking mask",
    )
    if set(image_paths) != set(mask_paths):
        raise RuntimeError(
            f"CTC {dataset}/{sequence_id} is incomplete: images without masks="
            f"{sorted(set(image_paths) - set(mask_paths))[:12]}, masks without images="
            f"{sorted(set(mask_paths) - set(image_paths))[:12]}"
        )

    tracks_by_id = _parse_ctc_lineage(lineage_path)
    _validate_track_graph(tracks_by_id)
    observed: dict[int, list[int]] = {track_id: [] for track_id in tracks_by_id}
    decoded: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    sequence_shape: tuple[int, int] | None = None
    for frame_index in sorted(image_paths):
        image = _read_ctc_tiff(image_paths[frame_index], "image")
        mask = _read_ctc_tiff(mask_paths[frame_index], "tracking mask")
        if not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0):
            raise RuntimeError(
                f"CTC tracking mask must contain non-negative integer IDs: "
                f"{mask_paths[frame_index]}"
            )
        if image.shape != mask.shape:
            raise RuntimeError(
                f"CTC image/mask shape mismatch at frame {frame_index}: "
                f"{image.shape} versus {mask.shape}"
            )
        if sequence_shape is None:
            sequence_shape = image.shape
        elif image.shape != sequence_shape:
            raise RuntimeError(f"CTC sequence changes shape at frame {frame_index}")
        frame_track_ids = sorted(int(value) for value in np.unique(mask) if value > 0)
        for track_id in frame_track_ids:
            if track_id not in tracks_by_id:
                raise RuntimeError(
                    f"CTC mask frame {frame_index} contains undeclared track {track_id}"
                )
            _validate_single_component(
                mask, track_id, f"CTC {dataset}/{sequence_id} frame {frame_index}"
            )
            observed[track_id].append(frame_index)
        decoded[frame_index] = (image, mask)

    declared_frames = set()
    for track_id, track in tracks_by_id.items():
        expected = tuple(range(track.first_frame, track.last_frame + 1))
        actual = tuple(observed[track_id])
        if actual != expected:
            raise RuntimeError(
                f"CTC track {track_id} observed in {actual}, declared for {expected}"
            )
        declared_frames.update(expected)
    if not declared_frames <= set(image_paths):
        raise RuntimeError("CTC lineage declares frames outside the raw sequence")

    lineage_sha256 = file_sha256(lineage_path)
    frames: list[TrackingFrame] = []
    for frame_index in sorted(image_paths):
        image, mask = decoded[frame_index]
        instances: list[FrameInstance] = []
        for track_id in sorted(int(value) for value in np.unique(mask) if value > 0):
            centroid_x, centroid_y, area = _component_geometry(mask, track_id)
            instances.append(
                FrameInstance(
                    track_id=track_id,
                    source_label=str(track_id),
                    component_id=track_id,
                    parent_track_id=tracks_by_id[track_id].parent_track_id,
                    centroid_x=centroid_x,
                    centroid_y=centroid_y,
                    area_pixels=area,
                )
            )
        frames.append(
            TrackingFrame(
                frame_index=frame_index,
                image_path=image_paths[frame_index].resolve(),
                instance_path=mask_paths[frame_index].resolve(),
                label_path=lineage_path.resolve(),
                image_sha256=file_sha256(image_paths[frame_index]),
                instance_sha256=file_sha256(mask_paths[frame_index]),
                label_sha256=lineage_sha256,
                height=int(image.shape[0]),
                width=int(image.shape[1]),
                image_dtype=str(image.dtype),
                instance_dtype=str(mask.dtype),
                instances=tuple(instances),
            )
        )

    tracks = tuple(tracks_by_id[key] for key in sorted(tracks_by_id))
    parent_links = tuple(
        ParentLink(
            parent_track_id=track.parent_track_id,
            child_track_id=track.track_id,
            parent_end_frame=tracks_by_id[track.parent_track_id].last_frame,
            child_start_frame=track.first_frame,
        )
        for track in tracks
        if track.parent_track_id > 0
    )
    role = acquisition_role(dataset, sequence_id)
    fingerprint = _fingerprint(
        _sequence_fingerprint_payload(
            dataset,
            sequence_id,
            role,
            "ctc_tra_v1",
            frames,
            tracks,
            parent_links,
            lineage_sha256,
        )
    )
    details = CTC_TRACKING_DATASETS[dataset]
    return TrackingSequence(
        schema_version=TRACKING_SCHEMA_VERSION,
        dataset=dataset,
        sequence_id=sequence_id,
        acquisition_group=acquisition_group(dataset, sequence_id),
        role=role,
        modality=details["modality"],
        organism=details["organism"],
        source_format="ctc_tra_v1",
        source_partition="official_training_ground_truth",
        root=archive_root.resolve(),
        frames=tuple(frames),
        tracks=tracks,
        parent_links=parent_links,
        lineage_sha256=lineage_sha256,
        sequence_fingerprint_sha256=fingerprint,
    )


def discover_ctc_training_tracking(root: Path, dataset: str) -> tuple[TrackingSequence, ...]:
    """Discover CTC training TRA sequences without consulting any test-result directory."""
    archive_root = _locate_ctc_training_root(root, dataset)
    lineage_paths = sorted(
        (
            path
            for path in archive_root.glob("*_GT/TRA/man_track.txt")
            if path.parent.name == "TRA" and path.parent.parent.name.endswith("_GT")
        ),
        key=_natural_key,
    )
    if not lineage_paths:
        raise RuntimeError(
            f"No CTC training lineage files matching *_GT/TRA/man_track.txt under "
            f"{archive_root}; *_ST and *_RES are intentionally unsupported"
        )
    sequences = tuple(
        _ctc_sequence_from_tra(archive_root, dataset, path.parent)
        for path in lineage_paths
    )
    groups = [sequence.acquisition_group for sequence in sequences]
    if len(groups) != len(set(groups)):
        raise RuntimeError(f"Duplicate CTC acquisition groups under {archive_root}")
    return sequences


@dataclass(frozen=True)
class _BoxRow:
    frame_index: int
    source_label: str
    left: float
    top: float
    width: float
    height: float
    parent_source_labels: tuple[str, ...] = ()


def _read_tracking_image(path: Path, *, label: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim not in (2, 3):
        raise RuntimeError(f"Could not decode {label} image {path}")
    return image


def _locate_ctmc_v1_train_root(root: Path) -> Path:
    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    candidates: list[Path] = []
    if resolved.name == "train":
        candidates.append(resolved)
    if (resolved / "train").is_dir():
        candidates.append(resolved / "train")
    candidates = sorted(set(candidates), key=_natural_key)
    if len(candidates) != 1:
        raise RuntimeError(
            f"CTMC-v1 requires exactly one extracted train/ directory under {resolved}; "
            f"found {[str(path) for path in candidates]}. The test partition is unsupported."
        )
    return candidates[0]


def _parse_ctmc_mot_gt(path: Path, *, sequence_length: int) -> tuple[_BoxRow, ...]:
    rows: list[_BoxRow] = []
    identities: set[tuple[int, int]] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for line_number, fields in enumerate(csv.reader(handle), start=1):
            if not fields or all(not field.strip() for field in fields):
                continue
            if len(fields) != 10:
                raise RuntimeError(
                    f"CTMC-v1 MOT row {path}:{line_number} must contain exactly 10 columns"
                )
            try:
                values = tuple(float(field.strip()) for field in fields)
            except ValueError as error:
                raise RuntimeError(
                    f"CTMC-v1 MOT row {path}:{line_number} contains a non-number"
                ) from error
            if not np.isfinite(values).all() or not values[0].is_integer() or not values[1].is_integer():
                raise RuntimeError(f"CTMC-v1 MOT frame/ID is invalid at {path}:{line_number}")
            frame_index, track_id = int(values[0]), int(values[1])
            left, top, width, height = values[2:6]
            if not 1 <= frame_index <= sequence_length or track_id <= 0:
                raise RuntimeError(f"CTMC-v1 MOT frame/ID is out of range at {path}:{line_number}")
            if width <= 0.0 or height <= 0.0:
                raise RuntimeError(f"CTMC-v1 MOT box is empty at {path}:{line_number}")
            identity = (frame_index, track_id)
            if identity in identities:
                raise RuntimeError(f"Duplicate CTMC-v1 frame/ID {identity} in {path}")
            identities.add(identity)
            rows.append(
                _BoxRow(
                    frame_index=frame_index,
                    source_label=str(track_id),
                    left=left,
                    top=top,
                    width=width,
                    height=height,
                )
            )
    if not rows:
        raise RuntimeError(f"CTMC-v1 ground truth is empty: {path}")
    return tuple(sorted(rows, key=lambda row: (row.frame_index, int(row.source_label))))


def _ctmc_cell_line(sequence_id: str) -> str:
    if not re.fullmatch(r".+-[^-]+", sequence_id):
        raise RuntimeError(f"CTMC-v1 sequence name lacks a run suffix: {sequence_id}")
    return sequence_id.rsplit("-", 1)[0]


def _ctmc_v1_sequence(train_root: Path, sequence_root: Path) -> TrackingSequence:
    sequence_id = sequence_root.name
    seqinfo_path = sequence_root / "seqinfo.ini"
    gt_path = sequence_root / "gt" / "gt.txt"
    lineage_path = sequence_root / "TRA" / "man_track.txt"
    if not seqinfo_path.is_file() or not gt_path.is_file() or not lineage_path.is_file():
        raise RuntimeError(
            f"Incomplete CTMC-v1 train sequence {sequence_root}: expected seqinfo.ini, "
            "gt/gt.txt, and TRA/man_track.txt"
        )
    parser = configparser.ConfigParser()
    try:
        parser.read(seqinfo_path, encoding="utf-8-sig")
        section = parser["Sequence"]
        declared_name = section.get("name", sequence_id)
        image_directory_name = section["imDir"]
        frame_rate = int(section["frameRate"])
        sequence_length = int(section["seqLength"])
        image_width = int(section["imWidth"])
        image_height = int(section["imHeight"])
        image_extension = section["imExt"]
    except (KeyError, ValueError, configparser.Error) as error:
        raise RuntimeError(f"Invalid CTMC-v1 seqinfo.ini: {seqinfo_path}") from error
    if (
        declared_name != sequence_id
        or image_directory_name != "img1"
        or frame_rate <= 0
        or sequence_length <= 0
        or image_width <= 0
        or image_height <= 0
        or not re.fullmatch(r"\.[A-Za-z0-9]+", image_extension)
    ):
        raise RuntimeError(f"CTMC-v1 seqinfo contract is invalid: {seqinfo_path}")
    image_directory = sequence_root / image_directory_name
    expected_names = {
        index: image_directory / f"{index:06d}{image_extension}"
        for index in range(1, sequence_length + 1)
    }
    missing = [index for index, path in expected_names.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"CTMC-v1 {sequence_id} is missing image frames {missing[:12]}")
    unexpected = [
        path.name
        for path in _image_files(image_directory)
        if path not in set(expected_names.values())
    ]
    if unexpected:
        raise RuntimeError(f"CTMC-v1 {sequence_id} has unexpected images {unexpected[:12]}")
    rows = _parse_ctmc_mot_gt(gt_path, sequence_length=sequence_length)
    tracks_by_id = _parse_ctc_lineage(lineage_path)
    _validate_track_graph(tracks_by_id)
    rows_by_frame: dict[int, list[_BoxRow]] = {}
    observed_by_track: dict[int, list[int]] = {track_id: [] for track_id in tracks_by_id}
    for row in rows:
        track_id = int(row.source_label)
        if track_id not in tracks_by_id:
            raise RuntimeError(f"CTMC-v1 box refers to undeclared TRA track {track_id}")
        rows_by_frame.setdefault(row.frame_index, []).append(row)
        observed_by_track[track_id].append(row.frame_index)
    for track_id, track in tracks_by_id.items():
        actual = tuple(observed_by_track[track_id])
        expected = tuple(range(track.first_frame, track.last_frame + 1))
        if actual != expected:
            raise RuntimeError(
                f"CTMC-v1 {sequence_id} track {track_id} boxes {actual[:8]} do not match "
                f"TRA lifetime {expected[:8]}"
            )
    gt_sha256 = file_sha256(gt_path)
    lineage_sha256 = file_sha256(lineage_path)
    seqinfo_sha256 = file_sha256(seqinfo_path)
    frames: list[TrackingFrame] = []
    for frame_index, image_path in expected_names.items():
        image = _read_tracking_image(image_path, label="CTMC-v1")
        if image.shape[:2] != (image_height, image_width):
            raise RuntimeError(
                f"CTMC-v1 {sequence_id} frame {frame_index} geometry {image.shape[:2]} "
                f"differs from {(image_height, image_width)}"
            )
        frame_rows = rows_by_frame.get(frame_index, [])
        proxy = _bbox_proxy_labels(
            (image_height, image_width),
            (
                (
                    int(row.source_label),
                    row.source_label,
                    row.left,
                    row.top,
                    row.width,
                    row.height,
                )
                for row in frame_rows
            ),
            context=f"CTMC-v1 {sequence_id} frame {frame_index}",
        )
        ids, counts = np.unique(proxy, return_counts=True)
        area_by_id = {
            int(component_id): int(count)
            for component_id, count in zip(ids, counts, strict=True)
            if component_id > 0
        }
        instances = tuple(
            FrameInstance(
                track_id=int(row.source_label),
                source_label=row.source_label,
                component_id=int(row.source_label),
                parent_track_id=tracks_by_id[int(row.source_label)].parent_track_id,
                centroid_x=row.left + 0.5 * row.width,
                centroid_y=row.top + 0.5 * row.height,
                area_pixels=area_by_id[int(row.source_label)],
                bbox_left=row.left,
                bbox_top=row.top,
                bbox_width=row.width,
                bbox_height=row.height,
            )
            for row in frame_rows
        )
        row_digest = _fingerprint(
            [
                (row.source_label, row.left, row.top, row.width, row.height)
                for row in frame_rows
            ]
        )
        frames.append(
            TrackingFrame(
                frame_index=frame_index,
                image_path=image_path.resolve(),
                instance_path=gt_path.resolve(),
                label_path=lineage_path.resolve(),
                image_sha256=file_sha256(image_path),
                instance_sha256=row_digest,
                label_sha256=lineage_sha256,
                height=image_height,
                width=image_width,
                image_dtype=str(image.dtype),
                instance_dtype="bbox_proxy_uint32_in_memory",
                instances=instances,
            )
        )
    tracks = tuple(tracks_by_id[key] for key in sorted(tracks_by_id))
    parent_links = tuple(
        ParentLink(
            parent_track_id=track.parent_track_id,
            child_track_id=track.track_id,
            parent_end_frame=tracks_by_id[track.parent_track_id].last_frame,
            child_start_frame=track.first_frame,
        )
        for track in tracks
        if track.parent_track_id > 0
    )
    child_counts = Counter(link.parent_track_id for link in parent_links)
    division_quarantine = tuple(
        sorted(parent for parent, count in child_counts.items() if count != 2)
    )
    cell_line = _ctmc_cell_line(sequence_id)
    dataset = "ctmc_v1_dic"
    role = acquisition_role(dataset, cell_line)
    combined_lineage = _fingerprint(
        {
            "gt": gt_sha256,
            "tra": lineage_sha256,
            "seqinfo": seqinfo_sha256,
        }
    )
    fingerprint = _fingerprint(
        _sequence_fingerprint_payload(
            dataset,
            sequence_id,
            role,
            CTMC_V1_SOURCE_FORMAT,
            frames,
            tracks,
            parent_links,
            combined_lineage,
            division_quarantined_parent_track_ids=division_quarantine,
        )
    )
    return TrackingSequence(
        schema_version=TRACKING_SCHEMA_VERSION,
        dataset=dataset,
        sequence_id=sequence_id,
        acquisition_group=acquisition_group(dataset, sequence_id),
        role=role,
        modality="differential interference contrast",
        organism=f"cultured eukaryotic cell line {cell_line}",
        source_format=CTMC_V1_SOURCE_FORMAT,
        source_partition="official_train_only",
        root=train_root.resolve(),
        frames=tuple(frames),
        tracks=tracks,
        parent_links=parent_links,
        lineage_sha256=combined_lineage,
        sequence_fingerprint_sha256=fingerprint,
        division_quarantined_parent_track_ids=division_quarantine,
    )


def discover_ctmc_v1_training(
    root: Path,
    *,
    enforce_expected_counts: bool = True,
) -> tuple[TrackingSequence, ...]:
    """Parse only CTMC-v1 official train; hidden-label test is never opened."""
    train_root = _locate_ctmc_v1_train_root(root)
    sequence_roots = sorted(
        (
            path
            for path in train_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ),
        key=_natural_key,
    )
    if enforce_expected_counts and len(sequence_roots) != CTMC_V1_EXPECTED["acquisitions"]:
        raise RuntimeError(
            f"CTMC-v1 train contains {len(sequence_roots)} acquisitions; expected "
            f"{CTMC_V1_EXPECTED['acquisitions']}"
        )
    sequences = tuple(_ctmc_v1_sequence(train_root, path) for path in sequence_roots)
    totals = {
        "acquisitions": len(sequences),
        "frames": sum(sequence.frame_count for sequence in sequences),
        "tracks": sum(len(sequence.tracks) for sequence in sequences),
        "boxes": sum(sequence.annotated_instance_frames for sequence in sequences),
    }
    if enforce_expected_counts and totals != CTMC_V1_EXPECTED:
        raise RuntimeError(
            f"CTMC-v1 current official train totals changed: {totals}; expected "
            f"{CTMC_V1_EXPECTED}"
        )
    return sequences


def _canonical_alfi_label(raw: str, *, allow_zero: bool, context: str) -> str:
    value = raw.strip()
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise RuntimeError(f"ALFI {context} is not a decimal identity: {raw!r}") from error
    if not number.is_finite() or number < 0 or (not allow_zero and number == 0):
        raise RuntimeError(f"ALFI {context} is outside its supported range: {raw!r}")
    canonical = format(number.normalize(), "f")
    return canonical.rstrip("0").rstrip(".") if "." in canonical else canonical


def _parse_alfi_dtl(path: Path, sequence_id: str) -> tuple[_BoxRow, ...]:
    expected_header = ["ImNo", "ID", "Class", "xmin", "ymin", "width", "height", "Parent"]
    rows: list[_BoxRow] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_header:
            raise RuntimeError(
                f"ALFI {sequence_id} DTLTruth header is {reader.fieldnames}; expected "
                f"{expected_header}"
            )
        for line_number, row in enumerate(reader, start=2):
            try:
                frame = int(row["ImNo"])
                source_label = _canonical_alfi_label(
                    row["ID"], allow_zero=False, context=f"{path}:{line_number} ID"
                )
                parent = _canonical_alfi_label(
                    row["Parent"], allow_zero=True, context=f"{path}:{line_number} Parent"
                )
                left = float(row["xmin"])
                top = float(row["ymin"])
                width = float(row["width"])
                height = float(row["height"])
            except (TypeError, ValueError) as error:
                raise RuntimeError(f"Invalid ALFI DTLTruth row {path}:{line_number}") from error
            if row["Class"] not in {"Interphase", "Mitosis"}:
                raise RuntimeError(f"Unknown ALFI Task-1 class at {path}:{line_number}")
            if frame <= 0 or width <= 0.0 or height <= 0.0 or not np.isfinite(
                (left, top, width, height)
            ).all():
                raise RuntimeError(f"Invalid ALFI geometry at {path}:{line_number}")
            rows.append(
                _BoxRow(
                    frame_index=frame,
                    source_label=source_label,
                    left=left,
                    top=top,
                    width=width,
                    height=height,
                    parent_source_labels=(() if parent == "0" else (parent,)),
                )
            )
    if not rows:
        raise RuntimeError(f"ALFI DTLTruth file is empty: {path}")
    return tuple(rows)


def _locate_alfi_data_root(root: Path) -> Path:
    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    candidates = []
    if resolved.name == "Data&Annotations":
        candidates.append(resolved)
    if (resolved / "Data&Annotations").is_dir():
        candidates.append(resolved / "Data&Annotations")
    candidates = sorted(set(candidates), key=_natural_key)
    if len(candidates) != 1:
        raise RuntimeError(
            f"ALFI requires exactly one Data&Annotations/ root at {resolved}; found {candidates}"
        )
    return candidates[0]


def _alfi_sequence(data_root: Path, sequence_id: str) -> TrackingSequence:
    sequence_root = data_root / sequence_id
    image_directory = sequence_root / "Images"
    dtl_path = sequence_root / f"{sequence_id}_DTLTruth.csv"
    if not image_directory.is_dir() or not dtl_path.is_file():
        raise RuntimeError(
            f"Incomplete ALFI Task-1 sequence {sequence_root}; expected Images/ and DTLTruth CSV"
        )
    image_paths = _index_numbered_files(
        _image_files(image_directory),
        re.compile(rf"I_{sequence_id}_(\d{{4}})\.png"),
        f"ALFI {sequence_id} image",
    )
    rows = _parse_alfi_dtl(dtl_path, sequence_id)
    if set(row.frame_index for row in rows) - set(image_paths):
        raise RuntimeError(f"ALFI {sequence_id} annotations refer to absent PNG frames")
    pair_counts = Counter((row.frame_index, row.source_label) for row in rows)
    ambiguous_events = {key for key, count in pair_counts.items() if count > 1}
    ambiguous_labels = sorted({label for _, label in ambiguous_events}, key=Decimal)
    exclusions = tuple(
        AnnotationExclusion(
            source_label=label,
            observation_count=sum(
                pair_counts[event] for event in ambiguous_events if event[1] == label
            ),
            duplicate_frame_count=sum(event[1] == label for event in ambiguous_events),
            reason=(
                "current official ALFI DTLTruth has multiple boxes for this frame/ID; only "
                "the affected events are excluded from identity/event supervision"
            ),
        )
        for label in ambiguous_labels
    )
    labels = sorted({row.source_label for row in rows}, key=Decimal)
    track_id_by_label = {label: index for index, label in enumerate(labels, start=1)}
    frames_by_label: dict[str, list[int]] = {label: [] for label in labels}
    parents_by_label: dict[str, set[str]] = {label: set() for label in labels}
    for row in rows:
        frames_by_label[row.source_label].append(row.frame_index)
        parents_by_label[row.source_label].update(row.parent_source_labels)
    observed_by_label = {
        label: tuple(sorted(set(frames))) for label, frames in frames_by_label.items()
    }
    direct_parent_by_label: dict[str, str] = {}
    unresolved_parent_labels: set[str] = set()
    for child in labels:
        first = observed_by_label[child][0]
        explicit = parents_by_label[child]
        candidates = sorted(
            (
                parent
                for parent in explicit
                if parent in observed_by_label and observed_by_label[parent][-1] == first - 1
            ),
            key=Decimal,
        )
        if len(candidates) > 1:
            raise RuntimeError(f"ALFI {sequence_id}/{child} has multiple adjacent explicit parents")
        if candidates:
            direct_parent_by_label[child] = candidates[0]
        elif explicit:
            unresolved_parent_labels.add(child)
    tracks = tuple(
        TrackLifetime(
            track_id=track_id_by_label[label],
            source_label=label,
            first_frame=observed_by_label[label][0],
            last_frame=observed_by_label[label][-1],
            parent_track_id=(
                track_id_by_label[direct_parent_by_label[label]]
                if label in direct_parent_by_label
                else 0
            ),
            observed_frames=observed_by_label[label],
        )
        for label in labels
    )
    tracks_by_id = {track.track_id: track for track in tracks}
    _validate_track_graph(tracks_by_id)
    parent_links = tuple(
        ParentLink(
            parent_track_id=track_id_by_label[parent],
            child_track_id=track_id_by_label[child],
            parent_end_frame=observed_by_label[parent][-1],
            child_start_frame=observed_by_label[child][0],
        )
        for child, parent in sorted(direct_parent_by_label.items(), key=lambda item: Decimal(item[0]))
    )
    child_counts = Counter(link.parent_track_id for link in parent_links)
    division_quarantine = tuple(
        sorted(parent for parent, count in child_counts.items() if count != 2)
    )
    event_quarantine: set[tuple[int, int]] = set()
    for label, observed in observed_by_label.items():
        track_id = track_id_by_label[label]
        safe = tuple(frame for frame in observed if (frame, label) not in ambiguous_events)
        for left, right in zip(safe, safe[1:]):
            if right != left + 1:
                event_quarantine.add((track_id, left))
                event_quarantine.add((track_id, right))
        for frame, candidate in ambiguous_events:
            if candidate == label:
                event_quarantine.add((track_id, frame))
        if label in unresolved_parent_labels:
            event_quarantine.add((track_id, observed[0]))
    rows_by_frame: dict[int, list[_BoxRow]] = {}
    for row in rows:
        rows_by_frame.setdefault(row.frame_index, []).append(row)
    csv_sha256 = file_sha256(dtl_path)
    frames: list[TrackingFrame] = []
    sequence_shape: tuple[int, int] | None = None
    for frame_index in sorted(image_paths):
        image = _read_tracking_image(image_paths[frame_index], label="ALFI DIC")
        if sequence_shape is None:
            sequence_shape = image.shape[:2]
        elif image.shape[:2] != sequence_shape:
            raise RuntimeError(f"ALFI {sequence_id} changes frame geometry")
        frame_rows = sorted(
            rows_by_frame.get(frame_index, []),
            key=lambda row: (Decimal(row.source_label), row.left, row.top),
        )
        proxy_rows = tuple(
            (index, row.source_label, row.left, row.top, row.width, row.height)
            for index, row in enumerate(frame_rows, start=1)
        )
        proxy = _bbox_proxy_labels(
            image.shape[:2], proxy_rows, context=f"ALFI {sequence_id} frame {frame_index}"
        )
        ids, counts = np.unique(proxy, return_counts=True)
        area_by_component = {
            int(component): int(count)
            for component, count in zip(ids, counts, strict=True)
            if component > 0
        }
        instances = tuple(
            FrameInstance(
                track_id=track_id_by_label[row.source_label],
                source_label=row.source_label,
                component_id=component,
                parent_track_id=tracks_by_id[track_id_by_label[row.source_label]].parent_track_id,
                centroid_x=row.left + 0.5 * row.width,
                centroid_y=row.top + 0.5 * row.height,
                area_pixels=area_by_component[component],
                bbox_left=row.left,
                bbox_top=row.top,
                bbox_width=row.width,
                bbox_height=row.height,
                identity_supervision_available=(
                    (frame_index, row.source_label) not in ambiguous_events
                ),
            )
            for component, row in enumerate(frame_rows, start=1)
        )
        row_digest = _fingerprint(
            [
                (
                    row.source_label,
                    row.left,
                    row.top,
                    row.width,
                    row.height,
                    (frame_index, row.source_label) not in ambiguous_events,
                )
                for row in frame_rows
            ]
        )
        frames.append(
            TrackingFrame(
                frame_index=frame_index,
                image_path=image_paths[frame_index].resolve(),
                instance_path=dtl_path.resolve(),
                label_path=dtl_path.resolve(),
                image_sha256=file_sha256(image_paths[frame_index]),
                instance_sha256=row_digest,
                label_sha256=csv_sha256,
                height=int(image.shape[0]),
                width=int(image.shape[1]),
                image_dtype=str(image.dtype),
                instance_dtype="bbox_proxy_uint32_in_memory",
                instances=instances,
            )
        )
    dataset = "alfi_task1_dic"
    role = acquisition_role(dataset, sequence_id)
    fingerprint = _fingerprint(
        _sequence_fingerprint_payload(
            dataset,
            sequence_id,
            role,
            ALFI_SOURCE_FORMAT,
            frames,
            tracks,
            parent_links,
            csv_sha256,
            exclusions,
            division_quarantined_parent_track_ids=division_quarantine,
            event_quarantined_track_frames=event_quarantine,
        )
    )
    return TrackingSequence(
        schema_version=TRACKING_SCHEMA_VERSION,
        dataset=dataset,
        sequence_id=sequence_id,
        acquisition_group=acquisition_group(dataset, sequence_id),
        role=role,
        modality="differential interference contrast",
        organism="human U2OS, HeLa, or hTERT RPE-1 cells",
        source_format=ALFI_SOURCE_FORMAT,
        source_partition="Task1_MI01_MI08_only",
        root=data_root.resolve(),
        frames=tuple(frames),
        tracks=tracks,
        parent_links=parent_links,
        lineage_sha256=csv_sha256,
        sequence_fingerprint_sha256=fingerprint,
        annotation_exclusions=exclusions,
        division_quarantined_parent_track_ids=division_quarantine,
        event_quarantined_track_frames=tuple(sorted(event_quarantine)),
    )


def discover_alfi_task1(
    root: Path,
    *,
    enforce_expected_counts: bool = True,
) -> tuple[TrackingSequence, ...]:
    data_root = _locate_alfi_data_root(root)
    missing = [sequence for sequence in ALFI_SEQUENCES if not (data_root / sequence).is_dir()]
    if missing:
        raise RuntimeError(f"ALFI Task-1 root is missing sequences {missing}")
    sequences = tuple(_alfi_sequence(data_root, sequence) for sequence in ALFI_SEQUENCES)
    raw_rows = sum(sequence.annotated_instance_frames for sequence in sequences)
    unique_keys = raw_rows - sum(
        exclusion.observation_count - exclusion.duplicate_frame_count
        for sequence in sequences
        for exclusion in sequence.annotation_exclusions
    )
    totals = {
        "acquisitions": len(sequences),
        "frames": sum(sequence.frame_count for sequence in sequences),
        "archive_dtl_rows": raw_rows,
        "archive_unique_frame_identity_keys": unique_keys,
        "tracks": sum(len(sequence.tracks) for sequence in sequences),
    }
    expected = {
        key: ALFI_EXPECTED[key]
        for key in totals
    }
    if enforce_expected_counts and totals != expected:
        raise RuntimeError(f"ALFI current official Task-1 totals changed: {totals}; expected {expected}")
    return sequences


def _deepsea_layout_error(root: Path, detail: str) -> RuntimeError:
    return RuntimeError(
        f"Unsupported or incomplete DeepSea tracking layout under {root}: {detail}. "
        "Expected tracking_dataset/train/<sequence>/images/<frame>.png, "
        "masks/<frame>_cell_area_masked.png, and "
        "labels/<frame>_cell_pos_labels.txt. Only the official train partition is read; "
        "convert other exports explicitly instead of renaming or guessing fields."
    )


def _locate_deepsea_tracking_root(root: Path) -> Path:
    resolved = root.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    # Google Drive may wrap the official directory as `track/tracking_dataset/train`, while
    # extracted challenge mirrors often start directly at `tracking_dataset/train`. Follow only a
    # short chain of known aliases and select the unique alias directory that *owns* `train`.
    # This is deliberately not an unrestricted recursive search: unrelated result/test folders
    # cannot become training data by merely containing a directory named `train`.
    frontier = [resolved]
    visited: set[Path] = set()
    aliases_seen: set[Path] = set()
    train_owners: set[Path] = set()
    for depth in range(4):
        next_frontier: list[Path] = []
        for current in frontier:
            canonical = current.resolve()
            if canonical != resolved and resolved not in canonical.parents:
                raise _deepsea_layout_error(
                    resolved,
                    f"tracking alias escaped the selected root through a link: {current}",
                )
            if canonical in visited:
                continue
            visited.add(canonical)
            is_alias = canonical.name.casefold() in DEEPSEA_TRACKING_ALIASES
            if is_alias:
                aliases_seen.add(canonical)
                if (canonical / "train").is_dir():
                    train_owners.add(canonical)
            if depth == 3:
                continue
            # The supplied root is the only non-alias node allowed in the traversal.
            if canonical != resolved and not is_alias:
                continue
            next_frontier.extend(
                child
                for child in canonical.iterdir()
                if child.is_dir()
                and not child.name.startswith(".")
                and child.name.casefold() in DEEPSEA_TRACKING_ALIASES
            )
        frontier = next_frontier

    unique = sorted(train_owners, key=_natural_key)
    if len(unique) != 1:
        alias_names = sorted(
            path.relative_to(resolved).as_posix() if path != resolved else "."
            for path in aliases_seen
        )
        owner_names = sorted(
            path.relative_to(resolved).as_posix() if path != resolved else "."
            for path in train_owners
        )
        raise _deepsea_layout_error(
            resolved,
            "expected one alias directory that directly owns the official train partition; "
            f"aliases={alias_names}, train owners={owner_names}",
        )
    return unique[0]


class _DeepSeaIrregularSampling(RuntimeError):
    """A DeepSea acquisition whose published frames are not evenly spaced in time."""


def _deepsea_frame_index(path: Path) -> int:
    match = re.search(r"(\d+)$", path.stem)
    if match is None:
        raise RuntimeError(
            f"DeepSea frame name has no terminal numeric index: {path.name}"
        )
    return int(match.group(1))


def _deepsea_frame_order(image_paths: Sequence[Path]) -> tuple[dict[Path, int], int]:
    """Order one acquisition's frames and return each frame's published index and the stride.

    DeepSea's tracking exports number frames in three different ways: a terminal counter
    (``…_005.png``), a mid-name slice counter (``A11_z003_c001.png``, where the terminal ``c``
    field is constant), and a time index sampled every fourth frame (``exp1_F0002-00006.png``).
    Reading the terminal number as a position would collapse the first family into duplicate
    indices and would misread the third as a gapped track.  Use the rightmost numeric field that
    is unique across the acquisition, then require an even stride so adjacency means the same
    amount of elapsed time everywhere in the sequence.
    """
    stems = [path.stem for path in image_paths]
    groups = [re.findall(r"\d+", stem) for stem in stems]
    if any(not group for group in groups):
        raise RuntimeError(
            f"DeepSea frame name has no numeric index: {image_paths[groups.index([])].name}"
        )
    field_counts = {len(group) for group in groups}
    unique_field = None
    if len(field_counts) == 1:
        for field in reversed(range(field_counts.copy().pop())):
            if len({group[field] for group in groups}) == len(groups):
                unique_field = field
                break
    if unique_field is None:
        # Fall back to the historical terminal-counter rule, which still has to be unique.
        published = {path: _deepsea_frame_index(path) for path in image_paths}
    else:
        published = {
            path: int(group[unique_field])
            for path, group in zip(image_paths, groups, strict=True)
        }
    if len(set(published.values())) != len(image_paths):
        raise RuntimeError(
            "DeepSea acquisition has duplicate frame indices: "
            f"{sorted(published.values())[:12]}"
        )
    ordered = sorted(image_paths, key=lambda path: published[path])
    indices = [published[path] for path in ordered]
    strides = {right - left for left, right in zip(indices, indices[1:])}
    if len(strides) > 1:
        raise _DeepSeaIrregularSampling(
            "DeepSea acquisition is sampled unevenly; adjacent frames would span different "
            f"amounts of time (observed strides {sorted(strides)[:8]})"
        )
    stride = strides.pop() if strides else 1
    if stride < 1:
        raise RuntimeError(f"DeepSea frame indices are not increasing: {indices[:12]}")
    return {path: position for position, path in enumerate(ordered)}, stride


def _parse_deepsea_labels(path: Path) -> tuple[tuple[str, int, int], ...]:
    """Read every published marker row in file order.

    A handful of frames give one identity two markers.  Those rows are kept rather than rejected
    so the caller can quarantine that identity the same way the ALFI and LiveCellTrack adapters
    do, instead of failing an entire acquisition.
    """
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as error:
        raise RuntimeError(f"Could not read DeepSea label file {path}: {error}") from error
    if not lines:
        raise RuntimeError(f"DeepSea label file is empty: {path}")
    rows: list[tuple[str, int, int]] = []
    for line_number, raw_line in enumerate(lines[1:], start=2):
        if not raw_line.strip():
            continue
        fields = raw_line.rstrip("\r\n").split("\t")
        if len(fields) < 3:
            raise RuntimeError(
                f"DeepSea label {path}:{line_number} must be tab-separated label, x, y"
            )
        source_label = fields[0].strip()
        if not source_label:
            raise RuntimeError(f"DeepSea label {path}:{line_number} has an empty identity")
        try:
            x, y = int(fields[1]), int(fields[2])
        except ValueError as error:
            raise RuntimeError(
                f"DeepSea label {path}:{line_number} has non-integer centroid coordinates"
            ) from error
        rows.append((source_label, x, y))
    if not rows:
        raise RuntimeError(f"DeepSea label file contains no cells after its header: {path}")
    return tuple(rows)


@dataclass(frozen=True)
class _DeepSeaDecodedFrame:
    frame_index: int
    image_path: Path
    mask_path: Path
    label_path: Path
    image: np.ndarray
    binary_mask: np.ndarray
    component_labels: np.ndarray
    # One entry per published marker: (source label, x, y, mask component id).  A label appears
    # more than once only where the publisher gave one identity two markers in a frame.
    matches: tuple[tuple[str, int, int, int], ...]
    unlabeled_components: tuple[UnlabeledMaskRegion, ...] = ()

    @property
    def ambiguous_labels(self) -> frozenset[str]:
        counted = Counter(source_label for source_label, _, _, _ in self.matches)
        return frozenset(label for label, count in counted.items() if count > 1)

    @property
    def labels(self) -> frozenset[str]:
        return frozenset(source_label for source_label, _, _, _ in self.matches)


def _decode_deepsea_frame(
    image_path: Path,
    mask_path: Path,
    label_path: Path,
    frame_index: int,
) -> _DeepSeaDecodedFrame:
    image = _read_deepsea_image(image_path)
    binary_mask = _read_deepsea_binary_mask(mask_path)
    if image.shape[:2] != binary_mask.shape:
        raise RuntimeError(
            f"DeepSea image/mask shape mismatch for {image_path.name}: "
            f"{image.shape[:2]} versus {binary_mask.shape}"
        )
    component_count, components, _, component_centroids = (
        cv2.connectedComponentsWithStats(
            np.ascontiguousarray(binary_mask, dtype=np.uint8), connectivity=8
        )
    )
    marker_rows = _parse_deepsea_labels(label_path)
    height, width = binary_mask.shape
    for source_label, x, y in marker_rows:
        if not 0 <= x < width or not 0 <= y < height:
            raise RuntimeError(
                f"DeepSea label {source_label!r} centroid ({x}, {y}) is outside "
                f"{image_path.name}"
            )
    expected_components = set(range(1, component_count))
    if len(marker_rows) > len(expected_components):
        raise RuntimeError(
            f"DeepSea lists {len(marker_rows)} tracked cells but {image_path.name} "
            f"segments only {len(expected_components)} mask components"
        )
    # A DeepSea position label is the rounded centroid of its cell's mask component, so the
    # faithful match is a one-to-one assignment onto component centroids rather than a
    # point-in-component test: the published integer marker legitimately lands a pixel or two
    # outside a curved or thin cell.  The assignment stays injective, and the tolerance keeps a
    # genuinely mismatched annotation loud instead of letting it snap to a neighbouring cell.
    ordered_markers = sorted(marker_rows)
    matches: list[tuple[str, int, int, int]] = []
    if ordered_markers:
        markers = np.asarray(
            [(x, y) for _, x, y in ordered_markers], dtype=np.float64
        )
        centroids = component_centroids[1:].astype(np.float64)
        distances = np.linalg.norm(markers[:, None, :] - centroids[None, :, :], axis=2)
        marker_indices, component_columns = linear_sum_assignment(distances)
        for row, column in zip(marker_indices, component_columns):
            separation = float(distances[row, column])
            source_label, x, y = ordered_markers[int(row)]
            if separation > DEEPSEA_CENTROID_TOLERANCE_PIXELS:
                raise RuntimeError(
                    f"DeepSea label {source_label!r} at ({x}, {y}) is {separation:.1f}px from "
                    f"the nearest free mask component centroid in {image_path.name}; the "
                    f"tolerance is {DEEPSEA_CENTROID_TOLERANCE_PIXELS:.1f}px"
                )
            matches.append((source_label, x, y, int(column) + 1))
    matches.sort()
    used_components = {component_id for _, _, _, component_id in matches}
    # Every labelled cell must own a component, but the reverse does not hold: the publisher
    # segments cells it never tracked, most often where one enters or leaves the field.  Record
    # them so they can be withheld from supervision rather than invented as tracks.
    unlabeled_components: list[UnlabeledMaskRegion] = []
    for component_id in sorted(expected_components - used_components):
        rows, columns = np.nonzero(components == component_id)
        unlabeled_components.append(
            UnlabeledMaskRegion(
                frame_index=frame_index,
                component_id=component_id,
                area_pixels=int(rows.size),
                touches_image_border=bool(
                    rows.min() == 0
                    or columns.min() == 0
                    or rows.max() == height - 1
                    or columns.max() == width - 1
                ),
            )
        )
    return _DeepSeaDecodedFrame(
        frame_index=frame_index,
        image_path=image_path,
        mask_path=mask_path,
        label_path=label_path,
        image=image,
        binary_mask=binary_mask,
        component_labels=components,
        matches=tuple(matches),
        unlabeled_components=tuple(unlabeled_components),
    )


def _deepsea_sequence(
    tracking_root: Path,
    sequence_root: Path,
) -> TrackingSequence:
    images_directory = sequence_root / DEEPSEA_IMAGE_DIRECTORY
    masks_directory = sequence_root / DEEPSEA_MASK_DIRECTORY
    labels_directory = sequence_root / DEEPSEA_LABEL_DIRECTORY
    missing = [
        name
        for name, path in (
            (DEEPSEA_IMAGE_DIRECTORY, images_directory),
            (DEEPSEA_MASK_DIRECTORY, masks_directory),
            (DEEPSEA_LABEL_DIRECTORY, labels_directory),
        )
        if not path.is_dir()
    ]
    if missing:
        raise _deepsea_layout_error(sequence_root, f"missing directories {missing}")

    image_paths = _image_files(images_directory)
    if not image_paths:
        raise _deepsea_layout_error(sequence_root, "images directory is empty")
    if any(path.suffix.casefold() != ".png" for path in image_paths):
        raise _deepsea_layout_error(
            sequence_root, "the official tracker adapter requires PNG image frames"
        )
    try:
        positions, frame_index_stride = _deepsea_frame_order(image_paths)
    except _DeepSeaIrregularSampling:
        raise
    except RuntimeError as error:
        raise _deepsea_layout_error(sequence_root, str(error)) from error
    # Frames are addressed by published order, so "adjacent" means one sampling interval for
    # every acquisition, whether the publisher numbered its frames 1,2,3 or 2,6,10.
    indexed_images = {position: path for path, position in positions.items()}
    indices = sorted(indexed_images)

    expected_masks = {
        path.with_name(path.stem + DEEPSEA_MASK_SUFFIX).name
        for path in image_paths
    }
    expected_labels = {
        path.with_name(path.stem + DEEPSEA_LABEL_SUFFIX).name
        for path in image_paths
    }
    actual_masks = {
        path.name
        for path in masks_directory.iterdir()
        if path.is_file() and not path.name.startswith(".")
    }
    actual_labels = {
        path.name
        for path in labels_directory.iterdir()
        if path.is_file() and not path.name.startswith(".")
    }
    missing_masks = expected_masks - actual_masks
    missing_labels = expected_labels - actual_labels
    if missing_masks or missing_labels:
        raise _deepsea_layout_error(
            sequence_root,
            "one or more selected image frames lack an official-suffix companion: "
            f"missing masks={sorted(missing_masks)[:8]}, "
            f"missing labels={sorted(missing_labels)[:8]}",
        )

    # Some official DeepSea acquisitions publish mask/label rows beyond their final raw image.
    # Match the publisher's BasicTrackerDataset semantics by anchoring the sequence on images.
    # Extra companions cannot supervise a frame without an image, but their names and bytes remain
    # hash-bound in the provenance instead of being silently discarded.
    source_file_exclusions: list[SourceFileExclusion] = []
    for directory, names, reason in (
        (
            masks_directory,
            actual_masks - expected_masks,
            "DeepSea mask has no matching official image frame; excluded from supervision",
        ),
        (
            labels_directory,
            actual_labels - expected_labels,
            "DeepSea label has no matching official image frame; excluded from supervision",
        ),
    ):
        for name in sorted(names, key=_natural_key):
            path = (directory / name).resolve()
            try:
                relative_path = path.relative_to(tracking_root.resolve()).as_posix()
            except ValueError as error:
                raise _deepsea_layout_error(
                    sequence_root, f"companion file escaped tracking root: {path}"
                ) from error
            source_file_exclusions.append(
                SourceFileExclusion(
                    relative_path=relative_path,
                    sha256=file_sha256(path),
                    reason=reason,
                )
            )

    decoded: list[_DeepSeaDecodedFrame] = []
    sequence_shape: tuple[int, int] | None = None
    for frame_index in indices:
        image_path = indexed_images[frame_index]
        frame = _decode_deepsea_frame(
            image_path,
            masks_directory / f"{image_path.stem}{DEEPSEA_MASK_SUFFIX}",
            labels_directory / f"{image_path.stem}{DEEPSEA_LABEL_SUFFIX}",
            frame_index,
        )
        if sequence_shape is None:
            sequence_shape = frame.image.shape[:2]
        elif frame.image.shape[:2] != sequence_shape:
            raise RuntimeError(
                f"DeepSea sequence {sequence_root.name} changes image shape at frame "
                f"{frame_index}"
            )
        decoded.append(frame)

    labels_by_frame = {frame.frame_index: set(frame.labels) for frame in decoded}
    # A frame that gives one identity two markers cannot say which cell is that track.  Follow the
    # ALFI and LiveCellTrack contract: keep both cells as detections, withhold identity
    # supervision for that frame, and record the quarantine.
    ambiguous_events = {
        (frame.frame_index, source_label)
        for frame in decoded
        for source_label in frame.ambiguous_labels
    }
    marker_counts = Counter(
        (frame.frame_index, source_label)
        for frame in decoded
        for source_label, _, _, _ in frame.matches
    )
    annotation_exclusions = tuple(
        AnnotationExclusion(
            source_label=source_label,
            observation_count=sum(
                marker_counts[event] for event in ambiguous_events if event[1] == source_label
            ),
            duplicate_frame_count=sum(
                event[1] == source_label for event in ambiguous_events
            ),
            reason=(
                "the published DeepSea label file gives this identity more than one marker in "
                "a frame; only the affected frames leave identity and event supervision"
            ),
        )
        for source_label in sorted({label for _, label in ambiguous_events}, key=_natural_key)
    )
    parent_by_label: dict[str, str | None] = {
        label: None for labels in labels_by_frame.values() for label in labels
    }
    for previous, current in zip(decoded, decoded[1:]):
        if current.frame_index != previous.frame_index + 1:
            raise RuntimeError("DeepSea decoded frames are not adjacent")

    observed_by_label: dict[str, list[int]] = {}
    for frame_index, labels in sorted(labels_by_frame.items()):
        for source_label in labels:
            observed_by_label.setdefault(source_label, []).append(frame_index)
    for observed_frames in observed_by_label.values():
        observed_frames.sort()

    # A DeepSea division is written into the label names: parent ``P`` becomes ``P_1`` and ``P_2``.
    # With staggered annotation the two daughters need not be annotated in the same frame, so the
    # pairing is resolved over the whole acquisition rather than across one frame boundary.
    incomplete_divisions: set[str] = set()
    for parent_label in sorted(observed_by_label):
        present_children = [
            child
            for child in (f"{parent_label}_1", f"{parent_label}_2")
            if child in observed_by_label
        ]
        # Some acquisitions write the same convention with a hyphen.  Those daughters are never
        # linked, because guessing a lineage edge from a second naming style would invent
        # divisions; the parent is quarantined instead so no division supervision is derived.
        if any(
            f"{parent_label}-{suffix}" in observed_by_label for suffix in ("1", "2")
        ):
            incomplete_divisions.add(parent_label)
        if not present_children:
            continue
        if len(present_children) == 1:
            # One annotated daughter is not a division: the sibling was never labelled, so no
            # parent link is created and the event stays out of supervision.
            incomplete_divisions.add(parent_label)
            continue
        parent_last_frame = observed_by_label[parent_label][-1]
        if any(
            observed_by_label[child][0] <= parent_last_frame
            for child in present_children
        ):
            # The annotation has the parent still present after a daughter starts, which cannot
            # both be true.  Neither reading is guessed: the division is quarantined and the three
            # tracks stay in the data as independent identities.
            incomplete_divisions.add(parent_label)
            continue
        for child in present_children:
            prior_parent = parent_by_label.get(child)
            if prior_parent not in (None, parent_label):
                raise RuntimeError(f"DeepSea child {child!r} has multiple parents")
            parent_by_label[child] = parent_label
    # DeepSea annotates identities on a staggered schedule rather than on every frame, so a track
    # legitimately reappears after a gap.  The gap is kept as published: the identity is the same
    # cell before and after, and the tokenizer already refuses to build a continuation edge across
    # a gap and quarantines its endpoints from birth/death supervision, so nothing is invented.
    for source_label, observed_frames in observed_by_label.items():
        if len(set(observed_frames)) != len(observed_frames):
            raise RuntimeError(
                f"DeepSea label {source_label!r} is annotated twice in one frame: "
                f"{observed_frames}"
            )

    ordered_labels = sorted(
        observed_by_label,
        key=lambda label: (observed_by_label[label][0], _natural_key(label)),
    )
    track_id_by_label = {
        source_label: index
        for index, source_label in enumerate(ordered_labels, start=1)
    }
    tracks: list[TrackLifetime] = []
    for source_label in ordered_labels:
        observed_frames = tuple(observed_by_label[source_label])
        parent_label = parent_by_label[source_label]
        parent_track_id = 0 if parent_label is None else track_id_by_label[parent_label]
        if parent_label is not None:
            parent_frames = observed_by_label[parent_label]
            # A daughter must start after its parent's last observation.  With staggered
            # annotation the first daughter observation can be several frames later, so require
            # order rather than immediate adjacency; the tokenizer keeps the division edge only
            # when both endpoints are unambiguously annotated.
            if observed_frames[0] <= parent_frames[-1]:
                raise RuntimeError(
                    f"DeepSea child {source_label!r} begins at frame {observed_frames[0]} but "
                    f"its parent {parent_label!r} is still observed at frame "
                    f"{parent_frames[-1]}"
                )
        tracks.append(
            TrackLifetime(
                track_id=track_id_by_label[source_label],
                source_label=source_label,
                first_frame=observed_frames[0],
                last_frame=observed_frames[-1],
                parent_track_id=parent_track_id,
                observed_frames=observed_frames,
            )
        )
    tracks_by_id = {track.track_id: track for track in tracks}
    _validate_track_graph(tracks_by_id)
    division_quarantined_parent_track_ids = tuple(
        sorted(
            track_id_by_label[parent_label]
            for parent_label in incomplete_divisions
            if parent_label in track_id_by_label
        )
    )
    parent_links = tuple(
        ParentLink(
            parent_track_id=track.parent_track_id,
            child_track_id=track.track_id,
            parent_end_frame=tracks_by_id[track.parent_track_id].last_frame,
            child_start_frame=track.first_frame,
        )
        for track in tracks
        if track.parent_track_id > 0
    )

    frames: list[TrackingFrame] = []
    label_hash_rows: list[dict[str, object]] = []
    for decoded_frame in decoded:
        instances: list[FrameInstance] = []
        for source_label, _, _, component_id in sorted(
            decoded_frame.matches,
            key=lambda match: (_natural_key(match[0]), match[3]),
        ):
            centroid_x, centroid_y, area = _component_geometry(
                decoded_frame.component_labels, component_id
            )
            track_id = track_id_by_label[source_label]
            instances.append(
                FrameInstance(
                    track_id=track_id,
                    source_label=source_label,
                    component_id=component_id,
                    parent_track_id=tracks_by_id[track_id].parent_track_id,
                    centroid_x=centroid_x,
                    centroid_y=centroid_y,
                    area_pixels=area,
                    identity_supervision_available=(
                        (decoded_frame.frame_index, source_label) not in ambiguous_events
                    ),
                )
            )
        label_sha256 = file_sha256(decoded_frame.label_path)
        label_hash_rows.append(
            {
                "frame_index": decoded_frame.frame_index,
                "name": decoded_frame.label_path.name,
                "sha256": label_sha256,
            }
        )
        frames.append(
            TrackingFrame(
                frame_index=decoded_frame.frame_index,
                image_path=decoded_frame.image_path.resolve(),
                instance_path=decoded_frame.mask_path.resolve(),
                label_path=decoded_frame.label_path.resolve(),
                image_sha256=file_sha256(decoded_frame.image_path),
                instance_sha256=file_sha256(decoded_frame.mask_path),
                label_sha256=label_sha256,
                height=int(decoded_frame.image.shape[0]),
                width=int(decoded_frame.image.shape[1]),
                image_dtype=str(decoded_frame.image.dtype),
                instance_dtype=str(decoded_frame.component_labels.dtype),
                instances=tuple(instances),
                unlabeled_component_ids=tuple(
                    region.component_id
                    for region in decoded_frame.unlabeled_components
                ),
            )
        )
    unlabeled_mask_regions = tuple(
        region
        for decoded_frame in decoded
        for region in decoded_frame.unlabeled_components
    )
    lineage_sha256 = _fingerprint(label_hash_rows)
    dataset = "deepsea_tracking"
    sequence_id = sequence_root.name
    role = acquisition_role(dataset, sequence_id)
    fingerprint = _fingerprint(
        _sequence_fingerprint_payload(
            dataset,
            sequence_id,
            role,
            "deepsea_basic_tracker_v1",
            frames,
            tracks,
            parent_links,
            lineage_sha256,
            annotation_exclusions=annotation_exclusions,
            source_file_exclusions=source_file_exclusions,
            division_quarantined_parent_track_ids=division_quarantined_parent_track_ids,
            frame_index_stride=frame_index_stride,
        )
    )
    return TrackingSequence(
        schema_version=TRACKING_SCHEMA_VERSION,
        dataset=dataset,
        sequence_id=sequence_id,
        acquisition_group=acquisition_group(dataset, sequence_id),
        role=role,
        modality="phase contrast",
        organism="mouse embryonic stem, bronchial epithelial, or C2C12 cells",
        source_format="deepsea_basic_tracker_v1",
        source_partition="official_train_only",
        root=tracking_root.resolve(),
        frames=tuple(frames),
        tracks=tuple(tracks),
        parent_links=parent_links,
        lineage_sha256=lineage_sha256,
        sequence_fingerprint_sha256=fingerprint,
        annotation_exclusions=annotation_exclusions,
        source_file_exclusions=tuple(source_file_exclusions),
        unlabeled_mask_regions=unlabeled_mask_regions,
        division_quarantined_parent_track_ids=division_quarantined_parent_track_ids,
        frame_index_stride=frame_index_stride,
    )


def deepsea_irregular_sequences(root: Path) -> tuple[tuple[str, str], ...]:
    """Name the training acquisitions withheld for uneven sampling, reading filenames only."""
    tracking_root = _locate_deepsea_tracking_root(root)
    train_root = tracking_root / "train"
    if not train_root.is_dir():
        return ()
    withheld: list[tuple[str, str]] = []
    for sequence_root in sorted(
        (path for path in train_root.iterdir() if path.is_dir()), key=_natural_key
    ):
        images_directory = sequence_root / DEEPSEA_IMAGE_DIRECTORY
        if not images_directory.is_dir():
            continue
        image_paths = _image_files(images_directory)
        if not image_paths:
            continue
        try:
            _deepsea_frame_order(image_paths)
        except _DeepSeaIrregularSampling as reason:
            withheld.append((sequence_root.name, str(reason)))
        except RuntimeError:
            continue
    return tuple(withheld)


def discover_deepsea_training_tracking(root: Path) -> tuple[TrackingSequence, ...]:
    """Discover only DeepSea tracker training sequences; the test directory is never opened."""
    tracking_root = _locate_deepsea_tracking_root(root)
    train_root = tracking_root / "train"
    if not train_root.is_dir():
        raise _deepsea_layout_error(tracking_root, "missing official train directory")
    sequence_roots = sorted(
        (
            path
            for path in train_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ),
        key=_natural_key,
    )
    if not sequence_roots:
        raise _deepsea_layout_error(train_root, "no sequence directories")
    collected: list[TrackingSequence] = []
    for sequence_root in sequence_roots:
        try:
            collected.append(_deepsea_sequence(tracking_root, sequence_root))
        except _DeepSeaIrregularSampling as reason:
            # Position addressing would present these frames as one sampling interval apart when
            # they are not, which would misstate motion.  Withhold the acquisition instead.
            print(f"DeepSea tracking: excluded {sequence_root.name}; {reason}")
    if not collected:
        raise _deepsea_layout_error(train_root, "no evenly sampled sequences")
    sequences = tuple(collected)
    groups = [sequence.acquisition_group for sequence in sequences]
    if len(groups) != len(set(groups)):
        raise RuntimeError(f"Duplicate DeepSea acquisition groups under {train_root}")
    return sequences


@dataclass(frozen=True)
class _LiveCellTrackBox:
    frame_index: int
    track_id: int
    left: float
    top: float
    width: float
    height: float
    confidence: float

    @property
    def centroid_x(self) -> float:
        return self.left + 0.5 * self.width

    @property
    def centroid_y(self) -> float:
        return self.top + 0.5 * self.height


def _locate_livecelltrack_imaging_root(root: Path) -> Path:
    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    candidates: list[Path] = []
    if resolved.name == "Cell_imaging":
        candidates.append(resolved)
    if (resolved / "Cell_imaging").is_dir():
        candidates.append(resolved / "Cell_imaging")
    candidates = sorted(set(candidates), key=_natural_key)
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one LiveCellTrack Cell_imaging/ root at {resolved}, found "
            f"{[str(path) for path in candidates]}"
        )
    imaging_root = candidates[0]
    missing_domains = [
        domain
        for domain in LIVECELLTRACK_PREVIEW_DOMAINS
        if not (imaging_root / domain / "data").is_dir()
    ]
    if missing_domains:
        raise RuntimeError(
            f"LiveCellTrack preview is missing strict domain data roots: {missing_domains}"
        )
    return imaging_root


def _parse_livecelltrack_gt(
    path: Path,
) -> tuple[tuple[_LiveCellTrackBox, ...], tuple[AnnotationExclusion, ...]]:
    """Parse human MOT rows and quarantine ambiguous IDs instead of guessing links."""
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as error:
        raise RuntimeError(f"Could not read LiveCellTrack ground truth {path}: {error}") from error
    rows: list[_LiveCellTrackBox] = []
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 10:
            raise RuntimeError(
                f"LiveCellTrack MOT row {path}:{line_number} must contain exactly 10 fields"
            )
        try:
            values = tuple(float(field) for field in fields)
        except ValueError as error:
            raise RuntimeError(
                f"LiveCellTrack MOT row {path}:{line_number} contains a non-number"
            ) from error
        if not np.isfinite(values).all():
            raise RuntimeError(
                f"LiveCellTrack MOT row {path}:{line_number} contains a non-finite number"
            )
        if not values[0].is_integer() or not values[1].is_integer():
            raise RuntimeError(
                f"LiveCellTrack frame and identity must be integers at {path}:{line_number}"
            )
        frame_index, track_id = int(values[0]), int(values[1])
        left, top, width, height, confidence = values[2:7]
        if frame_index < 0 or track_id <= 0 or track_id > np.iinfo(np.int32).max:
            raise RuntimeError(
                f"LiveCellTrack frame/identity is outside its supported range at "
                f"{path}:{line_number}"
            )
        if width <= 0.0 or height <= 0.0 or confidence <= 0.0:
            raise RuntimeError(
                f"LiveCellTrack box/confidence must be positive at {path}:{line_number}"
            )
        rows.append(
            _LiveCellTrackBox(
                frame_index=frame_index,
                track_id=track_id,
                left=left,
                top=top,
                width=width,
                height=height,
                confidence=confidence,
            )
        )
    if not rows:
        raise RuntimeError(f"LiveCellTrack ground truth is empty: {path}")
    pair_counts = Counter((row.frame_index, row.track_id) for row in rows)
    ambiguous_ids = sorted(
        {
            track_id
            for (_, track_id), count in pair_counts.items()
            if count > 1
        }
    )
    ambiguous_set = set(ambiguous_ids)
    exclusions = tuple(
        AnnotationExclusion(
            source_label=str(track_id),
            observation_count=sum(row.track_id == track_id for row in rows),
            duplicate_frame_count=sum(
                count > 1
                for (_, candidate), count in pair_counts.items()
                if candidate == track_id
            ),
            reason=(
                "source ID occurs in multiple human MOT rows within one frame; the entire "
                "identity is quarantined without inferred reassignment"
            ),
        )
        for track_id in ambiguous_ids
    )
    accepted = tuple(
        sorted(
            (row for row in rows if row.track_id not in ambiguous_set),
            key=lambda row: (row.frame_index, row.track_id),
        )
    )
    if not accepted:
        raise RuntimeError(
            f"Every LiveCellTrack identity was ambiguous after strict quarantine: {path}"
        )
    return accepted, exclusions


def _livecelltrack_bbox_proxy(
    shape: tuple[int, int],
    rows: Iterable[_LiveCellTrackBox],
) -> np.ndarray:
    """Rasterize deterministic inscribed ellipses without claiming dense manual masks."""
    height, width = shape
    labels = np.zeros((height, width), dtype=np.uint32)
    best_distance = np.full((height, width), np.inf, dtype=np.float32)
    row_list = sorted(rows, key=lambda row: row.track_id)
    for row in row_list:
        # A cell leaving the field has a box that straddles the border, so its center can sit a
        # fraction of a pixel outside the frame.  The ellipse is clipped to the image below and
        # the overlap check that follows is the real requirement; every retained row still has to
        # occupy pixels, which is verified after rasterization.
        x0 = max(0, int(np.floor(row.left)))
        y0 = max(0, int(np.floor(row.top)))
        x1 = min(width, int(np.ceil(row.left + row.width)))
        y1 = min(height, int(np.ceil(row.top + row.height)))
        if x0 >= x1 or y0 >= y1:
            raise RuntimeError(f"LiveCellTrack box for ID {row.track_id} has no image overlap")
        x_coordinates = np.arange(x0, x1, dtype=np.float32) + 0.5
        y_coordinates = np.arange(y0, y1, dtype=np.float32) + 0.5
        normalized_x = (x_coordinates - row.centroid_x) / max(0.5 * row.width, 1e-6)
        normalized_y = (y_coordinates - row.centroid_y) / max(0.5 * row.height, 1e-6)
        distance = normalized_y[:, None] ** 2 + normalized_x[None, :] ** 2
        inside = distance <= 1.0
        region_best = best_distance[y0:y1, x0:x1]
        # Sorted track IDs give deterministic ownership for exact-distance ties.
        update = inside & (distance < region_best)
        region_best[update] = distance[update]
        labels[y0:y1, x0:x1][update] = row.track_id
    retained = {int(value) for value in np.unique(labels) if value > 0}
    # Each proxy pixel belongs to exactly one identity, so a box that a heavily overlapping
    # neighbour covers completely keeps none.  That observation cannot be a detection, and
    # awarding it a contested pixel would invent ownership, so it is reported to the caller and
    # withheld from that frame instead.
    occluded = tuple(
        sorted({row.track_id for row in row_list if row.track_id not in retained})
    )
    return labels, occluded


def _materialize_livecelltrack_proxy(path: Path, labels: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = np.asarray(tifffile.imread(path))
        if existing.dtype != labels.dtype or not np.array_equal(existing, labels):
            raise RuntimeError(
                f"Existing LiveCellTrack proxy differs from deterministic v1 output: {path}. "
                f"Remove {path.parent} and rerun preflight."
            )
    else:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            tifffile.imwrite(
                temporary,
                labels,
                photometric="minisblack",
                compression="zlib",
                compressionargs={"level": 6},
                metadata=None,
            )
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    return file_sha256(path)


def _livecelltrack_sequence(
    imaging_root: Path,
    domain: str,
    acquisition_root: Path,
    *,
    expected_frames: int | None,
) -> TrackingSequence:
    if domain not in LIVECELLTRACK_PREVIEW_DOMAINS:
        raise ValueError(f"Unknown LiveCellTrack preview domain: {domain}")
    images_directory = acquisition_root / "img1"
    gt_path = acquisition_root / "gt" / "gt.txt"
    if not images_directory.is_dir() or not gt_path.is_file():
        raise RuntimeError(
            f"Incomplete LiveCellTrack acquisition {acquisition_root}: expected img1/ and gt/gt.txt"
        )
    image_paths = _index_numbered_files(
        _image_files(images_directory),
        re.compile(r"(\d+)\.tiff?", flags=re.I),
        f"LiveCellTrack {domain}/{acquisition_root.name} image",
    )
    if expected_frames is not None and len(image_paths) != expected_frames:
        raise RuntimeError(
            f"LiveCellTrack {domain}/{acquisition_root.name} has {len(image_paths)} frames; "
            f"the pinned preview declares {expected_frames}"
        )
    rows, annotation_exclusions = _parse_livecelltrack_gt(gt_path)
    rows_by_frame: dict[int, list[_LiveCellTrackBox]] = {}
    observed_by_track: dict[int, list[int]] = {}
    for row in rows:
        if row.frame_index not in image_paths:
            raise RuntimeError(
                f"LiveCellTrack gt frame {row.frame_index} has no matching TIFF in "
                f"{images_directory}"
            )
        rows_by_frame.setdefault(row.frame_index, []).append(row)
        observed_by_track.setdefault(row.track_id, []).append(row.frame_index)
    tracks: list[TrackLifetime] = []
    for track_id in sorted(observed_by_track):
        observed = tuple(observed_by_track[track_id])
        expected = tuple(range(observed[0], observed[-1] + 1))
        if observed != expected:
            raise RuntimeError(
                f"LiveCellTrack ID {track_id} disappears and reappears in "
                f"{domain}/{acquisition_root.name}: {observed}"
            )
        tracks.append(
            TrackLifetime(
                track_id=track_id,
                source_label=str(track_id),
                first_frame=observed[0],
                last_frame=observed[-1],
                parent_track_id=0,
                observed_frames=observed,
            )
        )

    gt_sha256 = file_sha256(gt_path)
    frames: list[TrackingFrame] = []
    # (frame, identity) pairs whose annotated box is completely covered by an overlapping
    # neighbour, so the proxy can give it no pixels of its own.
    occluded_observations: list[tuple[int, int]] = []
    sequence_shape: tuple[int, int] | None = None
    proxy_root = (
        imaging_root
        / LIVECELLTRACK_PROXY_DIRECTORY
        / domain
        / acquisition_root.name
    )
    for frame_index in sorted(image_paths):
        image = _read_ctc_tiff(image_paths[frame_index], "LiveCellTrack image")
        if sequence_shape is None:
            sequence_shape = image.shape
        elif image.shape != sequence_shape:
            raise RuntimeError(
                f"LiveCellTrack {domain}/{acquisition_root.name} changes image shape at "
                f"frame {frame_index}"
            )
        frame_rows = rows_by_frame.get(frame_index, [])
        proxy, occluded_ids = _livecelltrack_bbox_proxy(image.shape, frame_rows)
        if occluded_ids:
            occluded_observations.extend(
                (frame_index, track_id) for track_id in occluded_ids
            )
            frame_rows = [
                row for row in frame_rows if row.track_id not in set(occluded_ids)
            ]
        proxy_ids, proxy_counts = np.unique(proxy, return_counts=True)
        proxy_area = {
            int(component_id): int(count)
            for component_id, count in zip(proxy_ids, proxy_counts, strict=True)
            if component_id > 0
        }
        proxy_path = proxy_root / f"{image_paths[frame_index].stem}.tif"
        proxy_sha256 = _materialize_livecelltrack_proxy(proxy_path, proxy)
        instances = tuple(
            FrameInstance(
                track_id=row.track_id,
                source_label=str(row.track_id),
                component_id=row.track_id,
                parent_track_id=0,
                centroid_x=row.centroid_x,
                centroid_y=row.centroid_y,
                area_pixels=proxy_area[row.track_id],
            )
            for row in frame_rows
        )
        frames.append(
            TrackingFrame(
                frame_index=frame_index,
                image_path=image_paths[frame_index].resolve(),
                instance_path=proxy_path.resolve(),
                label_path=gt_path.resolve(),
                image_sha256=file_sha256(image_paths[frame_index]),
                instance_sha256=proxy_sha256,
                label_sha256=gt_sha256,
                height=int(image.shape[0]),
                width=int(image.shape[1]),
                image_dtype=str(image.dtype),
                instance_dtype=str(proxy.dtype),
                instances=instances,
            )
        )

    event_quarantined_track_frames = tuple(
        sorted((track_id, frame_index) for frame_index, track_id in occluded_observations)
    )
    details = LIVECELLTRACK_PREVIEW_DOMAINS[domain]
    dataset = details["dataset"]
    sequence_id = acquisition_root.name
    role = acquisition_role(dataset, sequence_id)
    parent_links: tuple[ParentLink, ...] = ()
    fingerprint = _fingerprint(
        _sequence_fingerprint_payload(
            dataset,
            sequence_id,
            role,
            LIVECELLTRACK_SOURCE_FORMAT,
            frames,
            tracks,
            parent_links,
            gt_sha256,
            annotation_exclusions,
            event_quarantined_track_frames=event_quarantined_track_frames,
        )
    )
    return TrackingSequence(
        schema_version=TRACKING_SCHEMA_VERSION,
        dataset=dataset,
        sequence_id=sequence_id,
        acquisition_group=acquisition_group(dataset, sequence_id),
        role=role,
        modality=details["modality"],
        organism=details["organism"],
        source_format=LIVECELLTRACK_SOURCE_FORMAT,
        source_partition="public_preview_human_ground_truth",
        root=imaging_root.resolve(),
        frames=tuple(frames),
        tracks=tuple(tracks),
        parent_links=parent_links,
        lineage_sha256=gt_sha256,
        sequence_fingerprint_sha256=fingerprint,
        annotation_exclusions=annotation_exclusions,
        event_quarantined_track_frames=event_quarantined_track_frames,
    )


def discover_livecelltrack_preview(root: Path) -> tuple[TrackingSequence, ...]:
    """Discover the complete pinned preview; no train.json or detector/test file is consulted."""
    imaging_root = _locate_livecelltrack_imaging_root(root)
    sequences: list[TrackingSequence] = []
    for domain in LIVECELLTRACK_PREVIEW_DOMAINS:
        expected_acquisitions, expected_frames = LIVECELLTRACK_PREVIEW_EXPECTED[domain]
        data_root = imaging_root / domain / "data"
        acquisition_roots = sorted(
            (
                path
                for path in data_root.iterdir()
                if path.is_dir() and not path.name.startswith(".")
            ),
            key=_natural_key,
        )
        if len(acquisition_roots) != expected_acquisitions:
            raise RuntimeError(
                f"LiveCellTrack {domain} preview contains {len(acquisition_roots)} "
                f"acquisitions; expected {expected_acquisitions}"
            )
        sequences.extend(
            _livecelltrack_sequence(
                imaging_root,
                domain,
                acquisition_root,
                expected_frames=expected_frames,
            )
            for acquisition_root in acquisition_roots
        )
    groups = [sequence.acquisition_group for sequence in sequences]
    if len(groups) != len(set(groups)):
        raise RuntimeError("LiveCellTrack preview produced duplicate acquisition groups")
    return tuple(sorted(sequences, key=lambda sequence: sequence.acquisition_group))


def discover_transmitted_light_tracking(
    *,
    deepsea_root: Path | None = None,
    ctc_roots: Mapping[str, Path] | None = None,
    livecelltrack_root: Path | None = None,
    ctmc_root: Path | None = None,
    alfi_root: Path | None = None,
) -> tuple[TrackingSequence, ...]:
    """Unified API for all requested tracking sources, sorted by acquisition group."""
    sequences: list[TrackingSequence] = []
    if deepsea_root is not None:
        sequences.extend(discover_deepsea_training_tracking(deepsea_root))
    if ctc_roots is not None:
        unknown = sorted(set(ctc_roots) - set(CTC_TRACKING_DATASETS))
        if unknown:
            raise ValueError(f"Unsupported CTC root keys: {unknown}")
        for dataset in sorted(ctc_roots):
            sequences.extend(discover_ctc_training_tracking(ctc_roots[dataset], dataset))
    if livecelltrack_root is not None:
        sequences.extend(discover_livecelltrack_preview(livecelltrack_root))
    if ctmc_root is not None:
        sequences.extend(discover_ctmc_v1_training(ctmc_root))
    if alfi_root is not None:
        sequences.extend(discover_alfi_task1(alfi_root))
    sequences.sort(key=lambda sequence: sequence.acquisition_group)
    groups = [sequence.acquisition_group for sequence in sequences]
    if len(groups) != len(set(groups)):
        raise RuntimeError("Tracking sources produced duplicate acquisition groups")
    return tuple(sequences)


def tracking_summary(sequences: Iterable[TrackingSequence]) -> dict[str, object]:
    sequence_list = list(sequences)
    role_counts = {role: 0 for role in TRACKING_ROLES}
    dataset_counts: dict[str, int] = {}
    for sequence in sequence_list:
        role_counts[sequence.role] += 1
        dataset_counts[sequence.dataset] = dataset_counts.get(sequence.dataset, 0) + 1
    return {
        "schema_version": TRACKING_SCHEMA_VERSION,
        "acquisitions": len(sequence_list),
        "frames": sum(sequence.frame_count for sequence in sequence_list),
        "instance_frames": sum(
            sequence.annotated_instance_frames for sequence in sequence_list
        ),
        "tracks": sum(len(sequence.tracks) for sequence in sequence_list),
        "parent_links": sum(len(sequence.parent_links) for sequence in sequence_list),
        "lineage_annotated_acquisitions": sum(
            sequence.source_format != LIVECELLTRACK_SOURCE_FORMAT
            for sequence in sequence_list
        ),
        "identity_only_acquisitions": sum(
            sequence.source_format == LIVECELLTRACK_SOURCE_FORMAT
            for sequence in sequence_list
        ),
        "quarantined_ambiguous_identities": sum(
            len(sequence.annotation_exclusions) for sequence in sequence_list
        ),
        "quarantined_ambiguous_observations": sum(
            exclusion.observation_count
            for sequence in sequence_list
            for exclusion in sequence.annotation_exclusions
        ),
        "excluded_unpaired_source_files": sum(
            len(sequence.source_file_exclusions) for sequence in sequence_list
        ),
        "division_quarantined_parent_tracks": sum(
            len(sequence.division_quarantined_parent_track_ids)
            for sequence in sequence_list
        ),
        "event_quarantined_track_frames": sum(
            len(sequence.event_quarantined_track_frames)
            for sequence in sequence_list
        ),
        # Staggered identity annotation, reported so a reader can see how much association
        # supervision is adjacent-frame and how much the publisher simply did not label.
        "tracks_with_observation_gaps": sum(
            any(
                right > left + 1
                for left, right in zip(track.observed_frames, track.observed_frames[1:])
            )
            for sequence in sequence_list
            for track in sequence.tracks
        ),
        "observation_gaps": sum(
            right > left + 1
            for sequence in sequence_list
            for track in sequence.tracks
            for left, right in zip(track.observed_frames, track.observed_frames[1:])
        ),
        "longest_observation_gap_frames": max(
            (
                right - left - 1
                for sequence in sequence_list
                for track in sequence.tracks
                for left, right in zip(track.observed_frames, track.observed_frames[1:])
                if right > left + 1
            ),
            default=0,
        ),
        "unlabeled_mask_regions": sum(
            len(sequence.unlabeled_mask_regions) for sequence in sequence_list
        ),
        "unlabeled_mask_region_pixels": sum(
            region.area_pixels
            for sequence in sequence_list
            for region in sequence.unlabeled_mask_regions
        ),
        "unlabeled_mask_regions_touching_image_border": sum(
            region.touches_image_border
            for sequence in sequence_list
            for region in sequence.unlabeled_mask_regions
        ),
        "ctmc_alfi_bbox_proxies_materialized_on_disk": False,
        "ctmc_v1": {
            "official_partition": "train_only",
            "expected_current_page_totals": dict(CTMC_V1_EXPECTED),
            "license": CTMC_V1_LICENSE,
        },
        "alfi_task1": {
            "expected_current_archive_totals": dict(ALFI_EXPECTED),
            "license_conflict": ALFI_LICENSE,
            "task2_parsed": False,
            "semantic_masks_used_as_whole_cell_instances": False,
        },
        "acquisitions_by_role": role_counts,
        "acquisitions_by_dataset": dict(sorted(dataset_counts.items())),
        "official_test_labels_parsed": False,
    }


def _write_deepsea_label(path: Path, rows: Iterable[tuple[str, int, int]]) -> None:
    lines = ["label\tx\ty"]
    lines.extend(f"{label}\t{x}\t{y}" for label, x, y in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def synthetic_self_test() -> dict[str, object]:
    """Run all adapters and archive-safety checks on bounded local synthetic data."""
    with tempfile.TemporaryDirectory(prefix="cellect-tracking-self-test-") as temporary:
        root = Path(temporary)

        download_payload = b"bounded-livecelltrack-download-contract" * 7
        download_destination = root / "download" / "fixture.zip"
        download_destination.parent.mkdir(parents=True)
        partial = download_destination.with_suffix(".zip.partial")
        resume_offset = 19
        partial.write_bytes(download_payload[:resume_offset])
        observed_ranges: list[str | None] = []

        class _FakeRangeResponse(io.BytesIO):
            status = 206

            def __init__(self, payload: bytes) -> None:
                super().__init__(payload)
                self.headers = {
                    "Content-Range": (
                        f"bytes {resume_offset}-{len(download_payload) - 1}/"
                        f"{len(download_payload)}"
                    )
                }

            def getcode(self) -> int:
                return self.status

        def fake_opener(request: urllib.request.Request, timeout: int) -> _FakeRangeResponse:
            if timeout != 120:
                raise AssertionError("Verified downloader timeout contract changed")
            observed_ranges.append(request.get_header("Range"))
            return _FakeRangeResponse(download_payload[resume_offset:])

        downloaded = _download_verified_file(
            download_destination,
            url="https://no-network.invalid/livecelltrack.zip",
            expected_bytes=len(download_payload),
            expected_sha256=hashlib.sha256(download_payload).hexdigest(),
            allow_network=True,
            opener=fake_opener,
        )
        if (
            downloaded.read_bytes() != download_payload
            or observed_ranges != [f"bytes={resume_offset}-"]
        ):
            raise AssertionError("Verified resumable download contract failed")

        alfi_fixture_destination = root / "download" / "alfi_fixture.zip"
        alfi_partial = alfi_fixture_destination.with_suffix(".zip.partial")
        alfi_partial.write_bytes(download_payload[:resume_offset])
        alfi_ranges: list[str | None] = []

        def fake_alfi_opener(
            request: urllib.request.Request, timeout: int
        ) -> _FakeRangeResponse:
            if timeout != 120:
                raise AssertionError("ALFI downloader timeout contract changed")
            alfi_ranges.append(request.get_header("Range"))
            return _FakeRangeResponse(download_payload[resume_offset:])

        alfi_fixture = _download_verified_md5_file(
            alfi_fixture_destination,
            url="https://no-network.invalid/alfi.zip",
            expected_bytes=len(download_payload),
            expected_md5=hashlib.md5(
                download_payload, usedforsecurity=False
            ).hexdigest(),
            allow_network=True,
            opener=fake_alfi_opener,
        )
        if (
            alfi_fixture.read_bytes() != download_payload
            or alfi_ranges != [f"bytes={resume_offset}-"]
        ):
            raise AssertionError("ALFI resumable size/MD5 contract failed")

        ctmc_fixture = root / "download" / "CTMCV1-fixture.zip"
        ctmc_fixture.write_bytes(b"first successful CTMC bytes")
        ctmc_marker_directory = root / "download" / ".ctmc_provenance"
        _verify_or_record_ctmc_archive(
            ctmc_fixture,
            provenance="synthetic_self_test",
            marker_directory=ctmc_marker_directory,
            minimum_bytes=1,
        )
        ctmc_fixture.write_bytes(b"changed CTMC archive bytes")
        try:
            _verify_or_record_ctmc_archive(
                ctmc_fixture,
                provenance="synthetic_self_test_changed",
                marker_directory=ctmc_marker_directory,
                minimum_bytes=1,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("CTMC immutable first-byte marker accepted changed bytes")

        ctc_root = root / "ctc"
        archive = ctc_root / "BF-C2DL-HSC"
        raw = archive / "01"
        tra = archive / "01_GT" / "TRA"
        raw.mkdir(parents=True)
        tra.mkdir(parents=True)
        for frame_index in range(3):
            image = np.full((24, 32), frame_index * 20 + 30, dtype=np.uint8)
            mask = np.zeros(image.shape, dtype=np.uint16)
            if frame_index < 2:
                mask[7:15, 10 + frame_index : 16 + frame_index] = 1
            else:
                mask[5:11, 9:14] = 2
                mask[13:19, 15:20] = 3
            tifffile.imwrite(raw / f"t{frame_index:03d}.tif", image)
            tifffile.imwrite(tra / f"man_track{frame_index:03d}.tif", mask)
        (tra / "man_track.txt").write_text(
            "1 0 1 0\n2 2 2 1\n3 2 2 1\n", encoding="utf-8"
        )
        ctc_sequences = discover_ctc_training_tracking(ctc_root, "ctc_bf_hsc")
        if len(ctc_sequences) != 1 or len(ctc_sequences[0].parent_links) != 2:
            raise AssertionError("Synthetic CTC lineage was not recovered")

        # Reproduce the actual split-Google-Drive layout `track/tracking_dataset/train`.
        deepsea_collection = root / "deepsea"
        deepsea = deepsea_collection / "track" / "tracking_dataset"
        sequence = deepsea / "train" / "set_01_MESC"
        images = sequence / "images"
        masks = sequence / "masks"
        labels = sequence / "labels"
        for directory in (images, masks, labels):
            directory.mkdir(parents=True)
        for frame_index in range(3):
            stem = f"img_{frame_index:04d}"
            image = np.full((24, 32), frame_index * 15 + 40, dtype=np.uint8)
            mask = np.zeros(image.shape, dtype=np.uint8)
            if frame_index < 2:
                mask[7:15, 10 + frame_index : 16 + frame_index] = 255
                label_rows = [("A", 13 + frame_index, 10)]
            else:
                mask[5:11, 9:14] = 255
                mask[13:19, 17:22] = 255
                label_rows = [("A_1", 11, 7), ("A_2", 19, 15)]
            if not cv2.imwrite(str(images / f"{stem}.png"), image):
                raise AssertionError("Could not write synthetic DeepSea image")
            if not cv2.imwrite(
                str(masks / f"{stem}{DEEPSEA_MASK_SUFFIX}"), mask
            ):
                raise AssertionError("Could not write synthetic DeepSea mask")
            _write_deepsea_label(
                labels / f"{stem}{DEEPSEA_LABEL_SUFFIX}", label_rows
            )
        extra_stem = "img_0003"
        extra_mask = np.zeros((24, 32), dtype=np.uint8)
        extra_mask[8:16, 12:20] = 255
        if not cv2.imwrite(
            str(masks / f"{extra_stem}{DEEPSEA_MASK_SUFFIX}"), extra_mask
        ):
            raise AssertionError("Could not write synthetic orphan DeepSea mask")
        _write_deepsea_label(
            labels / f"{extra_stem}{DEEPSEA_LABEL_SUFFIX}", [("unused", 15, 11)]
        )
        # This malformed test decoy proves discovery never walks the official test partition.
        (deepsea / "test" / "do_not_parse").mkdir(parents=True)
        deepsea_sequences = discover_deepsea_training_tracking(deepsea_collection)
        if len(deepsea_sequences) != 1 or len(deepsea_sequences[0].parent_links) != 2:
            raise AssertionError("Synthetic DeepSea lineage was not recovered")
        deepsea_exclusions = deepsea_sequences[0].source_file_exclusions
        if (
            len(deepsea_exclusions) != 2
            or not all("no matching official image frame" in row.reason for row in deepsea_exclusions)
            or not all(re.fullmatch(r"[0-9a-f]{64}", row.sha256) for row in deepsea_exclusions)
        ):
            raise AssertionError("Unpaired DeepSea companions were not provenance-sealed")
        required_label = labels / f"img_0000{DEEPSEA_LABEL_SUFFIX}"
        required_label_bytes = required_label.read_bytes()
        required_label.unlink()
        try:
            discover_deepsea_training_tracking(deepsea_collection)
        except RuntimeError as error:
            if "missing labels" not in str(error):
                raise
        else:
            raise AssertionError("DeepSea adapter accepted an image without a label companion")
        finally:
            required_label.write_bytes(required_label_bytes)
        direct_tracking = root / "deepsea_direct" / "tracking_dataset"
        (direct_tracking / "train").mkdir(parents=True)
        if _locate_deepsea_tracking_root(direct_tracking) != direct_tracking.resolve():
            raise AssertionError("Direct DeepSea tracking root compatibility failed")

        livecelltrack_root = root / "livecelltrack" / "Cell_imaging"
        live_acquisition = (
            livecelltrack_root
            / "scratch_wound"
            / "data"
            / "acquisition_01"
        )
        live_images = live_acquisition / "img1"
        live_gt = live_acquisition / "gt" / "gt.txt"
        live_images.mkdir(parents=True)
        live_gt.parent.mkdir(parents=True)
        # A malformed train.json decoy proves the adapter needs only human MOT gt.txt rows.
        (livecelltrack_root / "scratch_wound" / "annotation").mkdir(parents=True)
        (livecelltrack_root / "scratch_wound" / "annotation" / "train.json").write_text(
            "this file must not be parsed", encoding="utf-8"
        )
        (livecelltrack_root / "HeLa" / "data").mkdir(parents=True)
        live_rows: list[str] = []
        for frame_index in range(1, 4):
            image = np.full((24, 32), frame_index * 18 + 20, dtype=np.uint8)
            tifffile.imwrite(live_images / f"{frame_index:06d}.tif", image)
            live_rows.append(f"{frame_index},1,{8 + frame_index},6,8,10,1,-1,-1,-1")
            if frame_index >= 2:
                live_rows.append(f"{frame_index},2,20,12,7,8,1,-1,-1,-1")
        # Identity 99 is deliberately ambiguous in frame 1. The strict adapter must quarantine
        # all of its rows rather than inventing two identities or choosing one box.
        live_rows.extend(
            (
                "1,99,2,2,5,5,1,-1,-1,-1",
                "1,99,8,2,5,5,1,-1,-1,-1",
                "2,99,4,3,5,5,1,-1,-1,-1",
            )
        )
        live_gt.write_text("\n".join(live_rows) + "\n", encoding="utf-8")
        livecelltrack_sequence = _livecelltrack_sequence(
            livecelltrack_root,
            "scratch_wound",
            live_acquisition,
            expected_frames=3,
        )
        if livecelltrack_sequence.parent_links or len(livecelltrack_sequence.tracks) != 2:
            raise AssertionError("LiveCellTrack adapter invented or lost lineage")
        if (
            len(livecelltrack_sequence.annotation_exclusions) != 1
            or livecelltrack_sequence.annotation_exclusions[0].source_label != "99"
            or livecelltrack_sequence.annotation_exclusions[0].observation_count != 3
        ):
            raise AssertionError("LiveCellTrack ambiguous identity was not audited/quarantined")
        if any(
            frame.instance_dtype != "uint32" or not frame.instance_path.is_file()
            for frame in livecelltrack_sequence.frames
        ):
            raise AssertionError("LiveCellTrack bounding-box shape proxies were not materialized")

        # CTMC-v1: exact seqinfo/img1/10-column MOT/4-column TRA contract. A malformed test
        # decoy proves the adapter discovers only the official train/ subtree. Two runs sharing
        # the same cell-line prefix must remain in one scientific role.
        ctmc_root = root / "ctmc_v1"
        ctmc_train = ctmc_root / "train"

        def write_ctmc_sequence(sequence_id: str) -> None:
            sequence_root = ctmc_train / sequence_id
            image_root = sequence_root / "img1"
            gt_path = sequence_root / "gt" / "gt.txt"
            tra_path = sequence_root / "TRA" / "man_track.txt"
            image_root.mkdir(parents=True)
            gt_path.parent.mkdir(parents=True)
            tra_path.parent.mkdir(parents=True)
            (sequence_root / "seqinfo.ini").write_text(
                "[Sequence]\n"
                f"name={sequence_id}\n"
                "imDir=img1\nframeRate=1\nseqLength=3\n"
                "imWidth=32\nimHeight=24\nimExt=.tif\n",
                encoding="utf-8",
            )
            rows: list[str] = []
            for frame_index in range(1, 4):
                tifffile.imwrite(
                    image_root / f"{frame_index:06d}.tif",
                    np.full((24, 32), 30 + frame_index * 10, dtype=np.uint8),
                )
                if frame_index < 3:
                    rows.append(
                        f"{frame_index},1,{7 + frame_index},6,8,9,1,-1,-1,-1"
                    )
                else:
                    rows.extend(
                        (
                            "3,2,7,5,7,8,1,-1,-1,-1",
                            "3,3,18,12,7,8,1,-1,-1,-1",
                        )
                    )
            gt_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            tra_path.write_text("1 1 2 0\n2 3 3 1\n3 3 3 1\n", encoding="utf-8")

        write_ctmc_sequence("HeLa-01")
        write_ctmc_sequence("HeLa-02")
        (ctmc_root / "test" / "do_not_parse").mkdir(parents=True)
        (ctmc_root / "test" / "do_not_parse" / "seqinfo.ini").write_text(
            "malformed hidden-test decoy", encoding="utf-8"
        )
        ctmc_sequences = discover_ctmc_v1_training(
            ctmc_root, enforce_expected_counts=False
        )
        if (
            len(ctmc_sequences) != 2
            or {sequence.role for sequence in ctmc_sequences}
            != {ctmc_sequences[0].role}
            or any(len(sequence.parent_links) != 2 for sequence in ctmc_sequences)
            or any(sequence.division_quarantined_parent_track_ids for sequence in ctmc_sequences)
        ):
            raise AssertionError("Synthetic CTMC-v1 roles/lineage contract failed")
        if any(
            frame.instance_dtype != "bbox_proxy_uint32_in_memory"
            or frame.instance_path.name != "gt.txt"
            for sequence in ctmc_sequences
            for frame in sequence.frames
        ):
            raise AssertionError("CTMC-v1 adapter materialized or mislabeled its box proxy")
        if bbox_proxy_labels(ctmc_sequences[0].frames[0]).dtype != np.uint32:
            raise AssertionError("CTMC-v1 in-memory box proxy contract failed")

        # ALFI: Task-1 images plus DTLTruth only. This fixture covers exact adjacent parent
        # resolution, a one-child/non-binary quarantine, and event-level duplicate quarantine.
        alfi_data = root / "alfi" / "Data&Annotations"
        alfi_sequence_root = alfi_data / "MI01"
        alfi_images = alfi_sequence_root / "Images"
        alfi_images.mkdir(parents=True)
        for frame_index in range(1, 5):
            if not cv2.imwrite(
                str(alfi_images / f"I_MI01_{frame_index:04d}.png"),
                np.full((28, 36), 35 + frame_index * 12, dtype=np.uint8),
            ):
                raise AssertionError("Could not write synthetic ALFI image")
        alfi_rows = (
            "ImNo,ID,Class,xmin,ymin,width,height,Parent",
            "1,1,Interphase,4,5,7,8,0",
            "2,1,Mitosis,5,5,7,8,0",
            "3,1.1,Interphase,4,4,6,7,1",
            "4,1.1,Interphase,5,4,6,7,1",
            "3,1.2,Interphase,14,12,6,7,1",
            "4,1.2,Interphase,15,12,6,7,1",
            "1,5,Interphase,24,3,6,7,0",
            "2,5.1,Interphase,24,4,6,7,5",
            "2,9,Interphase,2,18,5,6,0",
            "2,9,Interphase,9,18,5,6,0",
        )
        (alfi_sequence_root / "MI01_DTLTruth.csv").write_text(
            "\n".join(alfi_rows) + "\n", encoding="utf-8"
        )
        (root / "alfi" / "Task2" / "do_not_parse").mkdir(parents=True)
        (root / "alfi" / "Task2" / "do_not_parse" / "bad.csv").write_text(
            "malformed Task-2 decoy", encoding="utf-8"
        )
        alfi_sequence = _alfi_sequence(alfi_data, "MI01")
        if (
            len(alfi_sequence.annotation_exclusions) != 1
            or alfi_sequence.annotation_exclusions[0].source_label != "9"
            or alfi_sequence.annotation_exclusions[0].observation_count != 2
            or len(alfi_sequence.parent_links) != 3
            or len(alfi_sequence.division_quarantined_parent_track_ids) != 1
        ):
            raise AssertionError("Synthetic ALFI parent/duplicate quarantine contract failed")
        ambiguous_instances = [
            instance
            for frame in alfi_sequence.frames
            for instance in frame.instances
            if instance.source_label == "9"
        ]
        if not ambiguous_instances or any(
            instance.identity_supervision_available for instance in ambiguous_instances
        ):
            raise AssertionError("ALFI duplicate events retained identity supervision")
        if any(
            frame.instance_dtype != "bbox_proxy_uint32_in_memory"
            or frame.instance_path.name != "MI01_DTLTruth.csv"
            for frame in alfi_sequence.frames
        ):
            raise AssertionError("ALFI adapter materialized or mislabeled its box proxy")
        if bbox_proxy_labels(alfi_sequence.frames[1]).dtype != np.uint32:
            raise AssertionError("ALFI in-memory box proxy contract failed")

        malicious_archive = root / "malicious_livecelltrack.zip"
        with zipfile.ZipFile(malicious_archive, "w") as archive:
            archive.writestr("../escape.txt", "unsafe")
            archive.writestr("Cell_imaging/placeholder.txt", "placeholder")
        try:
            _safe_extract_livecelltrack_archive(
                malicious_archive, root / "rejected_livecelltrack"
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError("LiveCellTrack safe extraction accepted path traversal")
        if (root / "escape.txt").exists():
            raise AssertionError("LiveCellTrack path traversal escaped its staging directory")

        all_sequences = tuple(
            ctc_sequences
            + deepsea_sequences
            + (livecelltrack_sequence,)
            + ctmc_sequences
            + (alfi_sequence,)
        )
        summary = tracking_summary(all_sequences)
        if summary["official_test_labels_parsed"] is not False:
            raise AssertionError("Synthetic tracking adapter opened test labels")
        if summary["frames"] != 19 or summary["parent_links"] != 11:
            raise AssertionError("Synthetic tracking totals are incorrect")
        return {
            "status": "PASS",
            "ctc_adapter": "PASS",
            "deepsea_official_adapter": "PASS",
            "deepsea_image_anchored_companion_provenance": "PASS",
            "livecelltrack_mot_adapter": "PASS",
            "livecelltrack_bbox_proxy": "PASS",
            "ctmc_v1_train_only_adapter": "PASS",
            "ctmc_v1_cell_line_role_grouping": "PASS",
            "ctmc_v1_in_memory_bbox_proxy": "PASS",
            "alfi_task1_only_adapter": "PASS",
            "alfi_duplicate_event_quarantine": "PASS",
            "alfi_in_memory_bbox_proxy": "PASS",
            "ambiguous_identity_quarantine": "PASS",
            "verified_archive_safe_extraction": "PASS",
            "verified_resumable_download": "PASS",
            "alfi_resumable_size_md5": "PASS",
            "ctmc_immutable_first_byte_sha": "PASS",
            "complete_sequence_integrity": "PASS",
            "lineage_consistency": "PASS",
            "test_label_seal": "PASS",
            "summary": summary,
            "sequence_fingerprints": [
                sequence.sequence_fingerprint_sha256 for sequence in all_sequences
            ],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help=(
            "run bounded synthetic CTC, DeepSea, LiveCellTrack, CTMC-v1, and ALFI adapter tests"
        ),
    )
    args = parser.parse_args()
    if not args.self_test:
        parser.error("no action selected; use --self-test")
    print(json.dumps(synthetic_self_test(), indent=2))


if __name__ == "__main__":
    main()
