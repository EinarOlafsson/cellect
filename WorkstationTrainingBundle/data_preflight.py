#!/usr/bin/env python3
"""Streaming, full-data integrity and split preflight for the Cellect v3 bundle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from pycocotools import mask as coco_mask
from pycocotools.coco import COCO
from tqdm import tqdm

from accuracy_data import ExternalSample, load_external_record
from mask_targets import internal_contact_boundary
from scientific_splits import (
    BUNDLE_VERSION,
    LIVECELL_ROLES,
    livecell_role,
    scientific_group,
    validation_role,
)


_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def _livecell_image(images_root: Path, file_name: str) -> Path:
    candidates = (
        images_root / file_name,
        images_root / "images" / file_name,
        images_root / "livecell_train_val_images" / file_name,
        images_root / "livecell_test_images" / file_name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = list(images_root.rglob(Path(file_name).name))
    if len(matches) == 1:
        return matches[0]
    for match in matches:
        if str(match).endswith(file_name):
            return match
    raise FileNotFoundError(f"Could not locate LIVECell image {file_name}")


def _fingerprint(rows: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows):
        digest.update(row.encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _livecell_test_image_files(images_root: Path) -> list[Path]:
    """Find official test images without consulting or decoding the test COCO labels."""
    roots: set[Path] = set()
    for candidate in (
        images_root / "livecell_test_images",
        images_root / "images" / "livecell_test_images",
    ):
        if candidate.is_dir():
            roots.add(candidate.resolve())
    for candidate in images_root.rglob("livecell_test_images"):
        if candidate.is_dir():
            roots.add(candidate.resolve())
    files = {
        path.resolve()
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
    }
    return sorted(files, key=lambda path: path.as_posix())


def _sealed_asset(path: Path | None, label: str) -> tuple[Path | None, int, str]:
    """Existence-check and hash an asset without interpreting its contents."""
    if path is None:
        return None, 0, "none"
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved, resolved.stat().st_size, _file_sha256(resolved)


def run_full_data_preflight(
    annotations_root: Path,
    images_root: Path,
    external_samples: list[ExternalSample],
    destination: Path,
) -> dict[str, object]:
    """Validate development labels while cryptographically sealing final-test labels."""
    started = time.time()
    errors: list[str] = []
    warnings: list[str] = []
    fingerprint_rows: list[str] = []
    manifest_rows: list[dict[str, object]] = []
    livecell_report: dict[str, object] = {}
    livecell_paths_by_split: dict[str, set[Path]] = {}
    livecell_hashes_by_split: dict[str, set[str]] = {}
    livecell_role_images: dict[str, int] = defaultdict(int)
    livecell_role_instances: dict[str, int] = defaultdict(int)
    livecell_group_roles: dict[str, set[str]] = defaultdict(set)

    # LIVECell train and validation form one acquisition-grouped development pool.  Parsing the
    # two development JSON files is intentional; the official test JSON is handled separately
    # below and is never passed to COCO.
    for split in ("train", "val"):
        annotation_path = annotations_root / f"livecell_coco_{split}.json"
        try:
            resolved_annotation, annotation_size, annotation_sha256 = _sealed_asset(
                annotation_path, f"LIVECell {split} annotation"
            )
            assert resolved_annotation is not None
            coco = COCO(str(resolved_annotation))
        except Exception as error:
            errors.append(f"LIVECell {split} annotation: {error}")
            livecell_paths_by_split[split] = set()
            livecell_hashes_by_split[split] = set()
            livecell_report[split] = {
                "annotation_parsed": False,
                "error": str(error),
            }
            continue
        image_ids = sorted(coco.getImgIds())
        split_paths: set[Path] = set()
        split_hashes: set[str] = set()
        annotation_count = 0
        split_role_images: dict[str, int] = defaultdict(int)
        split_role_instances: dict[str, int] = defaultdict(int)
        cell_lines: dict[str, int] = defaultdict(int)
        for image_id in tqdm(image_ids, desc=f"preflight LIVECell {split}"):
            metadata = coco.loadImgs([image_id])[0]
            try:
                path = _livecell_image(images_root, metadata["file_name"]).resolve()
                image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if image is None:
                    raise RuntimeError("OpenCV returned no pixels")
                if tuple(image.shape[:2]) != (
                    int(metadata["height"]),
                    int(metadata["width"]),
                ):
                    raise RuntimeError(
                        f"decoded shape {image.shape[:2]} differs from COCO "
                        f"{metadata['height']}x{metadata['width']}"
                    )
                annotation_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=None)
                annotations = coco.loadAnns(annotation_ids)
                if not annotations:
                    raise RuntimeError("image has no cell annotations")
                for annotation in annotations:
                    if float(annotation.get("area", 0)) <= 0:
                        raise RuntimeError("annotation has non-positive area")
                    if not annotation.get("segmentation"):
                        raise RuntimeError("annotation has no segmentation geometry")
                    try:
                        encoded = coco.annToRLE(annotation)
                        encoded_area = float(coco_mask.area(encoded))
                    except Exception as error:
                        raise RuntimeError(
                            f"annotation {annotation.get('id')} cannot be encoded: {error}"
                        ) from error
                    if not np.isfinite(encoded_area) or encoded_area <= 0:
                        raise RuntimeError(
                            f"annotation {annotation.get('id')} decodes to zero/invalid area"
                        )
                annotation_count += len(annotations)
                split_paths.add(path)
                cell_line = Path(metadata["file_name"]).name.split("_")[0]
                cell_lines[cell_line] += 1
                group = scientific_group("livecell", metadata["file_name"])
                role = livecell_role(metadata["file_name"])
                livecell_group_roles[group].add(role)
                livecell_role_images[role] += 1
                livecell_role_instances[role] += len(annotations)
                split_role_images[role] += 1
                split_role_instances[role] += len(annotations)
                stat = path.stat()
                image_sha256 = _file_sha256(path)
                split_hashes.add(image_sha256)
                manifest_rows.append(
                    {
                        "dataset": "livecell",
                        "sample_id": int(image_id),
                        "upstream_split": split,
                        "role": role,
                        "acquisition_group": group,
                        "image_path": str(path),
                        "image_size": stat.st_size,
                        "image_sha256": image_sha256,
                        "mask_source": str(resolved_annotation),
                        "mask_size": annotation_size,
                        "mask_sha256": annotation_sha256,
                        "labels_parsed": True,
                        "instance_count": len(annotations),
                        "height": int(image.shape[0]),
                        "width": int(image.shape[1]),
                    }
                )
                fingerprint_rows.append(
                    f"livecell:{split}:{path}:{stat.st_size}:{image_sha256}:"
                    f"{resolved_annotation}:{annotation_size}:{annotation_sha256}:"
                    f"{role}:{group}:{len(annotations)}"
                )
            except Exception as error:  # report all corrupt files in one pass
                errors.append(f"LIVECell {split} image {image_id}: {error}")
        livecell_paths_by_split[split] = split_paths
        livecell_hashes_by_split[split] = split_hashes
        livecell_report[split] = {
            "images": len(image_ids),
            "annotations": annotation_count,
            "annotation_path": str(resolved_annotation),
            "annotation_size": annotation_size,
            "annotation_sha256": annotation_sha256,
            "annotation_parsed": True,
            "cell_line_images": dict(sorted(cell_lines.items())),
            "development_role_images": dict(sorted(split_role_images.items())),
            "development_role_instances": dict(sorted(split_role_instances.items())),
        }

    crossing_livecell_groups = {
        group: sorted(roles)
        for group, roles in livecell_group_roles.items()
        if len(roles) > 1
    }
    for group, roles in list(sorted(crossing_livecell_groups.items()))[:25]:
        errors.append(
            f"LIVECell combined train/val acquisition group crosses roles: "
            f"{group} -> {roles}"
        )
    livecell_group_counts: dict[str, int] = defaultdict(int)
    for roles in livecell_group_roles.values():
        if len(roles) == 1:
            livecell_group_counts[next(iter(roles))] += 1
    livecell_report["combined_train_val"] = {
        "source_splits": ["train", "val"],
        "assignment": "deterministic acquisition-group hash; 70/10/10/10 target",
        "image_counts_by_role": dict(sorted(livecell_role_images.items())),
        "instance_counts_by_role": dict(sorted(livecell_role_instances.items())),
        "acquisition_group_counts_by_role": dict(sorted(livecell_group_counts.items())),
        "acquisition_groups": len(livecell_group_roles),
        "groups_crossing_roles": len(crossing_livecell_groups),
    }

    # Final-test label sealing: hash/size the JSON as opaque bytes.  Images are discovered from
    # the official image directory rather than by reading names or dimensions from that JSON.
    test_split = "test"
    test_annotation_path = annotations_root / "livecell_coco_test.json"
    test_paths: set[Path] = set()
    test_hashes: set[str] = set()
    try:
        (
            resolved_test_annotation,
            test_annotation_size,
            test_annotation_sha256,
        ) = _sealed_asset(test_annotation_path, "LIVECell test annotation")
        assert resolved_test_annotation is not None
        fingerprint_rows.append(
            f"livecell:test-annotation-sealed:{resolved_test_annotation}:"
            f"{test_annotation_size}:{test_annotation_sha256}"
        )
        test_image_files = _livecell_test_image_files(images_root)
        if not test_image_files:
            raise FileNotFoundError(
                f"no images found beneath a livecell_test_images directory in {images_root}"
            )
        total_test_image_bytes = 0
        for path in tqdm(test_image_files, desc="seal LIVECell test images"):
            try:
                # Decoding the image checks transport integrity.  No image dimensions or pixel
                # statistics are retained, and the sealed annotation remains unopened.
                decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                if decoded is None:
                    raise RuntimeError("OpenCV returned no pixels")
                del decoded
                stat = path.stat()
                image_sha256 = _file_sha256(path)
                group = scientific_group("livecell", path.name)
                test_paths.add(path)
                test_hashes.add(image_sha256)
                total_test_image_bytes += stat.st_size
                manifest_rows.append(
                    {
                        "dataset": "livecell",
                        "sample_id": path.name,
                        "upstream_split": test_split,
                        "role": "final_test",
                        "acquisition_group": group,
                        "image_path": str(path),
                        "image_size": stat.st_size,
                        "image_sha256": image_sha256,
                        "mask_source": str(resolved_test_annotation),
                        "mask_size": test_annotation_size,
                        "mask_sha256": test_annotation_sha256,
                        "labels_parsed": False,
                        "sealed_test_labels": True,
                    }
                )
                fingerprint_rows.append(
                    f"livecell:test-sealed:{path}:{stat.st_size}:{image_sha256}:"
                    f"{resolved_test_annotation}:{test_annotation_size}:"
                    f"{test_annotation_sha256}"
                )
            except Exception as error:
                errors.append(f"LIVECell test image {path}: {error}")
        livecell_report[test_split] = {
            "images_integrity_checked_and_hashed": len(test_paths),
            "image_bytes": total_test_image_bytes,
            "annotation_path": str(resolved_test_annotation),
            "annotation_size": test_annotation_size,
            "annotation_sha256": test_annotation_sha256,
            "annotation_parsed": False,
            "mask_statistics_exposed": False,
            "seal_status": "opaque label file hashed; no COCO parsing or mask decoding",
        }
    except Exception as error:
        errors.append(f"LIVECell test seal: {error}")
        livecell_report[test_split] = {
            "annotation_parsed": False,
            "mask_statistics_exposed": False,
            "seal_status": "failed",
            "error": str(error),
        }
    livecell_paths_by_split[test_split] = test_paths
    livecell_hashes_by_split[test_split] = test_hashes

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = livecell_paths_by_split[left] & livecell_paths_by_split[right]
        if overlap:
            errors.append(
                f"LIVECell physical image leakage between {left} and {right}: "
                f"{len(overlap)} files"
            )
        hash_overlap = livecell_hashes_by_split[left] & livecell_hashes_by_split[right]
        if hash_overlap:
            errors.append(
                f"LIVECell byte-identical content leakage between {left} and {right}: "
                f"{len(hash_overlap)} images"
            )

    external_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    external_cells: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    external_validation_roles: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    external_sealed_test_pairs: dict[str, int] = defaultdict(int)
    seen_external_paths: dict[Path, tuple[str, str]] = {}
    seen_external_hashes: dict[str, tuple[str, str, Path]] = {}
    for sample in tqdm(external_samples, desc="preflight all external pairs"):
        try:
            image_path, image_size, image_sha256 = _sealed_asset(
                sample.image_path, "external image"
            )
            mask_path, mask_size, mask_sha256 = _sealed_asset(
                sample.instance_path, "external instance mask"
            )
            class_path, class_size, class_sha256 = _sealed_asset(
                sample.class_path, "external class labels"
            )
            boundary_path, boundary_size, boundary_sha256 = _sealed_asset(
                sample.boundary_path, "external boundary mask"
            )
            assert image_path is not None
            assert mask_path is not None
            prior = seen_external_paths.get(image_path)
            if prior is not None and prior[1] != sample.split:
                raise RuntimeError(
                    f"same image appears in {prior[1]} and {sample.split}"
                )
            seen_external_paths[image_path] = (sample.dataset, sample.split)
            if sample.dataset == "deepsea_phase":
                if sample.boundary_path is None:
                    raise RuntimeError(
                        "missing official unetwmaps touching-edge file; refusing to infer "
                        "instances from DeepSea's binary mask alone"
                    )
                if boundary_path is None:
                    raise RuntimeError("DeepSea touching-edge file could not be sealed")
            if sample.split == "val":
                assigned_validation_role = validation_role(
                    sample.dataset, sample.image_path
                )
                role = assigned_validation_role
            else:
                role = "train" if sample.split == "train" else "final_test"
            prior_hash = seen_external_hashes.get(image_sha256)
            if prior_hash is not None and prior_hash[1] != sample.split:
                raise RuntimeError(
                    "byte-identical image content crosses splits: "
                    f"{prior_hash[2]} ({prior_hash[1]})"
                )
            seen_external_hashes[image_sha256] = (
                sample.dataset,
                sample.split,
                image_path,
            )

            group = scientific_group(sample.dataset, sample.image_path)
            manifest_row: dict[str, object] = {
                "dataset": sample.dataset,
                "sample_id": str(sample.image_path),
                "upstream_split": sample.split,
                "role": role,
                "acquisition_group": group,
                "image_path": str(image_path),
                "image_size": image_size,
                "image_sha256": image_sha256,
                "mask_path": str(mask_path),
                "mask_size": mask_size,
                "mask_sha256": mask_sha256,
                "class_path": str(class_path) if class_path is not None else None,
                "class_size": class_size,
                "class_sha256": class_sha256,
                "boundary_path": (
                    str(boundary_path) if boundary_path is not None else None
                ),
                "boundary_size": boundary_size,
                "boundary_sha256": boundary_sha256,
            }

            if sample.split == "test":
                # Never invoke any data loader on final-test assets during preflight: even a
                # seemingly harmless shape/count check can expose label information before lock.
                manifest_row.update(
                    {
                        "labels_parsed": False,
                        "sealed_test_labels": True,
                    }
                )
                cell_count_fingerprint = "sealed"
            else:
                image, labels, explicit_boundary = load_external_record(sample)
                if image.ndim != 2 or labels.ndim != 2:
                    raise RuntimeError(
                        f"expected 2-D image/labels, got {image.shape}/{labels.shape}"
                    )
                if image.shape != labels.shape:
                    raise RuntimeError(
                        f"image/label shape mismatch {image.shape}/{labels.shape}"
                    )
                if not np.isfinite(image).all():
                    raise RuntimeError("image contains NaN or infinity")
                cell_count = int(np.unique(labels[labels > 0]).size)
                if cell_count < 1:
                    raise RuntimeError("mask contains no cell instances")
                if explicit_boundary is not None:
                    if explicit_boundary.shape != labels.shape:
                        raise RuntimeError("explicit boundary has the wrong shape")
                    if (
                        explicit_boundary.any()
                        and not internal_contact_boundary(
                            labels, explicit_boundary
                        ).any()
                    ):
                        raise RuntimeError(
                            "DeepSea touching-edge map has no component adjacent to two "
                            "reconstructed cells"
                        )
                external_cells[sample.dataset][sample.split] += cell_count
                manifest_row.update(
                    {
                        "labels_parsed": True,
                        "instance_count": cell_count,
                        "height": int(image.shape[0]),
                        "width": int(image.shape[1]),
                    }
                )
                cell_count_fingerprint = str(cell_count)

            external_counts[sample.dataset][sample.split] += 1
            if sample.split == "test":
                external_sealed_test_pairs[sample.dataset] += 1
            elif sample.split == "val":
                external_validation_roles[sample.dataset][role] += 1
            manifest_rows.append(manifest_row)
            fingerprint_rows.append(
                f"{sample.dataset}:{sample.split}:{role}:{group}:"
                f"{image_path}:{image_size}:{image_sha256}:"
                f"{mask_path}:{mask_size}:{mask_sha256}:"
                f"{class_path or 'none'}:{class_size}:{class_sha256}:"
                f"{boundary_path or 'none'}:{boundary_size}:{boundary_sha256}:"
                f"{cell_count_fingerprint}"
            )
        except Exception as error:  # keep scanning to provide one actionable report
            errors.append(
                f"{sample.dataset} {sample.split} {sample.image_path}: {error}"
            )

    roles_by_acquisition: dict[tuple[str, str], set[str]] = defaultdict(set)
    content_locations: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for row in manifest_rows:
        roles_by_acquisition[(str(row["dataset"]), str(row["acquisition_group"]))].add(
            str(row["role"])
        )
        content_locations[str(row["image_sha256"])].append(
            (str(row["dataset"]), str(row["role"]), str(row["image_path"]))
        )
    leaking_groups = [
        (dataset, group, sorted(roles))
        for (dataset, group), roles in roles_by_acquisition.items()
        if len(roles) > 1
    ]
    for dataset, group, roles in leaking_groups[:25]:
        errors.append(
            f"acquisition group crosses scientific roles: {dataset} {group} -> {roles}"
        )
    for locations in content_locations.values():
        unique_paths = {location[2] for location in locations}
        roles = {location[1] for location in locations}
        if len(unique_paths) > 1 and len(roles) > 1:
            errors.append(
                "byte-identical image crosses scientific roles: "
                + "; ".join(
                    f"{dataset}/{role} {path}" for dataset, role, path in locations[:5]
                )
            )

    missing_livecell_roles = [
        role for role in LIVECELL_ROLES if int(livecell_role_images.get(role, 0)) == 0
    ]
    if missing_livecell_roles:
        errors.append(
            "LIVECell combined train/val development pool is missing roles: "
            + ", ".join(missing_livecell_roles)
        )
    for dataset, counts in external_counts.items():
        if not counts.get("train"):
            warnings.append(f"{dataset} contributes no training images")
        if not counts.get("test"):
            warnings.append(
                f"{dataset} has no independent test images; do not make a domain-specific "
                "generalization claim from its validation masks"
            )
    deepsea_total = sum(external_counts.get("deepsea_phase", {}).values())
    # The public sample is 100 triplets. DeepSea's manuscript component counts total 3,624,
    # while the complete redistributed segmentation archive is indexed as 3,686; accept only
    # those known complete layouts so a missing multi-part Google Drive ZIP fails before training.
    if deepsea_total not in {100, 3624, 3686}:
        errors.append(
            "DeepSea discovery is incomplete or unexpected: found "
            f"{deepsea_total} strict image/mask/wmap triplets; expected either the verified "
            "100-pair public sample or a known complete 3,624/3,686-pair local collection"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(destination.parent)
    free_gib = usage.free / (1024**3)
    minimum_free_gib = float(os.environ.get("CELLECT_MIN_FREE_GIB", "100"))
    if free_gib < minimum_free_gib:
        errors.append(
            f"Only {free_gib:.1f} GiB free; at least {minimum_free_gib:.1f} GiB is required"
        )
    elif free_gib < 150:
        warnings.append(
            f"Only {free_gib:.1f} GiB free. Best training may need more space for checkpoints "
            "and prediction caches."
        )

    manifest_path = destination.parent / "splits_v3.jsonl"
    manifest_temporary = manifest_path.with_suffix(".jsonl.tmp")
    manifest_temporary.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in manifest_rows
        )
    )
    manifest_temporary.replace(manifest_path)
    manifest_sha256 = _file_sha256(manifest_path)
    report: dict[str, object] = {
        "schema_version": 4,
        "bundle_version": BUNDLE_VERSION,
        "status": "passed" if not errors else "failed",
        "completed_unix_seconds": time.time(),
        "duration_seconds": time.time() - started,
        "dataset_fingerprint_sha256": _fingerprint(fingerprint_rows),
        "split_manifest": manifest_path.name,
        "split_manifest_sha256": manifest_sha256,
        "split_manifest_rows": len(manifest_rows),
        "livecell": livecell_report,
        "external_pairs": {
            dataset: dict(sorted(counts.items()))
            for dataset, counts in sorted(external_counts.items())
        },
        "external_instances_parsed_development_only": {
            dataset: dict(sorted(counts.items()))
            for dataset, counts in sorted(external_cells.items())
        },
        "external_test_labels": {
            "access_policy": (
                "final-test image/mask/class/boundary assets were existence-checked and "
                "SHA-256 sealed only; no label loader was called and no instance counts "
                "were exposed"
            ),
            "sealed_pairs_by_dataset": dict(sorted(external_sealed_test_pairs.items())),
        },
        "external_validation_roles": {
            dataset: dict(sorted(counts.items()))
            for dataset, counts in sorted(external_validation_roles.items())
        },
        "disk_free_gib": free_gib,
        "minimum_disk_free_gib": minimum_free_gib,
        "warnings": warnings,
        "errors": errors,
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(destination)
    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:25])
        suffix = "\n- ..." if len(errors) > 25 else ""
        raise RuntimeError(
            f"Full-data preflight failed with {len(errors)} error(s):\n{preview}{suffix}\n"
            f"Complete report: {destination}"
        )
    print(f"Full-data preflight passed: {destination}")
    return report
