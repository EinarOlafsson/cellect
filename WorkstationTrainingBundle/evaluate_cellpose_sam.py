#!/usr/bin/env python3
"""Evaluate a Cellpose-family checkpoint on COCO or materialized scientific roles."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from cellpose import metrics, models
from pycocotools.coco import COCO
from scipy import ndimage
from tqdm import tqdm

from scientific_splits import VALIDATION_ROLES, belongs_to_validation_role


PAIR_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_image(images_root: Path, file_name: str) -> Path:
    candidates = [
        images_root / file_name,
        images_root / "images" / file_name,
        images_root / "livecell_train_val_images" / file_name,
        images_root / "livecell_test_images" / file_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(images_root.rglob(Path(file_name).name))
    if not matches:
        raise FileNotFoundError(file_name)
    return matches[0]


def ground_truth(coco: COCO, image_id: int, height: int, width: int) -> np.ndarray:
    labels = np.zeros((height, width), dtype=np.int32)
    annotation_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=None)
    for label, annotation in enumerate(coco.loadAnns(annotation_ids), start=1):
        labels[coco.annToMask(annotation).astype(bool)] = label
    return labels


def pair_domain(image_path: Path) -> str:
    """Recover the materialized dataset name before ``_(train|val|test)_``."""
    stem = image_path.stem
    if not stem.lower().endswith("_img"):
        raise ValueError(f"Materialized image does not end in _img: {image_path}")
    source_stem = stem[:-4]
    match = re.match(r"^(?P<domain>.+?)_(?:train|val|test)_", source_stem, flags=re.I)
    if match is None:
        raise ValueError(
            "Cannot derive a dataset domain before _(train|val|test)_ from "
            f"{image_path.name}"
        )
    return match.group("domain").lower()


def discover_materialized_pairs(
    pairs_directory: Path,
    limit_per_domain: int | None = None,
) -> list[tuple[Path, Path, str]]:
    """Find strict ``*_img.*``/``*_masks.tif`` pairs in one scientific-role folder."""
    if not pairs_directory.is_dir():
        raise FileNotFoundError(
            f"Materialized pair directory does not exist: {pairs_directory}"
        )
    discovered: list[tuple[Path, Path, str]] = []
    seen_masks: set[Path] = set()
    domain_counts: dict[str, int] = defaultdict(int)
    image_paths = sorted(
        (
            path
            for path in pairs_directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in PAIR_IMAGE_EXTENSIONS
            and path.stem.lower().endswith("_img")
        ),
        key=lambda path: path.as_posix().casefold(),
    )
    for image_path in image_paths:
        mask_path = image_path.with_name(f"{image_path.stem[:-4]}_masks.tif")
        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Materialized Cellpose image has no matching mask: {image_path}"
            )
        resolved_mask = mask_path.resolve()
        if resolved_mask in seen_masks:
            raise RuntimeError(
                f"Materialized mask is paired more than once: {mask_path}"
            )
        domain = pair_domain(image_path)
        if limit_per_domain is not None and domain_counts[domain] >= limit_per_domain:
            continue
        seen_masks.add(resolved_mask)
        domain_counts[domain] += 1
        discovered.append((image_path, mask_path, domain))
    if not discovered:
        raise RuntimeError(
            f"No materialized *_img.*/*_masks.tif pairs in {pairs_directory}"
        )
    return discovered


def load_pair_truth(mask_path: Path, expected_shape: tuple[int, int]) -> np.ndarray:
    labels = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if labels is None:
        raise RuntimeError(f"Could not decode materialized mask {mask_path}")
    if labels.ndim == 3:
        labels = labels[..., 0]
    if tuple(labels.shape) != expected_shape:
        raise RuntimeError(
            f"Materialized image/mask shape mismatch for {mask_path}: "
            f"{expected_shape} versus {labels.shape}"
        )
    labels = np.asarray(labels, dtype=np.int32)
    if not np.any(labels > 0):
        raise RuntimeError(f"Materialized mask contains no instances: {mask_path}")
    return labels


def boundary_mask(labels: np.ndarray) -> np.ndarray:
    foreground = labels > 0
    boundary = np.zeros(labels.shape, dtype=bool)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted = np.roll(labels, shift=(dy, dx), axis=(0, 1))
        boundary |= foreground & (shifted != labels)
    boundary[[0, -1], :] |= foreground[[0, -1], :]
    boundary[:, [0, -1]] |= foreground[:, [0, -1]]
    return ndimage.binary_dilation(boundary, iterations=1) & foreground


def safe_ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 1.0
    return numerator / denominator


def summarize_rows(rows: list[dict[str, float]]) -> dict[str, object]:
    if not rows:
        raise ValueError("Cannot summarize an empty Cellpose evaluation role")

    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in rows]))

    pixel_dice = mean("pixel_dice")
    boundary_dice = mean("boundary_dice")
    ap50 = mean("instance_ap50")
    ap75 = mean("instance_ap75")
    return {
        "evaluation_images": len(rows),
        "pixel_dice": pixel_dice,
        "pixel_iou": mean("pixel_iou"),
        "foreground_precision": mean("foreground_precision"),
        "foreground_recall": mean("foreground_recall"),
        "boundary_dice": boundary_dice,
        "boundary_precision": mean("boundary_precision"),
        "boundary_recall": mean("boundary_recall"),
        "count_mae": mean("count_absolute_error"),
        "seconds_per_image": mean("seconds_per_image"),
        "instance_average_precision": {
            "0.5": ap50,
            "0.75": ap75,
            "0.9": mean("instance_ap90"),
        },
        "selection_score": (
            0.25 * pixel_dice + 0.15 * boundary_dice + 0.35 * ap50 + 0.25 * ap75
        ),
    }


def macro_average_domains(
    per_domain: dict[str, dict[str, object]],
) -> dict[str, object]:
    if not per_domain:
        raise ValueError("Cellpose evaluation produced no domains")

    def domain_mean(key: str) -> float:
        return float(np.mean([float(metrics[key]) for metrics in per_domain.values()]))

    def domain_ap(threshold: str) -> float:
        return float(
            np.mean(
                [
                    float(metrics["instance_average_precision"][threshold])
                    for metrics in per_domain.values()
                ]
            )
        )

    pixel_dice = domain_mean("pixel_dice")
    boundary_dice = domain_mean("boundary_dice")
    ap50 = domain_ap("0.5")
    ap75 = domain_ap("0.75")
    return {
        "domain_count": len(per_domain),
        "pixel_dice": pixel_dice,
        "pixel_iou": domain_mean("pixel_iou"),
        "foreground_precision": domain_mean("foreground_precision"),
        "foreground_recall": domain_mean("foreground_recall"),
        "boundary_dice": boundary_dice,
        "boundary_precision": domain_mean("boundary_precision"),
        "boundary_recall": domain_mean("boundary_recall"),
        "count_mae": domain_mean("count_mae"),
        "seconds_per_image": domain_mean("seconds_per_image"),
        "instance_average_precision": {
            "0.5": ap50,
            "0.75": ap75,
            "0.9": domain_ap("0.9"),
        },
        "selection_score": (
            0.25 * pixel_dice + 0.15 * boundary_dice + 0.35 * ap50 + 0.25 * ap75
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--annotations", type=Path)
    source.add_argument("--pairs-dir", type=Path)
    parser.add_argument("--images-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--flow-threshold", type=float, default=0.4)
    parser.add_argument("--cellprob-threshold", type=float, default=0.0)
    parser.add_argument("--validation-role", choices=VALIDATION_ROLES)
    parser.add_argument("--tile-size", type=int, default=256)
    parser.add_argument("--tile-overlap", type=float, default=0.25)
    parser.add_argument("--tile-batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.annotations is not None and args.images_root is None:
        parser.error("--images-root is required with --annotations")
    if args.pairs_dir is not None and args.images_root is not None:
        parser.error("--images-root is only valid with --annotations")

    coco: COCO | None = None
    evaluation_samples: list[tuple[object, Path, Path | None, str]] = []
    source_mode: str
    if args.pairs_dir is not None:
        source_mode = "materialized_pairs"
        for image_path, mask_path, domain in discover_materialized_pairs(
            args.pairs_dir,
            limit_per_domain=args.limit if args.limit else None,
        ):
            evaluation_samples.append((image_path.name, image_path, mask_path, domain))
    else:
        source_mode = "livecell_coco_compatibility"
        assert args.annotations is not None and args.images_root is not None
        coco = COCO(str(args.annotations))
        image_ids = sorted(coco.getImgIds())
        if args.validation_role is not None:
            image_ids = [
                image_id
                for image_id in image_ids
                if belongs_to_validation_role(
                    "livecell",
                    coco.loadImgs([image_id])[0]["file_name"],
                    args.validation_role,
                )
            ]
        if args.limit:
            image_ids = image_ids[: args.limit]
        for image_id in image_ids:
            metadata = coco.loadImgs([image_id])[0]
            image_path = locate_image(args.images_root, metadata["file_name"])
            evaluation_samples.append((image_id, image_path, None, "livecell"))
    if not evaluation_samples:
        raise RuntimeError(
            "Cellpose evaluation role contains no images; check the split manifest."
        )
    model = models.CellposeModel(gpu=True, pretrained_model=str(args.model))
    thresholds = (0.5, 0.75, 0.9)
    rows_by_domain: dict[str, list[dict[str, float]]] = defaultdict(list)

    for sample_id, image_path, mask_path, domain in tqdm(
        evaluation_samples,
        desc="evaluate Cellpose-SAM",
    ):
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Could not decode {image_path}")
        if mask_path is not None:
            truth = load_pair_truth(mask_path, image.shape)
        else:
            assert coco is not None and isinstance(sample_id, int)
            truth = ground_truth(coco, sample_id, image.shape[0], image.shape[1])

        started = time.perf_counter()
        prediction, *_ = model.eval(
            image,
            diameter=None,
            flow_threshold=args.flow_threshold,
            cellprob_threshold=args.cellprob_threshold,
            normalize=True,
            bsize=args.tile_size,
            tile_overlap=args.tile_overlap,
            batch_size=args.tile_batch_size,
        )
        runtime = time.perf_counter() - started
        prediction = np.asarray(prediction, dtype=np.int32)

        truth_fg = truth > 0
        prediction_fg = prediction > 0
        intersection = int((truth_fg & prediction_fg).sum())
        false_positive = int((prediction_fg & ~truth_fg).sum())
        false_negative = int((~prediction_fg & truth_fg).sum())
        union = int((truth_fg | prediction_fg).sum())
        denominator = int(truth_fg.sum() + prediction_fg.sum())
        pixel_iou = safe_ratio(intersection, union)
        pixel_dice = safe_ratio(2 * intersection, denominator)
        foreground_precision = safe_ratio(intersection, intersection + false_positive)
        foreground_recall = safe_ratio(intersection, intersection + false_negative)
        truth_boundary = boundary_mask(truth)
        prediction_boundary = boundary_mask(prediction)
        boundary_intersection = int((truth_boundary & prediction_boundary).sum())
        boundary_false_positive = int((prediction_boundary & ~truth_boundary).sum())
        boundary_false_negative = int((~prediction_boundary & truth_boundary).sum())
        boundary_dice = safe_ratio(
            2 * boundary_intersection,
            int(truth_boundary.sum() + prediction_boundary.sum()),
        )
        boundary_precision = safe_ratio(
            boundary_intersection,
            boundary_intersection + boundary_false_positive,
        )
        boundary_recall = safe_ratio(
            boundary_intersection,
            boundary_intersection + boundary_false_negative,
        )
        predicted_count = int(np.unique(prediction[prediction > 0]).size)
        truth_count = int(np.unique(truth[truth > 0]).size)
        average_precision, _, _, _ = metrics.average_precision(
            [truth],
            [prediction],
            threshold=list(thresholds),
        )
        ap_values = np.asarray(average_precision)[0]
        rows_by_domain[domain].append(
            {
                "pixel_dice": pixel_dice,
                "pixel_iou": pixel_iou,
                "foreground_precision": foreground_precision,
                "foreground_recall": foreground_recall,
                "boundary_dice": boundary_dice,
                "boundary_precision": boundary_precision,
                "boundary_recall": boundary_recall,
                "count_absolute_error": float(abs(predicted_count - truth_count)),
                "seconds_per_image": runtime,
                "instance_ap50": float(ap_values[0]),
                "instance_ap75": float(ap_values[1]),
                "instance_ap90": float(ap_values[2]),
            }
        )

    per_domain = {
        domain: summarize_rows(rows) for domain, rows in sorted(rows_by_domain.items())
    }
    macro = macro_average_domains(per_domain)
    result = {
        "model": str(args.model),
        "model_sha256": file_sha256(args.model),
        "source_mode": source_mode,
        "pairs_directory": str(args.pairs_dir) if args.pairs_dir is not None else None,
        "annotations": str(args.annotations) if args.annotations is not None else None,
        "inference_settings": {
            "diameter": None,
            "flow_threshold": args.flow_threshold,
            "cellprob_threshold": args.cellprob_threshold,
            "normalize": True,
            "tile_size": args.tile_size,
            "tile_overlap": args.tile_overlap,
            "tile_batch_size": args.tile_batch_size,
        },
        "validation_role": args.validation_role,
        "evaluation_images": len(evaluation_samples),
        # Retain the old key for existing result importers; it means evaluated images, not that
        # the held-out test was opened.
        "test_images": len(evaluation_samples),
        "domain_aggregation": "metrics averaged within domain, then macro-averaged across domains",
        "per_domain_metrics": per_domain,
        "macro_domain_metrics": macro,
        "pixel_dice": macro["pixel_dice"],
        "pixel_iou": macro["pixel_iou"],
        "foreground_precision": macro["foreground_precision"],
        "foreground_recall": macro["foreground_recall"],
        "boundary_dice": macro["boundary_dice"],
        "boundary_precision": macro["boundary_precision"],
        "boundary_recall": macro["boundary_recall"],
        "count_mae": macro["count_mae"],
        "seconds_per_image": macro["seconds_per_image"],
        "instance_average_precision": macro["instance_average_precision"],
        "selection_score": macro["selection_score"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
