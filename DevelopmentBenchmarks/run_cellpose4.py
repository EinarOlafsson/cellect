#!/usr/bin/env python3
"""Run Cellpose 4 / Cellpose-SAM models over the development image fixtures."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from cellpose import models


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    output = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for label in np.unique(labels):
        if label == 0:
            continue
        value = int(label)
        output[labels == label] = (
            (value * 67) % 205 + 50,
            (value * 109) % 205 + 50,
            (value * 149) % 205 + 50,
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["cpsam_v2"])
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    image_paths = sorted(args.images.glob("*.jpg"))
    if args.limit is not None:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit(f"No JPEG fixtures found in {args.images}")
    images = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in image_paths]
    if any(image is None for image in images):
        raise SystemExit("Could not read one or more fixture images")

    summary: dict[str, object] = {
        "framework": "cellpose 4",
        "images": [path.name for path in image_paths],
        "models": {},
    }
    for model_name in args.models:
        destination = args.output / model_name
        destination.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        model = models.CellposeModel(gpu=False, pretrained_model=model_name)
        masks, _, _ = model.eval(images)
        elapsed = time.perf_counter() - started
        counts = []
        for path, mask in zip(image_paths, masks):
            mask = np.asarray(mask, dtype=np.uint16)
            counts.append(int(mask.max()))
            cv2.imwrite(str(destination / f"{path.stem}_labels.png"), mask)
            cv2.imwrite(
                str(destination / f"{path.stem}_preview.png"),
                cv2.cvtColor(colorize_labels(mask), cv2.COLOR_RGB2BGR),
            )
        summary["models"][model_name] = {
            "elapsed_seconds": elapsed,
            "seconds_per_image": elapsed / len(images),
            "counts": counts,
        }
        print(f"{model_name}: {elapsed:.2f}s, counts={counts}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
