#!/usr/bin/env python3
"""Run legacy Cellpose 3 models over the development image fixture set."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from cellpose import models


def colorize_labels(labels: np.ndarray) -> np.ndarray:
    """Create a deterministic RGB preview without adding benchmark dependencies."""
    output = np.zeros((*labels.shape, 3), dtype=np.uint8)
    ids = np.unique(labels)
    ids = ids[ids != 0]
    for label in ids:
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
    parser.add_argument(
        "--diameter",
        type=float,
        default=None,
        help="Cell diameter in pixels. By default each model's training diameter is used.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["cyto3", "cyto2", "cyto", "nuclei"],
    )
    args = parser.parse_args()

    image_paths = sorted(args.images.glob("*.jpg"))
    if not image_paths:
        raise SystemExit(f"No JPEG fixtures found in {args.images}")

    images = []
    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise SystemExit(f"Could not read {path}")
        images.append(image)

    summary: dict[str, object] = {
        "framework": f"cellpose {models.__name__}",
        "images": [path.name for path in image_paths],
        "models": {},
    }

    for model_name in args.models:
        destination = args.output / model_name
        destination.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        model = models.CellposeModel(gpu=False, model_type=model_name)
        masks, _, _ = model.eval(
            images,
            diameter=args.diameter,
            channels=[0, 0],
        )
        elapsed = time.perf_counter() - started

        counts = []
        for path, mask in zip(image_paths, masks):
            mask = np.asarray(mask, dtype=np.uint16)
            count = int(mask.max())
            counts.append(count)
            cv2.imwrite(str(destination / f"{path.stem}_labels.png"), mask)
            cv2.imwrite(
                str(destination / f"{path.stem}_preview.png"),
                cv2.cvtColor(colorize_labels(mask), cv2.COLOR_RGB2BGR),
            )

        summary["models"][model_name] = {
            "elapsed_seconds": elapsed,
            "seconds_per_image": elapsed / len(images),
            "counts": counts,
            "requested_diameter": args.diameter,
            "model_training_diameter": float(model.diam_labels),
        }
        print(f"{model_name}: {elapsed:.2f}s, counts={counts}", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
