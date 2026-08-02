#!/usr/bin/env python3
"""Fine-tune one current Cellpose foundation model on all manual instance masks.

This follows the official Cellpose recommendation: always start from the built-in model, put all
training images into the same training round, use 1e-5 learning rate, 0.1 weight decay, batch size
1, and validate on a separate folder. A completion marker prevents accidentally fine-tuning an
already fine-tuned model again, which the Cellpose documentation warns can reweight data badly.

The official Python API returns every epoch's training loss.  This wrapper records that complete
history and leaves validation loss blank on epochs where Cellpose does not run validation, rather
than misreporting its zero-filled placeholders as measured values.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import tifffile

from cellpose_teacher import (
    PINNED_CELLPOSE_MODELS,
    ensure_pinned_model_file,
    instantiate_strict_cellpose_model,
    strict_checkpoint_report,
)

VALIDATION_EPOCHS = lambda epoch: epoch == 5 or epoch % 10 == 0
TRAINING_IMPLEMENTATION_VERSION = "cellect-cellpose-stream-resume-v2"
STATE_CHECKPOINT_SCHEMA = 2
STATE_CHECKPOINT_EVERY_EPOCHS = 5
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class TrainingPair:
    role: str
    dataset: str
    image: Path
    mask: Path


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json(value) + b"\n")
    os.replace(temporary, path)


def atomic_torch_save(path: Path, value: Mapping[str, object]) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    previous = path.with_suffix(path.suffix + ".prev")
    torch.save(dict(value), temporary)
    if path.is_file():
        os.replace(path, previous)
    os.replace(temporary, path)


def discover_training_pairs(directory: Path, role: str) -> tuple[TrainingPair, ...]:
    pairs: list[TrainingPair] = []
    for mask in sorted(directory.glob("*_masks.tif")):
        base = mask.stem.removesuffix("_masks")
        images = sorted(
            path
            for path in directory.glob(f"{base}_img.*")
            if path.suffix.casefold() in IMAGE_EXTENSIONS
        )
        if len(images) != 1:
            raise RuntimeError(
                f"Expected exactly one image for {mask}, found {[str(path) for path in images]}"
            )
        dataset = base
        for separator in ("_train_", "_val_", "_test_"):
            if separator in base:
                dataset = base.split(separator, 1)[0]
                break
        if dataset.startswith("livecell"):
            dataset = "livecell"
        pairs.append(
            TrainingPair(
                role=role,
                dataset=dataset,
                image=images[0].resolve(),
                mask=mask.resolve(),
            )
        )
    if not pairs:
        raise RuntimeError(f"No strict *_img/_masks Cellpose pairs found in {directory}")
    return tuple(pairs)


def _finite_integer_mask(path: Path) -> np.ndarray:
    mask = np.asarray(tifffile.imread(path))
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    if mask.ndim != 2 or not np.isfinite(mask).all() or np.any(mask < 0):
        raise RuntimeError(f"Cellpose mask must be a finite non-negative 2-D map: {path}")
    rounded = np.rint(mask)
    if not np.array_equal(mask, rounded):
        raise RuntimeError(f"Cellpose mask contains fractional identifiers: {path}")
    return np.ascontiguousarray(rounded, dtype=np.int32)


def _system_memory_bytes() -> int | None:
    try:
        return int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None


def write_ram_preflight(
    pairs: Sequence[TrainingPair],
    output_directory: Path,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    maximum = 0
    total_pixels = 0
    for pair in pairs:
        image = cv2.imread(str(pair.image), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Could not decode Cellpose training image {pair.image}")
        mask = _finite_integer_mask(pair.mask)
        if image.shape[:2] != mask.shape:
            raise RuntimeError(
                f"Cellpose image/mask geometry differs: {pair.image} / {pair.mask}"
            )
        pixels = int(mask.size)
        # Conservative host estimate for one streamed sample: source image+mask, 4-channel flow,
        # normalized 3-channel input, augmented input/labels, and working copies.
        estimated = int(image.nbytes + mask.nbytes + pixels * 4 * (4 + 3 + 3 + 4))
        maximum = max(maximum, estimated)
        total_pixels += pixels
        rows.append(
            {
                "role": pair.role,
                "dataset": pair.dataset,
                "image": str(pair.image),
                "mask": str(pair.mask),
                "shape": list(mask.shape),
                "estimated_streamed_host_bytes": estimated,
            }
        )
        del image, mask
    memory = _system_memory_bytes()
    if memory is not None and maximum > memory // 2:
        raise RuntimeError(
            f"One Cellpose sample may require {maximum:,} host bytes, more than half of "
            f"system RAM ({memory:,}); tile/materialize smaller source images first"
        )
    report = {
        "schema_version": 1,
        "implementation": TRAINING_IMPLEMENTATION_VERSION,
        "pair_count": len(pairs),
        "maximum_estimated_streamed_host_bytes": maximum,
        "system_physical_memory_bytes": memory,
        "estimated_uncompressed_flow_cache_bytes": total_pixels * 4 * 4,
        "all_images_or_flows_loaded_together": False,
        "rows": rows,
    }
    atomic_json(output_directory / "ram_preflight.json", report)
    return report


def _flow_cache_paths(output_directory: Path, pair: TrainingPair) -> tuple[Path, Path]:
    identity = hashlib.sha256(
        f"{pair.role}:{pair.dataset}:{pair.image}:{pair.mask}".encode("utf-8")
    ).hexdigest()
    root = output_directory / "flow_cache" / pair.role / identity[:2]
    return root / f"{identity}_flows.tif", root / f"{identity}.json"


def precompute_flow_files(
    pairs: Sequence[TrainingPair],
    output_directory: Path,
    dynamics: Any,
    device: Any,
) -> tuple[tuple[Path, ...], dict[str, object]]:
    paths: list[Path] = []
    entries: list[dict[str, object]] = []
    generated = 0
    for index, pair in enumerate(pairs):
        flow_path, marker_path = _flow_cache_paths(output_directory, pair)
        mask_sha256 = file_sha256(pair.mask)
        reusable = False
        if flow_path.is_file() and marker_path.is_file():
            try:
                marker = json.loads(marker_path.read_text(encoding="utf-8"))
                reusable = (
                    marker.get("implementation") == TRAINING_IMPLEMENTATION_VERSION
                    and marker.get("mask_sha256") == mask_sha256
                    and marker.get("flow_sha256") == file_sha256(flow_path)
                )
            except (OSError, ValueError, json.JSONDecodeError):
                reusable = False
        if not reusable:
            if flow_path.exists() or marker_path.exists():
                raise RuntimeError(
                    f"Incomplete/stale immutable flow cache for {pair.mask}; use a fresh "
                    "fingerprint-specific output directory"
                )
            mask = _finite_integer_mask(pair.mask)
            flow = np.asarray(
                dynamics.labels_to_flows(
                    [mask],
                    files=None,
                    device=device,
                    return_flows=True,
                )[0],
                dtype=np.float32,
            )
            if flow.shape != (4, *mask.shape) or not np.isfinite(flow).all():
                raise RuntimeError(f"Cellpose flow contract failed for {pair.mask}: {flow.shape}")
            if not np.array_equal(np.rint(flow[0]).astype(np.int32), mask):
                raise RuntimeError(f"Cellpose flow cache changed instance IDs for {pair.mask}")
            flow_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = flow_path.with_name(f".{flow_path.stem}.{os.getpid()}.tmp.tif")
            tifffile.imwrite(temporary, flow, compression="zlib")
            os.replace(temporary, flow_path)
            marker = {
                "schema_version": 1,
                "implementation": TRAINING_IMPLEMENTATION_VERSION,
                "role": pair.role,
                "dataset": pair.dataset,
                "image": str(pair.image),
                "mask": str(pair.mask),
                "mask_sha256": mask_sha256,
                "shape": list(flow.shape),
                "dtype": str(flow.dtype),
                "flow_sha256": file_sha256(flow_path),
            }
            atomic_json(marker_path, marker)
            generated += 1
            del mask, flow
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        paths.append(flow_path)
        entries.append(marker)
        if (index + 1) % 100 == 0:
            print(f"Prepared {index + 1}/{len(pairs)} Cellpose flow files")
    report = {
        "schema_version": 1,
        "implementation": TRAINING_IMPLEMENTATION_VERSION,
        "entry_count": len(entries),
        "generated_this_run": generated,
        "reused": len(entries) - generated,
        "serial_one_mask_at_a_time": True,
        "entries_sha256": hashlib.sha256(canonical_json(entries)).hexdigest(),
    }
    roles = sorted({pair.role for pair in pairs})
    manifest_name = (
        f"flow_cache_{roles[0]}_manifest.json"
        if len(roles) == 1
        else "flow_cache_manifest.json"
    )
    report["manifest"] = manifest_name
    atomic_json(output_directory / manifest_name, report)
    return tuple(paths), report


def _rng_payload(torch: Any) -> dict[str, object]:
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all(),
    }


def _restore_rng(payload: Mapping[str, object], torch: Any) -> None:
    required = (
        "python_random_state",
        "numpy_random_state",
        "torch_random_state",
        "cuda_random_state",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise RuntimeError("Cellpose resume lacks RNG state: " + ", ".join(missing))
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_random_state"])
    torch.cuda.set_rng_state_all(payload["cuda_random_state"])


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def learning_rate_at_epoch(epoch: int, epochs: int, base_learning_rate: float) -> float:
    schedule = np.linspace(0, base_learning_rate, 10)
    schedule = np.append(schedule, base_learning_rate * np.ones(max(0, epochs - 10)))
    if epochs > 300:
        schedule = schedule[:-100]
        for _ in range(10):
            schedule = np.append(schedule, schedule[-1] / 2 * np.ones(10))
    elif epochs > 99:
        schedule = schedule[:-50]
        for _ in range(10):
            schedule = np.append(schedule, schedule[-1] / 2 * np.ones(5))
    return float(schedule[epoch])


def write_history(
    train_losses: np.ndarray,
    validation_losses: np.ndarray,
    destination: Path,
    base_learning_rate: float,
) -> int:
    if train_losses.shape != validation_losses.shape:
        raise RuntimeError("Cellpose returned differently shaped train/validation histories")
    fieldnames = [
        "epoch",
        "train_loss",
        "validation_loss",
        "validation_evaluated",
        "learning_rate",
    ]
    with destination.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for epoch, train_loss in enumerate(train_losses):
            evaluated = VALIDATION_EPOCHS(epoch)
            writer.writerow(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(train_loss),
                    # Cellpose leaves unevaluated epochs as zero; a blank prevents those zeros
                    # from being misreported as excellent validation losses in the paper.
                    "validation_loss": (
                        float(validation_losses[epoch]) if evaluated else ""
                    ),
                    "validation_evaluated": evaluated,
                    "learning_rate": learning_rate_at_epoch(
                        epoch, len(train_losses), base_learning_rate
                    ),
                }
            )
    return len(train_losses)


def _diameters_and_keep(
    flow_files: Sequence[Path],
    io: Any,
    utils: Any,
    *,
    minimum_masks: int,
) -> tuple[np.ndarray, np.ndarray]:
    diameters = np.zeros(len(flow_files), dtype=np.float64)
    keep = np.zeros(len(flow_files), dtype=bool)
    for index, path in enumerate(flow_files):
        labels = np.asarray(io.imread(str(path)))[0]
        diameter, individual = utils.diameters(labels)
        diameters[index] = max(float(diameter), 5.0)
        keep[index] = len(individual) >= minimum_masks
        del labels
    return diameters, np.flatnonzero(keep)


def _training_sampling_probabilities(
    pairs: Sequence[TrainingPair],
) -> tuple[np.ndarray, dict[str, object]]:
    yeast = {"yeast_microstructures", "yeaz_phase", "yeaz_brightfield"}
    domains = ["yeast" if pair.dataset in yeast else pair.dataset for pair in pairs]
    counts = {domain: domains.count(domain) for domain in sorted(set(domains))}
    raw = np.asarray([1.0 / math.sqrt(counts[domain]) for domain in domains], dtype=np.float64)
    probabilities = raw / raw.sum()
    expected = {
        domain: float(probabilities[np.asarray(domains) == domain].sum())
        for domain in counts
    }
    return probabilities, {
        "unit": "image draw with replacement",
        "policy": "per-image inverse sqrt(dataset-domain count); all yeast sources share one domain",
        "draws_per_epoch": len(pairs),
        "domain_image_counts": counts,
        "expected_domain_probability": expected,
    }


def _load_resume_checkpoint(path: Path, torch: Any) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        previous = path.with_suffix(path.suffix + ".prev")
        if not previous.is_file():
            raise
        value = torch.load(previous, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise RuntimeError(f"Cellpose resume checkpoint is not a mapping: {path}")
    return value


def stateful_stream_train(
    *,
    net: Any,
    train_module: Any,
    io: Any,
    utils: Any,
    models: Any,
    torch: Any,
    train_pairs: Sequence[TrainingPair],
    train_flow_files: Sequence[Path],
    validation_pairs: Sequence[TrainingPair],
    validation_flow_files: Sequence[Path],
    output_directory: Path,
    training_contract_sha256: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    tile_size: int,
    model_stem: str,
) -> tuple[Path, np.ndarray, np.ndarray, dict[str, object]]:
    """File-streamed Cellpose loop with exact optimizer/RNG resume at saved epochs.

    This intentionally mirrors Cellpose 4.2.1.1's ``train_seg`` loop and private loss/augmentation
    functions.  The pinned package version and this source file are part of the outer experiment
    fingerprint, so a private-API change cannot silently resume an old run.
    """

    train_diameters, train_keep = _diameters_and_keep(
        train_flow_files, io, utils, minimum_masks=1
    )
    if len(train_keep) == 0:
        raise RuntimeError("Every Cellpose training image was rejected for having zero masks")
    train_pairs = tuple(train_pairs[index] for index in train_keep)
    train_flow_files = tuple(train_flow_files[index] for index in train_keep)
    train_diameters = train_diameters[train_keep]
    train_probabilities, sampling_report = _training_sampling_probabilities(train_pairs)
    validation_diameters, validation_keep = _diameters_and_keep(
        validation_flow_files, io, utils, minimum_masks=1
    )
    if len(validation_keep) == 0:
        raise RuntimeError("Every Cellpose checkpoint image was rejected for having zero masks")
    validation_pairs = tuple(validation_pairs[index] for index in validation_keep)
    validation_flow_files = tuple(validation_flow_files[index] for index in validation_keep)
    validation_diameters = validation_diameters[validation_keep]

    device = net.device
    net.diam_labels.data = torch.tensor(
        [float(train_diameters.mean())], dtype=net.diam_labels.dtype, device=device
    )
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    learning_rates = np.asarray(
        [learning_rate_at_epoch(epoch, epochs, learning_rate) for epoch in range(epochs)],
        dtype=np.float64,
    )
    train_losses = np.full(epochs, np.nan, dtype=np.float64)
    validation_losses = np.full(epochs, np.nan, dtype=np.float64)
    start_epoch = 0
    resume_path = output_directory / "stateful_resume.pt"
    resumed = False
    if resume_path.is_file() or resume_path.with_suffix(resume_path.suffix + ".prev").is_file():
        checkpoint = _load_resume_checkpoint(resume_path, torch)
        required = {
            "schema_version",
            "implementation",
            "training_contract_sha256",
            "completed_epoch",
            "model",
            "optimizer",
            "train_losses",
            "validation_losses",
            "python_random_state",
            "numpy_random_state",
            "torch_random_state",
            "cuda_random_state",
        }
        missing = sorted(required - set(checkpoint))
        if missing:
            raise RuntimeError("Cellpose resume is incomplete: " + ", ".join(missing))
        if (
            checkpoint["schema_version"] != STATE_CHECKPOINT_SCHEMA
            or checkpoint["implementation"] != TRAINING_IMPLEMENTATION_VERSION
            or checkpoint["training_contract_sha256"] != training_contract_sha256
        ):
            raise RuntimeError("Cellpose resume checkpoint contract changed")
        net.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        train_losses = np.asarray(checkpoint["train_losses"], dtype=np.float64)
        validation_losses = np.asarray(checkpoint["validation_losses"], dtype=np.float64)
        if train_losses.shape != (epochs,) or validation_losses.shape != (epochs,):
            raise RuntimeError("Cellpose resume history shape changed")
        start_epoch = int(checkpoint["completed_epoch"]) + 1
        if not 0 <= start_epoch <= epochs:
            raise RuntimeError("Cellpose resume epoch lies outside the configured schedule")
        _restore_rng(checkpoint, torch)
        resumed = True

    normalize_params = {**models.normalize_default, "normalize": True}
    train_image_files = [str(pair.image) for pair in train_pairs]
    train_label_files = [str(path) for path in train_flow_files]
    validation_image_files = [str(pair.image) for pair in validation_pairs]
    validation_label_files = [str(path) for path in validation_flow_files]
    checkpoint_directory = output_directory / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    started = time.time()
    for epoch in range(start_epoch, epochs):
        # Matches upstream Cellpose's stable per-epoch ordering and makes re-running an interrupted
        # unsaved epoch exact from the last state checkpoint.
        np.random.seed(epoch)
        order = np.random.choice(
            np.arange(len(train_pairs)),
            size=len(train_pairs),
            replace=True,
            p=train_probabilities,
        )
        optimizer.param_groups[0]["lr"] = float(learning_rates[epoch])
        net.train()
        epoch_loss = 0.0
        for index in order:
            images, labels = train_module._get_batch(
                np.asarray([index]),
                data=None,
                labels=None,
                files=train_image_files,
                labels_files=train_label_files,
                normalize_params=normalize_params,
            )
            augmented_image, augmented_label = train_module.random_rotate_and_resize(
                images,
                # Cellpose 4 renamed the label argument and the output-size argument, and it
                # returns tensors on ``device`` instead of NumPy arrays.
                lbls=labels,
                rescale=np.ones(1, dtype=np.float32),
                scale_range=0.5,
                bsize=(tile_size, tile_size),
                device=device,
            )[:2]
            image_tensor = augmented_image.to(device)
            label_tensor = augmented_label.to(device)
            if image_tensor.dtype != net.dtype:
                image_tensor = image_tensor.to(net.dtype)
                label_tensor = label_tensor.to(net.dtype)
            prediction = net(image_tensor)[0]
            loss = train_module._loss_fn_seg(label_tensor, prediction, device)
            if prediction.shape[1] > 3:
                loss = loss + train_module._loss_fn_class(
                    label_tensor, prediction, class_weights=None
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
        train_losses[epoch] = epoch_loss / len(train_pairs)

        if VALIDATION_EPOCHS(epoch):
            np.random.seed(42)
            order = np.random.permutation(len(validation_pairs))
            validation_loss = 0.0
            net.eval()
            with torch.inference_mode():
                for index in order:
                    images, labels = train_module._get_batch(
                        np.asarray([index]),
                        data=None,
                        labels=None,
                        files=validation_image_files,
                        labels_files=validation_label_files,
                        normalize_params=normalize_params,
                    )
                    augmented_image, augmented_label = train_module.random_rotate_and_resize(
                        images,
                        lbls=labels,
                        rescale=np.ones(1, dtype=np.float32),
                        scale_range=0.5,
                        bsize=(tile_size, tile_size),
                        device=device,
                    )[:2]
                    image_tensor = augmented_image.to(device)
                    label_tensor = augmented_label.to(device)
                    if image_tensor.dtype != net.dtype:
                        image_tensor = image_tensor.to(net.dtype)
                        label_tensor = label_tensor.to(net.dtype)
                    prediction = net(image_tensor)[0]
                    loss = train_module._loss_fn_seg(label_tensor, prediction, device)
                    if prediction.shape[1] > 3:
                        loss = loss + train_module._loss_fn_class(
                            label_tensor, prediction, class_weights=None
                        )
                    validation_loss += float(loss.detach())
            validation_losses[epoch] = validation_loss / len(validation_pairs)

        write_history(
            train_losses[: epoch + 1],
            validation_losses[: epoch + 1],
            output_directory / "history.csv",
            learning_rate,
        )
        logging.getLogger("cellpose").info(
            "epoch=%d train_loss=%.6f validation_loss=%s LR=%.8g elapsed=%.1fs",
            epoch + 1,
            train_losses[epoch],
            (
                f"{validation_losses[epoch]:.6f}"
                if np.isfinite(validation_losses[epoch])
                else "not_evaluated"
            ),
            learning_rates[epoch],
            time.time() - started,
        )
        if (epoch + 1) % 25 == 0 and epoch + 1 < epochs:
            snapshot = checkpoint_directory / f"{model_stem}_epoch_{epoch + 1:04d}.pt"
            net.save_model(snapshot)
        if (
            (epoch + 1) % STATE_CHECKPOINT_EVERY_EPOCHS == 0
            or epoch + 1 == epochs
        ):
            atomic_torch_save(
                resume_path,
                {
                    "schema_version": STATE_CHECKPOINT_SCHEMA,
                    "implementation": TRAINING_IMPLEMENTATION_VERSION,
                    "training_contract_sha256": training_contract_sha256,
                    "completed_epoch": epoch,
                    "model": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "train_losses": train_losses,
                    "validation_losses": validation_losses,
                    **_rng_payload(torch),
                },
            )

    final_path = output_directory / "final_model.pt"
    temporary_final = output_directory / ".final_model.tmp.pt"
    net.save_model(temporary_final)
    os.replace(temporary_final, final_path)
    return final_path, train_losses, validation_losses, {
        "resumed": resumed,
        "start_epoch": start_epoch,
        "completed_epochs": epochs,
        "state_checkpoint": resume_path.name,
        "state_checkpoint_every_epochs": STATE_CHECKPOINT_EVERY_EPOCHS,
        "training_sampling": sampling_report,
        "exact_resume_scope": (
            "model, AdamW moments, learning-rate epoch, histories, Python/NumPy/Torch/CUDA RNG; "
            "an interruption after the latest state checkpoint deterministically replays at most "
            f"{STATE_CHECKPOINT_EVERY_EPOCHS - 1} epochs"
        ),
    }


def run_resume_contract_self_test() -> dict[str, object]:
    """Simulate a process stop and require bit-exact model/optimizer/RNG continuation."""

    import inspect
    import torch

    parameters = inspect.signature(stateful_stream_train).parameters
    if "model_stem" not in parameters or "model_name" in parameters:
        raise AssertionError("Cellpose stateful trainer call contract changed")

    def execute(root: Path, total_epochs: int, stop_after: int | None) -> dict[str, Any]:
        torch.manual_seed(1701)
        np.random.seed(1701)
        random.seed(1701)
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        checkpoint_path = root / "state.pt"
        start = 0
        history: list[float] = []
        if checkpoint_path.is_file():
            checkpoint = _load_resume_checkpoint(checkpoint_path, torch)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            history = list(checkpoint["history"])
            start = int(checkpoint["completed_epoch"]) + 1
            _restore_rng(checkpoint, torch)
        for epoch in range(start, total_epochs):
            numpy_noise = torch.from_numpy(
                np.random.normal(size=(3, 4)).astype(np.float32)
            )
            tensor_noise = torch.rand(3, 4)
            target = torch.full((3, 2), random.random(), dtype=torch.float32)
            loss = torch.nn.functional.mse_loss(model(numpy_noise + tensor_noise), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            history.append(float(loss.detach()))
            atomic_torch_save(
                checkpoint_path,
                {
                    "completed_epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                    **_rng_payload(torch),
                },
            )
            if stop_after is not None and epoch + 1 == stop_after:
                break
        return {
            "model": {
                key: value.detach().clone() for key, value in model.state_dict().items()
            },
            "optimizer": optimizer.state_dict(),
            "history": history,
        }

    with tempfile.TemporaryDirectory(prefix="cellect-cellpose-resume-") as temporary:
        root = Path(temporary)
        uninterrupted = execute(root / "uninterrupted", 6, None)
        execute(root / "resumed", 6, 3)
        resumed = execute(root / "resumed", 6, None)
        if uninterrupted["history"] != resumed["history"]:
            raise AssertionError("Cellpose resume test changed the loss history")
        for key, expected in uninterrupted["model"].items():
            if not torch.equal(expected, resumed["model"][key]):
                raise AssertionError(f"Cellpose resume changed model tensor {key}")
        # Serialize the optimizer states canonically through torch to cover Adam moments/steps.
        first = root / "first_optimizer.pt"
        second = root / "second_optimizer.pt"
        torch.save(uninterrupted["optimizer"], first)
        torch.save(resumed["optimizer"], second)
        # Pickle byte order is not a stable equality contract; compare every nested tensor/value.
        def nested_equal(left: Any, right: Any) -> bool:
            if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
                return torch.equal(left, right)
            if isinstance(left, Mapping) and isinstance(right, Mapping):
                return set(left) == set(right) and all(
                    nested_equal(left[key], right[key]) for key in left
                )
            if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
                return len(left) == len(right) and all(
                    nested_equal(a, b) for a, b in zip(left, right)
                )
            return left == right
        if not nested_equal(uninterrupted["optimizer"], resumed["optimizer"]):
            raise AssertionError("Cellpose resume changed AdamW optimizer state")
    return {
        "status": "PASS",
        "implementation": TRAINING_IMPLEMENTATION_VERSION,
        "interruption_after_epochs": 3,
        "completed_epochs": 6,
        "model_optimizer_rng_exact": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-model",
        choices=["cpsam_v2", "cpdino", "cpdino-vitb"],
        default="cpsam_v2",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--stage-fingerprint", required=True)
    args = parser.parse_args()
    tile_size = 384 if args.base_model.startswith("cpdino") else 256

    train_directory = (args.data_root / "train").resolve()
    validation_directory = (args.data_root / "checkpoint").resolve()
    output_directory = args.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    completion_marker = output_directory / "completed.json"
    if completion_marker.is_file():
        try:
            completed = json.loads(completion_marker.read_text())
            completed_model = output_directory / str(completed["final_model"])
            reusable = (
                completed.get("stage_fingerprint") == args.stage_fingerprint
                and completed.get("base_model") == args.base_model
                and completed.get("implementation") == TRAINING_IMPLEMENTATION_VERSION
                and int(completed.get("epochs", -1)) == args.epochs
                and completed_model.is_file()
                and completed.get("final_model_sha256") == file_sha256(completed_model)
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            reusable = False
        if reusable:
            print(f"Reusing fingerprint-matched Cellpose fine-tune: {output_directory}")
            return
        raise RuntimeError(
            f"Refusing stale Cellpose completion marker in {output_directory}; "
            "use the fingerprint-specific v4 output directory."
        )
    if not any(train_directory.glob("*_masks.tif")):
        raise SystemExit(f"No Cellpose training masks found in {train_directory}")
    if not any(validation_directory.glob("*_masks.tif")):
        raise SystemExit(f"No Cellpose validation masks found in {validation_directory}")

    from cellpose import dynamics, io, models, train, utils
    import torch

    train_pairs = discover_training_pairs(train_directory, "train")
    validation_pairs = discover_training_pairs(validation_directory, "checkpoint")
    ram_preflight = write_ram_preflight(
        (*train_pairs, *validation_pairs), output_directory
    )

    model_stem = f"cellect_{args.base_model}_{args.stage_fingerprint[:12]}"
    pinned_path, pinned_report = ensure_pinned_model_file(
        args.data_root.parent / "pinned_cellpose_weights",
        args.base_model,
    )
    device, _ = models.assign_device(use_torch=True, gpu=True)
    if getattr(device, "type", None) != "cuda":
        raise RuntimeError("Pinned Cellpose fine-tuning requires CUDA")
    model, strict_initialization = instantiate_strict_cellpose_model(
        pinned_path,
        args.base_model,
        device=device,
        expected_sha256=PINNED_CELLPOSE_MODELS[args.base_model].sha256,
    )
    shared_flow_root = (
        args.data_root.parent
        / "cellpose_flow_cache_v4"
        / args.stage_fingerprint[:16]
    )
    train_flow_files, train_flow_report = precompute_flow_files(
        train_pairs,
        shared_flow_root,
        dynamics,
        device,
    )
    validation_flow_files, validation_flow_report = precompute_flow_files(
        validation_pairs,
        shared_flow_root,
        dynamics,
        device,
    )
    atomic_json(
        output_directory / "flow_cache_train_manifest.json",
        {**train_flow_report, "shared_cache_root": str(shared_flow_root)},
    )
    atomic_json(
        output_directory / "flow_cache_checkpoint_manifest.json",
        {**validation_flow_report, "shared_cache_root": str(shared_flow_root)},
    )
    training_call = {
        "api": "Cellect stateful file-stream loop mirroring cellpose.train.train_seg",
        "implementation": TRAINING_IMPLEMENTATION_VERSION,
        "cellpose_version": "4.2.1.1",
        "pretrained_model": str(pinned_path),
        "base_model": args.base_model,
        "learning_rate": 1e-5,
        "weight_decay": 0.1,
        "epochs": args.epochs,
        "batch_size": 1,
        "tile_size": tile_size,
        "minimum_masks_per_image": 1,
        "save_every": 25,
        "save_each": True,
        "load_files": False,
        "precomputed_flow_files": True,
        "sampling": (
            "image draws with replacement using inverse-sqrt dataset-domain frequency; "
            "all yeast sources share one domain"
        ),
        "state_checkpoint_every_epochs": STATE_CHECKPOINT_EVERY_EPOCHS,
    }
    pair_contract = [
        {
            "role": pair.role,
            "dataset": pair.dataset,
            "image": str(pair.image),
            "image_sha256": file_sha256(pair.image),
            "mask": str(pair.mask),
            "mask_sha256": file_sha256(pair.mask),
        }
        for pair in (*train_pairs, *validation_pairs)
    ]
    training_contract = {
        "schema_version": 2,
        "implementation": TRAINING_IMPLEMENTATION_VERSION,
        "stage_fingerprint": args.stage_fingerprint,
        "base_model": args.base_model,
        "pinned_model_sha256": PINNED_CELLPOSE_MODELS[args.base_model].sha256,
        "training_call": training_call,
        "pairs": pair_contract,
        "train_flow_entries_sha256": train_flow_report["entries_sha256"],
        "validation_flow_entries_sha256": validation_flow_report["entries_sha256"],
        "source_sha256": file_sha256(Path(__file__).resolve()),
    }
    training_contract_sha256 = hashlib.sha256(
        canonical_json(training_contract)
    ).hexdigest()
    contract_payload = {
        **training_contract,
        "training_contract_sha256": training_contract_sha256,
    }
    contract_path = output_directory / "training_contract.json"
    if contract_path.is_file():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract_payload:
            raise RuntimeError("Cellpose training contract changed; refusing stale resume")
    else:
        atomic_json(contract_path, contract_payload)
    print("Running:", json.dumps(training_call, sort_keys=True))
    log_path = output_directory / "training.log"
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    )
    cellpose_logger = logging.getLogger("cellpose")
    previous_level = cellpose_logger.level
    cellpose_logger.setLevel(logging.INFO)
    cellpose_logger.addHandler(file_handler)
    try:
        trained_model, train_losses, validation_losses, resume_report = stateful_stream_train(
            net=model.net,
            train_module=train,
            io=io,
            utils=utils,
            models=models,
            torch=torch,
            train_pairs=train_pairs,
            train_flow_files=train_flow_files,
            validation_pairs=validation_pairs,
            validation_flow_files=validation_flow_files,
            output_directory=output_directory,
            training_contract_sha256=training_contract_sha256,
            epochs=args.epochs,
            learning_rate=1e-5,
            weight_decay=0.1,
            tile_size=tile_size,
            model_stem=model_stem,
        )
    finally:
        cellpose_logger.removeHandler(file_handler)
        cellpose_logger.setLevel(previous_level)
        file_handler.close()

    destination = Path(trained_model)
    final_checkpoint_report, _ = strict_checkpoint_report(
        model.net,
        destination,
        expected_model_name=args.base_model,
    )
    checkpoint_directory = output_directory / "checkpoints"
    checkpoint_directory.mkdir(exist_ok=True)
    snapshots = [
        str(snapshot.relative_to(output_directory))
        for snapshot in sorted(checkpoint_directory.glob(f"{model_stem}_epoch_*.pt"))
    ]
    history_rows = write_history(
        np.asarray(train_losses, dtype=np.float64),
        np.asarray(validation_losses, dtype=np.float64),
        output_directory / "history.csv",
        1e-5,
    )
    payload = {
        "base_model": args.base_model,
        "base_model_pinned_source": pinned_report,
        "strict_initialization": strict_initialization,
        "stage_fingerprint": args.stage_fingerprint,
        "data_root": str(args.data_root.resolve()),
        "epochs": args.epochs,
        "learning_rate": 1e-5,
        "weight_decay": 0.1,
        "train_batch_size": 1,
        "tile_size": tile_size,
        "minimum_masks_per_image": 1,
        "final_model": destination.name,
        "final_model_sha256": file_sha256(destination),
        "strict_final_checkpoint": final_checkpoint_report,
        "checkpoint_interval_epochs": 25,
        "checkpoint_candidates": snapshots + [destination.name],
        "training_log": log_path.name,
        "history": "history.csv",
        "history_rows": history_rows,
        "validation_loss_schedule": "epoch index 0, 5, then every 10; blank otherwise",
        "training_call": training_call,
        "implementation": TRAINING_IMPLEMENTATION_VERSION,
        "training_contract": contract_path.name,
        "training_contract_sha256": training_contract_sha256,
        "ram_preflight": ram_preflight,
        "train_flow_cache": train_flow_report,
        "validation_flow_cache": validation_flow_report,
        "resume": resume_report,
    }
    atomic_json(completion_marker, payload)


if __name__ == "__main__":
    main()
