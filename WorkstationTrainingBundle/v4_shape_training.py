#!/usr/bin/env python3
"""Leakage-safe v4 shape-student training orchestration.

This module is intentionally callable from ``pipeline.py`` but does not depend on pipeline globals.
It consumes the existing materialized Cellpose-style role directories:

``train/``, ``checkpoint/``, ``calibration/``, and ``ensemble_selection/``.

The sealed final-test role is neither discovered nor accepted.  Parameter fitting and the
Ceb-inspired graph teacher use only ``train``; early stopping uses ``checkpoint``; branch threshold
calibration uses ``calibration``; and frozen model comparison uses ``ensemble_selection``.

Public entry point: :func:`run_v4_shape_training`.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import tempfile
import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from boundary_teacher import (
    BOUNDARY_TEACHER_VERSION,
    BoundaryGraphTeacher,
    BoundaryTeacherSpec,
    boundary_teacher_loss,
    build_candidate_patch,
    run_contract_self_test as boundary_contract_self_test,
)
from cellpose_teacher import (
    TEACHER_OUTPUT_CONTRACT,
    TeacherInferenceSettings,
    TeacherProvenance,
    TeacherSample,
    read_teacher_cache,
    run_cellpose_teacher,
    teacher_cache_directory,
    teacher_cache_key,
    write_teacher_cache,
)
from deployment_runtime import reconstruct_iphone_instances
from scientific_splits import BUNDLE_VERSION, SPLIT_PROTOCOL_VERSION, scientific_group
from shape_models import (
    EXTENDED_DEPLOYMENT_OUTPUT_SEMANTICS,
    LEGACY_DEPLOYMENT_OUTPUT_SEMANTICS,
    DeploymentAdapter,
    ExtendedDeploymentAdapter,
    ShapeModelSpec,
    V4ShapeNet,
    V4_SHAPE_MODEL_SPECS,
    gradient_audit,
    initialize_from_v3,
    multitask_shape_loss,
    audit_frozen_teacher,
    freeze_teacher,
    run_contract_self_test as model_contract_self_test,
    set_training_phase,
)
from shape_targets import (
    SHAPE_TARGET_VERSION,
    candidate_keep_merge_targets,
    generate_candidate_pairs,
    geometry_targets,
    run_contract_self_test as target_contract_self_test,
)
from v4_boundary_search import (
    BoundaryEvaluationRecord,
    BoundarySearchBounds,
    FiveChannelDiskCache,
    run_v4_boundary_search,
)


V4_SHAPE_TRAINING_VERSION = "cellect-v4-shape-orchestration-v2"
V4_SHAPE_CHECKPOINT_SCHEMA = 2
SCIENTIFIC_ROLES = (
    "train",
    "checkpoint",
    "calibration",
    "ensemble_selection",
)
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
BOUNDARY_OUTPUT_NAMES = (
    "fused_boundary_logit",
    "context_boundary_logit",
    "flow_boundary_logit",
    "shape_boundary_logit",
)
CALIBRATED_OUTPUT_NAMES = ("foreground_logit", *BOUNDARY_OUTPUT_NAMES)
V3_SOURCE_MODEL_BY_SHAPE = {
    "shape_mobile_mobilenetv3_v4": "mobilenetv3_small_unet_accuracy_v2",
    "shape_context_segformer_b5_v4": "segformer_b5_accuracy_v2",
}


@dataclass(frozen=True)
class _ProposalReconstructionSettings:
    foreground_threshold: float = 0.50
    boundary_threshold: float = 0.45
    min_area_fraction: float = 0.0

__all__ = [
    "CellposeTeacherSource",
    "MaterializedPair",
    "PhaseSpec",
    "ShapeTrainingConfig",
    "V4_SHAPE_TRAINING_VERSION",
    "discover_materialized_pairs",
    "materialize_cellpose_teacher_caches",
    "run_contract_self_test",
    "run_v4_shape_training",
]


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def file_sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, label: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", value.casefold()) is None:
        raise ValueError(f"{label} must be one lowercase SHA-256 digest")


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_json(payload) + b"\n")
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(_jsonable(row), sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_numpy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npy")
    np.save(temporary, np.ascontiguousarray(array), allow_pickle=False)
    temporary.replace(path)


def _atomic_torch_save(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    previous = path.with_suffix(path.suffix + ".prev")
    torch.save(dict(payload), temporary)
    if path.is_file():
        shutil.copy2(path, previous)
    temporary.replace(path)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        previous = path.with_suffix(path.suffix + ".prev")
        if not previous.is_file():
            raise
        value = torch.load(previous, map_location="cpu", weights_only=False)
    if not isinstance(value, dict):
        raise RuntimeError(f"Checkpoint root is not a dictionary: {path}")
    return value


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class PhaseSpec:
    name: str
    epochs: int
    learning_rate: float

    def __post_init__(self) -> None:
        if self.name not in {"heads", "decoder", "full"}:
            raise ValueError(f"Unknown v4 training phase: {self.name}")
        if self.epochs < 1:
            raise ValueError("Every enabled training phase needs at least one epoch")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Phase learning rate must be positive and finite")


@dataclass(frozen=True)
class ShapeTrainingConfig:
    mode: str
    phases: tuple[PhaseSpec, ...]
    seed: int = 1701
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    accumulation_steps_mobile: int = 4
    accumulation_steps_heavy: int = 8
    batch_size_mobile: int = 4
    batch_size_heavy: int = 1
    workers: int = 4
    early_stopping_patience: int = 30
    minimum_full_epochs: int = 40
    minimum_improvement: float = 1e-4
    role_limit_per_domain: int | None = None
    boundary_teacher_epochs: int = 40
    boundary_teacher_patience: int = 8
    maximum_boundary_candidates: int = 16
    boundary_patch_size: int = 128
    require_cuda: bool = True
    require_v3_initialization: bool = True
    export_onnx: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {"smoke", "full"}:
            raise ValueError("ShapeTrainingConfig.mode must be smoke or full")
        if tuple(phase.name for phase in self.phases) != (
            "heads",
            "decoder",
            "full",
        ):
            raise ValueError("v4 phases must be heads, decoder, full in that order")
        for value in (
            self.accumulation_steps_mobile,
            self.accumulation_steps_heavy,
            self.batch_size_mobile,
            self.batch_size_heavy,
        ):
            if value < 1:
                raise ValueError("Batch and accumulation settings must be positive")
        if self.role_limit_per_domain is not None and self.role_limit_per_domain < 1:
            raise ValueError("role_limit_per_domain must be positive")
        full_epochs = next(phase.epochs for phase in self.phases if phase.name == "full")
        if not 1 <= self.minimum_full_epochs <= full_epochs:
            raise ValueError("minimum_full_epochs must be within the configured full phase")
        if self.boundary_patch_size < 32 or self.boundary_patch_size % 16:
            raise ValueError("boundary_patch_size must be a multiple of 16")

    @classmethod
    def for_mode(cls, mode: str) -> ShapeTrainingConfig:
        if mode == "smoke":
            return cls(
                mode="smoke",
                phases=(
                    PhaseSpec("heads", 1, 1e-3),
                    PhaseSpec("decoder", 1, 3e-4),
                    PhaseSpec("full", 1, 5e-5),
                ),
                accumulation_steps_mobile=1,
                accumulation_steps_heavy=1,
                batch_size_mobile=1,
                batch_size_heavy=1,
                workers=0,
                early_stopping_patience=3,
                minimum_full_epochs=1,
                role_limit_per_domain=2,
                boundary_teacher_epochs=1,
                boundary_teacher_patience=1,
                maximum_boundary_candidates=3,
                boundary_patch_size=32,
                require_v3_initialization=False,
                export_onnx=True,
            )
        if mode != "full":
            raise ValueError("mode must be smoke or full")
        return cls(
            mode="full",
            phases=(
                PhaseSpec("heads", 15, 1e-3),
                PhaseSpec("decoder", 25, 3e-4),
                PhaseSpec("full", 160, 5e-5),
            ),
        )


@dataclass(frozen=True)
class MaterializedPair:
    dataset: str
    role: str
    image_path: Path
    mask_path: Path
    sample_id: str
    acquisition_group: str
    image_sha256: str
    mask_sha256: str
    image_size_bytes: int
    mask_size_bytes: int

    def as_manifest(self, root: Path) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "role": self.role,
            "sample_id": self.sample_id,
            "acquisition_group": self.acquisition_group,
            "image_path": str(self.image_path.relative_to(root)),
            "mask_path": str(self.mask_path.relative_to(root)),
            "image_sha256": self.image_sha256,
            "mask_sha256": self.mask_sha256,
            "image_size_bytes": self.image_size_bytes,
            "mask_size_bytes": self.mask_size_bytes,
        }


def _materialized_domain(image_path: Path) -> str:
    stem = image_path.stem
    if not stem.casefold().endswith("_img"):
        raise ValueError(f"Materialized image must end in _img: {image_path}")
    source = stem[:-4]
    match = re.match(r"^(?P<domain>.+?)_(?:train|val|test)_", source, re.I)
    if match is None:
        raise ValueError(
            "Cannot recover dataset before _(train|val|test)_ in "
            f"{image_path.name}"
        )
    return match.group("domain").casefold()


def discover_materialized_pairs(data_root: Path) -> tuple[list[MaterializedPair], dict[str, object]]:
    """Index only development roles and fail on any cross-role identity overlap."""

    root = data_root.resolve()
    records: list[MaterializedPair] = []
    seen_images: dict[str, tuple[str, Path]] = {}
    seen_masks: dict[str, tuple[str, Path]] = {}
    group_roles: dict[tuple[str, str], set[str]] = defaultdict(set)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for role in SCIENTIFIC_ROLES:
        directory = root / role
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing materialized scientific role: {directory}")
        image_paths = sorted(
            (
                path.resolve()
                for path in directory.rglob("*")
                if path.is_file()
                and path.suffix.casefold() in IMAGE_EXTENSIONS
                and path.stem.casefold().endswith("_img")
            ),
            key=lambda path: path.as_posix().casefold(),
        )
        if not image_paths:
            raise RuntimeError(f"Scientific role has no materialized pairs: {directory}")
        for image_path in image_paths:
            mask_path = image_path.with_name(f"{image_path.stem[:-4]}_masks.tif")
            if not mask_path.is_file():
                raise FileNotFoundError(
                    f"Materialized image has no matching instance mask: {image_path}"
                )
            mask_path = mask_path.resolve()
            dataset = _materialized_domain(image_path)
            group = scientific_group(dataset, image_path.name)
            image_digest = file_sha256(image_path)
            mask_digest = file_sha256(mask_path)
            prior_image = seen_images.get(image_digest)
            if prior_image is not None and prior_image[0] != role:
                raise RuntimeError(
                    "Byte-identical image crosses scientific roles: "
                    f"{prior_image[1]} ({prior_image[0]}) and {image_path} ({role})"
                )
            prior_mask = seen_masks.get(mask_digest)
            if prior_mask is not None and prior_mask[0] != role:
                raise RuntimeError(
                    "Byte-identical instance mask crosses scientific roles: "
                    f"{prior_mask[1]} ({prior_mask[0]}) and {mask_path} ({role})"
                )
            seen_images[image_digest] = (role, image_path)
            seen_masks[mask_digest] = (role, mask_path)
            group_roles[(dataset, group)].add(role)
            sample_id = hashlib.sha256(
                f"{dataset}:{role}:{image_digest}:{mask_digest}".encode()
            ).hexdigest()
            record = MaterializedPair(
                dataset=dataset,
                role=role,
                image_path=image_path,
                mask_path=mask_path,
                sample_id=sample_id,
                acquisition_group=group,
                image_sha256=image_digest,
                mask_sha256=mask_digest,
                image_size_bytes=image_path.stat().st_size,
                mask_size_bytes=mask_path.stat().st_size,
            )
            records.append(record)
            counts[dataset][role] += 1
    crossing = [
        (dataset, group, sorted(roles))
        for (dataset, group), roles in group_roles.items()
        if len(roles) > 1
    ]
    if crossing:
        dataset, group, roles = crossing[0]
        raise RuntimeError(
            f"Acquisition group crosses materialized roles: {dataset}/{group} -> {roles}"
        )
    rows = [record.as_manifest(root) for record in records]
    fingerprint = _json_sha256(rows)
    return records, {
        "schema_version": 1,
        "training_version": V4_SHAPE_TRAINING_VERSION,
        "split_protocol": SPLIT_PROTOCOL_VERSION,
        "data_root": str(root),
        "record_count": len(records),
        "counts": {
            dataset: dict(sorted(role_counts.items()))
            for dataset, role_counts in sorted(counts.items())
        },
        "dataset_fingerprint_sha256": fingerprint,
        "records": rows,
    }


def _fold_index(group: str, fold_count: int) -> int:
    return int.from_bytes(hashlib.sha256(group.encode()).digest()[:8], "big") % fold_count


@dataclass
class CellposeTeacherSource:
    """Optional pinned or OOF teacher cache used by the student dataset.

    ``models_by_checkpoint_sha256`` is needed only when ``generate_missing`` is true.  It contains
    already-instantiated, strictly loaded Cellpose models; this module never downloads weights.
    """

    cache_root: Path
    settings: TeacherInferenceSettings
    validation_provenance: TeacherProvenance
    train_provenance_by_fold: Mapping[int, TeacherProvenance] = field(default_factory=dict)
    models_by_checkpoint_sha256: Mapping[str, object] = field(default_factory=dict)
    generate_missing: bool = False

    def _provenance(self, record: MaterializedPair) -> tuple[TeacherProvenance, int | None]:
        if record.role != "train":
            if self.validation_provenance.source_kind == "oof_finetuned":
                raise ValueError("OOF teachers cannot generate validation-role caches")
            return self.validation_provenance, None
        if self.train_provenance_by_fold:
            fold_counts = {
                int(provenance.fold_count)
                for provenance in self.train_provenance_by_fold.values()
                if provenance.fold_count is not None
            }
            if len(fold_counts) != 1:
                raise ValueError("OOF teacher mapping must use one common fold count")
            fold_count = next(iter(fold_counts))
            fold = _fold_index(record.acquisition_group, fold_count)
            try:
                provenance = self.train_provenance_by_fold[fold]
            except KeyError as error:
                raise ValueError(f"No OOF teacher excludes training fold {fold}") from error
            if provenance.excluded_fold != fold:
                raise ValueError("OOF provenance key does not match its excluded fold")
            return provenance, fold
        if self.validation_provenance.source_kind == "full_train_finetuned":
            raise ValueError("Full-train teachers cannot predict their fitting samples")
        return self.validation_provenance, None

    def load_or_generate(
        self,
        record: MaterializedPair,
        image: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, object]]:
        provenance, fold = self._provenance(record)
        height, width = image.shape[:2]
        sample = TeacherSample(
            dataset=record.dataset,
            sample_id=record.sample_id,
            role=record.role,
            acquisition_group=record.acquisition_group,
            image_sha256=record.image_sha256,
            image_size_bytes=record.image_size_bytes,
            height=height,
            width=width,
            channels=1 if image.ndim == 2 else image.shape[2],
            fold_index=fold,
        )
        key = teacher_cache_key(sample, provenance, self.settings)
        directory = teacher_cache_directory(
            self.cache_root, provenance.base_model, key
        )
        if directory.is_dir():
            arrays, manifest = read_teacher_cache(
                directory,
                expected_sample=sample,
                expected_provenance=provenance,
                expected_settings=self.settings,
            )
            return arrays, manifest
        if not self.generate_missing:
            raise FileNotFoundError(
                f"Pinned Cellpose teacher cache is missing and generation is disabled: {directory}"
            )
        try:
            model = self.models_by_checkpoint_sha256[provenance.checkpoint_sha256]
        except KeyError as error:
            raise RuntimeError(
                "No strictly loaded Cellpose model was supplied for teacher checkpoint "
                f"{provenance.checkpoint_sha256}"
            ) from error
        outputs = run_cellpose_teacher(
            model,
            image,
            provenance.base_model,
            self.settings,
        )
        _, manifest, _ = write_teacher_cache(
            self.cache_root,
            sample,
            provenance,
            self.settings,
            outputs,
        )
        arrays, _ = read_teacher_cache(
            directory,
            expected_sample=sample,
            expected_provenance=provenance,
            expected_settings=self.settings,
        )
        return arrays, manifest

    def manifest(self) -> dict[str, object]:
        return {
            "cache_root": str(self.cache_root.resolve()),
            "settings": self.settings.as_manifest(),
            "validation_provenance": self.validation_provenance.as_manifest(),
            "train_provenance_by_fold": {
                str(fold): provenance.as_manifest()
                for fold, provenance in sorted(self.train_provenance_by_fold.items())
            },
            "generate_missing": self.generate_missing,
            "output_contract": dict(TEACHER_OUTPUT_CONTRACT),
        }


def materialize_cellpose_teacher_caches(
    records: Sequence[MaterializedPair],
    source: CellposeTeacherSource,
    *,
    roles: Sequence[str] = ("train", "checkpoint"),
) -> dict[str, object]:
    """Generate/validate teacher caches serially, then make worker access read-only.

    Cellpose model objects are deliberately discarded before any multi-worker DataLoader is
    constructed. This prevents forked workers from touching a CUDA context or duplicating the
    foundation model in GPU memory.
    """

    role_set = set(roles)
    if not role_set or not role_set <= {"train", "checkpoint"}:
        raise ValueError("Teacher caches may be materialized only for train/checkpoint")
    selected = sorted(
        (record for record in records if record.role in role_set),
        key=lambda record: (record.role, record.sample_id),
    )
    if not selected:
        raise RuntimeError("No train/checkpoint records were selected for teacher caching")
    generation_was_enabled = source.generate_missing
    cache_keys: list[str] = []
    role_counts: dict[str, int] = defaultdict(int)
    for record in selected:
        image, _ = _read_pair(record)
        arrays, manifest = source.load_or_generate(record, image)
        for key in TEACHER_OUTPUT_CONTRACT["cached_arrays"]:
            value = arrays[key]
            if value.shape != image.shape or value.dtype != np.float32:
                raise RuntimeError(
                    f"Teacher cache {key} contract mismatch for {record.sample_id}: "
                    f"{value.shape}/{value.dtype}"
                )
        cache_keys.append(str(manifest["cache_key"]))
        role_counts[record.role] += 1

    # This state transition is intentional and is part of the public helper's contract.
    source.generate_missing = False
    source.models_by_checkpoint_sha256 = {}
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "status": "materialized_and_read_only",
        "roles": sorted(role_set),
        "role_counts": dict(sorted(role_counts.items())),
        "cache_count": len(cache_keys),
        "cache_keys_sha256": _json_sha256(sorted(cache_keys)),
        "generation_was_enabled": generation_was_enabled,
        "models_cleared_before_dataloader": True,
    }


def _read_pair(record: MaterializedPair) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(record.image_path), cv2.IMREAD_UNCHANGED)
    labels = cv2.imread(str(record.mask_path), cv2.IMREAD_UNCHANGED)
    if image is None or labels is None:
        raise RuntimeError(f"Could not decode {record.image_path} / {record.mask_path}")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if labels.ndim == 3:
        labels = labels[..., 0]
    if image.shape != labels.shape:
        raise RuntimeError(
            f"Image/mask shape mismatch: {record.image_path} {image.shape}/{labels.shape}"
        )
    if not np.isfinite(image).all() or not np.isfinite(labels).all():
        raise RuntimeError(f"Non-finite materialized pair: {record.image_path}")
    labels = np.rint(labels).astype(np.int32)
    if np.any(labels < 0) or not np.any(labels > 0):
        raise RuntimeError(f"Invalid/non-cell instance mask: {record.mask_path}")
    if image.dtype != np.uint8:
        image_float = image.astype(np.float32)
        low, high = np.percentile(image_float, (0.5, 99.5))
        image = (
            np.zeros(image.shape, dtype=np.uint8)
            if high <= low
            else np.clip((image_float - low) * 255.0 / (high - low), 0, 255).astype(
                np.uint8
            )
        )
    return np.ascontiguousarray(image), np.ascontiguousarray(labels)


def _d4_scalar(array: np.ndarray, rotation: int, flip_x: bool) -> np.ndarray:
    transformed = np.rot90(array, rotation, axes=(-2, -1))
    if flip_x:
        transformed = np.flip(transformed, axis=-1)
    return np.ascontiguousarray(transformed)


def _d4_flow(
    flow_y: np.ndarray,
    flow_x: np.ndarray,
    rotation: int,
    flip_x: bool,
) -> tuple[np.ndarray, np.ndarray]:
    y = _d4_scalar(flow_y, rotation, flip_x)
    x = _d4_scalar(flow_x, rotation, flip_x)
    if rotation == 1:
        y, x = -x, y
    elif rotation == 2:
        y, x = -y, -x
    elif rotation == 3:
        y, x = x, -y
    if flip_x:
        x = -x
    return np.ascontiguousarray(y), np.ascontiguousarray(x)


def _resize_scalar(array: np.ndarray, size: int, nearest: bool = False) -> np.ndarray:
    return cv2.resize(
        array,
        (size, size),
        interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
    )


class MaterializedShapeDataset(Dataset):
    def __init__(
        self,
        records: Sequence[MaterializedPair],
        role: str,
        image_size: int,
        *,
        train: bool,
        seed: int,
        limit_per_domain: int | None = None,
        teacher_source: CellposeTeacherSource | None = None,
    ) -> None:
        selected = sorted(
            (record for record in records if record.role == role),
            key=lambda record: record.sample_id,
        )
        if limit_per_domain is not None:
            counts: dict[str, int] = defaultdict(int)
            retained = []
            for record in selected:
                if counts[record.dataset] >= limit_per_domain:
                    continue
                counts[record.dataset] += 1
                retained.append(record)
            selected = retained
        if not selected:
            raise RuntimeError(f"No materialized records for scientific role {role}")
        self.records = selected
        self.role = role
        self.image_size = image_size
        self.train_mode = train
        self.seed = seed
        self.epoch = 0
        self.teacher_source = teacher_source

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        image, labels = _read_pair(record)
        cached_teacher: dict[str, np.ndarray] | None = None
        teacher_cache_key_value = "none"
        if self.teacher_source is not None:
            cached_teacher, manifest = self.teacher_source.load_or_generate(record, image)
            teacher_cache_key_value = str(manifest["cache_key"])

        digest = hashlib.sha256(
            f"{self.seed}:{self.epoch}:{record.sample_id}".encode()
        ).digest()
        rotation = digest[0] % 4 if self.train_mode else 0
        flip_x = bool(digest[1] & 1) if self.train_mode else False
        image = _d4_scalar(image, rotation, flip_x)
        labels = _d4_scalar(labels, rotation, flip_x)
        if self.train_mode:
            # Deterministic image-only tone/noise augmentation; geometric targets are generated
            # below from already transformed labels.
            rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
            gamma = float(rng.uniform(0.75, 1.35))
            exposure = float(2.0 ** rng.uniform(-0.35, 0.35))
            normalized = np.clip(image.astype(np.float32) / 255.0 * exposure, 0, 1)
            normalized = normalized**gamma
            if digest[2] % 3 == 0:
                normalized += rng.normal(0.0, 0.012, normalized.shape).astype(np.float32)
            image = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)

        image = _resize_scalar(image, self.image_size)
        labels = _resize_scalar(labels, self.image_size, nearest=True).astype(np.int32)
        targets_numpy = geometry_targets(labels)
        target_tensors = {
            key: torch.from_numpy(np.ascontiguousarray(value)).float()
            for key, value in targets_numpy.items()
            if key != "affinity_offsets"
        }
        target_tensors["valid"] = torch.ones(
            1, self.image_size, self.image_size, dtype=torch.float32
        )
        for key in (
            "foreground",
            "internal_contact",
            "signed_distance",
            "centroid_flow_y",
            "centroid_flow_x",
        ):
            target_tensors[key] = target_tensors[key][None]

        teacher_tensors = {
            "flow_y_raw": torch.zeros(1, self.image_size, self.image_size),
            "flow_x_raw": torch.zeros(1, self.image_size, self.image_size),
            "cellprob_logit": torch.zeros(1, self.image_size, self.image_size),
        }
        teacher_available = False
        if cached_teacher is not None:
            flow_y, flow_x = _d4_flow(
                cached_teacher["flow_y_raw"],
                cached_teacher["flow_x_raw"],
                rotation,
                flip_x,
            )
            cellprob = _d4_scalar(
                cached_teacher["cellprob_logit"], rotation, flip_x
            )
            teacher_tensors = {
                "flow_y_raw": torch.from_numpy(
                    _resize_scalar(flow_y, self.image_size)[None]
                ).float(),
                "flow_x_raw": torch.from_numpy(
                    _resize_scalar(flow_x, self.image_size)[None]
                ).float(),
                "cellprob_logit": torch.from_numpy(
                    _resize_scalar(cellprob, self.image_size)[None]
                ).float(),
            }
            teacher_available = True

        image_tensor = torch.from_numpy(
            np.repeat(image[None], 3, axis=0).copy()
        ).float() / 255.0
        return {
            "image": image_tensor,
            "instances": torch.from_numpy(labels.astype(np.int64)),
            "targets": target_tensors,
            "cellpose_teacher": teacher_tensors,
            "teacher_available": torch.tensor(teacher_available),
            "teacher_cache_key": teacher_cache_key_value,
            "sample_id": record.sample_id,
            "dataset": record.dataset,
            "role": record.role,
            "acquisition_group": record.acquisition_group,
        }


def _synthetic_proposals(labels: np.ndarray, maximum_splits: int = 2) -> np.ndarray:
    proposals = np.asarray(labels, dtype=np.int32).copy()
    values, counts = np.unique(proposals[proposals > 0], return_counts=True)
    next_label = int(proposals.max()) + 1
    for value in values[np.argsort(-counts)[:maximum_splits]]:
        ys, xs = np.nonzero(proposals == int(value))
        if len(ys) < 16:
            continue
        if np.ptp(xs) >= np.ptp(ys):
            split = np.median(xs)
            side = xs > split
        else:
            split = np.median(ys)
            side = ys > split
        if not side.any() or side.all():
            continue
        proposals[ys[side], xs[side]] = next_label
        next_label += 1
    return proposals


def _candidate_line(labels: np.ndarray, left: int, right: int) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    region_left = labels == left
    region_right = labels == right
    for iterations in (1, 2, 3):
        line = cv2.dilate(
            region_left.astype(np.uint8), kernel, iterations=iterations
        ).astype(bool) & cv2.dilate(
            region_right.astype(np.uint8), kernel, iterations=iterations
        ).astype(bool)
        if line.any():
            return line
    return np.zeros(labels.shape, dtype=bool)


def _candidate_examples(
    image: np.ndarray,
    proposals: np.ndarray,
    *,
    truth: np.ndarray | None,
    foreground_probability: np.ndarray,
    context_boundary_probability: np.ndarray,
    flow_y: np.ndarray,
    flow_x: np.ndarray,
    spec: BoundaryTeacherSpec,
    maximum_candidates: int,
) -> dict[str, object] | None:
    proposals = np.asarray(proposals, dtype=np.int32)
    pairs = generate_candidate_pairs(proposals, max_background_gap=2)
    # Candidate membership, order, and cap must be proposal-only because graph attention lets
    # every output observe the other candidates.  Ground-truth-balanced or utility-ranked sets
    # would leak labels indirectly and create a different graph distribution at inference.
    proposal_ranked_pairs = sorted(
        pairs,
        key=lambda pair: (
            -int(_candidate_line(proposals, pair[0], pair[1]).sum()),
            -min(
                int(np.count_nonzero(proposals == pair[0])),
                int(np.count_nonzero(proposals == pair[1])),
            ),
            pair[0],
            pair[1],
        ),
    )[:maximum_candidates]
    if not proposal_ranked_pairs:
        return None
    if truth is not None:
        decisions = candidate_keep_merge_targets(
            proposals,
            truth,
            proposal_ranked_pairs,
            max_background_gap=2,
        )
        selected = [
            (
                decision.region_a,
                decision.region_b,
                float(decision.label),
                float(decision.utility),
            )
            for decision in decisions
        ]
    else:
        selected = [
            (int(left), int(right), math.nan, math.nan)
            for left, right in proposal_ranked_pairs
        ]
    if not selected:
        return None
    grayscale = image.mean(axis=0) if image.ndim == 3 else image
    patches = []
    geometries = []
    lines = []
    labels_out = []
    utilities = []
    for region_a, region_b, label, utility in selected:
        patch, geometry = build_candidate_patch(
            grayscale,
            proposals,
            region_a,
            region_b,
            foreground_probability=foreground_probability,
            context_boundary_probability=context_boundary_probability,
            cellpose_flow_y=flow_y,
            cellpose_flow_x=flow_x,
            patch_size=spec.patch_size,
        )
        patches.append(patch)
        geometries.append(geometry)
        lines.append(
            _candidate_line(proposals, region_a, region_b)
        )
        labels_out.append(label)
        utilities.append(utility)
    result: dict[str, object] = {
        "patches": np.stack(patches).astype(np.float32),
        "geometry": np.stack(geometries).astype(np.float32),
        "lines": lines,
        "candidate_pairs": [(left, right) for left, right, _, _ in selected],
        "proposal_instance_count": int(proposals.max(initial=0)),
    }
    if truth is not None:
        result["labels"] = np.asarray(labels_out, dtype=np.float32)
        result["utility"] = np.asarray(utilities, dtype=np.float32)
    return result


@torch.inference_mode()
def _frozen_proposal_probabilities(
    proposal_model: V4ShapeNet,
    image: Tensor,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    outputs = proposal_model(image[None].to(device))
    # Exact imported v3 semantic channels are used; new v4 refiners cannot leak their own labels
    # into the teacher's training proposals.
    base = torch.sigmoid(outputs["base_segmentation_logits"][0]).float().cpu().numpy()
    return base[0], base[1]


def _proposal_instances(
    foreground_probability: np.ndarray,
    context_boundary_probability: np.ndarray,
) -> np.ndarray:
    reconstructed = reconstruct_iphone_instances(
        foreground_probability,
        context_boundary_probability,
        _ProposalReconstructionSettings(),
    ).astype(np.int32)
    # Add deterministic counterfactual cuts to the *model proposal*, never to ground truth.  They
    # create merge-negative examples even when a frozen model happens to predict only one region;
    # occasionally a cut repairs a true model merge and correctly becomes a keep-positive.
    return _synthetic_proposals(reconstructed, maximum_splits=2)


def _boundary_examples_from_dataset(
    dataset: MaterializedShapeDataset,
    index: int,
    spec: BoundaryTeacherSpec,
    maximum_candidates: int,
    proposal_model: V4ShapeNet | None,
    device: torch.device,
    *,
    contract_fixture_only: bool,
) -> dict[str, object] | None:
    sample = dataset[index]
    image = sample["image"].numpy()
    truth = sample["instances"].numpy().astype(np.int32)
    targets = sample["targets"]
    if proposal_model is not None:
        foreground, context = _frozen_proposal_probabilities(
            proposal_model,
            sample["image"],
            device,
        )
        proposals = _proposal_instances(foreground, context)
    elif contract_fixture_only:
        # The orchestration contract intentionally has no v3 checkpoint.  This isolated path
        # exercises graph tensor plumbing only and is explicitly recorded in its manifest; full
        # and best-smoke runs are forbidden from entering it.
        foreground = targets["foreground"][0].numpy()
        context = targets["internal_contact"][0].numpy()
        proposals = _synthetic_proposals(truth)
    else:
        raise RuntimeError("Boundary teacher requires a frozen v3 proposal model")
    if bool(sample["teacher_available"]):
        flow_y = sample["cellpose_teacher"]["flow_y_raw"][0].numpy() / 5.0
        flow_x = sample["cellpose_teacher"]["flow_x_raw"][0].numpy() / 5.0
    else:
        # Never substitute ground-truth centroid flow into a teacher input.
        flow_y = np.zeros_like(foreground, dtype=np.float32)
        flow_x = np.zeros_like(foreground, dtype=np.float32)
    return _candidate_examples(
        image,
        proposals,
        truth=truth,
        foreground_probability=foreground,
        context_boundary_probability=context,
        flow_y=flow_y,
        flow_x=flow_x,
        spec=spec,
        maximum_candidates=maximum_candidates,
    )


def _assert_candidate_graph_is_truth_independent() -> None:
    proposals = np.zeros((40, 48), dtype=np.int32)
    proposals[8:32, 4:17] = 1
    proposals[8:32, 17:30] = 2
    proposals[8:32, 30:43] = 3
    truth_keep = proposals.copy()
    truth_merge = proposals.copy()
    truth_merge[truth_merge == 2] = 1
    truth_merge[truth_merge == 3] = 2
    image = np.linspace(0.0, 1.0, proposals.size, dtype=np.float32).reshape(
        proposals.shape
    )
    foreground = (proposals > 0).astype(np.float32)
    context = np.zeros_like(foreground)
    zeros = np.zeros_like(foreground)
    spec = BoundaryTeacherSpec(
        patch_size=32,
        embedding_size=32,
        graph_layers=1,
        graph_heads=4,
        dropout=0.0,
    )
    first = _candidate_examples(
        image,
        proposals,
        truth=truth_keep,
        foreground_probability=foreground,
        context_boundary_probability=context,
        flow_y=zeros,
        flow_x=zeros,
        spec=spec,
        maximum_candidates=2,
    )
    second = _candidate_examples(
        image,
        proposals,
        truth=truth_merge,
        foreground_probability=foreground,
        context_boundary_probability=context,
        flow_y=zeros,
        flow_x=zeros,
        spec=spec,
        maximum_candidates=2,
    )
    if first is None or second is None:
        raise AssertionError("Truth-independence fixture produced no candidates")
    for key in ("patches", "geometry"):
        if not np.array_equal(first[key], second[key]):
            raise AssertionError(f"Ground truth changed boundary candidate {key}")
    if first["candidate_pairs"] != second["candidate_pairs"]:
        raise AssertionError("Ground truth changed boundary candidate membership/order")
    if any(
        not np.array_equal(left, right)
        for left, right in zip(first["lines"], second["lines"])
    ):
        raise AssertionError("Ground truth changed boundary candidate lines")


def _boundary_epoch(
    model: BoundaryGraphTeacher,
    dataset: MaterializedShapeDataset,
    device: torch.device,
    spec: BoundaryTeacherSpec,
    maximum_candidates: int,
    optimizer: torch.optim.Optimizer | None,
    proposal_model: V4ShapeNet | None,
    *,
    contract_fixture_only: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = defaultdict(float)
    examples = 0
    correct = 0
    for index in range(len(dataset)):
        candidate = _boundary_examples_from_dataset(
            dataset,
            index,
            spec,
            maximum_candidates,
            proposal_model,
            device,
            contract_fixture_only=contract_fixture_only,
        )
        if candidate is None:
            continue
        patches = torch.from_numpy(candidate["patches"])[None].to(device)
        geometry = torch.from_numpy(candidate["geometry"])[None].to(device)
        labels = torch.from_numpy(candidate["labels"])[None].to(device)
        utility = torch.from_numpy(candidate["utility"])[None].to(device)
        graph_valid = torch.ones(labels.shape, dtype=torch.bool, device=device)
        supervision_valid = labels >= 0
        if not bool(supervision_valid.any()):
            continue
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(patches, geometry, graph_valid)
            loss = boundary_teacher_loss(
                outputs,
                labels.clamp_min(0),
                utility,
                supervision_valid,
            )
            if optimizer is not None:
                loss["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        candidates = int(supervision_valid.sum())
        examples += candidates
        correct += int(
            ((torch.sigmoid(outputs["keep_logit"]) >= 0.5) == labels.bool())
            .logical_and(supervision_valid)
            .sum()
        )
        for name, value in loss.items():
            totals[name] += float(value.detach()) * candidates
    if examples == 0:
        raise RuntimeError(
            f"No valid keep/merge candidates in boundary-teacher role {dataset.role}"
        )
    return {
        **{name: value / examples for name, value in totals.items()},
        "accuracy": correct / examples,
        "candidates": float(examples),
    }


def _train_boundary_teacher(
    train_dataset: MaterializedShapeDataset,
    checkpoint_dataset: MaterializedShapeDataset,
    output_directory: Path,
    run_contract_sha256: str,
    config: ShapeTrainingConfig,
    device: torch.device,
    *,
    proposal_model: V4ShapeNet | None,
    proposal_source: Mapping[str, object],
    compact_contract: bool = False,
) -> tuple[BoundaryGraphTeacher, dict[str, object]]:
    spec = (
        BoundaryTeacherSpec(
            patch_size=config.boundary_patch_size,
            embedding_size=32,
            graph_layers=1,
            graph_heads=4,
            dropout=0.0,
        )
        if compact_contract
        else BoundaryTeacherSpec(patch_size=config.boundary_patch_size)
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "manifest.json"
    manifest_core = {
        "schema_version": 1,
        "training_version": V4_SHAPE_TRAINING_VERSION,
        "boundary_teacher_version": BOUNDARY_TEACHER_VERSION,
        "run_contract_sha256": run_contract_sha256,
        "spec": asdict(spec),
        "epochs": config.boundary_teacher_epochs,
        "train_role": "train",
        "selection_role": "checkpoint",
        "proposal_source": proposal_source,
        "ground_truth_input_policy": (
            "contract-fixture tensor-plumbing exception; not a scientific result"
            if compact_contract
            else "ground truth labels score frozen-model keep/merge proposals only; they are "
            "never rendered as teacher input probabilities, flows, or proposal instances"
        ),
    }
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        if existing != manifest_core:
            raise RuntimeError("Boundary-teacher manifest changed; refusing stale resume")
    else:
        _atomic_json(manifest_path, manifest_core)

    model = BoundaryGraphTeacher(spec).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    last_path = output_directory / "last.pt"
    best_path = output_directory / "best.pt"
    history: list[dict[str, object]] = []
    start_epoch = 0
    best_score = -math.inf
    stale = 0
    if last_path.is_file():
        checkpoint = _load_checkpoint(last_path)
        if checkpoint.get("run_contract_sha256") != run_contract_sha256:
            raise RuntimeError("Boundary-teacher checkpoint contract mismatch")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = list(checkpoint["history"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_score = float(checkpoint["best_score"])
        stale = int(checkpoint["stale"])
        torch.set_rng_state(checkpoint["torch_random_state"])
        if stale >= config.boundary_teacher_patience:
            # The previous process already reached the deterministic early-stop condition.
            start_epoch = config.boundary_teacher_epochs
    for epoch in range(start_epoch, config.boundary_teacher_epochs):
        train_dataset.set_epoch(epoch)
        train_metrics = _boundary_epoch(
            model,
            train_dataset,
            device,
            spec,
            config.maximum_boundary_candidates,
            optimizer,
            proposal_model,
            contract_fixture_only=compact_contract,
        )
        checkpoint_dataset.set_epoch(0)
        validation_metrics = _boundary_epoch(
            model,
            checkpoint_dataset,
            device,
            spec,
            config.maximum_boundary_candidates,
            None,
            proposal_model,
            contract_fixture_only=compact_contract,
        )
        score = validation_metrics["accuracy"] - 0.05 * validation_metrics["total"]
        row: dict[str, object] = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "checkpoint": validation_metrics,
            "selection_score": score,
        }
        history.append(row)
        if score > best_score + 1e-5:
            best_score = score
            stale = 0
            _atomic_torch_save(
                {
                    "schema_version": V4_SHAPE_CHECKPOINT_SCHEMA,
                    "run_contract_sha256": run_contract_sha256,
                    "spec": asdict(spec),
                    "model": model.state_dict(),
                    "selection_score": score,
                },
                best_path,
            )
        else:
            stale += 1
        _atomic_torch_save(
            {
                "schema_version": V4_SHAPE_CHECKPOINT_SCHEMA,
                "run_contract_sha256": run_contract_sha256,
                "epoch": epoch,
                "spec": asdict(spec),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "best_score": best_score,
                "stale": stale,
                "torch_random_state": torch.get_rng_state(),
            },
            last_path,
        )
        _atomic_jsonl(output_directory / "history.jsonl", history)
        if stale >= config.boundary_teacher_patience:
            break
    if not best_path.is_file():
        raise RuntimeError("Boundary teacher completed without a best checkpoint")
    best = _load_checkpoint(best_path)
    model.load_state_dict(best["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    proposal_audit = (
        audit_frozen_teacher(proposal_model, str(proposal_source["state_sha256"]))
        if proposal_model is not None
        else {"status": "contract fixture only; no proposal model"}
    )
    return model, {
        "spec": asdict(spec),
        "checkpoint": str(best_path),
        "checkpoint_sha256": file_sha256(best_path),
        "best_selection_score": float(best["selection_score"]),
        "epochs_completed": len(history),
        "history": str(output_directory / "history.jsonl"),
        "proposal_source": dict(proposal_source),
        "proposal_source_post_training_audit": proposal_audit,
    }


def _dense_boundary_teacher_probability(
    teacher: BoundaryGraphTeacher,
    images: Tensor,
    outputs: Mapping[str, Tensor],
    cellpose: Mapping[str, Tensor],
    teacher_available: Tensor,
    device: torch.device,
    maximum_candidates: int,
) -> Tensor:
    dense_maps = []
    spec = teacher.spec
    for index in range(images.shape[0]):
        image = images[index].detach().cpu().numpy()
        foreground = torch.sigmoid(outputs["foreground_logit"][index, 0]).detach().cpu().numpy()
        context = torch.sigmoid(
            outputs["context_boundary_logit"][index, 0]
        ).detach().cpu().numpy()
        proposals = _proposal_instances(foreground, context)
        if bool(teacher_available[index]):
            flow_y = cellpose["flow_y_raw"][index, 0].cpu().numpy() / 5.0
            flow_x = cellpose["flow_x_raw"][index, 0].cpu().numpy() / 5.0
        else:
            flow_y = np.zeros_like(foreground, dtype=np.float32)
            flow_x = np.zeros_like(foreground, dtype=np.float32)
        candidate = _candidate_examples(
            image,
            proposals,
            truth=None,
            foreground_probability=foreground,
            context_boundary_probability=context,
            flow_y=flow_y,
            flow_x=flow_x,
            spec=spec,
            maximum_candidates=maximum_candidates,
        )
        # Outside candidate lines the teacher is neutral/current-model preserving.  Crucially,
        # ground-truth contact is never copied into this teacher output; only learned keep/merge
        # corrections can alter the detached context probability.
        dense = np.asarray(context, dtype=np.float32).copy()
        if candidate is not None:
            patches = torch.from_numpy(candidate["patches"])[None].to(device)
            geometry = torch.from_numpy(candidate["geometry"])[None].to(device)
            valid = torch.ones(
                (1, patches.shape[1]), dtype=torch.bool, device=device
            )
            with torch.inference_mode():
                probability = torch.sigmoid(
                    teacher(patches, geometry, valid)["keep_logit"][0]
                ).cpu().numpy()
            if len(candidate["lines"]) != len(probability):
                raise RuntimeError("Boundary-teacher candidate/probability count changed")
            for line, value in zip(candidate["lines"], probability):
                dense[line] = float(value)
        dense_maps.append(dense)
    return torch.from_numpy(np.stack(dense_maps)[:, None]).to(
        device=device,
        dtype=images.dtype,
    )


def _to_device_mapping(value: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {
        key: tensor.to(device, non_blocking=device.type == "cuda")
        for key, tensor in value.items()
    }


def _metric_counts(probability: Tensor, truth: Tensor, threshold: float) -> dict[str, float]:
    prediction = probability >= threshold
    expected = truth >= 0.5
    intersection = float((prediction & expected).sum())
    predicted = float(prediction.sum())
    actual = float(expected.sum())
    union = predicted + actual - intersection
    return {
        "dice": (2 * intersection + 1.0) / (predicted + actual + 1.0),
        "iou": (intersection + 1.0) / (union + 1.0),
        "precision": (intersection + 1.0) / (predicted + 1.0),
        "recall": (intersection + 1.0) / (actual + 1.0),
    }


@torch.inference_mode()
def _evaluate_student(
    model: V4ShapeNet,
    loader: DataLoader,
    device: torch.device,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, object]:
    model.eval()
    threshold_map = {name: 0.5 for name in CALIBRATED_OUTPUT_NAMES}
    if thresholds is not None:
        threshold_map.update({name: float(value) for name, value in thresholds.items()})
    rows_by_domain: dict[str, list[dict[str, float]]] = defaultdict(list)
    loss_rows: dict[str, list[float]] = defaultdict(list)
    fusion_weight_sum = np.zeros(3, dtype=np.float64)
    fusion_pixels = 0
    for batch in loader:
        images = batch["image"].to(device)
        targets = _to_device_mapping(batch["targets"], device)
        outputs = model(images)
        loss = multitask_shape_loss(outputs, targets, teacher=None)
        components = loss["components"]
        for name, value in components.items():
            loss_rows[name].append(float(value.detach()))
        weights = outputs["boundary_fusion_weights"].detach().cpu().numpy()
        fusion_weight_sum += weights.sum(axis=(0, 2, 3))
        fusion_pixels += weights.shape[0] * weights.shape[2] * weights.shape[3]
        for index, domain in enumerate(batch["dataset"]):
            row: dict[str, float] = {}
            for name in CALIBRATED_OUTPUT_NAMES:
                truth_name = (
                    "foreground" if name == "foreground_logit" else "internal_contact"
                )
                metrics = _metric_counts(
                    torch.sigmoid(outputs[name][index, 0]),
                    targets[truth_name][index, 0],
                    threshold_map[name],
                )
                for metric_name, value in metrics.items():
                    row[f"{name}_{metric_name}"] = value
            rows_by_domain[str(domain)].append(row)
    if not rows_by_domain:
        raise RuntimeError("Student evaluation loader was empty")
    per_domain = {
        domain: {
            key: float(np.mean([row[key] for row in rows]))
            for key in rows[0]
        }
        for domain, rows in sorted(rows_by_domain.items())
    }
    macro = {
        key: float(np.mean([metrics[key] for metrics in per_domain.values()]))
        for key in next(iter(per_domain.values()))
    }
    score = (
        0.25 * macro["foreground_logit_dice"]
        + 0.35 * macro["fused_boundary_logit_dice"]
        + 0.40
        / 3.0
        * sum(
            macro[f"{name}_dice"]
            for name in (
                "context_boundary_logit",
                "flow_boundary_logit",
                "shape_boundary_logit",
            )
        )
    )
    result: dict[str, object] = {
        "thresholds": dict(threshold_map),
        "selection_score": score,
        "macro_domain_metrics": macro,
        "per_domain_metrics": per_domain,
        "domain_image_counts": {
            domain: len(rows) for domain, rows in sorted(rows_by_domain.items())
        },
        "mean_loss_components": {
            name: float(np.mean(values)) for name, values in sorted(loss_rows.items())
        },
        "mean_boundary_fusion_weights": (
            (fusion_weight_sum / max(fusion_pixels, 1)).tolist()
        ),
    }
    return result


@torch.inference_mode()
def _calibrate_thresholds(
    model: V4ShapeNet,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    """Select per-branch thresholds without retaining full-resolution prediction maps."""

    model.eval()
    candidates = tuple(round(0.10 + 0.05 * index, 2) for index in range(17))
    domain_scores: dict[tuple[str, float, str], list[float]] = defaultdict(list)
    examples = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=device.type == "cuda")
        targets = _to_device_mapping(batch["targets"], device)
        outputs = model(images)
        for index, domain_value in enumerate(batch["dataset"]):
            domain = str(domain_value)
            for name in CALIBRATED_OUTPUT_NAMES:
                truth_name = (
                    "foreground" if name == "foreground_logit" else "internal_contact"
                )
                probability = torch.sigmoid(outputs[name][index, 0])
                truth = targets[truth_name][index, 0]
                for threshold in candidates:
                    domain_scores[(name, threshold, domain)].append(
                        _metric_counts(probability, truth, threshold)["dice"]
                    )
            examples += 1
    if examples == 0:
        raise ValueError("Calibration loader was empty")
    selected: dict[str, float] = {}
    curves: dict[str, list[dict[str, float]]] = {}
    for name in CALIBRATED_OUTPUT_NAMES:
        curve = []
        for threshold in candidates:
            domains = sorted(
                domain
                for candidate_name, candidate_threshold, domain in domain_scores
                if candidate_name == name and candidate_threshold == threshold
            )
            macro_dice = float(
                np.mean(
                    [
                        np.mean(domain_scores[(name, threshold, domain)])
                        for domain in domains
                    ]
                )
            )
            curve.append({"threshold": threshold, "macro_domain_dice": macro_dice})
        winner = max(curve, key=lambda row: (row["macro_domain_dice"], -abs(row["threshold"] - 0.5)))
        selected[name] = float(winner["threshold"])
        curves[name] = curve
    return {
        "selection_role": "calibration",
        "candidate_thresholds": list(candidates),
        "selected_thresholds": selected,
        "curves": curves,
        "examples": examples,
    }


@torch.inference_mode()
def _run_frozen_boundary_search(
    model: V4ShapeNet,
    datasets: Mapping[str, MaterializedShapeDataset],
    output_directory: Path,
    device: torch.device,
    config: ShapeTrainingConfig,
    batch_size: int,
) -> dict[str, object]:
    """Cache real frozen-student outputs, then search without touching final-test data."""

    model.eval()
    search_directory = output_directory / "boundary_search"
    probability_cache = FiveChannelDiskCache(search_directory / "five_channel_cache")
    truth_directory = search_directory / "truth_cache"
    descriptors: list[BoundaryEvaluationRecord] = []
    descriptor_rows: list[dict[str, object]] = []
    for role in ("calibration", "ensemble_selection"):
        loader = _make_loader(
            datasets[role],
            min(2, batch_size),
            min(2, config.workers),
            shuffle=False,
            seed=config.seed,
        )
        for batch in loader:
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            outputs = model(images)
            five_channel_probability = torch.sigmoid(
                torch.cat(
                    (
                        outputs["foreground_logit"],
                        outputs["fused_boundary_logit"],
                        outputs["context_boundary_logit"],
                        outputs["flow_boundary_logit"],
                        outputs["shape_boundary_logit"],
                    ),
                    dim=1,
                )
            ).detach().cpu().numpy().astype(np.float32)
            instances = batch["instances"].numpy().astype(np.int32)
            for index, sample_id_value in enumerate(batch["sample_id"]):
                sample_id = str(sample_id_value)
                record_id = f"{role}:{sample_id}"
                probability = five_channel_probability[index]
                existed = probability_cache.verify(record_id)
                probability_path = probability_cache.write(
                    record_id,
                    probability,
                    from_logits=False,
                )
                if existed:
                    cached = probability_cache.load(record_id, mmap=True)
                    if cached.shape != probability.shape or not np.allclose(
                        cached, probability, rtol=0.0, atol=1e-6
                    ):
                        raise RuntimeError(
                            f"Frozen boundary cache changed for {record_id}; refusing stale reuse"
                        )

                truth = instances[index]
                truth_key = hashlib.sha256(record_id.encode()).hexdigest()
                truth_path = truth_directory / truth_key[:2] / f"{truth_key}.npy"
                if truth_path.is_file():
                    cached_truth = np.load(truth_path, mmap_mode="r", allow_pickle=False)
                    if cached_truth.shape != truth.shape or not np.array_equal(
                        cached_truth, truth
                    ):
                        raise RuntimeError(
                            f"Frozen truth cache changed for {record_id}; refusing stale reuse"
                        )
                else:
                    _atomic_numpy(truth_path, truth)

                descriptor = probability_cache.record(
                    record_id,
                    role=role,
                    domain=str(batch["dataset"][index]),
                    truth_source=truth_path,
                    group_id=str(batch["acquisition_group"][index]),
                )
                descriptors.append(descriptor)
                descriptor_rows.append(
                    {
                        "record_id": record_id,
                        "role": role,
                        "domain": descriptor.domain,
                        "group_id": descriptor.group_id,
                        "probability_path": str(probability_path),
                        "probability_sha256": file_sha256(probability_path),
                        "truth_path": str(truth_path),
                        "truth_sha256": file_sha256(truth_path),
                    }
                )
    descriptor_manifest = {
        "schema_version": 1,
        "roles": ["calibration", "ensemble_selection"],
        "final_test": "not instantiated or read",
        "five_channel_order": [
            "foreground",
            "learned_fused_boundary",
            "context_boundary",
            "flow_boundary",
            "shape_boundary",
        ],
        "records": descriptor_rows,
    }
    descriptor_manifest["records_sha256"] = _json_sha256(descriptor_rows)
    _atomic_json(search_directory / "frozen_output_manifest.json", descriptor_manifest)
    bounds = (
        BoundarySearchBounds.smoke()
        if config.mode == "smoke"
        else BoundarySearchBounds()
    )
    report = run_v4_boundary_search(
        descriptors,
        output_directory=search_directory,
        bounds=bounds,
        smoke=config.mode == "smoke",
    )
    report["frozen_output_manifest"] = str(
        search_directory / "frozen_output_manifest.json"
    )
    report["frozen_output_manifest_sha256"] = file_sha256(
        search_directory / "frozen_output_manifest.json"
    )
    return report


def _make_loader(
    dataset: MaterializedShapeDataset,
    batch_size: int,
    workers: int,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    arguments: dict[str, object] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": torch.cuda.is_available(),
        "generator": generator,
        "persistent_workers": False,
    }
    if workers:
        arguments["prefetch_factor"] = 1
    return DataLoader(**arguments)


def _phase_schedule(config: ShapeTrainingConfig) -> list[tuple[str, int, float]]:
    return [
        (phase.name, phase_epoch, phase.learning_rate)
        for phase in config.phases
        for phase_epoch in range(phase.epochs)
    ]


def _should_stop_full_phase(
    phase: str,
    phase_epoch: int,
    phase_stale: int,
    config: ShapeTrainingConfig,
) -> bool:
    """Apply patience only to progress measured within a sufficiently long full phase."""

    return (
        phase == "full"
        and phase_epoch + 1 >= config.minimum_full_epochs
        and phase_stale >= config.early_stopping_patience
    )


def _model_batch_settings(
    spec: ShapeModelSpec, config: ShapeTrainingConfig
) -> tuple[int, int]:
    heavy = spec.tier in {"high_accuracy", "contract_heavy"}
    return (
        config.batch_size_heavy if heavy else config.batch_size_mobile,
        config.accumulation_steps_heavy if heavy else config.accumulation_steps_mobile,
    )


def _restore_rng(checkpoint: Mapping[str, object], device: torch.device) -> None:
    required = ("python_random_state", "numpy_random_state", "torch_random_state")
    missing = [name for name in required if name not in checkpoint]
    if missing:
        raise RuntimeError("Resume checkpoint lacks RNG state: " + ", ".join(missing))
    random.setstate(checkpoint["python_random_state"])
    np.random.set_state(checkpoint["numpy_random_state"])
    torch.set_rng_state(checkpoint["torch_random_state"])
    if device.type == "cuda":
        if "cuda_random_state" not in checkpoint:
            raise RuntimeError("CUDA resume checkpoint lacks CUDA RNG state")
        torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])


def _train_student(
    spec: ShapeModelSpec,
    records: Sequence[MaterializedPair],
    output_directory: Path,
    run_contract_sha256: str,
    config: ShapeTrainingConfig,
    device: torch.device,
    boundary_teacher: BoundaryGraphTeacher,
    *,
    v3_checkpoint: Path | None,
    teacher_source: CellposeTeacherSource | None,
) -> tuple[V4ShapeNet, dict[str, object]]:
    output_directory.mkdir(parents=True, exist_ok=True)
    batch_size, accumulation_steps = _model_batch_settings(spec, config)
    datasets = {
        role: MaterializedShapeDataset(
            records,
            role,
            spec.image_size,
            train=role == "train",
            seed=config.seed,
            limit_per_domain=config.role_limit_per_domain,
            teacher_source=teacher_source if role in {"train", "checkpoint"} else None,
        )
        for role in SCIENTIFIC_ROLES
    }
    model = V4ShapeNet(spec).to(device)
    initialization: dict[str, object]
    if v3_checkpoint is not None:
        initialization = initialize_from_v3(model, v3_checkpoint)
    elif config.require_v3_initialization:
        raise RuntimeError(f"Full v4 run requires a compatible v3 checkpoint for {spec.name}")
    else:
        initialization = {"status": "not supplied; smoke-only random backbone"}

    schedule = _phase_schedule(config)
    manifest_core = {
        "schema_version": 1,
        "training_version": V4_SHAPE_TRAINING_VERSION,
        "run_contract_sha256": run_contract_sha256,
        "spec": asdict(spec),
        "config": asdict(config),
        "schedule": schedule,
        "initialization": initialization,
        "v3_checkpoint_sha256": file_sha256(v3_checkpoint) if v3_checkpoint else None,
        "roles": {
            role: [record.sample_id for record in dataset.records]
            for role, dataset in datasets.items()
        },
    }
    manifest_path = output_directory / "manifest.json"
    if manifest_path.is_file():
        if json.loads(manifest_path.read_text()) != _jsonable(manifest_core):
            raise RuntimeError(f"{spec.name} manifest changed; refusing stale resume")
    else:
        _atomic_json(manifest_path, manifest_core)

    last_path = output_directory / "last.pt"
    best_path = output_directory / "best.pt"
    history: list[dict[str, object]] = []
    start_step = 0
    best_score = -math.inf
    phase_best_score = -math.inf
    phase_stale = 0
    stale_phase: str | None = None
    resume_checkpoint: dict[str, Any] | None = None
    if last_path.is_file():
        resume_checkpoint = _load_checkpoint(last_path)
        required = {
            "schema_version",
            "run_contract_sha256",
            "schedule_step",
            "phase",
            "phase_epoch",
            "model",
            "optimizer",
            "scheduler",
            "scaler",
            "history",
            "best_score",
            "phase_best_score",
            "phase_stale",
            "stale_phase",
        }
        missing = sorted(required - set(resume_checkpoint))
        if missing:
            raise RuntimeError(
                f"{spec.name} resume checkpoint is incomplete: {', '.join(missing)}"
            )
        if resume_checkpoint["schema_version"] != V4_SHAPE_CHECKPOINT_SCHEMA:
            raise RuntimeError(f"{spec.name} checkpoint schema mismatch")
        if resume_checkpoint["run_contract_sha256"] != run_contract_sha256:
            raise RuntimeError(f"{spec.name} checkpoint run contract mismatch")
        model.load_state_dict(resume_checkpoint["model"])
        history = list(resume_checkpoint["history"])
        start_step = int(resume_checkpoint["schedule_step"]) + 1
        if len(history) != start_step:
            raise RuntimeError(
                f"{spec.name} history/checkpoint epoch mismatch: {len(history)} != {start_step}"
            )
        best_score = float(resume_checkpoint["best_score"])
        phase_best_score = float(resume_checkpoint["phase_best_score"])
        phase_stale = int(resume_checkpoint["phase_stale"])
        stale_phase = str(resume_checkpoint["stale_phase"])
        if stale_phase != str(resume_checkpoint["phase"]):
            raise RuntimeError(f"{spec.name} phase-local patience state is inconsistent")
        _restore_rng(resume_checkpoint, device)
        if _should_stop_full_phase(
            str(resume_checkpoint["phase"]),
            int(resume_checkpoint["phase_epoch"]),
            phase_stale,
            config,
        ):
            # Do not silently add epochs when a completed early-stopped run is invoked again.
            start_step = len(schedule)

    active_phase: str | None = None
    optimizer: torch.optim.Optimizer | None = None
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    for schedule_step in range(start_step, len(schedule)):
        phase, phase_epoch, learning_rate = schedule[schedule_step]
        phase_changed = phase != active_phase
        if phase_changed:
            set_training_phase(model, phase)
            optimizer = torch.optim.AdamW(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=learning_rate,
                weight_decay=config.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=max(1, min(8, config.early_stopping_patience // 3)),
                min_lr=1e-7,
            )
            scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
            active_phase = phase
            resuming_same_phase = (
                resume_checkpoint is not None
                and schedule_step == start_step
                and start_step > 0
                and resume_checkpoint["phase"] == phase
            )
            if resuming_same_phase:
                optimizer.load_state_dict(resume_checkpoint["optimizer"])
                scheduler.load_state_dict(resume_checkpoint["scheduler"])
                scaler.load_state_dict(resume_checkpoint["scaler"])
            else:
                # Patience describes convergence of the current optimization phase.  Carrying
                # non-improvements from frozen-head training into end-to-end fine-tuning can stop
                # the full phase almost immediately, so phase state resets at each transition.
                phase_best_score = -math.inf
                phase_stale = 0
                stale_phase = phase
        assert optimizer is not None and scheduler is not None
        train_dataset = datasets["train"]
        train_dataset.set_epoch(schedule_step)
        train_loader = _make_loader(
            train_dataset,
            batch_size,
            config.workers,
            shuffle=True,
            seed=config.seed + schedule_step,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        component_sums: dict[str, float] = defaultdict(float)
        batches = 0
        teacher_keys: set[str] = set()
        first_gradient_audit: dict[str, object] | None = None
        started = time.time()
        for batch_index, batch in enumerate(
            tqdm(train_loader, desc=f"{spec.name} {phase} {phase_epoch + 1}", leave=False)
        ):
            images = batch["image"].to(device, non_blocking=device.type == "cuda")
            targets = _to_device_mapping(batch["targets"], device)
            cellpose = batch["cellpose_teacher"]
            available = batch["teacher_available"]
            if bool(available.any()) and not bool(available.all()):
                raise RuntimeError("A batch mixed present and missing Cellpose teacher caches")
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                outputs = model(images)
                # Retained v3 knowledge enters through the immutable, hash-audited
                # initialization and the staged freeze schedule.  Do not manufacture a
                # "teacher" by detaching this same forward pass: that is self-consistency,
                # not distillation, and contributes no independent v3 information.
                teacher_values: dict[str, Any] = {}
                if bool(available.all()):
                    teacher_values.update(_to_device_mapping(cellpose, device))
                    teacher_values["flow_scale_divisor"] = 5.0
                    teacher_keys.update(str(key) for key in batch["teacher_cache_key"])
                teacher_values["ceb_boundary_probability"] = (
                    _dense_boundary_teacher_probability(
                        boundary_teacher,
                        images,
                        outputs,
                        cellpose,
                        available,
                        device,
                        config.maximum_boundary_candidates,
                    )
                )
                loss_report = multitask_shape_loss(outputs, targets, teacher_values)
                loss = loss_report["total"] / accumulation_steps
            scaler.scale(loss).backward()
            should_step = (
                (batch_index + 1) % accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                if first_gradient_audit is None:
                    first_gradient_audit = gradient_audit(model, phase=phase)
                nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    config.gradient_clip_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            for name, value in loss_report["components"].items():
                component_sums[name] += float(value.detach())
            batches += 1
        checkpoint_loader = _make_loader(
            datasets["checkpoint"],
            min(2, batch_size),
            min(2, config.workers),
            shuffle=False,
            seed=config.seed,
        )
        checkpoint_metrics = _evaluate_student(model, checkpoint_loader, device)
        score = float(checkpoint_metrics["selection_score"])
        scheduler.step(score)
        row: dict[str, object] = {
            "schedule_step": schedule_step,
            "phase": phase,
            "phase_epoch": phase_epoch + 1,
            "global_epoch": schedule_step + 1,
            "train_loss_components": {
                name: value / max(batches, 1)
                for name, value in sorted(component_sums.items())
            },
            "checkpoint_metrics": checkpoint_metrics,
            "selection_score": score,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "duration_seconds": time.time() - started,
            "teacher_cache_keys_used": sorted(teacher_keys),
            "gradient_audit": first_gradient_audit,
        }
        history.append(row)
        if score > best_score + config.minimum_improvement:
            best_score = score
            _atomic_torch_save(
                {
                    "schema_version": V4_SHAPE_CHECKPOINT_SCHEMA,
                    "run_contract_sha256": run_contract_sha256,
                    "spec": asdict(spec),
                    "model": model.state_dict(),
                    "selection_score": score,
                    "schedule_step": schedule_step,
                },
                best_path,
            )
        if score > phase_best_score + config.minimum_improvement:
            phase_best_score = score
            phase_stale = 0
        else:
            phase_stale += 1
        row["global_best_selection_score"] = best_score
        row["phase_best_selection_score"] = phase_best_score
        row["phase_stale_epochs"] = phase_stale
        _atomic_torch_save(
            {
                "schema_version": V4_SHAPE_CHECKPOINT_SCHEMA,
                "run_contract_sha256": run_contract_sha256,
                "schedule_step": schedule_step,
                "phase": phase,
                "phase_epoch": phase_epoch,
                "spec": asdict(spec),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "history": history,
                "best_score": best_score,
                "phase_best_score": phase_best_score,
                "phase_stale": phase_stale,
                "stale_phase": stale_phase,
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_random_state": torch.get_rng_state(),
                "cuda_random_state": (
                    torch.cuda.get_rng_state_all() if device.type == "cuda" else None
                ),
            },
            last_path,
        )
        _atomic_jsonl(output_directory / "history.jsonl", history)
        # Early stopping is permitted only in the final end-to-end phase.  All three knowledge
        # transfer phases are always exercised at least once.
        if _should_stop_full_phase(phase, phase_epoch, phase_stale, config):
            break
        resume_checkpoint = None
    if not best_path.is_file():
        raise RuntimeError(f"{spec.name} produced no best checkpoint")
    best = _load_checkpoint(best_path)
    model.load_state_dict(best["model"])
    model.to(device).eval()

    calibration_loader = _make_loader(
        datasets["calibration"],
        min(2, batch_size),
        min(2, config.workers),
        shuffle=False,
        seed=config.seed,
    )
    calibration_baseline = _evaluate_student(model, calibration_loader, device)
    calibration = _calibrate_thresholds(model, calibration_loader, device)
    calibration["baseline_metrics"] = calibration_baseline
    _atomic_json(output_directory / "calibration.json", calibration)

    selection_loader = _make_loader(
        datasets["ensemble_selection"],
        min(2, batch_size),
        min(2, config.workers),
        shuffle=False,
        seed=config.seed,
    )
    selection = _evaluate_student(
        model,
        selection_loader,
        device,
        thresholds=calibration["selected_thresholds"],
    )
    selection["role"] = "ensemble_selection"
    selection["threshold_source"] = "calibration"
    _atomic_json(output_directory / "ensemble_selection.json", selection)
    boundary_search = _run_frozen_boundary_search(
        model,
        datasets,
        output_directory,
        device,
        config,
        batch_size,
    )
    return model, {
        "spec": asdict(spec),
        "initialization": initialization,
        "best_checkpoint": str(best_path),
        "best_checkpoint_sha256": file_sha256(best_path),
        "best_checkpoint_selection_score": float(best["selection_score"]),
        "best_checkpoint_schedule_step": int(best["schedule_step"]),
        "epochs_completed": len(history),
        "history": str(output_directory / "history.jsonl"),
        "calibration": calibration,
        "ensemble_selection": selection,
        "boundary_search": boundary_search,
    }


def _export_adapter(
    adapter: nn.Module,
    sample: Tensor,
    output_directory: Path,
    stem: str,
    output_name: str,
    expected_channels: int,
    export_onnx: bool,
) -> dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    # Export is the terminal use of each trained model. Moving it to CPU avoids briefly keeping
    # two complete copies of the heavy student on a 24 GB workstation GPU.
    adapter = adapter.cpu().eval()
    sample = sample.cpu()
    with torch.inference_mode():
        eager = adapter(sample)
    expected_shape = (1, expected_channels, sample.shape[2], sample.shape[3])
    if tuple(eager.shape) != expected_shape or not torch.isfinite(eager).all():
        raise RuntimeError(
            f"{stem} eager export contract failed: {tuple(eager.shape)} != {expected_shape}"
        )
    traced = torch.jit.trace(adapter, sample, strict=False)
    traced = torch.jit.freeze(traced.eval())
    torchscript_path = output_directory / f"{stem}.torchscript.pt"
    torch.jit.save(traced, torchscript_path)
    with torch.inference_mode():
        traced_output = traced(sample)
    maximum_error = float((eager - traced_output).abs().max())
    if maximum_error > 1e-4:
        raise RuntimeError(f"{stem} TorchScript parity error is {maximum_error}")
    result: dict[str, object] = {
        "output_name": output_name,
        "output_shape": list(expected_shape),
        "torchscript": str(torchscript_path),
        "torchscript_sha256": file_sha256(torchscript_path),
        "torchscript_maximum_absolute_error": maximum_error,
        "onnx": None,
        "onnx_status": "disabled",
    }
    if export_onnx:
        onnx_path = output_directory / f"{stem}.onnx"
        try:
            torch.onnx.export(
                adapter,
                sample,
                onnx_path,
                input_names=["image"],
                output_names=[output_name],
                dynamic_axes={
                    "image": {0: "batch"},
                    output_name: {0: "batch"},
                },
                opset_version=17,
            )
            try:
                import onnx

                onnx.checker.check_model(onnx.load(str(onnx_path)))
            except ImportError:
                pass
            result.update(
                {
                    "onnx": str(onnx_path),
                    "onnx_sha256": file_sha256(onnx_path),
                    "onnx_status": "exported",
                }
            )
        except Exception as error:  # transformer export can be backend/version dependent
            if onnx_path.exists():
                onnx_path.unlink()
            result["onnx_status"] = "unsupported"
            result["onnx_error"] = f"{type(error).__name__}: {error}"
    return result


def export_shape_model(
    model: V4ShapeNet,
    spec: ShapeModelSpec,
    output_directory: Path,
    *,
    export_onnx: bool,
) -> dict[str, object]:
    sample = torch.zeros(1, 3, spec.image_size, spec.image_size)
    legacy = _export_adapter(
        DeploymentAdapter(model),
        sample,
        output_directory,
        f"{spec.name}_{spec.image_size}_legacy2",
        "segmentation_logits",
        2,
        export_onnx,
    )
    extended = _export_adapter(
        ExtendedDeploymentAdapter(model),
        sample,
        output_directory,
        f"{spec.name}_{spec.image_size}_extended5",
        "extended_segmentation_logits",
        5,
        export_onnx,
    )
    return {
        "legacy": {
            **legacy,
            "semantics": list(LEGACY_DEPLOYMENT_OUTPUT_SEMANTICS),
        },
        "extended": {
            **extended,
            "semantics": list(EXTENDED_DEPLOYMENT_OUTPUT_SEMANTICS),
        },
    }


def _source_contract_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = (
        "v4_shape_training.py",
        "shape_models.py",
        "shape_targets.py",
        "cellpose_teacher.py",
        "boundary_teacher.py",
        "v4_boundary_search.py",
        "deployment_runtime.py",
        "mask_targets.py",
        "scientific_splits.py",
    )
    return {name: file_sha256(root / name) for name in names}


def _resolve_v3_checkpoint(
    spec: ShapeModelSpec,
    checkpoints: Mapping[str, Path] | None,
) -> Path | None:
    if not checkpoints:
        return None
    keys = (spec.name, V3_SOURCE_MODEL_BY_SHAPE.get(spec.name, ""))
    for key in keys:
        if key and key in checkpoints:
            path = Path(checkpoints[key]).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            return path
    return None


def _build_boundary_proposal_model(
    specs: Sequence[ShapeModelSpec],
    checkpoints: Mapping[str, Path] | None,
    device: torch.device,
    *,
    compact_contract: bool,
) -> tuple[V4ShapeNet | None, dict[str, object]]:
    mobile = next((spec for spec in specs if "mobile" in spec.name), specs[0])
    checkpoint = _resolve_v3_checkpoint(mobile, checkpoints)
    if checkpoint is None:
        if compact_contract:
            return None, {
                "kind": "contract_fixture_only",
                "ground_truth_proposals_allowed": True,
                "deployable_or_scientific_result": False,
            }
        raise RuntimeError(
            "The production boundary graph teacher requires a frozen v3 proposal checkpoint"
        )
    model = V4ShapeNet(mobile).to(device)
    initialization = initialize_from_v3(model, checkpoint)
    set_training_phase(model, "frozen")
    state_sha256 = freeze_teacher(model)
    return model, {
        "kind": "frozen_v3_prediction_proposals",
        "model_spec": asdict(mobile),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "initialization": initialization,
        "state_sha256": state_sha256,
        "probability_channels": [
            "base foreground probability",
            "base internal-contact probability",
        ],
        "proposal_reconstruction": asdict(_ProposalReconstructionSettings()),
        "counterfactual_splits": (
            "deterministic cuts of frozen-model proposals only; ground truth labels score the "
            "keep/merge utility after proposal construction"
        ),
    }


def run_v4_shape_training(
    data_root: Path,
    output_root: Path,
    *,
    stage_fingerprint: str,
    split_manifest_sha256: str,
    mode: str = "smoke",
    v3_checkpoints: Mapping[str, Path] | None = None,
    cellpose_teacher_source: CellposeTeacherSource | None = None,
    specs: Sequence[ShapeModelSpec] = V4_SHAPE_MODEL_SPECS,
    config: ShapeTrainingConfig | None = None,
    device: str | torch.device | None = None,
    compact_boundary_teacher: bool = False,
) -> dict[str, object]:
    """Train, calibrate, compare, and export v4 shape students.

    This is the stable pipeline entry point.  It returns the same manifest written to
    ``output_root/v4_shape_training_manifest.json``.
    """

    _require_sha256(stage_fingerprint, "stage_fingerprint")
    _require_sha256(split_manifest_sha256, "split_manifest_sha256")
    resolved_config = config or ShapeTrainingConfig.for_mode(mode)
    if resolved_config.mode != mode:
        raise ValueError("ShapeTrainingConfig.mode does not match mode")
    resolved_device = torch.device(
        device
        if device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if resolved_config.require_cuda and resolved_device.type != "cuda":
        raise RuntimeError("Full v4 shape training requires CUDA")
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    _seed_everything(resolved_config.seed)
    if resolved_device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    records, data_manifest = discover_materialized_pairs(Path(data_root))
    source_hashes = _source_contract_hashes()
    cellpose_teacher_contract: dict[str, object] | None = None
    if cellpose_teacher_source is not None:
        # Cache generation is an operational state that this module intentionally flips to
        # read-only. Excluding that mutable flag keeps the scientific resume contract stable when
        # the same source object is reused after a successful preflight.
        cellpose_teacher_contract = cellpose_teacher_source.manifest()
        cellpose_teacher_contract.pop("generate_missing", None)
        cellpose_teacher_contract["cache_access"] = (
            "serial_preflight_then_read_only_dataloader"
        )
    checkpoint_hashes = {
        key: file_sha256(Path(path).resolve())
        for key, path in sorted((v3_checkpoints or {}).items())
    }
    contract_core = {
        "schema_version": 1,
        "training_version": V4_SHAPE_TRAINING_VERSION,
        "bundle_version": BUNDLE_VERSION,
        "split_protocol": SPLIT_PROTOCOL_VERSION,
        "stage_fingerprint": stage_fingerprint,
        "split_manifest_sha256": split_manifest_sha256,
        "materialized_dataset_fingerprint_sha256": data_manifest[
            "dataset_fingerprint_sha256"
        ],
        "config": asdict(resolved_config),
        "specs": [asdict(spec) for spec in specs],
        "source_sha256": source_hashes,
        "v3_checkpoint_sha256": checkpoint_hashes,
        "cellpose_teacher": (
            cellpose_teacher_contract
        ),
        "shape_target_version": SHAPE_TARGET_VERSION,
        "boundary_teacher_version": BOUNDARY_TEACHER_VERSION,
    }
    run_contract_sha256 = _json_sha256(contract_core)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "run_contract.json"
    contract_payload = {**contract_core, "run_contract_sha256": run_contract_sha256}
    if contract_path.is_file():
        if json.loads(contract_path.read_text()) != _jsonable(contract_payload):
            raise RuntimeError("v4 shape run contract changed; refusing stale output directory")
    else:
        _atomic_json(contract_path, contract_payload)
    _atomic_json(output / "materialized_data_manifest.json", data_manifest)

    cellpose_cache_report: dict[str, object]
    if cellpose_teacher_source is None:
        cellpose_cache_report = {"status": "disabled"}
    else:
        cellpose_cache_report = materialize_cellpose_teacher_caches(
            records,
            cellpose_teacher_source,
        )
        _atomic_json(output / "cellpose_teacher_cache_preflight.json", cellpose_cache_report)

    # One common, train-only boundary teacher supplies the whole-boundary knowledge distilled by
    # both mobile and heavy dense students. Candidate regions and input probabilities come from a
    # separately frozen v3 model; labels are consulted only after proposals exist to score whether
    # keeping or merging each candidate improves matched instance IoU.
    proposal_model, proposal_source = _build_boundary_proposal_model(
        specs,
        v3_checkpoints,
        resolved_device,
        compact_contract=compact_boundary_teacher,
    )
    boundary_size = min(spec.image_size for spec in specs)
    boundary_train = MaterializedShapeDataset(
        records,
        "train",
        boundary_size,
        train=True,
        seed=resolved_config.seed,
        limit_per_domain=resolved_config.role_limit_per_domain,
        teacher_source=cellpose_teacher_source,
    )
    boundary_checkpoint = MaterializedShapeDataset(
        records,
        "checkpoint",
        boundary_size,
        train=False,
        seed=resolved_config.seed,
        limit_per_domain=resolved_config.role_limit_per_domain,
        teacher_source=cellpose_teacher_source,
    )
    boundary_teacher, boundary_report = _train_boundary_teacher(
        boundary_train,
        boundary_checkpoint,
        output / "boundary_teacher",
        run_contract_sha256,
        resolved_config,
        resolved_device,
        proposal_model=proposal_model,
        proposal_source=proposal_source,
        compact_contract=compact_boundary_teacher,
    )
    del proposal_model
    if resolved_device.type == "cuda":
        torch.cuda.empty_cache()

    model_reports: dict[str, object] = {}
    for spec in specs:
        checkpoint = _resolve_v3_checkpoint(spec, v3_checkpoints)
        model, report = _train_student(
            spec,
            records,
            output / spec.name,
            run_contract_sha256,
            resolved_config,
            resolved_device,
            boundary_teacher,
            v3_checkpoint=checkpoint,
            teacher_source=cellpose_teacher_source,
        )
        report["exports"] = export_shape_model(
            model,
            spec,
            output / spec.name / "exports",
            export_onnx=resolved_config.export_onnx,
        )
        model_reports[spec.name] = report
        del model
        if resolved_device.type == "cuda":
            torch.cuda.empty_cache()
    # Select what the app will actually deploy: instance reconstruction after the complete
    # frozen boundary-fusion search, not the earlier semantic-pixel checkpoint proxy.
    selected_model = max(
        model_reports,
        key=lambda name: model_reports[name]["boundary_search"]["selected_candidate"][
            "selection_score"
        ],
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "training_version": V4_SHAPE_TRAINING_VERSION,
        "bundle_version": BUNDLE_VERSION,
        "status": "complete",
        "mode": mode,
        "run_contract_sha256": run_contract_sha256,
        "stage_fingerprint": stage_fingerprint,
        "split_manifest_sha256": split_manifest_sha256,
        "scientific_roles": {
            "fit": "train",
            "early_stopping": "checkpoint",
            "branch_thresholds": "calibration",
            "frozen_model_comparison": "ensemble_selection",
            "final_test": "sealed and never discovered by this module",
        },
        "materialized_data": data_manifest,
        "boundary_teacher": boundary_report,
        "cellpose_teacher": (
            {
                **(cellpose_teacher_contract or {}),
                "runtime_generate_missing_after_preflight": (
                    cellpose_teacher_source.generate_missing
                ),
                "cache_preflight": cellpose_cache_report,
            }
            if cellpose_teacher_source is not None
            else {"status": "disabled"}
        ),
        "models": model_reports,
        "selected_model": selected_model,
        "selected_model_score": model_reports[selected_model]["boundary_search"][
            "selected_candidate"
        ]["selection_score"],
        "selected_model_score_source": (
            "frozen deployed instance reconstruction on ensemble_selection"
        ),
        "legacy_output_semantics": list(LEGACY_DEPLOYMENT_OUTPUT_SEMANTICS),
        "extended_output_semantics": list(EXTENDED_DEPLOYMENT_OUTPUT_SEMANTICS),
        "completed_unix_seconds": time.time(),
    }
    manifest_path = output / "v4_shape_training_manifest.json"
    _atomic_json(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = file_sha256(manifest_path)
    return manifest


def _write_contract_fixture(root: Path) -> None:
    for role_index, role in enumerate(SCIENTIFIC_ROLES):
        directory = root / role
        directory.mkdir(parents=True)
        for sample_index in range(2):
            height, width = 36, 44
            image = np.linspace(0, 255, height * width, dtype=np.uint8).reshape(
                height, width
            )
            image = np.roll(image, role_index * 3 + sample_index, axis=1)
            labels = np.zeros((height, width), dtype=np.uint16)
            # Deliberately vary every role's mask bytes: the scientific-role leakage guard must
            # remain active in this contract fixture, rather than being bypassed for a self-test.
            y_shift = role_index + sample_index
            x_shift = role_index
            labels[5 + y_shift : 27 + y_shift, 3 + x_shift : 18 + x_shift] = 1
            labels[5 + y_shift : 27 + y_shift, 18 + x_shift : 35 + x_shift] = 2
            # Materialized filenames retain their source dataset split token independently of
            # the outer scientific role directory.
            stem = f"fixture_train_scientific-{role}_field{role_index}_{sample_index}"
            if not cv2.imwrite(str(directory / f"{stem}_img.tif"), image):
                raise RuntimeError("Could not write shape-training contract image")
            if not cv2.imwrite(str(directory / f"{stem}_masks.tif"), labels):
                raise RuntimeError("Could not write shape-training contract mask")


def _cellpose_source_contract_test(
    cache_root: Path,
    record: MaterializedPair,
) -> dict[str, object]:
    """Exercise the generation and cache-reuse branch with a local fake Cellpose model."""

    provenance = TeacherProvenance.pinned_foundation("cpsam_v2")
    settings = TeacherInferenceSettings.for_model("cpsam_v2", model_precision="float32")

    class FakeCellposeModel:
        backbone = "sam_vitl"

        def eval(self, image: np.ndarray, **kwargs: object) -> object:
            height, width = image.shape[:2]
            if kwargs.get("compute_masks") is not False or kwargs.get("bsize") != 256:
                raise AssertionError("Cellpose source changed its pinned inference contract")
            flow = np.zeros((2, height, width), dtype=np.float32)
            cellprob = np.zeros((height, width), dtype=np.float32)
            return (
                np.zeros((0,), dtype=np.uint16),
                [
                    np.zeros((height, width, 3), dtype=np.uint8),
                    flow,
                    cellprob,
                    np.zeros((2, height, width), dtype=np.float32),
                ],
                np.zeros(256, dtype=np.float32),
            )

    source = CellposeTeacherSource(
        cache_root=cache_root,
        settings=settings,
        validation_provenance=provenance,
        models_by_checkpoint_sha256={
            provenance.checkpoint_sha256: FakeCellposeModel()
        },
        generate_missing=True,
    )
    cache_report = materialize_cellpose_teacher_caches((record,), source, roles=("train",))
    if source.generate_missing or source.models_by_checkpoint_sha256:
        raise AssertionError("Cellpose cache helper did not lock worker access to read-only")
    image, _ = _read_pair(record)
    reused, reused_manifest = source.load_or_generate(record, image)
    for key in TEACHER_OUTPUT_CONTRACT["cached_arrays"]:
        if reused[key].shape != image.shape or not np.all(reused[key] == 0):
            raise AssertionError(f"Cellpose source cache changed {key}")
    return {
        "base_model": provenance.base_model,
        "cache_key": str(reused_manifest["cache_key"]),
        "preflight": cache_report,
        "generation_and_reuse": "passed",
    }


def run_contract_self_test() -> dict[str, object]:
    """Run a bounded CPU orchestration path without downloading models or datasets."""

    full_config = ShapeTrainingConfig.for_mode("full")
    # Forty preceding non-improvements from heads/decoder must never consume full-phase
    # patience, and even full-phase patience cannot fire before the configured minimum.
    if _should_stop_full_phase("decoder", 24, 999, full_config):
        raise AssertionError("Pre-full epochs incorrectly consume full-phase patience")
    if _should_stop_full_phase(
        "full",
        full_config.minimum_full_epochs - 2,
        full_config.early_stopping_patience,
        full_config,
    ):
        raise AssertionError("Full phase stopped before minimum_full_epochs")
    if not _should_stop_full_phase(
        "full",
        full_config.minimum_full_epochs - 1,
        full_config.early_stopping_patience,
        full_config,
    ):
        raise AssertionError("Full phase ignored phase-local patience after its minimum")

    target_contract_self_test()
    model_report = model_contract_self_test()
    boundary_report = boundary_contract_self_test()
    _assert_candidate_graph_is_truth_independent()
    with tempfile.TemporaryDirectory(prefix="cellect-v4-shape-training-") as temporary:
        root = Path(temporary)
        data = root / "data"
        output = root / "output"
        _write_contract_fixture(data)
        fixture_records, _ = discover_materialized_pairs(data)
        cellpose_source_report = _cellpose_source_contract_test(
            root / "teacher_cache",
            next(record for record in fixture_records if record.role == "train"),
        )
        tiny_spec = ShapeModelSpec(
            name="shape_training_contract",
            tier="contract",
            encoder="contract",
            architecture="contract",
            image_size=32,
            refinement_channels=16,
            refinement_blocks=2,
            encoder_weights=None,
        )
        config = replace(
            ShapeTrainingConfig.for_mode("smoke"),
            role_limit_per_domain=1,
            export_onnx=False,
            require_cuda=False,
        )
        digest_a = hashlib.sha256(b"v4-shape-self-test-stage").hexdigest()
        digest_b = hashlib.sha256(b"v4-shape-self-test-splits").hexdigest()
        manifest = run_v4_shape_training(
            data,
            output,
            stage_fingerprint=digest_a,
            split_manifest_sha256=digest_b,
            mode="smoke",
            specs=(tiny_spec,),
            config=config,
            device="cpu",
            compact_boundary_teacher=True,
        )
        if manifest["status"] != "complete":
            raise AssertionError("v4 shape orchestration did not complete")
        model = manifest["models"][tiny_spec.name]
        if model["epochs_completed"] != 3:
            raise AssertionError("Smoke contract did not exercise all three phases")
        if model["exports"]["legacy"]["output_shape"][1] != 2:
            raise AssertionError("Legacy export channel contract failed")
        if model["exports"]["extended"]["output_shape"][1] != 5:
            raise AssertionError("Extended export channel contract failed")
        search = model["boundary_search"]
        if search["status"] != "complete" or search["mode"] != "smoke":
            raise AssertionError("Frozen five-channel boundary search did not complete")
        if set(search["records"]) != {"calibration", "ensemble_selection"}:
            raise AssertionError("Boundary search used the wrong scientific roles")
        # A second invocation must resume/reuse the exact completed checkpoints without adding
        # duplicate history rows or changing the scientific contract.
        second = run_v4_shape_training(
            data,
            output,
            stage_fingerprint=digest_a,
            split_manifest_sha256=digest_b,
            mode="smoke",
            specs=(tiny_spec,),
            config=config,
            device="cpu",
            compact_boundary_teacher=True,
        )
        if second["models"][tiny_spec.name]["epochs_completed"] != 3:
            raise AssertionError("Completed smoke resume duplicated or lost history")
        return {
            "status": "PASS",
            "training_version": V4_SHAPE_TRAINING_VERSION,
            "model_contract": model_report["status"],
            "boundary_contract": boundary_report["status"],
            "cellpose_source_contract": cellpose_source_report,
            "run_contract_sha256": manifest["run_contract_sha256"],
            "roles": manifest["scientific_roles"],
            "student_epochs": model["epochs_completed"],
            "legacy_channels": model["exports"]["legacy"]["output_shape"][1],
            "extended_channels": model["exports"]["extended"]["output_shape"][1],
            "boundary_search": "passed",
            "resume": "passed",
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--stage-fingerprint")
    parser.add_argument("--split-manifest-sha256")
    parser.add_argument("--mobile-v3-checkpoint", type=Path)
    parser.add_argument("--heavy-v3-checkpoint", type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(run_contract_self_test(), indent=2))
        return
    required = {
        "--data-root": args.data_root,
        "--output-root": args.output_root,
        "--stage-fingerprint": args.stage_fingerprint,
        "--split-manifest-sha256": args.split_manifest_sha256,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        parser.error("Missing required arguments: " + ", ".join(missing))
    checkpoints = {}
    if args.mobile_v3_checkpoint is not None:
        checkpoints["mobilenetv3_small_unet_accuracy_v2"] = args.mobile_v3_checkpoint
    if args.heavy_v3_checkpoint is not None:
        checkpoints["segformer_b5_accuracy_v2"] = args.heavy_v3_checkpoint
    result = run_v4_shape_training(
        args.data_root,
        args.output_root,
        stage_fingerprint=args.stage_fingerprint,
        split_manifest_sha256=args.split_manifest_sha256,
        mode=args.mode,
        v3_checkpoints=checkpoints,
    )
    print(json.dumps(_jsonable(result), indent=2))


if __name__ == "__main__":
    main()
