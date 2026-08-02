#!/usr/bin/env python3
"""Pinned Cellpose teacher inference and leakage-safe, content-addressed caches.

This module deliberately imports neither Cellpose nor PyTorch at module import time.  The cache
contract and ``--self-test`` therefore remain usable during the workstation preflight before the
large foundation-model dependencies are installed.

Cellpose 4.2.1.1 calls its third network output ``cellprob``, but it is a raw logit rather than a
probability.  Its first two network outputs are Y/row and X/column flows scaled by five.  Cellect
stores those three raw outputs losslessly as float32; consumers may derive ``sigmoid(cellprob)``
and ``flow / 5`` without silently changing the teacher target.

The license metadata below is intentionally conservative.  It is provenance documentation, not
legal advice: teacher-derived artifacts remain research-only until the combined Cellpose data,
SAM, and DINOv3 obligations have been reviewed for the intended distribution.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np


CELLPOSE_CONTRACT_VERSION = "4.2.1.1"
# Registered as non-gradient parameters holding the ROI diameter statistics, not learned weights.
# The published cpsam checkpoints omit them and Cellpose falls back to its 30px defaults.
CELLPOSE_DIAMETER_STATISTIC_KEYS = ("diam_labels", "diam_mean")
CELLPOSE_HUGGINGFACE_REPOSITORY = "mouseland/cellpose-sam"
CELLPOSE_WEIGHTS_REVISION = "7c61431b5fbb078f3296754bd15d9f51b320f837"
TEACHER_CACHE_SCHEMA_VERSION = 1
TEACHER_CACHE_IMPLEMENTATION = "cellect-cellpose-teacher-v1"
TEACHER_PAYLOAD_NAME = "teacher_outputs.npz"
TEACHER_MANIFEST_NAME = "manifest.json"
TEACHER_OUTPUT_KEYS = ("flow_y_raw", "flow_x_raw", "cellprob_logit")
TEACHER_OUTPUT_CONTRACT = MappingProxyType(
    {
        "cellpose_version": CELLPOSE_CONTRACT_VERSION,
        "array_dtype": "float32",
        "array_shape": "H x W",
        "flow_axis_order": ["y_row", "x_column"],
        "flow_scale_divisor": 5.0,
        "cellprob_representation": "raw_logit",
        "derived_probability": "sigmoid(cellprob_logit)",
        "cached_arrays": list(TEACHER_OUTPUT_KEYS),
        "excluded_outputs": ["hsv_flow_visualization", "style_vector", "masks"],
    }
)

SCIENTIFIC_ROLES = (
    "train",
    "checkpoint",
    "calibration",
    "ensemble_selection",
    "test",
)
TEACHER_SOURCE_KINDS = (
    "pinned_foundation",
    "oof_finetuned",
    "full_train_finetuned",
)


@dataclass(frozen=True)
class PinnedCellposeModel:
    """Immutable identity for one upstream foundation checkpoint."""

    name: str
    backbone: str
    tile_size: int
    size_bytes: int
    sha256: str

    @property
    def revision(self) -> str:
        return CELLPOSE_WEIGHTS_REVISION

    @property
    def download_url(self) -> str:
        return (
            "https://huggingface.co/"
            f"{CELLPOSE_HUGGINGFACE_REPOSITORY}/resolve/{self.revision}/{self.name}"
        )

    def as_manifest(self) -> dict[str, object]:
        return {
            **asdict(self),
            "revision": self.revision,
            "download_url": self.download_url,
        }


PINNED_CELLPOSE_MODELS: Mapping[str, PinnedCellposeModel] = MappingProxyType(
    {
        "cpsam_v2": PinnedCellposeModel(
            name="cpsam_v2",
            backbone="sam_vitl",
            tile_size=256,
            size_bytes=1_233_586_851,
            sha256="0f1cc3f7ecdd8a037a57c6c48d9d8921391be4cbce3fa9f13c3e3a2e1253c667",
        ),
        "cpdino": PinnedCellposeModel(
            name="cpdino",
            backbone="dino_vitl",
            tile_size=384,
            size_bytes=1_213_987_717,
            sha256="fad0dda262554922da1c5ab0e0004b00c1b14b8a48bea6781b89222567577573",
        ),
        "cpdino-vitb": PinnedCellposeModel(
            name="cpdino-vitb",
            backbone="dino_vitb",
            tile_size=384,
            size_bytes=343_599_997,
            sha256="3ed4c06a3963ab13ff377d4e2957174aaaf637434eb031ae9931fcbfaf9a217f",
        ),
    }
)


_RESEARCH_ONLY_LICENSE_METADATA = {
    "distribution_classification": "research_only_pending_legal_review",
    "notice": (
        "Do not ship teacher-derived weights in an App Store or commercial distribution until "
        "the combined upstream code, model-weight, and training-data obligations are reviewed."
    ),
    "cellpose_code": {
        "license": "BSD-3-Clause",
        "source": "https://github.com/MouseLand/cellpose/blob/v4.2.1.1/LICENSE",
        "required_actions": [
            "retain copyright and license notice",
            "do not use HHMI or contributor names for endorsement",
        ],
    },
    "cellpose_weights": {
        "declared_license": "BSD-3-Clause",
        "source": f"https://huggingface.co/{CELLPOSE_HUGGINGFACE_REPOSITORY}",
        "provenance_warning": (
            "Cellpose states that all Cellpose models were trained on CC-BY-NC data."
        ),
    },
    "cpsam_dependency": {
        "name": "Segment Anything",
        "license": "Apache-2.0",
        "source": "https://github.com/facebookresearch/segment-anything/blob/main/LICENSE",
    },
    "cpdino_dependency": {
        "name": "DINOv3",
        "license": "DINOv3 License",
        "source": (
            "https://github.com/facebookresearch/dinov3/blob/"
            "6876159a11b4df116f30f667f8c9888617df0751/LICENSE.md"
        ),
        "required_actions": [
            "redistribute DINO Materials and derivatives under the DINOv3 agreement",
            "provide the DINOv3 agreement with redistributed DINO Materials",
            "acknowledge DINOv3 in research publications that use DINO Materials",
        ],
    },
    "ceb_mechanism": {
        "license": "MIT",
        "source": "https://github.com/pxliang/Ceb/blob/main/LICENSE",
        "checkpoint_status": (
            "The official repository contains no pretrained boundary-classifier checkpoint; "
            "Cellect trains its own weights and does not claim Ceb weight transfer."
        ),
    },
}


def research_only_license_metadata() -> dict[str, object]:
    """Return an isolated, JSON-safe copy so callers cannot mutate module policy."""
    return json.loads(json.dumps(_RESEARCH_ONLY_LICENSE_METADATA))


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def file_sha256(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str | None, label: str, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{label} is not hexadecimal") from error


def _require_nonempty(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")


def pinned_model(name: str) -> PinnedCellposeModel:
    try:
        return PINNED_CELLPOSE_MODELS[name]
    except KeyError as error:
        choices = ", ".join(PINNED_CELLPOSE_MODELS)
        raise ValueError(f"Unknown pinned Cellpose model {name!r}; choose {choices}") from error


def verify_pinned_model_file(path: Path, model_name: str) -> dict[str, object]:
    """Verify both size and SHA before Cellpose deserializes a downloaded foundation model."""
    spec = pinned_model(model_name)
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    size = resolved.stat().st_size
    if size != spec.size_bytes:
        raise RuntimeError(
            f"Pinned {model_name} size mismatch: expected {spec.size_bytes}, found {size}"
        )
    digest = file_sha256(resolved)
    if digest != spec.sha256:
        raise RuntimeError(
            f"Pinned {model_name} SHA-256 mismatch: expected {spec.sha256}, found {digest}"
        )
    return {
        **spec.as_manifest(),
        "resolved_path": str(resolved),
        "verified": True,
    }


def ensure_pinned_model_file(directory: Path, model_name: str) -> tuple[Path, dict[str, object]]:
    """Download one revision-qualified checkpoint with resume, then verify it byte-for-byte."""
    spec = pinned_model(model_name)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / spec.name
    if destination.is_file():
        return destination, verify_pinned_model_file(destination, model_name)

    partial = destination.with_suffix(".partial")
    existing = partial.stat().st_size if partial.is_file() else 0
    if existing > spec.size_bytes:
        partial.unlink()
        existing = 0
    request = urllib.request.Request(spec.download_url)
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    print(f"Downloading pinned {model_name} weights ({spec.revision})")
    with urllib.request.urlopen(request) as response:
        resumed = existing > 0 and getattr(response, "status", None) == 206
        mode = "ab" if resumed else "wb"
        if not resumed:
            existing = 0
        with partial.open(mode) as target:
            while True:
                chunk = response.read(4 * 1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
    if partial.stat().st_size != spec.size_bytes:
        raise RuntimeError(
            f"Incomplete pinned {model_name} download: expected {spec.size_bytes} bytes, "
            f"found {partial.stat().st_size}"
        )
    partial.replace(destination)
    try:
        report = verify_pinned_model_file(destination, model_name)
    except Exception:
        destination.replace(partial)
        raise
    return destination, report


def installed_cellpose_version() -> str | None:
    try:
        return package_version("cellpose")
    except PackageNotFoundError:
        return None


def require_cellpose_contract_version() -> str:
    installed = installed_cellpose_version()
    if installed is None:
        raise RuntimeError(
            f"Cellpose {CELLPOSE_CONTRACT_VERSION} is required but is not installed"
        )
    if installed != CELLPOSE_CONTRACT_VERSION:
        raise RuntimeError(
            "Unsupported Cellpose output contract: expected "
            f"{CELLPOSE_CONTRACT_VERSION}, found {installed}"
        )
    return installed


@dataclass(frozen=True)
class TeacherSample:
    """Scientific identity of one image receiving teacher predictions."""

    dataset: str
    sample_id: str
    role: str
    acquisition_group: str
    image_sha256: str
    image_size_bytes: int
    height: int
    width: int
    channels: int
    fold_index: int | None = None

    def __post_init__(self) -> None:
        _require_nonempty(self.dataset, "sample.dataset")
        _require_nonempty(self.sample_id, "sample.sample_id")
        _require_nonempty(self.acquisition_group, "sample.acquisition_group")
        _require_sha256(self.image_sha256, "sample.image_sha256")
        if self.role not in SCIENTIFIC_ROLES:
            raise ValueError(f"Unknown scientific role: {self.role}")
        if self.image_size_bytes < 0:
            raise ValueError("sample.image_size_bytes must be non-negative")
        if self.height <= 0 or self.width <= 0 or self.channels <= 0:
            raise ValueError("sample image dimensions and channels must be positive")
        if self.fold_index is not None and self.fold_index < 0:
            raise ValueError("sample.fold_index must be non-negative")

    def as_manifest(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_manifest(cls, value: Mapping[str, object]) -> "TeacherSample":
        return cls(**dict(value))  # type: ignore[arg-type]


@dataclass(frozen=True)
class TeacherProvenance:
    """Checkpoint lineage needed to prove that a cached training prediction is OOF."""

    base_model: str
    source_kind: str
    checkpoint_sha256: str
    cellpose_version: str
    base_weights_revision: str
    base_weights_sha256: str
    training_dataset_fingerprint_sha256: str | None = None
    training_stage_fingerprint_sha256: str | None = None
    training_roles: tuple[str, ...] = ()
    fold_count: int | None = None
    excluded_fold: int | None = None
    checkpoint_epoch: int | None = None
    deployment_lock_sha256: str | None = None

    def __post_init__(self) -> None:
        spec = pinned_model(self.base_model)
        if self.source_kind not in TEACHER_SOURCE_KINDS:
            raise ValueError(f"Unknown teacher source kind: {self.source_kind}")
        _require_sha256(self.checkpoint_sha256, "teacher.checkpoint_sha256")
        _require_sha256(self.base_weights_sha256, "teacher.base_weights_sha256")
        _require_sha256(
            self.training_dataset_fingerprint_sha256,
            "teacher.training_dataset_fingerprint_sha256",
            optional=True,
        )
        _require_sha256(
            self.training_stage_fingerprint_sha256,
            "teacher.training_stage_fingerprint_sha256",
            optional=True,
        )
        _require_sha256(
            self.deployment_lock_sha256,
            "teacher.deployment_lock_sha256",
            optional=True,
        )
        if self.cellpose_version != CELLPOSE_CONTRACT_VERSION:
            raise ValueError(
                f"Teacher uses Cellpose {self.cellpose_version}, not the pinned "
                f"{CELLPOSE_CONTRACT_VERSION} contract"
            )
        if self.base_weights_revision != CELLPOSE_WEIGHTS_REVISION:
            raise ValueError("Teacher base-weight revision is not pinned")
        if self.base_weights_sha256 != spec.sha256:
            raise ValueError("Teacher base-weight SHA does not match the pinned registry")
        if any(role not in SCIENTIFIC_ROLES for role in self.training_roles):
            raise ValueError("Teacher provenance contains an unknown training role")
        if self.checkpoint_epoch is not None and self.checkpoint_epoch < 0:
            raise ValueError("teacher.checkpoint_epoch must be non-negative")

        if self.source_kind == "pinned_foundation":
            if self.checkpoint_sha256 != spec.sha256:
                raise ValueError("Pinned-foundation checkpoint must equal its registry SHA")
            if any(
                value is not None
                for value in (
                    self.training_dataset_fingerprint_sha256,
                    self.training_stage_fingerprint_sha256,
                    self.fold_count,
                    self.excluded_fold,
                    self.checkpoint_epoch,
                )
            ) or self.training_roles:
                raise ValueError("Pinned-foundation provenance cannot claim Cellect training")
        else:
            if self.training_dataset_fingerprint_sha256 is None:
                raise ValueError("Fine-tuned provenance requires a dataset fingerprint")
            if self.training_stage_fingerprint_sha256 is None:
                raise ValueError("Fine-tuned provenance requires a training-stage fingerprint")
            if tuple(self.training_roles) != ("train",):
                raise ValueError("Fine-tuned teachers may fit only the scientific train role")

        if self.source_kind == "oof_finetuned":
            if self.fold_count is None or self.fold_count < 2:
                raise ValueError("OOF provenance requires at least two folds")
            if self.excluded_fold is None or not 0 <= self.excluded_fold < self.fold_count:
                raise ValueError("OOF provenance requires an in-range excluded fold")
        elif self.fold_count is not None or self.excluded_fold is not None:
            raise ValueError("Only OOF provenance may declare fold metadata")

    @classmethod
    def pinned_foundation(
        cls,
        model_name: str,
        *,
        deployment_lock_sha256: str | None = None,
    ) -> "TeacherProvenance":
        spec = pinned_model(model_name)
        return cls(
            base_model=model_name,
            source_kind="pinned_foundation",
            checkpoint_sha256=spec.sha256,
            cellpose_version=CELLPOSE_CONTRACT_VERSION,
            base_weights_revision=CELLPOSE_WEIGHTS_REVISION,
            base_weights_sha256=spec.sha256,
            deployment_lock_sha256=deployment_lock_sha256,
        )

    @classmethod
    def from_manifest(cls, value: Mapping[str, object]) -> "TeacherProvenance":
        fields = dict(value)
        fields["training_roles"] = tuple(fields.get("training_roles", ()))
        return cls(**fields)  # type: ignore[arg-type]

    def as_manifest(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TeacherInferenceSettings:
    """Every numerical choice that can change a teacher cache."""

    tile_size: int
    tile_overlap: float = 0.25
    tile_batch_size: int = 1
    diameter: float | None = None
    resample: bool = True
    augment: bool = False
    normalize: bool = True
    normalize_percentiles: tuple[float, float] = (1.0, 99.0)
    norm_3d: bool = False
    tile_norm_blocksize: int = 0
    invert: bool = False
    compute_masks: bool = False
    model_precision: str = "bfloat16"

    def __post_init__(self) -> None:
        if self.tile_size <= 0:
            raise ValueError("teacher tile_size must be positive")
        if not 0.0 <= self.tile_overlap < 1.0:
            raise ValueError("teacher tile_overlap must be in [0, 1)")
        if self.tile_batch_size <= 0:
            raise ValueError("teacher tile_batch_size must be positive")
        low, high = self.normalize_percentiles
        if not 0.0 <= low < high <= 100.0:
            raise ValueError("teacher normalization percentiles are invalid")
        if self.tile_norm_blocksize < 0:
            raise ValueError("teacher tile_norm_blocksize must be non-negative")
        if self.diameter is not None and self.diameter <= 0:
            raise ValueError("teacher diameter must be positive or None")
        if not self.resample or self.augment or not self.normalize or self.compute_masks:
            raise ValueError(
                "The v4 teacher contract requires resample=True, augment=False, "
                "normalize=True, and compute_masks=False"
            )
        if self.model_precision not in {"bfloat16", "float32"}:
            raise ValueError("teacher model_precision must be bfloat16 or float32")

    @classmethod
    def for_model(
        cls,
        model_name: str,
        *,
        tile_overlap: float = 0.25,
        tile_batch_size: int = 1,
        model_precision: str = "bfloat16",
    ) -> "TeacherInferenceSettings":
        return cls(
            tile_size=pinned_model(model_name).tile_size,
            tile_overlap=tile_overlap,
            tile_batch_size=tile_batch_size,
            model_precision=model_precision,
        )

    @classmethod
    def from_manifest(cls, value: Mapping[str, object]) -> "TeacherInferenceSettings":
        fields = dict(value)
        fields["normalize_percentiles"] = tuple(
            fields.get("normalize_percentiles", (1.0, 99.0))
        )
        return cls(**fields)  # type: ignore[arg-type]

    def as_manifest(self) -> dict[str, object]:
        return asdict(self)


def validate_provenance_for_sample(
    sample: TeacherSample,
    provenance: TeacherProvenance,
    *,
    allow_sealed_test: bool = False,
) -> None:
    """Reject in-sample teachers and premature access to the sealed final-test role."""
    if sample.role == "test":
        if not allow_sealed_test:
            raise ValueError("Teacher caching for the sealed test role is disabled")
        if provenance.deployment_lock_sha256 is None:
            raise ValueError("Test-role caching requires a deployment-lock SHA-256")

    if sample.role == "train":
        if provenance.source_kind == "full_train_finetuned":
            raise ValueError(
                "A full-train fine-tuned teacher cannot predict its own training sample"
            )
        if provenance.source_kind == "oof_finetuned":
            if sample.fold_index is None:
                raise ValueError("OOF training sample has no fold assignment")
            if sample.fold_index >= int(provenance.fold_count):
                raise ValueError("Training sample fold is outside the teacher fold count")
            if sample.fold_index != provenance.excluded_fold:
                raise ValueError(
                    "OOF teacher did not exclude the training sample's acquisition-group fold"
                )
    else:
        if provenance.source_kind == "oof_finetuned":
            raise ValueError("OOF fold teachers are valid only for train-role cache samples")
        if sample.fold_index is not None:
            raise ValueError("Only train-role samples may carry an OOF fold assignment")


def extract_cellpose_teacher_outputs(
    eval_result: object,
    expected_image_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    """Validate and unpack the Cellpose 4.2.1.1 single-image output contract."""
    if not isinstance(eval_result, tuple) or len(eval_result) < 3:
        raise RuntimeError("Cellpose eval must return at least (masks, flows, styles)")
    flow_bundle = eval_result[1]
    # A list containing one flow bundle is accepted defensively, although the public runner below
    # always invokes Cellpose with one ndarray and therefore receives the unwrapped bundle.
    if (
        isinstance(flow_bundle, (list, tuple))
        and len(flow_bundle) == 1
        and isinstance(flow_bundle[0], (list, tuple))
    ):
        flow_bundle = flow_bundle[0]
    if not isinstance(flow_bundle, (list, tuple)) or len(flow_bundle) < 3:
        raise RuntimeError(
            "Cellpose flows must contain [HSV visualization, Y/X flow, cellprob logit]"
        )

    height, width = expected_image_shape
    d_p = np.asarray(flow_bundle[1], dtype=np.float32)
    cellprob = np.asarray(flow_bundle[2], dtype=np.float32)
    if d_p.shape == (1, 2, height, width):
        d_p = d_p[0]
    if cellprob.shape == (1, height, width):
        cellprob = cellprob[0]
    if d_p.shape != (2, height, width):
        raise RuntimeError(
            f"Cellpose Y/X flow shape must be (2, {height}, {width}), found {d_p.shape}"
        )
    if cellprob.shape != (height, width):
        raise RuntimeError(
            f"Cellpose cellprob shape must be ({height}, {width}), found {cellprob.shape}"
        )
    if not np.isfinite(d_p).all() or not np.isfinite(cellprob).all():
        raise RuntimeError("Cellpose teacher produced NaN or infinite values")

    return {
        "flow_y_raw": np.ascontiguousarray(d_p[0], dtype=np.float32),
        "flow_x_raw": np.ascontiguousarray(d_p[1], dtype=np.float32),
        "cellprob_logit": np.ascontiguousarray(cellprob, dtype=np.float32),
    }


def run_cellpose_teacher(
    model: object,
    image: np.ndarray,
    model_name: str,
    settings: TeacherInferenceSettings | None = None,
) -> dict[str, np.ndarray]:
    """Run a strictly shaped, masks-disabled teacher pass on one image."""
    if image.ndim not in (2, 3):
        raise ValueError(f"Teacher image must be 2-D or channels-last 3-D, found {image.shape}")
    height, width = image.shape[:2]
    spec = pinned_model(model_name)
    settings = settings or TeacherInferenceSettings.for_model(model_name)
    if settings.tile_size != spec.tile_size:
        raise ValueError(
            f"{model_name} teacher tile size must be {spec.tile_size}, found "
            f"{settings.tile_size}"
        )
    backbone = getattr(model, "backbone", None)
    if backbone != spec.backbone:
        raise RuntimeError(
            f"{model_name} resolved to backbone {backbone!r}, expected {spec.backbone!r}"
        )
    evaluate = getattr(model, "eval", None)
    if not callable(evaluate):
        raise TypeError("Cellpose teacher object has no callable eval method")

    result = evaluate(
        image,
        diameter=settings.diameter,
        resample=settings.resample,
        normalize={
            "normalize": settings.normalize,
            "percentile": settings.normalize_percentiles,
            "norm3D": settings.norm_3d,
            "tile_norm_blocksize": settings.tile_norm_blocksize,
            "invert": settings.invert,
        },
        augment=settings.augment,
        bsize=settings.tile_size,
        tile_overlap=settings.tile_overlap,
        batch_size=settings.tile_batch_size,
        compute_masks=settings.compute_masks,
    )
    return extract_cellpose_teacher_outputs(result, (height, width))


def _load_plain_cellpose_state_dict(checkpoint_path: Path) -> Mapping[str, Any]:
    """Load only tensor weights; never fall back to unsafe pickle deserialization."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required for checkpoint validation") from error

    resolved = checkpoint_path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    try:
        state = torch.load(
            resolved,
            map_location=torch.device("cpu"),
            weights_only=True,
            mmap=True,
        )
    except TypeError:
        # ``weights_only`` is mandatory.  PyTorch releases that do not support it are rejected
        # instead of silently deserializing an untrusted pickle.
        raise RuntimeError("This PyTorch release does not support safe weights_only loading")
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("Cellpose checkpoint is not a non-empty plain state_dict")
    if not all(isinstance(key, str) for key in state):
        raise RuntimeError("Cellpose state_dict contains a non-string key")
    if all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    if "W2" not in state:
        raise RuntimeError("Checkpoint is not a Cellpose 4 model: W2 is absent")
    if not all(torch.is_tensor(value) for value in state.values()):
        raise RuntimeError("Cellpose state_dict contains a non-tensor value")
    return state


def infer_cellpose_backbone_from_state_dict(state: Mapping[str, Any]) -> str:
    """Mirror Cellpose 4.2.1.1's architecture inference with explicit failure modes."""
    if "encoder.cls_token" not in state:
        return "sam_vitl"
    patch_weight = state.get("encoder.patch_embed.proj.weight")
    if patch_weight is None or not hasattr(patch_weight, "shape"):
        raise RuntimeError("DINO checkpoint has no patch-embedding weight")
    feature_dimension = int(patch_weight.shape[0])
    if feature_dimension == 1024:
        return "dino_vitl"
    if feature_dimension == 768:
        return "dino_vitb"
    raise RuntimeError(f"Unknown Cellpose DINO feature dimension: {feature_dimension}")


def strict_checkpoint_report(
    network: object,
    checkpoint_path: Path,
    *,
    expected_sha256: str | None = None,
    expected_model_name: str | None = None,
) -> tuple[dict[str, object], Mapping[str, Any]]:
    """Compare every checkpoint key and tensor shape before any strict load."""
    if expected_sha256 is not None:
        _require_sha256(expected_sha256, "expected checkpoint SHA-256")
    resolved = checkpoint_path.resolve()
    actual_sha256 = file_sha256(resolved)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Checkpoint SHA-256 mismatch: expected {expected_sha256}, found {actual_sha256}"
        )
    state = _load_plain_cellpose_state_dict(resolved)
    state_dict_method = getattr(network, "state_dict", None)
    if not callable(state_dict_method):
        raise TypeError("Cellpose network has no state_dict method")
    expected_state = state_dict_method()
    expected_keys = set(expected_state)
    actual_keys = set(state)
    # ``diam_mean`` and ``diam_labels`` are ROI-diameter statistics that Cellpose registers as
    # non-gradient parameters rather than learned weights, and the published cpsam checkpoints do
    # not carry them.  Cellpose's own loader tolerates that and leaves its 30px defaults in place,
    # so their absence is not a weight mismatch.  Every other key, and every shape, stays strict.
    absent_statistics = sorted(
        (expected_keys - actual_keys) & set(CELLPOSE_DIAMETER_STATISTIC_KEYS)
    )
    missing = sorted(
        expected_keys - actual_keys - set(CELLPOSE_DIAMETER_STATISTIC_KEYS)
    )
    unexpected = sorted(actual_keys - expected_keys)
    shape_mismatches = sorted(
        {
            key: {
                "expected": list(expected_state[key].shape),
                "actual": list(state[key].shape),
            }
            for key in expected_keys & actual_keys
            if tuple(expected_state[key].shape) != tuple(state[key].shape)
        }.items()
    )
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            "Strict Cellpose checkpoint mismatch: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}, "
            f"shape_mismatches={shape_mismatches[:8]}"
        )
    inferred_backbone = infer_cellpose_backbone_from_state_dict(state)
    if expected_model_name is not None:
        expected_backbone = pinned_model(expected_model_name).backbone
        if inferred_backbone != expected_backbone:
            raise RuntimeError(
                f"Checkpoint backbone {inferred_backbone} does not match "
                f"{expected_model_name} ({expected_backbone})"
            )
    report = {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": actual_sha256,
        "tensor_count": len(state),
        "backbone": inferred_backbone,
        "strict_key_and_shape_match": True,
        "diameter_statistics_defaulted": absent_statistics,
        "diameter_statistics_values": {
            key: [float(value) for value in expected_state[key].flatten().tolist()]
            for key in absent_statistics
        },
    }
    if absent_statistics:
        # Carry the network's own defaults into the state so the load below stays strict=True for
        # every learned tensor instead of being downgraded to a permissive load.
        state = {
            **state,
            **{key: expected_state[key] for key in absent_statistics},
        }
    return report, state


def strict_load_cellpose_checkpoint(
    network: object,
    checkpoint_path: Path,
    *,
    expected_sha256: str | None = None,
    expected_model_name: str | None = None,
) -> dict[str, object]:
    """Validate, then invoke PyTorch's strict loader on an instantiated Cellpose network."""
    report, state = strict_checkpoint_report(
        network,
        checkpoint_path,
        expected_sha256=expected_sha256,
        expected_model_name=expected_model_name,
    )
    load_state_dict = getattr(network, "load_state_dict", None)
    if not callable(load_state_dict):
        raise TypeError("Cellpose network has no load_state_dict method")
    load_state_dict(state, strict=True)
    return {**report, "strict_load_complete": True}


def instantiate_strict_cellpose_model(
    checkpoint_path: Path,
    model_name: str,
    *,
    device: object,
    expected_sha256: str,
    use_bfloat16: bool = True,
) -> tuple[object, dict[str, object]]:
    """Instantiate the pinned API, then replace Cellpose's permissive load with a strict load."""
    require_cellpose_contract_version()
    try:
        from cellpose import models
    except ImportError as error:
        raise RuntimeError("Cellpose is unavailable after its version check passed") from error
    model = models.CellposeModel(
        device=device,
        pretrained_model=str(checkpoint_path.resolve()),
        use_bfloat16=use_bfloat16,
    )
    if getattr(model, "backbone", None) != pinned_model(model_name).backbone:
        raise RuntimeError("Instantiated Cellpose model has the wrong backbone")
    report = strict_load_cellpose_checkpoint(
        model.net,
        checkpoint_path,
        expected_sha256=expected_sha256,
        expected_model_name=model_name,
    )
    return model, report


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    header = {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
    }
    digest = hashlib.sha256(_canonical_json_bytes(header))
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _normalized_outputs(
    outputs: Mapping[str, np.ndarray],
    expected_shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    if set(outputs) != set(TEACHER_OUTPUT_KEYS):
        raise ValueError(
            f"Teacher cache requires exactly {TEACHER_OUTPUT_KEYS}, found {sorted(outputs)}"
        )
    normalized: dict[str, np.ndarray] = {}
    for key in TEACHER_OUTPUT_KEYS:
        array = np.ascontiguousarray(outputs[key], dtype=np.float32)
        if array.shape != expected_shape:
            raise ValueError(
                f"Teacher output {key} has shape {array.shape}, expected {expected_shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"Teacher output {key} contains NaN or infinite values")
        normalized[key] = array
    return normalized


def _output_descriptors(outputs: Mapping[str, np.ndarray]) -> dict[str, object]:
    return {
        key: {
            "dtype": str(outputs[key].dtype),
            "shape": list(outputs[key].shape),
            "array_sha256": _array_sha256(outputs[key]),
        }
        for key in TEACHER_OUTPUT_KEYS
    }


def cache_identity(
    sample: TeacherSample,
    provenance: TeacherProvenance,
    settings: TeacherInferenceSettings,
    *,
    allow_sealed_test: bool = False,
) -> dict[str, object]:
    validate_provenance_for_sample(
        sample,
        provenance,
        allow_sealed_test=allow_sealed_test,
    )
    spec = pinned_model(provenance.base_model)
    if settings.tile_size != spec.tile_size:
        raise ValueError(
            f"Teacher settings tile size {settings.tile_size} does not match "
            f"{provenance.base_model} ({spec.tile_size})"
        )
    identity = {
        "schema_version": TEACHER_CACHE_SCHEMA_VERSION,
        "implementation": TEACHER_CACHE_IMPLEMENTATION,
        "sample": sample.as_manifest(),
        "teacher_provenance": provenance.as_manifest(),
        "inference_settings": settings.as_manifest(),
        "output_contract": dict(TEACHER_OUTPUT_CONTRACT),
        "licensing": research_only_license_metadata(),
    }
    # Normalize tuples and any NumPy scalar-like values through the same canonical JSON form
    # used on disk.  A manifest returned by a first writer is then byte-for-byte equivalent to
    # the manifest returned by a later cache reader/reuser.
    return json.loads(_canonical_json_bytes(identity))


def teacher_cache_key(
    sample: TeacherSample,
    provenance: TeacherProvenance,
    settings: TeacherInferenceSettings,
    *,
    allow_sealed_test: bool = False,
) -> str:
    return _json_fingerprint(
        cache_identity(
            sample,
            provenance,
            settings,
            allow_sealed_test=allow_sealed_test,
        )
    )


def teacher_cache_directory(cache_root: Path, model_name: str, cache_key: str) -> Path:
    _require_sha256(cache_key, "teacher cache key")
    pinned_model(model_name)
    return cache_root.resolve() / model_name / cache_key[:2] / cache_key


def _manifest_identity(manifest: Mapping[str, object]) -> dict[str, object]:
    return {
        key: manifest[key]
        for key in (
            "schema_version",
            "implementation",
            "sample",
            "teacher_provenance",
            "inference_settings",
            "output_contract",
            "licensing",
        )
    }


def validate_cache_manifest(
    manifest: Mapping[str, object],
    *,
    expected_identity: Mapping[str, object] | None = None,
    cache_directory: Path | None = None,
    verify_payload: bool = True,
    allow_sealed_test: bool = False,
) -> dict[str, object]:
    """Validate schema, scientific lineage, content address, and optional on-disk payload."""
    try:
        identity = _manifest_identity(manifest)
        cache_key = str(manifest["cache_key"])
        payload = manifest["payload"]
        outputs = manifest["outputs"]
    except (KeyError, TypeError) as error:
        raise ValueError("Teacher cache manifest is incomplete") from error
    if identity["schema_version"] != TEACHER_CACHE_SCHEMA_VERSION:
        raise ValueError("Teacher cache schema version mismatch")
    if identity["implementation"] != TEACHER_CACHE_IMPLEMENTATION:
        raise ValueError("Teacher cache implementation mismatch")
    if identity["output_contract"] != dict(TEACHER_OUTPUT_CONTRACT):
        raise ValueError("Teacher output contract mismatch")
    if identity["licensing"] != research_only_license_metadata():
        raise ValueError("Teacher licensing metadata is missing or changed")
    if _json_fingerprint(identity) != cache_key:
        raise ValueError("Teacher cache key does not match its canonical identity")
    if expected_identity is not None and _canonical_json_bytes(identity) != _canonical_json_bytes(
        expected_identity
    ):
        raise ValueError("Teacher cache identity does not match the requested sample")

    if not isinstance(identity["sample"], Mapping) or not isinstance(
        identity["teacher_provenance"], Mapping
    ) or not isinstance(identity["inference_settings"], Mapping):
        raise ValueError("Teacher cache identity components must be mappings")
    sample = TeacherSample.from_manifest(identity["sample"])
    provenance = TeacherProvenance.from_manifest(identity["teacher_provenance"])
    settings = TeacherInferenceSettings.from_manifest(identity["inference_settings"])
    validate_provenance_for_sample(
        sample,
        provenance,
        allow_sealed_test=allow_sealed_test,
    )
    if settings.tile_size != pinned_model(provenance.base_model).tile_size:
        raise ValueError("Teacher cache tile size does not match its base model")

    if not isinstance(outputs, Mapping) or set(outputs) != set(TEACHER_OUTPUT_KEYS):
        raise ValueError("Teacher output descriptors are incomplete")
    expected_shape = [sample.height, sample.width]
    for key in TEACHER_OUTPUT_KEYS:
        descriptor = outputs[key]
        if not isinstance(descriptor, Mapping):
            raise ValueError(f"Teacher output descriptor {key} is invalid")
        if descriptor.get("dtype") != "float32":
            raise ValueError(f"Teacher output {key} is not float32")
        if descriptor.get("shape") != expected_shape:
            raise ValueError(f"Teacher output {key} shape does not match its sample")
        _require_sha256(str(descriptor.get("array_sha256")), f"{key} array SHA-256")

    if not isinstance(payload, Mapping):
        raise ValueError("Teacher payload descriptor is invalid")
    if payload.get("file") != TEACHER_PAYLOAD_NAME:
        raise ValueError("Teacher payload filename is not canonical")
    _require_sha256(str(payload.get("sha256")), "teacher payload SHA-256")
    if not isinstance(payload.get("size_bytes"), int) or int(payload["size_bytes"]) <= 0:
        raise ValueError("Teacher payload size is invalid")

    if cache_directory is not None:
        resolved_directory = cache_directory.resolve()
        if resolved_directory.name != cache_key:
            raise ValueError("Teacher cache directory name does not match its key")
        if verify_payload:
            payload_path = resolved_directory / TEACHER_PAYLOAD_NAME
            if not payload_path.is_file():
                raise FileNotFoundError(payload_path)
            if payload_path.stat().st_size != int(payload["size_bytes"]):
                raise RuntimeError("Teacher payload size changed after cache commit")
            if file_sha256(payload_path) != payload["sha256"]:
                raise RuntimeError("Teacher payload SHA-256 changed after cache commit")
    return {
        "cache_key": cache_key,
        "sample": sample,
        "provenance": provenance,
        "settings": settings,
    }


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        # Some filesystems do not support directory fsync.  Both payload files were already
        # fsynced, and the same validation is repeated whenever the cache is opened.
        pass
    finally:
        os.close(descriptor)


def _read_manifest_file(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not read teacher cache manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise RuntimeError("Teacher cache manifest root is not an object")
    return value


def _validate_existing_cache(
    directory: Path,
    identity: Mapping[str, object],
    output_descriptors: Mapping[str, object],
    *,
    allow_sealed_test: bool,
) -> dict[str, object]:
    manifest = _read_manifest_file(directory / TEACHER_MANIFEST_NAME)
    validate_cache_manifest(
        manifest,
        expected_identity=identity,
        cache_directory=directory,
        verify_payload=True,
        allow_sealed_test=allow_sealed_test,
    )
    if manifest.get("outputs") != output_descriptors:
        raise RuntimeError(
            "A cache with the same scientific identity produced different teacher arrays"
        )
    return manifest


def write_teacher_cache(
    cache_root: Path,
    sample: TeacherSample,
    provenance: TeacherProvenance,
    settings: TeacherInferenceSettings,
    outputs: Mapping[str, np.ndarray],
    *,
    allow_sealed_test: bool = False,
) -> tuple[Path, dict[str, object], bool]:
    """Atomically commit one complete content-addressed cache directory.

    Returns ``(directory, manifest, reused)``.  Existing entries are never overwritten: an
    identity collision with different numerical outputs is treated as a reproducibility error.
    """
    normalized = _normalized_outputs(outputs, (sample.height, sample.width))
    identity = cache_identity(
        sample,
        provenance,
        settings,
        allow_sealed_test=allow_sealed_test,
    )
    cache_key = _json_fingerprint(identity)
    destination = teacher_cache_directory(cache_root, provenance.base_model, cache_key)
    descriptors = _output_descriptors(normalized)
    if destination.is_dir():
        manifest = _validate_existing_cache(
            destination,
            identity,
            descriptors,
            allow_sealed_test=allow_sealed_test,
        )
        return destination, manifest, True
    if destination.exists():
        raise RuntimeError(f"Teacher cache destination is not a directory: {destination}")

    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / (
        f".{cache_key}.tmp-{os.getpid()}-{uuid.uuid4().hex}"
    )
    temporary.mkdir()
    try:
        payload_path = temporary / TEACHER_PAYLOAD_NAME
        with payload_path.open("wb") as payload_file:
            np.savez_compressed(payload_file, **normalized)
            payload_file.flush()
            os.fsync(payload_file.fileno())
        payload_descriptor = {
            "file": TEACHER_PAYLOAD_NAME,
            "size_bytes": payload_path.stat().st_size,
            "sha256": file_sha256(payload_path),
        }
        manifest: dict[str, object] = {
            **identity,
            "cache_key": cache_key,
            "outputs": descriptors,
            "payload": payload_descriptor,
        }
        manifest_path = temporary / TEACHER_MANIFEST_NAME
        with manifest_path.open("wb") as manifest_file:
            manifest_file.write(_canonical_json_bytes(manifest) + b"\n")
            manifest_file.flush()
            os.fsync(manifest_file.fileno())
        _fsync_directory(temporary)
        try:
            os.rename(temporary, destination)
        except OSError as error:
            if error.errno not in {errno.EEXIST, errno.ENOTEMPTY} or not destination.is_dir():
                raise
            # Another worker committed the same key first.  It is reusable only if the complete
            # identity and every numerical output digest agree with this worker.
            existing = _validate_existing_cache(
                destination,
                identity,
                descriptors,
                allow_sealed_test=allow_sealed_test,
            )
            shutil.rmtree(temporary)
            return destination, existing, True
        _fsync_directory(parent)
        validate_cache_manifest(
            manifest,
            expected_identity=identity,
            cache_directory=destination,
            verify_payload=True,
            allow_sealed_test=allow_sealed_test,
        )
        return destination, manifest, False
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def read_teacher_cache(
    directory: Path,
    *,
    expected_sample: TeacherSample | None = None,
    expected_provenance: TeacherProvenance | None = None,
    expected_settings: TeacherInferenceSettings | None = None,
    allow_sealed_test: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Verify all digests and lineage before returning cached arrays."""
    resolved = directory.resolve()
    manifest = _read_manifest_file(resolved / TEACHER_MANIFEST_NAME)
    expected_identity = None
    supplied = (
        expected_sample is not None,
        expected_provenance is not None,
        expected_settings is not None,
    )
    if any(supplied) and not all(supplied):
        raise ValueError(
            "expected_sample, expected_provenance, and expected_settings must be supplied together"
        )
    if all(supplied):
        assert expected_sample is not None
        assert expected_provenance is not None
        assert expected_settings is not None
        expected_identity = cache_identity(
            expected_sample,
            expected_provenance,
            expected_settings,
            allow_sealed_test=allow_sealed_test,
        )
    validate_cache_manifest(
        manifest,
        expected_identity=expected_identity,
        cache_directory=resolved,
        verify_payload=True,
        allow_sealed_test=allow_sealed_test,
    )
    payload_path = resolved / TEACHER_PAYLOAD_NAME
    with np.load(payload_path, allow_pickle=False) as archive:
        arrays = _normalized_outputs(
            {key: archive[key] for key in archive.files},
            (
                int(manifest["sample"]["height"]),  # type: ignore[index]
                int(manifest["sample"]["width"]),  # type: ignore[index]
            ),
        )
    descriptors = _output_descriptors(arrays)
    if manifest.get("outputs") != descriptors:
        raise RuntimeError("Teacher array digests do not match the committed manifest")
    return arrays, manifest


def contract_self_test() -> dict[str, object]:
    """Exercise the bounded cache/output contract without importing Cellpose or PyTorch."""
    height, width = 5, 7
    d_p = np.stack(
        (
            np.linspace(-5.0, 5.0, height * width, dtype=np.float32).reshape(
                height, width
            ),
            np.linspace(5.0, -5.0, height * width, dtype=np.float32).reshape(
                height, width
            ),
        )
    )
    cellprob = np.linspace(-3.0, 3.0, height * width, dtype=np.float32).reshape(
        height, width
    )
    fake_eval_result = (
        np.zeros(0, dtype=np.uint16),
        [
            np.zeros((height, width, 3), dtype=np.uint8),
            d_p,
            cellprob,
            np.zeros((2, height, width), dtype=np.float32),
        ],
        np.zeros(256, dtype=np.float32),
    )

    class FakeCellposeModel:
        backbone = "sam_vitl"

        def eval(self, image: np.ndarray, **kwargs: object) -> object:
            if image.shape != (height, width):
                raise AssertionError("Self-test runner changed the input image shape")
            if kwargs.get("compute_masks") is not False or kwargs.get("bsize") != 256:
                raise AssertionError("Self-test runner violated the masks/tile contract")
            return fake_eval_result

    outputs = run_cellpose_teacher(
        FakeCellposeModel(),
        np.zeros((height, width), dtype=np.uint8),
        "cpsam_v2",
    )

    sample = TeacherSample(
        dataset="self_test",
        sample_id="sample-001",
        role="train",
        acquisition_group="self_test/acquisition-001",
        image_sha256=hashlib.sha256(b"bounded-self-test-image").hexdigest(),
        image_size_bytes=height * width,
        height=height,
        width=width,
        channels=1,
        fold_index=2,
    )
    base = pinned_model("cpsam_v2")
    provenance = TeacherProvenance(
        base_model=base.name,
        source_kind="oof_finetuned",
        checkpoint_sha256=hashlib.sha256(b"bounded-oof-checkpoint").hexdigest(),
        cellpose_version=CELLPOSE_CONTRACT_VERSION,
        base_weights_revision=CELLPOSE_WEIGHTS_REVISION,
        base_weights_sha256=base.sha256,
        training_dataset_fingerprint_sha256=hashlib.sha256(
            b"bounded-training-data"
        ).hexdigest(),
        training_stage_fingerprint_sha256=hashlib.sha256(
            b"bounded-training-code"
        ).hexdigest(),
        training_roles=("train",),
        fold_count=5,
        excluded_fold=2,
        checkpoint_epoch=25,
    )
    settings = TeacherInferenceSettings.for_model("cpsam_v2")
    key = teacher_cache_key(sample, provenance, settings)

    leakage_rejected = False
    wrong_fold = TeacherSample(
        **{**sample.as_manifest(), "fold_index": 1}  # type: ignore[arg-type]
    )
    try:
        validate_provenance_for_sample(wrong_fold, provenance)
    except ValueError:
        leakage_rejected = True
    if not leakage_rejected:
        raise AssertionError("OOF fold mismatch was not rejected")

    with tempfile.TemporaryDirectory(prefix="cellect-cellpose-teacher-self-test-") as root:
        directory, first_manifest, first_reused = write_teacher_cache(
            Path(root), sample, provenance, settings, outputs
        )
        loaded, second_manifest = read_teacher_cache(
            directory,
            expected_sample=sample,
            expected_provenance=provenance,
            expected_settings=settings,
        )
        reused_directory, third_manifest, second_reused = write_teacher_cache(
            Path(root), sample, provenance, settings, outputs
        )
        if first_reused or not second_reused:
            raise AssertionError("Teacher cache reuse flags are incorrect")
        if directory != reused_directory:
            raise AssertionError("Content address changed between identical writes")
        if first_manifest != second_manifest or first_manifest != third_manifest:
            raise AssertionError("Teacher manifest changed between identical operations")
        for output_key in TEACHER_OUTPUT_KEYS:
            if not np.array_equal(outputs[output_key], loaded[output_key]):
                raise AssertionError(f"Teacher cache changed {output_key}")

    return {
        "status": "PASS",
        "cellpose_runtime_required": False,
        "pytorch_runtime_required": False,
        "cache_key": key,
        "output_shape": [height, width],
        "oof_leakage_guard": "PASS",
        "atomic_cache_round_trip": "PASS",
        "pinned_models": list(PINNED_CELLPOSE_MODELS),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the bounded contract/cache test without importing Cellpose or PyTorch",
    )
    args = parser.parse_args()
    if not args.self_test:
        parser.error("no action selected; use --self-test")
    print(json.dumps(contract_self_test(), indent=2))


if __name__ == "__main__":
    main()
