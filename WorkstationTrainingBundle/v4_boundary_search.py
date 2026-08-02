#!/usr/bin/env python3
"""Leakage-safe calibration and fusion search for Cellect v4 five-channel outputs.

The v4 deployment contract is channel-first and ordered as::

    foreground, learned fused boundary, context boundary,
    flow boundary, shape boundary

This module compares the learned fusion with each single boundary head, fuzzy any/majority/all
votes, and normalized weighted combinations.  Context, flow, and shape have independent
calibration thresholds.  Every resulting boundary map is passed to the same deterministic
``reconstruct_iphone_instances`` implementation used by the deployment reference.

The scientific split contract is deliberately strict:

* operating points and boundary-cutoff curves are selected on ``calibration`` only;
* those complete configurations are frozen before ``ensemble_selection`` is read;
* ``ensemble_selection`` only ranks the frozen candidates;
* test/final-test records are rejected before any probability or label source is opened.

The normalized boundary-cutoff AUC reported here is an area under *labelled validation metrics*.
It measures robustness of a candidate family during calibration; it is not an accuracy estimate
that can be computed from an unlabelled image at inference time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:  # Support both ``python v4_boundary_search.py`` and package-style imports.
    from .deployment_runtime import POSTPROCESS_VERSION, reconstruct_iphone_instances
    from .mask_targets import BOUNDARY_TARGET_VERSION, internal_contact_boundary
except ImportError:  # pragma: no cover - direct script execution uses this branch
    from deployment_runtime import POSTPROCESS_VERSION, reconstruct_iphone_instances
    from mask_targets import BOUNDARY_TARGET_VERSION, internal_contact_boundary


V4_BOUNDARY_SEARCH_VERSION = "cellect-v4-boundary-search-v2"
V4_OUTPUT_SEMANTICS = (
    "foreground_probability",
    "learned_fused_boundary_probability",
    "context_boundary_probability",
    "flow_boundary_probability",
    "shape_boundary_probability",
)
VALIDATION_ROLES = ("calibration", "ensemble_selection")
BOUNDARY_BRANCH_NAMES = ("context", "flow", "shape")
FUSION_MODES = ("learned", "single", "any", "majority", "all", "weighted")
SELECTION_OBJECTIVE = (
    "0.25 foreground Dice + 0.15 reconstructed contact-boundary Dice + "
    "0.35 instance AP50 + 0.25 instance AP75"
)
AUC_SEMANTICS = (
    "Normalized trapezoidal area under labelled validation metrics as the deployment boundary "
    "cutoff changes. This is calibration robustness, not unlabeled inference accuracy."
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return value


def _object_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_json(value) + b"\n")
    os.replace(temporary, path)


def _atomic_numpy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npy")
    np.save(temporary, np.ascontiguousarray(value), allow_pickle=False)
    os.replace(temporary, path)


def _threshold_tuple(
    values: Sequence[float],
    name: str,
    *,
    require_multiple: bool = False,
) -> tuple[float, ...]:
    converted = tuple(float(value) for value in values)
    if not converted or (require_multiple and len(converted) < 2):
        qualifier = "at least two" if require_multiple else "one or more"
        raise ValueError(f"{name} requires {qualifier} values")
    if any(not math.isfinite(value) or not 0.0 < value < 1.0 for value in converted):
        raise ValueError(f"{name} values must be finite and strictly within (0, 1)")
    if len(set(converted)) != len(converted):
        raise ValueError(f"{name} contains duplicate values")
    return tuple(sorted(converted))


@dataclass(frozen=True)
class BoundarySearchBounds:
    """Finite, serializable search space.

    Full bounds are deliberately broad.  :meth:`smoke` uses fewer deterministic records and a
    reduced, but still structurally complete, search that exercises every fusion family.
    """

    foreground_thresholds: tuple[float, ...] = (0.40, 0.50, 0.60)
    context_thresholds: tuple[float, ...] = (0.35, 0.50, 0.65)
    flow_thresholds: tuple[float, ...] = (0.35, 0.50, 0.65)
    shape_thresholds: tuple[float, ...] = (0.35, 0.50, 0.65)
    boundary_cutoffs: tuple[float, ...] = (
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    )
    minimum_area_fractions: tuple[float, ...] = (
        0.0,
        20.0 / (512.0 * 512.0),
    )
    weighted_combinations: tuple[tuple[float, float, float], ...] = (
        (1.0, 1.0, 1.0),
        (2.0, 1.0, 1.0),
        (1.0, 2.0, 1.0),
        (1.0, 1.0, 2.0),
        (3.0, 1.0, 1.0),
        (1.0, 3.0, 1.0),
        (1.0, 1.0, 3.0),
    )
    # Validation is deliberately sampled within every role/domain.  Exhaustively running a
    # watershed over every validation frame and every hyperparameter is both statistically
    # redundant and capable of eclipsing the actual model training by days.  The deterministic
    # SHA-256 sample and domain-macro aggregation keep the search reproducible and balanced.
    maximum_records_per_role_domain: int | None = 24
    branch_threshold_shortlist_per_head: int = 2

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "foreground_thresholds",
            _threshold_tuple(self.foreground_thresholds, "foreground_thresholds"),
        )
        for name in (
            "context_thresholds",
            "flow_thresholds",
            "shape_thresholds",
        ):
            object.__setattr__(self, name, _threshold_tuple(getattr(self, name), name))
        cutoffs = _threshold_tuple(
            self.boundary_cutoffs,
            "boundary_cutoffs",
            require_multiple=True,
        )
        object.__setattr__(self, "boundary_cutoffs", cutoffs)
        areas = tuple(float(value) for value in self.minimum_area_fractions)
        if not areas or any(
            not math.isfinite(value) or not 0.0 <= value < 1.0 for value in areas
        ):
            raise ValueError("minimum_area_fractions must contain finite values in [0, 1)")
        object.__setattr__(self, "minimum_area_fractions", tuple(sorted(set(areas))))
        normalized_weights: list[tuple[float, float, float]] = []
        observed: set[tuple[float, float, float]] = set()
        for raw_weights in self.weighted_combinations:
            if len(raw_weights) != 3:
                raise ValueError("Every weighted combination needs context/flow/shape weights")
            weights = tuple(float(value) for value in raw_weights)
            if any(not math.isfinite(value) or value < 0 for value in weights):
                raise ValueError("Fusion weights must be finite and non-negative")
            total = sum(weights)
            if total <= 0:
                raise ValueError("At least one fusion weight must be positive")
            normalized = tuple(round(value / total, 12) for value in weights)
            if normalized not in observed:
                observed.add(normalized)
                normalized_weights.append(normalized)
        if not normalized_weights:
            raise ValueError("weighted_combinations must not be empty")
        object.__setattr__(self, "weighted_combinations", tuple(normalized_weights))
        limit = self.maximum_records_per_role_domain
        if limit is not None and limit < 1:
            raise ValueError("maximum_records_per_role_domain must be positive")
        if self.branch_threshold_shortlist_per_head < 1:
            raise ValueError("branch_threshold_shortlist_per_head must be positive")

    @classmethod
    def smoke(cls) -> "BoundarySearchBounds":
        return cls(
            foreground_thresholds=(0.45, 0.55),
            context_thresholds=(0.40, 0.60),
            flow_thresholds=(0.40, 0.60),
            shape_thresholds=(0.40, 0.60),
            boundary_cutoffs=(0.25, 0.50, 0.75),
            minimum_area_fractions=(0.0,),
            weighted_combinations=((1.0, 1.0, 1.0), (2.0, 1.0, 1.0)),
            maximum_records_per_role_domain=2,
            branch_threshold_shortlist_per_head=2,
        )


@dataclass(frozen=True)
class BoundaryEvaluationRecord:
    """One probability/label descriptor; paths keep evaluation memory bounded."""

    record_id: str
    role: str
    domain: str
    probability_source: Path | np.ndarray
    truth_source: Path | np.ndarray
    group_id: str | None = None
    value_kind: str = "probability"

    def __post_init__(self) -> None:
        if not self.record_id.strip():
            raise ValueError("record_id must not be empty")
        if not self.domain.strip():
            raise ValueError("domain must not be empty")
        normalized_role = self.role.strip().casefold().replace("-", "_")
        object.__setattr__(self, "role", normalized_role)
        if self.group_id is None:
            object.__setattr__(self, "group_id", self.record_id)
        if self.value_kind not in {"probability", "logit"}:
            raise ValueError("value_kind must be 'probability' or 'logit'")
        if isinstance(self.probability_source, (str, os.PathLike)):
            object.__setattr__(self, "probability_source", Path(self.probability_source))
        if isinstance(self.truth_source, (str, os.PathLike)):
            object.__setattr__(self, "truth_source", Path(self.truth_source))


@dataclass(frozen=True)
class BoundaryFusionSpec:
    """A boundary-map recipe independent of foreground and reconstruction cutoffs."""

    mode: str
    context_threshold: float = 0.50
    flow_threshold: float = 0.50
    shape_threshold: float = 0.50
    head: str | None = None
    weights: tuple[float, float, float] = (1.0 / 3.0,) * 3

    def __post_init__(self) -> None:
        if self.mode not in FUSION_MODES:
            raise ValueError(f"Unknown fusion mode: {self.mode}")
        for name in (
            "context_threshold",
            "flow_threshold",
            "shape_threshold",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be strictly within (0, 1)")
        if self.mode == "single" and self.head not in BOUNDARY_BRANCH_NAMES:
            raise ValueError("single fusion requires head=context, flow, or shape")
        if self.mode != "single" and self.head is not None:
            raise ValueError("Only single-head fusion accepts head")
        if len(self.weights) != 3:
            raise ValueError("weights must contain context, flow, and shape")
        weights = tuple(float(value) for value in self.weights)
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("weights must be finite and non-negative")
        total = sum(weights)
        if total <= 0:
            raise ValueError("At least one weight must be positive")
        object.__setattr__(self, "weights", tuple(value / total for value in weights))

    @property
    def family(self) -> str:
        if self.mode == "single":
            return f"single_{self.head}"
        if self.mode in {"any", "majority", "all"}:
            return f"vote_{self.mode}"
        if self.mode == "learned":
            return "learned_fusion"
        return "weighted_branches"

    @property
    def identifier(self) -> str:
        return f"fusion-{_object_sha256(asdict(self))[:16]}"


@dataclass(frozen=True)
class BoundaryOperatingPoint:
    """Complete deployable reconstruction configuration."""

    fusion: BoundaryFusionSpec
    foreground_threshold: float
    boundary_cutoff: float
    min_area_fraction: float

    def __post_init__(self) -> None:
        for name in ("foreground_threshold", "boundary_cutoff"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be strictly within (0, 1)")
        if not math.isfinite(self.min_area_fraction) or not (
            0.0 <= self.min_area_fraction < 1.0
        ):
            raise ValueError("min_area_fraction must be within [0, 1)")

    @property
    def identifier(self) -> str:
        return f"candidate-{_object_sha256(self.as_dict())[:20]}"

    @property
    def curve_identifier(self) -> str:
        value = self.as_dict()
        value.pop("boundary_cutoff")
        return f"curve-{_object_sha256(value)[:20]}"

    def as_dict(self) -> dict[str, object]:
        return {
            "fusion": asdict(self.fusion),
            "fusion_family": self.fusion.family,
            "foreground_threshold": float(self.foreground_threshold),
            "boundary_cutoff": float(self.boundary_cutoff),
            "min_area_fraction": float(self.min_area_fraction),
        }


@dataclass(frozen=True)
class _ReconstructionSettings:
    foreground_threshold: float
    boundary_threshold: float
    min_area_fraction: float


class FiveChannelDiskCache:
    """Content-verified, pickle-free probability cache for streaming evaluation."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(record_id: str) -> str:
        return hashlib.sha256(record_id.encode("utf-8")).hexdigest()

    def paths(self, record_id: str) -> tuple[Path, Path]:
        key = self._key(record_id)
        directory = self.root / key[:2]
        return directory / f"{key}.npy", directory / f"{key}.json"

    def write(
        self,
        record_id: str,
        five_channel_output: Any,
        *,
        from_logits: bool,
        overwrite: bool = False,
    ) -> Path:
        probabilities = normalize_five_channel_output(
            five_channel_output,
            from_logits=from_logits,
        )
        array_path, marker_path = self.paths(record_id)
        if array_path.exists() or marker_path.exists():
            if not overwrite and self.verify(record_id):
                return array_path
            if not overwrite:
                raise FileExistsError(
                    f"Refusing to replace incomplete or invalid cache entry for {record_id!r}"
                )
        _atomic_numpy(array_path, probabilities)
        marker = {
            "schema_version": 1,
            "search_version": V4_BOUNDARY_SEARCH_VERSION,
            "record_id": record_id,
            "semantics": list(V4_OUTPUT_SEMANTICS),
            "shape": list(probabilities.shape),
            "dtype": str(probabilities.dtype),
            "source_value_kind": "logit" if from_logits else "probability",
            "array_sha256": _file_sha256(array_path),
        }
        _atomic_json(marker_path, marker)
        return array_path

    def verify(self, record_id: str) -> bool:
        array_path, marker_path = self.paths(record_id)
        if not array_path.is_file() or not marker_path.is_file():
            return False
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            array = np.load(array_path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError, json.JSONDecodeError):
            return False
        expected = {
            "schema_version": 1,
            "search_version": V4_BOUNDARY_SEARCH_VERSION,
            "record_id": record_id,
            "semantics": list(V4_OUTPUT_SEMANTICS),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "array_sha256": _file_sha256(array_path),
        }
        return all(marker.get(key) == value for key, value in expected.items()) and (
            array.ndim == 3
            and array.shape[0] == len(V4_OUTPUT_SEMANTICS)
            and array.dtype == np.float32
            and bool(np.isfinite(array).all())
            and float(array.min()) >= 0.0
            and float(array.max()) <= 1.0
        )

    def load(self, record_id: str, *, mmap: bool = True) -> np.ndarray:
        if not self.verify(record_id):
            raise RuntimeError(f"Invalid five-channel cache entry for {record_id!r}")
        array_path, _ = self.paths(record_id)
        return np.load(
            array_path,
            mmap_mode="r" if mmap else None,
            allow_pickle=False,
        )

    def record(
        self,
        record_id: str,
        *,
        role: str,
        domain: str,
        truth_source: Path | np.ndarray,
        group_id: str | None = None,
    ) -> BoundaryEvaluationRecord:
        array_path, _ = self.paths(record_id)
        if not self.verify(record_id):
            raise RuntimeError(f"Cache entry is not verified: {record_id!r}")
        return BoundaryEvaluationRecord(
            record_id=record_id,
            role=role,
            domain=domain,
            probability_source=array_path,
            truth_source=truth_source,
            group_id=group_id,
            value_kind="probability",
        )


def normalize_five_channel_output(
    output: Any,
    *,
    from_logits: bool,
) -> np.ndarray:
    """Return finite ``float32 [5,H,W]`` probabilities with explicit logit handling."""

    if hasattr(output, "detach"):
        output = output.detach().cpu().numpy()
    array = np.asarray(output)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError("Cache one sample at a time; four-dimensional output needs batch=1")
        array = array[0]
    if array.ndim == 3 and array.shape[0] != 5 and array.shape[-1] == 5:
        array = np.moveaxis(array, -1, 0)
    if array.ndim != 3 or array.shape[0] != 5:
        raise ValueError(
            f"Five-channel output must have shape [5,H,W], [1,5,H,W], or [H,W,5]; "
            f"received {array.shape}"
        )
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Five-channel output contains non-finite values")
    if from_logits:
        array = 1.0 / (1.0 + np.exp(-np.clip(array, -80.0, 80.0)))
    elif float(array.min()) < 0.0 or float(array.max()) > 1.0:
        raise ValueError("Probability input must be within [0, 1]; set from_logits=True for logits")
    return np.ascontiguousarray(array, dtype=np.float32)


def _load_probability_source(record: BoundaryEvaluationRecord) -> np.ndarray:
    source = record.probability_source
    if isinstance(source, np.ndarray):
        value: Any = source
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Missing probability source for {record.record_id}: {path}")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
    return normalize_five_channel_output(value, from_logits=record.value_kind == "logit")


def _load_truth_source(record: BoundaryEvaluationRecord) -> np.ndarray:
    source = record.truth_source
    if isinstance(source, np.ndarray):
        labels = source
    else:
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Missing truth source for {record.record_id}: {path}")
        if path.suffix.casefold() == ".npy":
            labels = np.load(path, mmap_mode="r", allow_pickle=False)
        else:
            labels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if labels is None:
                raise RuntimeError(f"Could not decode truth labels: {path}")
    labels = np.asarray(labels)
    if labels.ndim == 3 and labels.shape[2] == 1:
        labels = labels[..., 0]
    if labels.ndim != 2:
        raise ValueError(f"Truth labels must be a two-dimensional instance map: {labels.shape}")
    if not np.isfinite(labels).all() or np.any(labels < 0):
        raise ValueError("Truth labels must be finite and non-negative")
    rounded = np.rint(labels)
    if not np.array_equal(labels, rounded):
        raise ValueError("Truth labels must contain integer instance identifiers")
    return np.ascontiguousarray(rounded, dtype=np.int32)


def _threshold_calibrated_probability(
    probability: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Monotonically map a branch threshold to 0.5 while preserving confidence margins."""

    lower = 0.5 * probability / threshold
    upper = 0.5 + 0.5 * (probability - threshold) / (1.0 - threshold)
    return np.where(probability < threshold, lower, upper).astype(np.float32)


def fuse_boundary_probability(
    probabilities: np.ndarray,
    fusion: BoundaryFusionSpec,
) -> np.ndarray:
    """Build one continuous boundary map; 0.5 retains exact vote/threshold semantics."""

    values = normalize_five_channel_output(probabilities, from_logits=False)
    if fusion.mode == "learned":
        return np.ascontiguousarray(values[1], dtype=np.float32)
    branch_maps = np.stack(
        [
            _threshold_calibrated_probability(values[2], fusion.context_threshold),
            _threshold_calibrated_probability(values[3], fusion.flow_threshold),
            _threshold_calibrated_probability(values[4], fusion.shape_threshold),
        ]
    )
    if fusion.mode == "single":
        return branch_maps[BOUNDARY_BRANCH_NAMES.index(str(fusion.head))]
    if fusion.mode == "any":
        return branch_maps.max(axis=0)
    if fusion.mode == "all":
        return branch_maps.min(axis=0)
    if fusion.mode == "majority":
        return np.median(branch_maps, axis=0).astype(np.float32)
    weights = np.asarray(fusion.weights, dtype=np.float32)[:, None, None]
    return np.sum(branch_maps * weights, axis=0, dtype=np.float32)


def _fusion_specs(bounds: BoundarySearchBounds) -> list[BoundaryFusionSpec]:
    specifications = [BoundaryFusionSpec(mode="learned")]
    branch_thresholds = {
        "context": bounds.context_thresholds,
        "flow": bounds.flow_thresholds,
        "shape": bounds.shape_thresholds,
    }
    for head in BOUNDARY_BRANCH_NAMES:
        for threshold in branch_thresholds[head]:
            values = {name: 0.50 for name in BOUNDARY_BRANCH_NAMES}
            values[head] = threshold
            specifications.append(
                BoundaryFusionSpec(
                    mode="single",
                    head=head,
                    context_threshold=values["context"],
                    flow_threshold=values["flow"],
                    shape_threshold=values["shape"],
                )
            )
    triples = product(
        bounds.context_thresholds,
        bounds.flow_thresholds,
        bounds.shape_thresholds,
    )
    for context_threshold, flow_threshold, shape_threshold in triples:
        common = {
            "context_threshold": context_threshold,
            "flow_threshold": flow_threshold,
            "shape_threshold": shape_threshold,
        }
        for mode in ("any", "majority", "all"):
            specifications.append(BoundaryFusionSpec(mode=mode, **common))
        for weights in bounds.weighted_combinations:
            specifications.append(
                BoundaryFusionSpec(mode="weighted", weights=weights, **common)
            )
    unique = {specification.identifier: specification for specification in specifications}
    return [unique[key] for key in sorted(unique)]


def generate_operating_points(
    bounds: BoundarySearchBounds,
) -> list[BoundaryOperatingPoint]:
    """Generate the exhaustive grid (kept as a public audit/debug helper).

    Production calibration uses the bounded staged search in
    :func:`run_v4_boundary_search`; callers can still materialize this full grid explicitly for
    ablation work on a suitably small record set.
    """

    points = [
        BoundaryOperatingPoint(
            fusion=fusion,
            foreground_threshold=foreground_threshold,
            boundary_cutoff=boundary_cutoff,
            min_area_fraction=min_area_fraction,
        )
        for fusion in _fusion_specs(bounds)
        for foreground_threshold in bounds.foreground_thresholds
        for min_area_fraction in bounds.minimum_area_fractions
        for boundary_cutoff in bounds.boundary_cutoffs
    ]
    return sorted(points, key=lambda point: point.identifier)


def _closest_to_half(values: Sequence[float]) -> float:
    return min((float(value) for value in values), key=lambda value: (abs(value - 0.5), value))


def _points_for_specs(
    specifications: Sequence[BoundaryFusionSpec],
    foreground_thresholds: Sequence[float],
    minimum_area_fractions: Sequence[float],
    boundary_cutoffs: Sequence[float],
) -> list[BoundaryOperatingPoint]:
    unique_specs = {
        specification.identifier: specification for specification in specifications
    }
    points = [
        BoundaryOperatingPoint(
            fusion=specification,
            foreground_threshold=float(foreground_threshold),
            boundary_cutoff=float(boundary_cutoff),
            min_area_fraction=float(minimum_area_fraction),
        )
        for specification in unique_specs.values()
        for foreground_threshold in foreground_thresholds
        for minimum_area_fraction in minimum_area_fractions
        for boundary_cutoff in boundary_cutoffs
    ]
    return sorted(points, key=lambda point: point.identifier)


def _branch_screen_specs(bounds: BoundarySearchBounds) -> list[BoundaryFusionSpec]:
    specifications = [BoundaryFusionSpec(mode="learned")]
    thresholds = {
        "context": bounds.context_thresholds,
        "flow": bounds.flow_thresholds,
        "shape": bounds.shape_thresholds,
    }
    for head in BOUNDARY_BRANCH_NAMES:
        for threshold in thresholds[head]:
            values = {name: 0.5 for name in BOUNDARY_BRANCH_NAMES}
            values[head] = float(threshold)
            specifications.append(
                BoundaryFusionSpec(
                    mode="single",
                    head=head,
                    context_threshold=values["context"],
                    flow_threshold=values["flow"],
                    shape_threshold=values["shape"],
                )
            )
    return specifications


def _shortlist_branch_thresholds(
    curves: Sequence[Mapping[str, object]],
    bounds: BoundarySearchBounds,
) -> dict[str, tuple[float, ...]]:
    shortlisted: dict[str, tuple[float, ...]] = {}
    for head in BOUNDARY_BRANCH_NAMES:
        family = f"single_{head}"
        candidates = [curve for curve in curves if curve["fusion_family"] == family]
        ranked = sorted(
            candidates,
            key=lambda curve: (
                -float(curve["normalized_selection_score_auc"]),
                -float(curve["best_operating_point"]["selection_score"]),  # type: ignore[index]
                str(curve["curve_id"]),
            ),
        )
        if not ranked:
            raise RuntimeError(f"Branch screen produced no candidates for {head}")
        values: list[float] = []
        for curve in ranked[: bounds.branch_threshold_shortlist_per_head]:
            fusion = curve["base_config"]["fusion"]  # type: ignore[index]
            threshold = float(fusion[f"{head}_threshold"])
            if threshold not in values:
                values.append(threshold)
        shortlisted[head] = tuple(values)
    return shortlisted


def _fusion_screen_specs(
    bounds: BoundarySearchBounds,
    shortlisted: Mapping[str, Sequence[float]],
) -> list[BoundaryFusionSpec]:
    specifications = [BoundaryFusionSpec(mode="learned")]
    for head in BOUNDARY_BRANCH_NAMES:
        for threshold in shortlisted[head]:
            values = {name: 0.5 for name in BOUNDARY_BRANCH_NAMES}
            values[head] = float(threshold)
            specifications.append(
                BoundaryFusionSpec(
                    mode="single",
                    head=head,
                    context_threshold=values["context"],
                    flow_threshold=values["flow"],
                    shape_threshold=values["shape"],
                )
            )
    for context_threshold, flow_threshold, shape_threshold in product(
        shortlisted["context"],
        shortlisted["flow"],
        shortlisted["shape"],
    ):
        common = {
            "context_threshold": float(context_threshold),
            "flow_threshold": float(flow_threshold),
            "shape_threshold": float(shape_threshold),
        }
        for mode in ("any", "majority", "all"):
            specifications.append(BoundaryFusionSpec(mode=mode, **common))
        for weights in bounds.weighted_combinations:
            specifications.append(
                BoundaryFusionSpec(mode="weighted", weights=weights, **common)
            )
    unique = {specification.identifier: specification for specification in specifications}
    return [unique[key] for key in sorted(unique)]


def _safe_ratio(numerator: float, denominator: float, empty_value: float) -> float:
    return float(numerator / denominator) if denominator else float(empty_value)


def _binary_metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    true_positive = int((prediction & truth).sum())
    false_positive = int((prediction & ~truth).sum())
    false_negative = int((~prediction & truth).sum())
    predicted = true_positive + false_positive
    actual = true_positive + false_negative
    union = true_positive + false_positive + false_negative
    return {
        "dice": _safe_ratio(2 * true_positive, predicted + actual, 1.0),
        "iou": _safe_ratio(true_positive, union, 1.0),
        "precision": _safe_ratio(true_positive, predicted, 1.0 if actual == 0 else 0.0),
        "recall": _safe_ratio(true_positive, actual, 1.0),
    }


def _instance_iou_matrix(
    truth_labels: np.ndarray,
    prediction_labels: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    truth_values = np.unique(truth_labels)
    truth_values = truth_values[truth_values > 0]
    prediction_values = np.unique(prediction_labels)
    prediction_values = prediction_values[prediction_values > 0]
    truth_count = len(truth_values)
    prediction_count = len(prediction_values)
    matrix = np.zeros((truth_count, prediction_count), dtype=np.float64)
    if truth_count == 0 or prediction_count == 0:
        return matrix, truth_count, prediction_count
    truth_area = np.asarray(
        [(truth_labels == value).sum() for value in truth_values], dtype=np.float64
    )
    prediction_area = np.asarray(
        [(prediction_labels == value).sum() for value in prediction_values],
        dtype=np.float64,
    )
    overlap = (truth_labels > 0) & (prediction_labels > 0)
    if overlap.any():
        truth_index = np.searchsorted(truth_values, truth_labels[overlap])
        prediction_index = np.searchsorted(
            prediction_values, prediction_labels[overlap]
        )
        intersections = np.bincount(
            truth_index * prediction_count + prediction_index,
            minlength=truth_count * prediction_count,
        ).reshape(truth_count, prediction_count)
        unions = truth_area[:, None] + prediction_area[None, :] - intersections
        matrix = intersections / np.maximum(unions, 1.0)
    return matrix, truth_count, prediction_count


def _maximum_iou_matches(iou: np.ndarray, threshold: float) -> int:
    """Maximum-cardinality bipartite matching with deterministic IoU-first traversal."""

    if iou.size == 0:
        return 0
    adjacency = [
        [
            int(index)
            for index in np.argsort(-iou[row], kind="stable")
            if iou[row, index] >= threshold
        ]
        for row in range(iou.shape[0])
    ]
    prediction_match = [-1] * iou.shape[1]

    def augment(truth_index: int, visited: set[int]) -> bool:
        for prediction_index in adjacency[truth_index]:
            if prediction_index in visited:
                continue
            visited.add(prediction_index)
            previous = prediction_match[prediction_index]
            if previous < 0 or augment(previous, visited):
                prediction_match[prediction_index] = truth_index
                return True
        return False

    truth_order = sorted(
        range(iou.shape[0]),
        key=lambda row: (-float(iou[row].max(initial=0.0)), row),
    )
    return sum(augment(row, set()) for row in truth_order)


def segmentation_metrics(
    prediction_labels: np.ndarray,
    truth_labels: np.ndarray,
) -> dict[str, float]:
    """Foreground/background, reconstructed contact, object AP, and count metrics."""

    prediction = np.asarray(prediction_labels, dtype=np.int32)
    truth = np.asarray(truth_labels, dtype=np.int32)
    if prediction.shape != truth.shape or prediction.ndim != 2:
        raise ValueError(
            f"Prediction and truth must be matching 2-D maps: {prediction.shape}/{truth.shape}"
        )
    prediction_foreground = prediction > 0
    truth_foreground = truth > 0
    foreground = _binary_metrics(prediction_foreground, truth_foreground)
    background = _binary_metrics(~prediction_foreground, ~truth_foreground)
    prediction_contact = internal_contact_boundary(prediction)
    truth_contact = internal_contact_boundary(truth)
    contact = _binary_metrics(prediction_contact, truth_contact)
    iou, truth_count, prediction_count = _instance_iou_matrix(truth, prediction)

    metrics: dict[str, float] = {}
    metrics.update({f"foreground_{key}": value for key, value in foreground.items()})
    metrics.update({f"background_{key}": value for key, value in background.items()})
    metrics.update(
        {f"contact_boundary_{key}": value for key, value in contact.items()}
    )
    metrics["pixel_accuracy"] = float(
        (prediction_foreground == truth_foreground).mean()
    )
    metrics["false_positive_area_fraction"] = float(
        (prediction_foreground & ~truth_foreground).mean()
    )
    metrics["false_negative_area_fraction"] = float(
        (~prediction_foreground & truth_foreground).mean()
    )
    for threshold, name in ((0.50, "instance_ap50"), (0.75, "instance_ap75")):
        true_positive = _maximum_iou_matches(iou, threshold)
        denominator = true_positive + (prediction_count - true_positive) + (
            truth_count - true_positive
        )
        metrics[name] = _safe_ratio(true_positive, denominator, 1.0)
    absolute_error = abs(prediction_count - truth_count)
    metrics["predicted_count"] = float(prediction_count)
    metrics["truth_count"] = float(truth_count)
    metrics["count_absolute_error"] = float(absolute_error)
    metrics["count_relative_absolute_error"] = float(
        absolute_error / max(truth_count, 1)
    )
    metrics["count_score"] = float(1.0 / (1.0 + absolute_error))
    return metrics


def selection_score(metrics: Mapping[str, float]) -> float:
    return float(
        0.25 * metrics["foreground_dice"]
        + 0.15 * metrics["contact_boundary_dice"]
        + 0.35 * metrics["instance_ap50"]
        + 0.25 * metrics["instance_ap75"]
    )


class _DomainMetricAccumulator:
    def __init__(self) -> None:
        self.sums: dict[str, dict[str, float]] = defaultdict(dict)
        self.counts: dict[str, int] = defaultdict(int)

    def add(self, domain: str, metrics: Mapping[str, float]) -> None:
        self.counts[domain] += 1
        sums = self.sums[domain]
        for name, value in metrics.items():
            numeric = float(value)
            if not math.isfinite(numeric):
                raise RuntimeError(f"Non-finite metric {name} for domain {domain}")
            sums[name] = sums.get(name, 0.0) + numeric

    def report(self) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        if not self.counts:
            raise RuntimeError("Metric accumulator is empty")
        per_domain = {
            domain: {
                name: value / self.counts[domain]
                for name, value in sorted(self.sums[domain].items())
            }
            for domain in sorted(self.counts)
        }
        keys = tuple(next(iter(per_domain.values())))
        if any(tuple(metrics) != keys for metrics in per_domain.values()):
            raise RuntimeError("Metric keys differ between domains")
        macro = {
            name: float(np.mean([metrics[name] for metrics in per_domain.values()]))
            for name in keys
        }
        return macro, per_domain


def _evaluate_operating_points(
    records: Sequence[BoundaryEvaluationRecord],
    points: Sequence[BoundaryOperatingPoint],
) -> dict[str, dict[str, object]]:
    if not records or not points:
        raise ValueError("Evaluation requires records and operating points")
    point_by_id = {point.identifier: point for point in points}
    if len(point_by_id) != len(points):
        raise ValueError("Duplicate operating points")
    by_fusion: dict[str, list[BoundaryOperatingPoint]] = defaultdict(list)
    fusion_by_id: dict[str, BoundaryFusionSpec] = {}
    for point in points:
        by_fusion[point.fusion.identifier].append(point)
        fusion_by_id[point.fusion.identifier] = point.fusion
    accumulators = {
        identifier: _DomainMetricAccumulator() for identifier in point_by_id
    }

    for record in records:
        probabilities = _load_probability_source(record)
        truth = _load_truth_source(record)
        if probabilities.shape[1:] != truth.shape:
            raise ValueError(
                f"Probability/truth grid mismatch for {record.record_id}: "
                f"{probabilities.shape[1:]} versus {truth.shape}"
            )
        foreground = probabilities[0]
        for fusion_id in sorted(by_fusion):
            boundary = fuse_boundary_probability(probabilities, fusion_by_id[fusion_id])
            for point in by_fusion[fusion_id]:
                settings = _ReconstructionSettings(
                    foreground_threshold=point.foreground_threshold,
                    boundary_threshold=point.boundary_cutoff,
                    min_area_fraction=point.min_area_fraction,
                )
                prediction = reconstruct_iphone_instances(
                    foreground,
                    boundary,
                    settings,
                )
                accumulators[point.identifier].add(
                    record.domain,
                    segmentation_metrics(prediction, truth),
                )

    results: dict[str, dict[str, object]] = {}
    for identifier in sorted(point_by_id):
        macro, per_domain = accumulators[identifier].report()
        point = point_by_id[identifier]
        results[identifier] = {
            "candidate_id": identifier,
            "curve_id": point.curve_identifier,
            "config": point.as_dict(),
            "macro_domain_metrics": macro,
            "per_domain_metrics": per_domain,
            "selection_score": selection_score(macro),
        }
    return results


def _normalized_auc(x: Sequence[float], y: Sequence[float]) -> float:
    coordinates = np.asarray(x, dtype=np.float64)
    values = np.asarray(y, dtype=np.float64)
    if coordinates.ndim != 1 or values.shape != coordinates.shape or len(coordinates) < 2:
        raise ValueError("AUC requires matching one-dimensional arrays with at least two points")
    order = np.argsort(coordinates, kind="stable")
    coordinates = coordinates[order]
    values = values[order]
    if np.any(np.diff(coordinates) <= 0):
        raise ValueError("AUC coordinates must be unique")
    span = float(coordinates[-1] - coordinates[0])
    area = float(np.sum(np.diff(coordinates) * (values[:-1] + values[1:]) * 0.5))
    return area / span


def _calibration_curves(
    results: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for result in results.values():
        grouped[str(result["curve_id"])].append(result)
    curves: list[dict[str, object]] = []
    for curve_id, raw_points in sorted(grouped.items()):
        points = sorted(
            raw_points,
            key=lambda row: float(row["config"]["boundary_cutoff"]),  # type: ignore[index]
        )
        cutoffs = [
            float(row["config"]["boundary_cutoff"])  # type: ignore[index]
            for row in points
        ]
        metric_names = tuple(points[0]["macro_domain_metrics"])
        normalized_metric_auc = {
            metric: _normalized_auc(
                cutoffs,
                [float(row["macro_domain_metrics"][metric]) for row in points],  # type: ignore[index]
            )
            for metric in metric_names
        }
        normalized_selection_auc = _normalized_auc(
            cutoffs,
            [float(row["selection_score"]) for row in points],
        )
        per_domain_names = tuple(points[0]["per_domain_metrics"])
        per_domain_selection_auc = {
            domain: _normalized_auc(
                cutoffs,
                [
                    selection_score(row["per_domain_metrics"][domain])  # type: ignore[index]
                    for row in points
                ],
            )
            for domain in per_domain_names
        }
        best = max(
            points,
            key=lambda row: (
                float(row["selection_score"]),
                -abs(float(row["config"]["boundary_cutoff"]) - 0.5),  # type: ignore[index]
                str(row["candidate_id"]),
            ),
        )
        curves.append(
            {
                "curve_id": curve_id,
                "fusion_family": best["config"]["fusion_family"],  # type: ignore[index]
                "base_config": {
                    key: value
                    for key, value in best["config"].items()  # type: ignore[union-attr]
                    if key != "boundary_cutoff"
                },
                "boundary_cutoffs": cutoffs,
                "normalized_metric_auc": normalized_metric_auc,
                "normalized_selection_score_auc": normalized_selection_auc,
                "per_domain_normalized_selection_score_auc": per_domain_selection_auc,
                "auc_semantics": AUC_SEMANTICS,
                "best_operating_point": best,
                "operating_points": points,
            }
        )
    return sorted(
        curves,
        key=lambda row: (
            -float(row["normalized_selection_score_auc"]),
            str(row["curve_id"]),
        ),
    )


def _freeze_family_candidates(
    curves: Sequence[Mapping[str, object]],
) -> tuple[list[BoundaryOperatingPoint], list[dict[str, object]]]:
    family_curves: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for curve in curves:
        family_curves[str(curve["fusion_family"])].append(curve)
    frozen: list[BoundaryOperatingPoint] = []
    provenance: list[dict[str, object]] = []
    for family in sorted(family_curves):
        winning_curve = max(
            family_curves[family],
            key=lambda row: (
                float(row["normalized_selection_score_auc"]),
                float(row["best_operating_point"]["selection_score"]),  # type: ignore[index]
                str(row["curve_id"]),
            ),
        )
        best = winning_curve["best_operating_point"]
        config = best["config"]  # type: ignore[index]
        fusion_value = config["fusion"]
        fusion = BoundaryFusionSpec(
            mode=str(fusion_value["mode"]),
            context_threshold=float(fusion_value["context_threshold"]),
            flow_threshold=float(fusion_value["flow_threshold"]),
            shape_threshold=float(fusion_value["shape_threshold"]),
            head=fusion_value["head"],
            weights=tuple(float(value) for value in fusion_value["weights"]),
        )
        point = BoundaryOperatingPoint(
            fusion=fusion,
            foreground_threshold=float(config["foreground_threshold"]),
            boundary_cutoff=float(config["boundary_cutoff"]),
            min_area_fraction=float(config["min_area_fraction"]),
        )
        if point.identifier != best["candidate_id"]:  # type: ignore[index]
            raise RuntimeError("Frozen candidate changed during serialization round trip")
        frozen.append(point)
        provenance.append(
            {
                "fusion_family": family,
                "frozen_candidate_id": point.identifier,
                "selected_on_role": "calibration",
                "curve_id": winning_curve["curve_id"],
                "normalized_selection_score_auc": winning_curve[
                    "normalized_selection_score_auc"
                ],
                "calibration_operating_point": best,
            }
        )
    return frozen, provenance


def _materialize_descriptors(
    records: Iterable[BoundaryEvaluationRecord]
    | Callable[[], Iterable[BoundaryEvaluationRecord]],
) -> list[BoundaryEvaluationRecord]:
    values = list(records() if callable(records) else records)
    if not values:
        raise ValueError("Boundary search received no records")
    if not all(isinstance(record, BoundaryEvaluationRecord) for record in values):
        raise TypeError("Every boundary-search record must be BoundaryEvaluationRecord")
    return values


def _audit_validation_roles(records: Sequence[BoundaryEvaluationRecord]) -> None:
    """Check every descriptor before any array or label source can be opened."""

    invalid_roles = sorted({record.role for record in records} - set(VALIDATION_ROLES))
    if invalid_roles:
        raise PermissionError(
            "v4 boundary search accepts calibration and ensemble_selection only; "
            "held-out labels stay sealed. Rejected roles: "
            + ", ".join(invalid_roles)
        )
    identifiers = [record.record_id for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("record_id values must be unique across validation roles")
    groups = {
        role: {str(record.group_id) for record in records if record.role == role}
        for role in VALIDATION_ROLES
    }
    overlap = groups["calibration"] & groups["ensemble_selection"]
    if overlap:
        raise ValueError(
            "Calibration and ensemble-selection group IDs overlap: "
            + ", ".join(sorted(overlap)[:10])
        )
    missing = [role for role in VALIDATION_ROLES if not groups[role]]
    if missing:
        raise ValueError("Missing required validation roles: " + ", ".join(missing))


def _bounded_records(
    records: Sequence[BoundaryEvaluationRecord],
    maximum_per_role_domain: int | None,
) -> list[BoundaryEvaluationRecord]:
    if maximum_per_role_domain is None:
        return sorted(records, key=lambda record: (record.role, record.domain, record.record_id))
    grouped: dict[tuple[str, str], list[BoundaryEvaluationRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.role, record.domain)].append(record)
    selected: list[BoundaryEvaluationRecord] = []
    for key in sorted(grouped):
        ranked = sorted(
            grouped[key],
            key=lambda record: (
                hashlib.sha256(record.record_id.encode("utf-8")).hexdigest(),
                record.record_id,
            ),
        )
        selected.extend(ranked[:maximum_per_role_domain])
    return selected


def _cache_in_memory_probabilities(
    records: Sequence[BoundaryEvaluationRecord],
    cache: FiveChannelDiskCache | None,
) -> list[BoundaryEvaluationRecord]:
    if cache is None:
        return list(records)
    output: list[BoundaryEvaluationRecord] = []
    for record in records:
        if isinstance(record.probability_source, np.ndarray):
            path = cache.write(
                record.record_id,
                record.probability_source,
                from_logits=record.value_kind == "logit",
            )
            output.append(
                replace(record, probability_source=path, value_kind="probability")
            )
        else:
            output.append(record)
    return output


def _role_summary(records: Sequence[BoundaryEvaluationRecord]) -> dict[str, object]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for record in records:
        counts[record.role][record.domain] += 1
    return {
        role: {
            "record_count": sum(domain_counts.values()),
            "domain_counts": dict(sorted(domain_counts.items())),
        }
        for role, domain_counts in sorted(counts.items())
    }


def run_v4_boundary_search(
    records: Iterable[BoundaryEvaluationRecord]
    | Callable[[], Iterable[BoundaryEvaluationRecord]],
    *,
    output_directory: Path | str | None = None,
    cache_directory: Path | str | None = None,
    bounds: BoundarySearchBounds | None = None,
    smoke: bool = False,
) -> dict[str, object]:
    """Calibrate all fusion families, freeze them, then rank on a disjoint role.

    ``records`` should normally contain paths returned by :class:`FiveChannelDiskCache`; the
    evaluator then memory-maps one probability map and loads one truth map at a time.  In-memory
    probabilities are automatically moved into ``cache_directory`` when one is supplied.
    """

    descriptors = _materialize_descriptors(records)
    _audit_validation_roles(descriptors)
    resolved_bounds = bounds if bounds is not None else (
        BoundarySearchBounds.smoke() if smoke else BoundarySearchBounds()
    )
    if smoke and resolved_bounds.maximum_records_per_role_domain is None:
        resolved_bounds = replace(
            resolved_bounds,
            maximum_records_per_role_domain=2,
        )
    descriptors = _bounded_records(
        descriptors,
        resolved_bounds.maximum_records_per_role_domain,
    )
    output_path = (
        Path(output_directory).expanduser().resolve()
        if output_directory is not None
        else None
    )
    if cache_directory is not None:
        cache = FiveChannelDiskCache(cache_directory)
    elif output_path is not None and any(
        isinstance(record.probability_source, np.ndarray) for record in descriptors
    ):
        cache = FiveChannelDiskCache(output_path / "probability_cache")
    else:
        cache = None
    descriptors = _cache_in_memory_probabilities(descriptors, cache)
    calibration_records = [
        record for record in descriptors if record.role == "calibration"
    ]
    selection_records = [
        record for record in descriptors if record.role == "ensemble_selection"
    ]

    # Three bounded calibration stages retain full boundary-cutoff curves while avoiding the
    # combinatorial product of every branch threshold, foreground threshold, area threshold, and
    # fusion weight.  Stage 1 independently screens each evidence head.  Stage 2 tries every
    # fusion family and configured weight over the Cartesian product of the best head thresholds.
    # Stage 3 tunes foreground/minimum-area settings only for the winning recipe in each family.
    # No ensemble-selection record is opened until all three stages and their candidates freeze.
    reference_foreground = _closest_to_half(resolved_bounds.foreground_thresholds)
    reference_area = min(resolved_bounds.minimum_area_fractions)

    branch_specs = _branch_screen_specs(resolved_bounds)
    branch_points = _points_for_specs(
        branch_specs,
        (reference_foreground,),
        (reference_area,),
        resolved_bounds.boundary_cutoffs,
    )
    branch_results = _evaluate_operating_points(calibration_records, branch_points)
    branch_curves = _calibration_curves(branch_results)
    shortlisted_thresholds = _shortlist_branch_thresholds(
        branch_curves,
        resolved_bounds,
    )

    fusion_specs = _fusion_screen_specs(resolved_bounds, shortlisted_thresholds)
    fusion_points = _points_for_specs(
        fusion_specs,
        (reference_foreground,),
        (reference_area,),
        resolved_bounds.boundary_cutoffs,
    )
    fusion_results = _evaluate_operating_points(calibration_records, fusion_points)
    fusion_curves = _calibration_curves(fusion_results)
    screened_family_points, fusion_screen_provenance = _freeze_family_candidates(
        fusion_curves
    )
    finalist_specs = [point.fusion for point in screened_family_points]

    finalist_points = _points_for_specs(
        finalist_specs,
        resolved_bounds.foreground_thresholds,
        resolved_bounds.minimum_area_fractions,
        resolved_bounds.boundary_cutoffs,
    )
    calibration_results = _evaluate_operating_points(
        calibration_records,
        finalist_points,
    )
    curves = _calibration_curves(calibration_results)
    frozen_candidates, frozen_provenance = _freeze_family_candidates(curves)
    # No configuration construction or tuning occurs after this point.  The second role only
    # receives immutable objects selected above and ranks their measured performance.
    selection_results = _evaluate_operating_points(
        selection_records,
        frozen_candidates,
    )
    ranking = sorted(
        selection_results.values(),
        key=lambda row: (-float(row["selection_score"]), str(row["candidate_id"])),
    )
    frozen_ids = {point.identifier for point in frozen_candidates}
    if {str(row["candidate_id"]) for row in ranking} != frozen_ids:
        raise RuntimeError("Ensemble-selection evaluation changed the frozen candidate set")

    report: dict[str, object] = {
        "schema_version": 1,
        "search_version": V4_BOUNDARY_SEARCH_VERSION,
        "status": "complete",
        "mode": "smoke" if smoke else "full",
        "five_channel_semantics": list(V4_OUTPUT_SEMANTICS),
        "deployment_postprocess_version": POSTPROCESS_VERSION,
        "boundary_target_version": BOUNDARY_TARGET_VERSION,
        "scientific_roles": {
            "candidate_and_threshold_selection": "calibration",
            "frozen_candidate_ranking": "ensemble_selection",
            "test": "sealed; test/final-test descriptors are rejected before labels are opened",
        },
        "selection_objective": SELECTION_OBJECTIVE,
        "domain_aggregation": (
            "image metrics averaged within each domain, then domains macro-averaged equally"
        ),
        "boundary_cutoff_auc_semantics": AUC_SEMANTICS,
        "inference_warning": (
            "Boundary-cutoff AUC requires labelled validation masks and must never be presented "
            "as per-image inference accuracy."
        ),
        "voting_semantics": (
            "Independent branch thresholds map to 0.5; max/median/min therefore implement "
            "continuous any/majority/all votes while retaining confidence for cutoff curves."
        ),
        "weighted_semantics": (
            "Context/flow/shape confidence margins are averaged with non-negative normalized "
            "weights after independent branch-threshold calibration."
        ),
        "bounds": asdict(resolved_bounds),
        "records": _role_summary(descriptors),
        "streaming": {
            "disk_backed_probability_cache": cache is not None,
            "cache_directory": str(cache.root) if cache is not None else None,
            "one_probability_and_truth_grid_loaded_at_a_time": True,
            "smoke_record_selection": (
                "SHA-256(record_id) deterministic within role/domain"
                if resolved_bounds.maximum_records_per_role_domain is not None
                else "not bounded"
            ),
        },
        "calibration": {
            "role": "calibration",
            "search_strategy": (
                "bounded three-stage head-threshold screening, fusion-family screening, and "
                "final foreground/area tuning; every stage retains the complete configured "
                "boundary-cutoff curve"
            ),
            "reference_foreground_threshold": reference_foreground,
            "reference_min_area_fraction": reference_area,
            "deterministic_record_cap_per_role_domain": (
                resolved_bounds.maximum_records_per_role_domain
            ),
            "branch_threshold_shortlist": {
                key: list(value) for key, value in shortlisted_thresholds.items()
            },
            "fusion_spec_count": len(fusion_specs),
            "operating_point_count": (
                len(branch_points) + len(fusion_points) + len(finalist_points)
            ),
            "unique_operating_point_count": len(
                {
                    point.identifier
                    for point in (*branch_points, *fusion_points, *finalist_points)
                }
            ),
            "stage_operating_point_counts": {
                "independent_branch_threshold_screen": len(branch_points),
                "fusion_family_screen": len(fusion_points),
                "final_foreground_area_tuning": len(finalist_points),
            },
            "branch_screen_curve_count": len(branch_curves),
            "fusion_screen_curve_count": len(fusion_curves),
            "curve_count": len(curves),
            "curves": curves,
            "fusion_screen_family_winners": fusion_screen_provenance,
            "frozen_family_candidates": frozen_provenance,
        },
        "ensemble_selection": {
            "role": "ensemble_selection",
            "configuration_policy": (
                "rank only exact candidate IDs frozen on calibration; no threshold, weight, "
                "fusion, or cutoff is changed on this role"
            ),
            "ranking": ranking,
        },
        "selected_candidate": ranking[0],
    }
    report["report_sha256"] = _object_sha256(report)
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
        report_path = output_path / "v4_boundary_search_report.json"
        _atomic_json(report_path, report)
        report["report_path"] = str(report_path)
    return report


# Short alias for pipeline callers that already include the v4 context in their namespace.
evaluate_boundary_fusions = run_v4_boundary_search


def _synthetic_case(offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    height, width = 28, 36
    labels = np.zeros((height, width), dtype=np.int32)
    labels[5:23, 4 + offset : 18 + offset] = 1
    labels[5:23, 18 + offset : 32 + offset] = 2
    foreground = labels > 0
    contact = internal_contact_boundary(labels)
    yy, xx = np.indices(labels.shape)
    texture = ((yy * 7 + xx * 11 + offset) % 17).astype(np.float32) / 17.0
    probabilities = np.empty((5, height, width), dtype=np.float32)
    probabilities[0] = np.where(foreground, 0.94, 0.04)
    probabilities[1] = np.where(contact, 0.91, 0.06)
    probabilities[2] = np.where(contact, 0.88, 0.05) + 0.015 * texture
    probabilities[3] = np.where(contact, 0.84, 0.07) + 0.020 * (1.0 - texture)
    probabilities[4] = np.where(contact, 0.93, 0.04) + 0.010 * texture
    return np.clip(probabilities, 0.0, 1.0), labels


def run_contract_self_test() -> dict[str, object]:
    """Exercise every fusion, disk streaming, role separation, metrics, and AUC on CPU."""

    with tempfile.TemporaryDirectory(prefix="cellect-v4-boundary-search-") as name:
        root = Path(name)
        cache = FiveChannelDiskCache(root / "cache")
        records: list[BoundaryEvaluationRecord] = []
        index = 0
        for role in VALIDATION_ROLES:
            for domain_index, domain in enumerate(("phase", "brightfield")):
                probabilities, labels = _synthetic_case(domain_index)
                record_id = f"{role}:{domain}:{index}"
                cache.write(
                    record_id,
                    probabilities,
                    from_logits=False,
                )
                truth_path = root / "truth" / f"{index}.npy"
                _atomic_numpy(truth_path, labels)
                records.append(
                    cache.record(
                        record_id,
                        role=role,
                        domain=domain,
                        truth_source=truth_path,
                        group_id=f"{role}-group-{index}",
                    )
                )
                index += 1
        report = run_v4_boundary_search(
            records,
            output_directory=root / "report",
            cache_directory=cache.root,
            smoke=True,
        )
        families = {
            row["config"]["fusion_family"]
            for row in report["ensemble_selection"]["ranking"]
        }
        expected_families = {
            "learned_fusion",
            "single_context",
            "single_flow",
            "single_shape",
            "vote_any",
            "vote_majority",
            "vote_all",
            "weighted_branches",
        }
        if families != expected_families:
            raise AssertionError(f"Fusion family coverage failed: {families}")
        if report["calibration"]["operating_point_count"] <= len(families):
            raise AssertionError("Calibration grid did not search multiple operating points")
        if not report["calibration"]["curves"]:
            raise AssertionError("Boundary-cutoff curves were not reported")
        if any(
            curve["auc_semantics"] != AUC_SEMANTICS
            for curve in report["calibration"]["curves"]
        ):
            raise AssertionError("AUC validation semantics were lost")
        if not Path(report["report_path"]).is_file():
            raise AssertionError("Atomic boundary-search report was not written")
        if not all(cache.verify(record.record_id) for record in records):
            raise AssertionError("Five-channel disk cache verification failed")
        selected_metrics = report["selected_candidate"]["macro_domain_metrics"]
        required_metrics = {
            "foreground_dice",
            "background_dice",
            "contact_boundary_dice",
            "instance_ap50",
            "instance_ap75",
            "count_absolute_error",
        }
        if not required_metrics <= set(selected_metrics):
            raise AssertionError("Required segmentation metrics are missing")

        # The nonexistent label is a tripwire: role auditing must fail before it is opened.
        forbidden = records + [
            BoundaryEvaluationRecord(
                record_id="sealed-test-tripwire",
                role="test",
                domain="sealed",
                probability_source=root / "also-does-not-exist.npy",
                truth_source=root / "must-not-be-opened.npy",
            )
        ]
        try:
            run_v4_boundary_search(forbidden, bounds=BoundarySearchBounds.smoke())
        except PermissionError as error:
            if "held-out labels stay sealed" not in str(error):
                raise
        else:
            raise AssertionError("Test-role descriptor was not rejected")

        return {
            "status": "passed",
            "search_version": V4_BOUNDARY_SEARCH_VERSION,
            "fusion_families": sorted(families),
            "calibration_operating_points": report["calibration"][
                "operating_point_count"
            ],
            "frozen_candidates": len(report["ensemble_selection"]["ranking"]),
            "selected_score": report["selected_candidate"]["selection_score"],
            "disk_cache": "verified",
            "test_label_tripwire": "rejected_before_open",
        }


__all__ = [
    "AUC_SEMANTICS",
    "BOUNDARY_BRANCH_NAMES",
    "BoundaryEvaluationRecord",
    "BoundaryFusionSpec",
    "BoundaryOperatingPoint",
    "BoundarySearchBounds",
    "FiveChannelDiskCache",
    "FUSION_MODES",
    "SELECTION_OBJECTIVE",
    "V4_BOUNDARY_SEARCH_VERSION",
    "V4_OUTPUT_SEMANTICS",
    "VALIDATION_ROLES",
    "evaluate_boundary_fusions",
    "fuse_boundary_probability",
    "generate_operating_points",
    "normalize_five_channel_output",
    "run_contract_self_test",
    "run_v4_boundary_search",
    "segmentation_metrics",
    "selection_score",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the deterministic CPU contract test",
    )
    arguments = parser.parse_args()
    if not arguments.self_test:
        parser.error("use --self-test, or import run_v4_boundary_search from Python")
    print(json.dumps(run_contract_self_test(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
