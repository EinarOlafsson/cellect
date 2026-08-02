#!/usr/bin/env python3
"""Fail-closed CellectTrack v4 training, selection, and deployment export.

This module consumes only the strict development annotations exposed by ``tracking_data.py``.
It never searches CTC result/test directories or DeepSea's test partition.  LiveCellTrack uses
only its human MOT ``gt.txt`` identity boxes, with no invented division parents.  Entire
acquisitions retain their deterministic train/checkpoint/calibration/ensemble-selection roles.
Checkpoint loss selects weights, calibration selects operating thresholds, and
ensemble-selection chooses between the frozen heavy/mobile models and their mean ensemble.

``best-smoke`` validates every discovered acquisition before running a bounded, genuine
forward/backward/optimizer update in every training phase.  ``best`` requires a matching smoke
marker and refuses incompatible or unverifiable resume checkpoints.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset, Sampler

from tracking_data import (
    ALFI_SOURCE_FORMAT,
    CTC_TRACKING_DATASETS,
    CTMC_V1_SOURCE_FORMAT,
    LIVECELLTRACK_SOURCE_FORMAT,
    TRACKING_SCHEMA_VERSION,
    TRACKING_ROLES,
    FrameInstance,
    ParentLink as SourceParentLink,
    TrackLifetime,
    TrackingFrame,
    TrackingSequence,
    bbox_proxy_labels,
    discover_transmitted_light_tracking,
    file_sha256,
    prepare_alfi_task1,
    prepare_ctmc_v1,
    resolve_omitted_tracking_sources,
    tracking_summary,
)
from tracking_metrics import (
    Detection,
    DetectionMatch,
    DirectedLink,
    OfficialTrackingEvaluator,
    TrackingEvaluationCase,
    TrackingGraph,
    evaluate_tracking_dataset,
    run_official_evaluator,
)
from tracking_models import (
    CELL_FEATURE_DIM,
    CELL_FEATURE_NAMES,
    CELLECT_TRACK_VERSION,
    CellectTrack,
    CellectTrackConfig,
    CellectTrackDeploymentAdapter,
    CellectTrackHeavy,
    CellectTrackMobile,
    assert_gradient_audit,
    differentiable_assignment_loss,
    differentiable_division_loss,
    load_trackastra_teacher,
    set_training_phase,
    teacher_artifact,
    tracking_distillation_losses,
)


TRACKING_TRAINING_SCHEMA_VERSION = 4
TRACKING_FEATURE_VERSION = "cellect-track-features-v4.2-bbox-in-memory"
TRACKING_WINDOW_VERSION = "temporal-spatial-repair-windows-v2-gap2"
TRACKING_SAMPLER_VERSION = "modality-dataset-acquisition-window-balanced-v1"
TRACKASTRA_CHUNK_VERSION = "trackastra-adjacent-chunk64-overlap1-v1"
OOF_CELLECT_TOKEN_CACHE_VERSION = "cellect-oof-tracking-token-cache-v1"
TRACKING_RUN_VERSION = "cellect-v4-tracking-training-v2"
DEFAULT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = DEFAULT_ROOT / "data"
DEFAULT_OUTPUT_ROOT = DEFAULT_ROOT / "output" / "tracking_v4"
CTC_ENVIRONMENT_NAMES: Mapping[str, str] = {
    "ctc_bf_hsc": "CELLECT_CTC_BF_HSC_ROOT",
    "ctc_bf_musc": "CELLECT_CTC_BF_MUSC_ROOT",
    "ctc_dic_hela": "CELLECT_CTC_DIC_HELA_ROOT",
    "ctc_phc_u373": "CELLECT_CTC_PHC_U373_ROOT",
    "ctc_phc_psc": "CELLECT_CTC_PHC_PSC_ROOT",
}
LIVECELLTRACK_ENVIRONMENT_NAME = "CELLECT_LIVECELLTRACK_ROOT"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(_canonical_json(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _code_fingerprint() -> str:
    names = (
        "tracking_data.py",
        "tracking_metrics.py",
        "tracking_models.py",
        "v4_tracking_training.py",
        "requirements.txt",
    )
    return _fingerprint(
        {name: file_sha256(DEFAULT_ROOT / name) for name in names}
    )


@dataclass(frozen=True)
class TrackingTrainingConfig:
    max_tokens: int = 128
    window_frames: int = 3
    window_overlap_fraction: float = 0.25
    batch_size: int = 2
    optimizer_steps_per_epoch: int = 512
    checkpoint_steps_per_epoch: int = 128
    heads_epochs: int = 6
    adapters_epochs: int = 12
    full_epochs: int = 120
    distillation_epochs: int = 40
    early_stopping_patience: int = 18
    minimum_improvement: float = 1e-4
    heads_learning_rate: float = 3e-4
    adapters_learning_rate: float = 1e-4
    full_learning_rate: float = 3e-5
    distillation_learning_rate: float = 2e-5
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    feature_jitter_std: float = 0.025
    coordinate_jitter_std: float = 0.003
    token_dropout_probability: float = 0.05
    forced_gap_dropout_probability: float = 0.35
    morphology_perturbation_probability: float = 0.20
    merge_proposal_probability: float = 0.10
    split_proposal_probability: float = 0.10
    false_positive_proposal_probability: float = 0.10
    maximum_frame_gap: int = 2
    external_teacher_weight: float = 0.35
    internal_teacher_weight: float = 0.50
    seed: int = 20260731

    def __post_init__(self) -> None:
        if self.max_tokens < 4 or self.window_frames < 2:
            raise ValueError("Tracking windows require >=4 tokens and >=2 frames")
        if not 0.0 <= self.window_overlap_fraction < 1.0:
            raise ValueError("Window overlap fraction must be in [0, 1)")
        if not 0.0 <= self.token_dropout_probability < 0.5:
            raise ValueError("Token dropout probability must be in [0, 0.5)")
        if self.batch_size < 1:
            raise ValueError("Tracking batch size must be positive")
        if self.optimizer_steps_per_epoch < 1 or self.checkpoint_steps_per_epoch < 1:
            raise ValueError("Tracking sampler steps per epoch must be positive")
        if not 0.0 <= self.forced_gap_dropout_probability <= 1.0:
            raise ValueError("Forced gap dropout probability must be in [0, 1]")
        proposal_probabilities = (
            self.morphology_perturbation_probability,
            self.merge_proposal_probability,
            self.split_proposal_probability,
            self.false_positive_proposal_probability,
        )
        if min(proposal_probabilities) < 0.0 or sum(proposal_probabilities) > 1.0:
            raise ValueError(
                "Deployment perturbation probabilities must be non-negative and sum to <=1"
            )
        if self.maximum_frame_gap != 2:
            raise ValueError("The mobile deployment contract supports frame gaps 1--2 only")
        epoch_values = (
            self.heads_epochs,
            self.adapters_epochs,
            self.full_epochs,
            self.distillation_epochs,
            self.early_stopping_patience,
        )
        if min(epoch_values) < 1:
            raise ValueError("Every tracking phase and patience must be positive")
        if min(
            self.heads_learning_rate,
            self.adapters_learning_rate,
            self.full_learning_rate,
            self.distillation_learning_rate,
        ) <= 0:
            raise ValueError("Tracking learning rates must be positive")


@dataclass(frozen=True)
class TokenRecord:
    token_id: str
    frame_index: int
    component_id: int
    track_id: int
    parent_track_id: int
    time_yx: tuple[float, float, float]
    features: tuple[float, ...]
    identity_supervision_available: bool = True


@dataclass(frozen=True)
class SequenceTokens:
    acquisition_group: str
    role: str
    domain: str
    sequence: TrackingSequence
    records: tuple[TokenRecord, ...]
    continuation_edges: tuple[tuple[int, int], ...]
    parent_edges: tuple[tuple[int, int], ...]
    recovery_edges: tuple[tuple[int, int], ...] = ()
    event_supervision_quarantined: tuple[int, ...] = ()
    division_supervision_quarantined: tuple[int, ...] = ()

    @property
    def all_edges(self) -> tuple[tuple[int, int], ...]:
        return self.continuation_edges + self.parent_edges

    @property
    def training_positive_edges(self) -> tuple[tuple[int, int], ...]:
        return self.all_edges + self.recovery_edges


@dataclass(frozen=True)
class TokenWindow:
    window_id: str
    acquisition_group: str
    role: str
    token_indices: tuple[int, ...]


@dataclass(frozen=True)
class FeatureNormalization:
    names: tuple[str, ...]
    mean: tuple[float, ...]
    standard_deviation: tuple[float, ...]
    fitted_role: str
    fingerprint_sha256: str


@dataclass(frozen=True)
class ModelProbabilityOutput:
    association: Mapping[str, Mapping[tuple[str, str], float]]
    division: Mapping[str, Mapping[str, float]]
    birth: Mapping[str, Mapping[str, float]]
    death: Mapping[str, Mapping[str, float]]


def _read_image(path: Path) -> np.ndarray:
    if path.suffix.casefold() in {".tif", ".tiff"}:
        image = np.asarray(tifffile.imread(path))
    else:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Could not decode tracking image {path}")
    if image.ndim == 3:
        image = image.astype(np.float32).mean(axis=-1)
    if image.ndim != 2 or not np.isfinite(image).all():
        raise RuntimeError(f"Tracking image must be finite 2-D data: {path}")
    image = image.astype(np.float32)
    low, high = np.percentile(image, (0.5, 99.5))
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0).astype(np.float32)


def _read_instance_labels(frame: TrackingFrame, source_format: str) -> np.ndarray:
    if source_format in {CTMC_V1_SOURCE_FORMAT, ALFI_SOURCE_FORMAT}:
        return bbox_proxy_labels(frame).astype(np.int32)
    if frame.instance_path.suffix.casefold() in {".tif", ".tiff"}:
        mask = np.asarray(tifffile.imread(frame.instance_path))
    else:
        mask = cv2.imread(str(frame.instance_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Could not decode tracking mask {frame.instance_path}")
    if mask.ndim == 3:
        if not all(np.array_equal(mask[..., 0], mask[..., index]) for index in range(1, mask.shape[2])):
            raise RuntimeError(f"Tracking mask channels disagree: {frame.instance_path}")
        mask = mask[..., 0]
    if mask.ndim != 2 or mask.shape != (frame.height, frame.width):
        raise RuntimeError(f"Tracking mask geometry changed after preflight: {frame.instance_path}")
    if source_format == "deepsea_basic_tracker_v1":
        _, labels = cv2.connectedComponents((mask > 0).astype(np.uint8), connectivity=8)
        return labels.astype(np.int32)
    if source_format in {"ctc_tra_v1", LIVECELLTRACK_SOURCE_FORMAT}:
        if not np.issubdtype(mask.dtype, np.integer):
            raise RuntimeError(
                f"Strict tracking mask lost integer labels: {frame.instance_path}"
            )
        return mask.astype(np.int32)
    raise RuntimeError(f"Unsupported strict tracking source format: {source_format}")


def _component_shape_features(binary: np.ndarray) -> tuple[float, ...]:
    area = float(binary.sum())
    contours, _ = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    points = np.column_stack(np.nonzero(binary)).astype(np.float64)
    if len(points) >= 2:
        covariance = np.cov(points, rowvar=False)
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 1e-8)
        axis_ratio = float(math.sqrt(eigenvalues[0] / eigenvalues[-1]))
        eccentricity = float(math.sqrt(max(0.0, 1.0 - eigenvalues[0] / eigenvalues[-1])))
    else:
        axis_ratio, eccentricity = 1.0, 0.0
    y, x, height, width = cv2.boundingRect(binary.astype(np.uint8))
    del y, x
    extent = area / max(1.0, float(height * width))
    if contours:
        all_points = np.concatenate(contours, axis=0)
        hull_area = float(cv2.contourArea(cv2.convexHull(all_points)))
    else:
        hull_area = area
    solidity = min(1.0, area / max(area, hull_area, 1.0))
    compactness = min(1.0, 4.0 * math.pi * area / max(perimeter * perimeter, 1e-8))
    distance = cv2.distanceTransform(binary.astype(np.uint8), cv2.DIST_L2, 3)
    values = distance[binary]
    distance_mean = float(values.mean()) if values.size else 0.0
    distance_max = float(values.max()) if values.size else 0.0
    gradient_y, gradient_x = np.gradient(distance.astype(np.float32))
    magnitude = np.sqrt(gradient_y**2 + gradient_x**2) + 1e-6
    unit_y = gradient_y / magnitude
    unit_x = gradient_x / magnitude
    flow_y = float(unit_y[binary].mean()) if area else 0.0
    flow_x = float(unit_x[binary].mean()) if area else 0.0
    flow_coherence = float(math.sqrt(flow_y * flow_y + flow_x * flow_x))
    divergence = np.gradient(unit_y, axis=0) + np.gradient(unit_x, axis=1)
    flow_divergence = float(divergence[binary].mean()) if area else 0.0
    return (
        math.log1p(area),
        math.log1p(perimeter),
        compactness,
        eccentricity,
        solidity,
        extent,
        axis_ratio,
        distance_mean,
        distance_max,
        flow_y,
        flow_x,
        flow_coherence,
        flow_divergence,
    )


@dataclass(frozen=True)
class _FrameFeatureContext:
    image: np.ndarray
    labels: np.ndarray
    gradient: np.ndarray
    laplacian: np.ndarray
    diagonal: float


def _frame_feature_context(frame: TrackingFrame, source_format: str) -> _FrameFeatureContext:
    image = _read_image(frame.image_path)
    labels = _read_instance_labels(frame, source_format)
    gradient_x = cv2.Sobel(image, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image, cv2.CV_32F, 0, 1, ksize=3)
    return _FrameFeatureContext(
        image=image,
        labels=labels,
        gradient=np.sqrt(gradient_x**2 + gradient_y**2),
        laplacian=cv2.Laplacian(image, cv2.CV_32F),
        diagonal=math.hypot(frame.height, frame.width),
    )


def _features_for_binary(
    context: _FrameFeatureContext,
    binary: np.ndarray,
    *,
    centroid_y: float,
    centroid_x: float,
    peer_centroids_yx: np.ndarray,
    owned_component_ids: Sequence[int],
) -> tuple[float, ...]:
    if binary.shape != context.image.shape or not bool(binary.any()):
        raise RuntimeError("Deployment perturbation produced an empty/wrong-shape instance")
    intensities = context.image[binary]
    appearance = (
        float(intensities.mean()),
        float(intensities.std()),
        float(np.percentile(intensities, 10)),
        float(np.median(intensities)),
        float(np.percentile(intensities, 90)),
        float(context.gradient[binary].mean()),
        float(np.square(context.laplacian[binary]).mean()),
    )
    shape = list(_component_shape_features(binary))
    shape[7] /= max(context.diagonal, 1.0)
    shape[8] /= max(context.diagonal, 1.0)
    own_centroid = np.asarray((centroid_y, centroid_x), dtype=np.float64)
    if len(peer_centroids_yx):
        distances = np.sqrt(np.square(peer_centroids_yx - own_centroid).sum(axis=1))
        positive = distances[distances > 1e-12]
        nearest = float(positive.min()) if len(positive) else 1.0
        neighbor_count = float((positive < 0.10).sum())
        local_density = float(np.exp(-positive / 0.10).sum())
    else:
        nearest, neighbor_count, local_density = 1.0, 0.0, 0.0
    dilated = cv2.dilate(binary.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    owned = np.isin(context.labels, np.asarray(tuple(owned_component_ids), dtype=np.int32))
    other_instances = (context.labels > 0) & ~owned
    rim = dilated & ~binary
    contact_fraction = float((rim & other_instances).sum() / max(1, int(rim.sum())))
    features = appearance + tuple(shape) + (
        neighbor_count,
        nearest,
        local_density,
        contact_fraction,
    )
    if len(features) != CELL_FEATURE_DIM or not np.isfinite(features).all():
        raise RuntimeError("Tracking feature extractor violated the model feature contract")
    return tuple(float(value) for value in features)


def _frame_features(
    frame: TrackingFrame,
    source_format: str,
) -> dict[int, tuple[float, ...]]:
    context = _frame_feature_context(frame, source_format)
    centroids = np.asarray(
        [[item.centroid_y / frame.height, item.centroid_x / frame.width] for item in frame.instances],
        dtype=np.float64,
    )
    output: dict[int, tuple[float, ...]] = {}
    for index, instance in enumerate(frame.instances):
        binary = context.labels == instance.component_id
        if int(binary.sum()) != instance.area_pixels:
            raise RuntimeError(
                f"Instance component changed after preflight: {frame.instance_path} "
                f"component {instance.component_id}"
            )
        output[instance.component_id] = _features_for_binary(
            context,
            binary,
            centroid_y=float(instance.centroid_y / frame.height),
            centroid_x=float(instance.centroid_x / frame.width),
            peer_centroids_yx=centroids,
            owned_component_ids=(instance.component_id,),
        )
    return output


def extract_sequence_tokens(sequence: TrackingSequence) -> SequenceTokens:
    records: list[TokenRecord] = []
    indices_by_frame_track: dict[tuple[int, int], list[int]] = defaultdict(list)
    for frame in sequence.frames:
        features = _frame_features(frame, sequence.source_format)
        for instance in sorted(frame.instances, key=lambda item: item.track_id):
            token_id = (
                f"{sequence.acquisition_group}:{frame.frame_index}:"
                f"{instance.component_id}"
            )
            index = len(records)
            indices_by_frame_track[(frame.frame_index, instance.track_id)].append(index)
            records.append(
                TokenRecord(
                    token_id=token_id,
                    frame_index=frame.frame_index,
                    component_id=instance.component_id,
                    track_id=instance.track_id,
                    parent_track_id=instance.parent_track_id,
                    time_yx=(
                        float(frame.frame_index),
                        2.0 * instance.centroid_y / max(1, frame.height - 1) - 1.0,
                        2.0 * instance.centroid_x / max(1, frame.width - 1) - 1.0,
                    ),
                    features=features[instance.component_id],
                    identity_supervision_available=(
                        instance.identity_supervision_available
                    ),
                )
            )
    safe_index: dict[tuple[int, int], int] = {}
    for key, indices in indices_by_frame_track.items():
        available = [index for index in indices if records[index].identity_supervision_available]
        if len(available) > 1:
            raise RuntimeError(
                f"Multiple identity-supervised cells share frame/track {key} in "
                f"{sequence.acquisition_group}"
            )
        if available:
            safe_index[key] = available[0]
    continuation: list[tuple[int, int]] = []
    recovery: list[tuple[int, int]] = []
    event_quarantined: set[int] = set()
    for track in sequence.tracks:
        available_frames = sorted(
            frame
            for frame in track.observed_frames
            if (frame, track.track_id) in safe_index
        )
        for left, right in zip(available_frames, available_frames[1:]):
            if right == left + 1:
                continuation.append(
                    (safe_index[(left, track.track_id)], safe_index[(right, track.track_id)])
                )
            elif right > left + 1:
                event_quarantined.update(
                    (safe_index[(left, track.track_id)], safe_index[(right, track.track_id)])
                )
        for left in available_frames:
            right = left + 2
            if (right, track.track_id) in safe_index:
                middle_key = (left + 1, track.track_id)
                if middle_key in indices_by_frame_track and middle_key not in safe_index:
                    # An ambiguous annotated midpoint is not a simulated missed detection.
                    continue
                recovery.append(
                    (safe_index[(left, track.track_id)], safe_index[(right, track.track_id)])
                )
    parents_list: list[tuple[int, int]] = []
    for link in sequence.parent_links:
        source = safe_index.get((link.parent_end_frame, link.parent_track_id))
        target = safe_index.get((link.child_start_frame, link.child_track_id))
        if source is None or target is None:
            for key in (
                (link.parent_end_frame, link.parent_track_id),
                (link.child_start_frame, link.child_track_id),
            ):
                event_quarantined.update(indices_by_frame_track.get(key, ()))
            continue
        parents_list.append((source, target))
    for track_id, frame_index in sequence.event_quarantined_track_frames:
        event_quarantined.update(indices_by_frame_track.get((frame_index, track_id), ()))
    division_quarantined: set[int] = set()
    for track_id in sequence.division_quarantined_parent_track_ids:
        track = next(item for item in sequence.tracks if item.track_id == track_id)
        division_quarantined.update(
            indices_by_frame_track.get((track.last_frame, track_id), ())
        )
    return SequenceTokens(
        acquisition_group=sequence.acquisition_group,
        role=sequence.role,
        domain=f"{sequence.dataset}:{sequence.modality}",
        sequence=sequence,
        records=tuple(records),
        continuation_edges=tuple(sorted(continuation)),
        parent_edges=tuple(sorted(parents_list)),
        recovery_edges=tuple(sorted(set(recovery))),
        event_supervision_quarantined=tuple(sorted(event_quarantined)),
        division_supervision_quarantined=tuple(sorted(division_quarantined)),
    )


def fit_feature_normalization(sequences: Sequence[SequenceTokens]) -> FeatureNormalization:
    training = [sequence for sequence in sequences if sequence.role == "train"]
    if not training:
        raise RuntimeError("Feature normalization requires train-role acquisitions")
    values = np.asarray(
        [record.features for sequence in training for record in sequence.records],
        dtype=np.float64,
    )
    if values.ndim != 2 or values.shape[1] != CELL_FEATURE_DIM:
        raise RuntimeError("Train tracking features have an invalid matrix shape")
    mean = values.mean(axis=0)
    standard_deviation = values.std(axis=0)
    standard_deviation = np.maximum(standard_deviation, 1e-6)
    payload = {
        "version": TRACKING_FEATURE_VERSION,
        "names": CELL_FEATURE_NAMES,
        "mean": mean.tolist(),
        "standard_deviation": standard_deviation.tolist(),
        "fitted_role": "train",
    }
    return FeatureNormalization(
        names=CELL_FEATURE_NAMES,
        mean=tuple(float(value) for value in mean),
        standard_deviation=tuple(float(value) for value in standard_deviation),
        fitted_role="train",
        fingerprint_sha256=_fingerprint(payload),
    )


def _spatial_order(records: Sequence[TokenRecord], indices: Sequence[int]) -> list[int]:
    return sorted(
        indices,
        key=lambda index: (
            round(records[index].time_yx[1], 6),
            round(records[index].time_yx[2], 6),
            records[index].frame_index,
            records[index].component_id,
        ),
    )


def build_token_windows(
    sequence: SequenceTokens,
    *,
    max_tokens: int,
    window_frames: int,
    overlap_fraction: float,
) -> tuple[TokenWindow, ...]:
    """Pack temporal windows and add repair windows for every positive lineage edge.

    Large acquisitions are deterministically spatially tiled.  A second pass creates bounded
    context windows whenever tiling separated a true continuation or a complete daughter set.
    Thus token limits never silently discard positive supervision.
    """
    if max_tokens < 4 or window_frames < 2:
        raise ValueError("Invalid temporal token-window configuration")
    frame_indices = sorted({record.frame_index for record in sequence.records})
    if not frame_indices:
        raise RuntimeError(f"Tracking acquisition has no tokens: {sequence.acquisition_group}")
    starts = (
        frame_indices[:1]
        if len(frame_indices) < window_frames
        else frame_indices[: -window_frames + 1]
    )
    raw_windows: list[tuple[int, ...]] = []
    step = max(1, int(round(max_tokens * (1.0 - overlap_fraction))))
    for start in starts:
        allowed_frames = set(range(start, start + window_frames))
        indices = [
            index
            for index, record in enumerate(sequence.records)
            if record.frame_index in allowed_frames
        ]
        if not indices:
            continue
        ordered = _spatial_order(sequence.records, indices)
        if len(ordered) <= max_tokens:
            raw_windows.append(tuple(ordered))
        else:
            for offset in range(0, len(ordered), step):
                chunk = tuple(ordered[offset : offset + max_tokens])
                if chunk:
                    raw_windows.append(chunk)
                if offset + max_tokens >= len(ordered):
                    break

    # Make complete division groups atomic for positive repair.
    children_by_parent: dict[int, set[int]] = defaultdict(set)
    for source, target in sequence.parent_edges:
        children_by_parent[source].add(target)
    positive_groups: list[set[int]] = [
        set(edge) for edge in sequence.continuation_edges + sequence.recovery_edges
    ]
    positive_groups.extend({parent, *children} for parent, children in children_by_parent.items())
    for group in positive_groups:
        if len(group) > max_tokens:
            raise RuntimeError(
                f"A positive tracking lineage group exceeds max_tokens={max_tokens}: "
                f"{sequence.acquisition_group}"
            )
        if any(group <= set(window) for window in raw_windows):
            continue
        center_y = sum(sequence.records[index].time_yx[1] for index in group) / len(group)
        center_x = sum(sequence.records[index].time_yx[2] for index in group) / len(group)
        relevant_frames = {
            sequence.records[index].frame_index for index in group
        }
        candidates = sorted(
            (
                index
                for index, record in enumerate(sequence.records)
                if min(abs(record.frame_index - frame) for frame in relevant_frames)
                < window_frames
            ),
            key=lambda index: (
                0 if index in group else 1,
                (sequence.records[index].time_yx[1] - center_y) ** 2
                + (sequence.records[index].time_yx[2] - center_x) ** 2,
                sequence.records[index].frame_index,
                sequence.records[index].component_id,
            ),
        )
        repair = tuple(candidates[:max_tokens])
        if not group <= set(repair):
            raise RuntimeError("Positive-edge repair window could not retain its lineage group")
        raw_windows.append(repair)

    covered = set(index for window in raw_windows for index in window)
    missing = set(range(len(sequence.records))) - covered
    if missing:
        raise RuntimeError(
            f"Temporal tiling omitted {len(missing)} tokens in {sequence.acquisition_group}"
        )
    for edge in sequence.training_positive_edges:
        if not any(set(edge) <= set(window) for window in raw_windows):
            raise RuntimeError(f"Temporal tiling omitted positive edge {edge}")

    unique: dict[tuple[int, ...], TokenWindow] = {}
    for indices in raw_windows:
        canonical = tuple(sorted(set(indices), key=lambda index: (
            sequence.records[index].frame_index,
            sequence.records[index].component_id,
        )))
        identity = _fingerprint(
            {
                "version": TRACKING_WINDOW_VERSION,
                "acquisition": sequence.acquisition_group,
                "tokens": [sequence.records[index].token_id for index in canonical],
                "max_tokens": max_tokens,
            }
        )
        unique[canonical] = TokenWindow(
            window_id=identity,
            acquisition_group=sequence.acquisition_group,
            role=sequence.role,
            token_indices=canonical,
        )
    return tuple(sorted(unique.values(), key=lambda window: window.window_id))


DEPLOYMENT_PERTURBATION_CODES: Mapping[str, int] = {
    "clean": 0,
    "morphology": 1,
    "merge": 2,
    "split": 3,
    "false_positive": 4,
}


def _apply_deployment_perturbation(
    sequence: SequenceTokens,
    window: TokenWindow,
    *,
    coordinates: np.ndarray,
    features: np.ndarray,
    token_mask: np.ndarray,
    token_count: int,
    maximum: int,
    mean: np.ndarray,
    std: np.ndarray,
    config: TrackingTrainingConfig,
    rng: np.random.Generator,
    force: bool,
) -> tuple[int, set[int], set[int], str]:
    """Inject one image-derived detector error and return masked/dropped token slots.

    Morphology preserves a known identity. Merge/split/false-positive proposals are deliberately
    identity-ambiguous: they participate in transformer context but every dependent supervised
    event/link is masked. This avoids turning a simulated detector failure into false identity
    ground truth.
    """
    probabilities = (
        ("morphology", config.morphology_perturbation_probability),
        ("merge", config.merge_proposal_probability),
        ("split", config.split_proposal_probability),
        ("false_positive", config.false_positive_proposal_probability),
    )
    total_probability = sum(value for _, value in probabilities)
    if total_probability <= 0.0:
        return token_count, set(), set(), "clean"
    draw = float(rng.random())
    if not force and draw >= total_probability:
        return token_count, set(), set(), "clean"
    position = draw * total_probability if force else draw
    cumulative = 0.0
    kind = probabilities[-1][0]
    for candidate, probability in probabilities:
        cumulative += probability
        if position < cumulative:
            kind = candidate
            break

    records = [sequence.records[index] for index in window.token_indices]
    local_by_frame: dict[int, list[int]] = defaultdict(list)
    for local, record in enumerate(records):
        local_by_frame[record.frame_index].append(local)
    frame_by_index = {frame.frame_index: frame for frame in sequence.sequence.frames}

    def context_for(frame_index: int) -> tuple[TrackingFrame, _FrameFeatureContext]:
        frame = frame_by_index[frame_index]
        return frame, _frame_feature_context(frame, sequence.sequence.source_format)

    def standardized(raw_features: tuple[float, ...]) -> np.ndarray:
        return (np.asarray(raw_features, dtype=np.float32) - mean) / std

    def proposal_features(
        frame: TrackingFrame,
        context: _FrameFeatureContext,
        binary: np.ndarray,
        owned_components: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        y_values, x_values = np.nonzero(binary)
        if len(y_values) < 4:
            raise RuntimeError("Deployment proposal contains fewer than four pixels")
        centroid_y_pixels = float(y_values.mean())
        centroid_x_pixels = float(x_values.mean())
        owned = set(owned_components)
        peers = [
            (item.centroid_y / frame.height, item.centroid_x / frame.width)
            for item in frame.instances
            if item.component_id not in owned
        ]
        peers.append(
            (centroid_y_pixels / frame.height, centroid_x_pixels / frame.width)
        )
        raw = _features_for_binary(
            context,
            binary,
            centroid_y=centroid_y_pixels / frame.height,
            centroid_x=centroid_x_pixels / frame.width,
            peer_centroids_yx=np.asarray(peers, dtype=np.float64),
            owned_component_ids=tuple(owned_components),
        )
        coordinate = np.asarray(
            (
                float(frame.frame_index),
                2.0 * centroid_y_pixels / max(1, frame.height - 1) - 1.0,
                2.0 * centroid_x_pixels / max(1, frame.width - 1) - 1.0,
            ),
            dtype=np.float32,
        )
        return coordinate, standardized(raw)

    ambiguous: set[int] = set()
    structurally_dropped: set[int] = set()
    if kind == "morphology":
        candidates = [local for local in range(len(records)) if token_mask[local] > 0]
        if not candidates:
            return token_count, ambiguous, structurally_dropped, "clean"
        local = candidates[int(rng.integers(len(candidates)))]
        record = records[local]
        frame, context = context_for(record.frame_index)
        binary = context.labels == record.component_id
        kernel = np.ones((3, 3), dtype=np.uint8)
        if int(rng.integers(2)) == 0:
            changed = cv2.erode(binary.astype(np.uint8), kernel, iterations=1) > 0
        else:
            changed = cv2.dilate(binary.astype(np.uint8), kernel, iterations=1) > 0
            changed &= (context.labels == 0) | binary
        if int(changed.sum()) < 4:
            return token_count, ambiguous, structurally_dropped, "clean"
        coordinates[local], features[local] = proposal_features(
            frame, context, changed, (record.component_id,)
        )
        return token_count, ambiguous, structurally_dropped, kind

    if kind == "merge":
        possible: list[tuple[float, int, int]] = []
        for locals_in_frame in local_by_frame.values():
            for offset, left in enumerate(locals_in_frame):
                for right in locals_in_frame[offset + 1 :]:
                    distance = float(
                        np.square(coordinates[left, 1:] - coordinates[right, 1:]).sum()
                    )
                    possible.append((distance, left, right))
        if not possible:
            return token_count, ambiguous, structurally_dropped, "clean"
        _, left, right = sorted(possible)[0]
        left_record, right_record = records[left], records[right]
        frame, context = context_for(left_record.frame_index)
        binary = (context.labels == left_record.component_id) | (
            context.labels == right_record.component_id
        )
        p1 = tuple(
            np.rint(
                np.column_stack(np.nonzero(context.labels == left_record.component_id)).mean(
                    axis=0
                )
            ).astype(int)
        )
        p2 = tuple(
            np.rint(
                np.column_stack(np.nonzero(context.labels == right_record.component_id)).mean(
                    axis=0
                )
            ).astype(int)
        )
        bridge = np.zeros_like(binary, dtype=np.uint8)
        cv2.line(bridge, (p1[1], p1[0]), (p2[1], p2[0]), 1, thickness=3)
        binary |= bridge > 0
        coordinates[left], features[left] = proposal_features(
            frame,
            context,
            binary,
            (left_record.component_id, right_record.component_id),
        )
        token_mask[right] = 0.0
        ambiguous.add(left)
        structurally_dropped.add(right)
        return token_count, ambiguous, structurally_dropped, kind

    if kind == "split":
        if token_count >= maximum:
            return token_count, ambiguous, structurally_dropped, "clean"
        candidates = sorted(
            range(len(records)),
            key=lambda local: sequence.records[window.token_indices[local]].features[7],
            reverse=True,
        )
        for local in candidates:
            record = records[local]
            frame, context = context_for(record.frame_index)
            binary = context.labels == record.component_id
            y_values, x_values = np.nonzero(binary)
            if len(y_values) < 12:
                continue
            if np.ptp(x_values) >= np.ptp(y_values):
                cut = float(np.median(x_values))
                first = binary & (np.indices(binary.shape)[1] <= cut)
            else:
                cut = float(np.median(y_values))
                first = binary & (np.indices(binary.shape)[0] <= cut)
            second = binary & ~first
            if int(first.sum()) < 4 or int(second.sum()) < 4:
                continue
            added = token_count
            coordinates[local], features[local] = proposal_features(
                frame, context, first, (record.component_id,)
            )
            coordinates[added], features[added] = proposal_features(
                frame, context, second, (record.component_id,)
            )
            token_mask[added] = 1.0
            ambiguous.update((local, added))
            return token_count + 1, ambiguous, structurally_dropped, kind
        return token_count, ambiguous, structurally_dropped, "clean"

    if kind == "false_positive":
        if token_count >= maximum:
            return token_count, ambiguous, structurally_dropped, "clean"
        frame_index = sorted(local_by_frame)[int(rng.integers(len(local_by_frame)))]
        frame, context = context_for(frame_index)
        background = context.labels == 0
        distance = cv2.distanceTransform(background.astype(np.uint8), cv2.DIST_L2, 3)
        y, x = np.unravel_index(int(np.argmax(distance)), distance.shape)
        available_radius = int(distance[y, x]) - 1
        if available_radius < 1:
            return token_count, ambiguous, structurally_dropped, "clean"
        radius = max(1, min(4, available_radius))
        false_mask = np.zeros_like(background, dtype=np.uint8)
        cv2.circle(false_mask, (int(x), int(y)), radius, 1, thickness=-1)
        binary = (false_mask > 0) & background
        if int(binary.sum()) < 4:
            return token_count, ambiguous, structurally_dropped, "clean"
        added = token_count
        coordinates[added], features[added] = proposal_features(
            frame, context, binary, ()
        )
        token_mask[added] = 1.0
        ambiguous.add(added)
        return token_count + 1, ambiguous, structurally_dropped, kind

    raise AssertionError(kind)


class TrackingWindowDataset(Dataset[dict[str, torch.Tensor]]):
    """Materialize fixed tensors while retaining global token/edge metadata."""

    def __init__(
        self,
        sequences: Sequence[SequenceTokens],
        normalization: FeatureNormalization,
        config: TrackingTrainingConfig,
        *,
        role: str,
        augment: bool,
        external_teacher: Mapping[str, Mapping[tuple[str, str], float]] | None = None,
        deterministic_augmentation_seed: int | None = None,
        force_deployment_perturbation: bool = False,
    ) -> None:
        if role not in TRACKING_ROLES:
            raise ValueError(f"Unknown tracking role {role!r}")
        self.sequence_by_group = {
            sequence.acquisition_group: sequence
            for sequence in sequences
            if sequence.role == role
        }
        self.windows = tuple(
            window
            for sequence in self.sequence_by_group.values()
            for window in build_token_windows(
                sequence,
                max_tokens=config.max_tokens,
                window_frames=config.window_frames,
                overlap_fraction=config.window_overlap_fraction,
            )
        )
        if not self.windows:
            raise RuntimeError(f"Tracking role {role!r} produced no temporal windows")
        self.windows = tuple(sorted(self.windows, key=lambda item: item.window_id))
        self.mean = np.asarray(normalization.mean, dtype=np.float32)
        self.std = np.asarray(normalization.standard_deviation, dtype=np.float32)
        self.config = config
        self.role = role
        self.augment = augment
        self.external_teacher = external_teacher or {}
        self.deterministic_augmentation_seed = deterministic_augmentation_seed
        self.force_deployment_perturbation = force_deployment_perturbation
        if force_deployment_perturbation and not augment:
            raise ValueError("Forced deployment perturbation requires augmentation")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        window = self.windows[item]
        sequence = self.sequence_by_group[window.acquisition_group]
        token_count = len(window.token_indices)
        maximum = self.config.max_tokens
        coordinates = np.zeros((maximum, 3), dtype=np.float32)
        features = np.zeros((maximum, CELL_FEATURE_DIM), dtype=np.float32)
        token_mask = np.zeros(maximum, dtype=np.float32)
        association = np.zeros((maximum, maximum), dtype=np.float32)
        association_supervision = np.zeros((maximum, maximum), dtype=np.float32)
        recovery_supervision = np.zeros((maximum, maximum), dtype=np.float32)
        division = np.zeros(maximum, dtype=np.float32)
        division_supervision = np.zeros(maximum, dtype=np.float32)
        birth = np.zeros(maximum, dtype=np.float32)
        birth_supervision = np.zeros(maximum, dtype=np.float32)
        death = np.zeros(maximum, dtype=np.float32)
        death_supervision = np.zeros(maximum, dtype=np.float32)
        teacher = np.full((maximum, maximum), -1.0, dtype=np.float32)
        local_by_global = {
            global_index: local_index
            for local_index, global_index in enumerate(window.token_indices)
        }
        for local_index, global_index in enumerate(window.token_indices):
            record = sequence.records[global_index]
            coordinates[local_index] = record.time_yx
            features[local_index] = (
                np.asarray(record.features, dtype=np.float32) - self.mean
            ) / self.std
            token_mask[local_index] = 1.0
        if self.augment:
            if self.deterministic_augmentation_seed is None:
                augmentation_seed = int(
                    np.random.randint(0, np.iinfo(np.int64).max, dtype=np.int64)
                )
            else:
                augmentation_seed = int.from_bytes(
                    hashlib.sha256(
                        f"{self.deterministic_augmentation_seed}:{window.window_id}".encode()
                    ).digest()[:8],
                    "big",
                )
            rng = np.random.default_rng(augmentation_seed)
            (
                token_count,
                ambiguous_perturbation_slots,
                structurally_dropped_slots,
                perturbation_kind,
            ) = _apply_deployment_perturbation(
                sequence,
                window,
                coordinates=coordinates,
                features=features,
                token_mask=token_mask,
                token_count=token_count,
                maximum=maximum,
                mean=self.mean,
                std=self.std,
                config=self.config,
                rng=rng,
                force=self.force_deployment_perturbation,
            )
        else:
            rng = None
            ambiguous_perturbation_slots = set()
            structurally_dropped_slots = set()
            perturbation_kind = "clean"
        if self.augment:
            assert rng is not None
            features[:token_count] += rng.normal(
                0.0,
                self.config.feature_jitter_std,
                size=features[:token_count].shape,
            ).astype(np.float32)
            coordinates[:token_count, 1:] += rng.normal(
                0.0,
                self.config.coordinate_jitter_std,
                size=coordinates[:token_count, 1:].shape,
            ).astype(np.float32)

        positive_edges = set(sequence.all_edges)
        recovery_edges = set(sequence.recovery_edges)
        window_globals = set(window.token_indices)
        incoming: dict[int, set[int]] = defaultdict(set)
        outgoing: dict[int, set[int]] = defaultdict(set)
        for source, target in positive_edges:
            incoming[target].add(source)
            outgoing[source].add(target)
        parent_children: dict[int, set[int]] = defaultdict(set)
        for source, target in sequence.parent_edges:
            parent_children[source].add(target)
        teacher_targets = self.external_teacher.get(sequence.acquisition_group)
        for source_local, source_global in enumerate(window.token_indices):
            source_record = sequence.records[source_global]
            for target_local, target_global in enumerate(window.token_indices):
                target_record = sequence.records[target_global]
                if (
                    source_record.identity_supervision_available
                    and target_record.identity_supervision_available
                    and target_record.frame_index == source_record.frame_index + 1
                ):
                    association_supervision[source_local, target_local] = 1.0
                    association[source_local, target_local] = float(
                        (source_global, target_global) in positive_edges
                    )
                    if teacher_targets is not None:
                        # The pinned teacher returns only edges above its tiny
                        # sparse threshold. Missing adjacent-frame edges are
                        # therefore supervised as zero, while an absent
                        # acquisition remains -1 (no teacher supervision).
                        teacher[source_local, target_local] = teacher_targets.get(
                            (source_record.token_id, target_record.token_id),
                            0.0,
                        )

            global_incoming = incoming.get(source_global, set())
            included_incoming = global_incoming & window_globals
            # A first observation is censored (entry/start of acquisition), not an observed
            # biological birth. We supervise only known non-births with an annotated predecessor.
            if global_incoming and included_incoming == global_incoming:
                birth_supervision[source_local] = 1.0

            global_outgoing = outgoing.get(source_global, set())
            included_outgoing = global_outgoing & window_globals
            # Likewise, a last observation is censored. Only an annotated successor proves that
            # this token is a non-death example in these sources.
            if global_outgoing and included_outgoing == global_outgoing:
                death_supervision[source_local] = 1.0

            if source_global in sequence.event_supervision_quarantined:
                birth_supervision[source_local] = 0.0
                death_supervision[source_local] = 0.0

            # LiveCellTrack MOT rows contain identities but no parent field.  Absence of a
            # parent link is therefore unknown, not a negative division label.
            if sequence.sequence.source_format != LIVECELLTRACK_SOURCE_FORMAT:
                children = parent_children.get(source_global, set())
                if source_global in sequence.division_supervision_quarantined:
                    division_supervision[source_local] = 0.0
                elif len(children) == 2:
                    if children <= window_globals:
                        division[source_local] = 1.0
                        division_supervision[source_local] = 1.0
                elif not children and global_outgoing and global_outgoing <= window_globals:
                    division_supervision[source_local] = 1.0

        # Merge/split/false-positive proposals have no unique source identity. They remain real
        # input tokens for contextual robustness, but all link/event/teacher labels touching them
        # are unknown and therefore masked. Structural merge removals are absent detections.
        for local_index in sorted(ambiguous_perturbation_slots):
            association_supervision[local_index, :] = 0.0
            association_supervision[:, local_index] = 0.0
            division_supervision[local_index] = 0.0
            birth_supervision[local_index] = 0.0
            death_supervision[local_index] = 0.0
            teacher[local_index, :] = -1.0
            teacher[:, local_index] = -1.0
        for local_index in sorted(structurally_dropped_slots):
            token_mask[local_index] = 0.0
            association_supervision[local_index, :] = 0.0
            association_supervision[:, local_index] = 0.0
            division_supervision[local_index] = 0.0
            birth_supervision[local_index] = 0.0
            death_supervision[local_index] = 0.0
            teacher[local_index, :] = -1.0
            teacher[:, local_index] = -1.0

        # Detection-noise augmentation is applied after labels are built. Dropped tokens and every
        # dependent edge/event target are masked, so a simulated missed detection never creates a
        # contradictory positive or negative label.
        dropped = np.zeros(token_count, dtype=bool)
        if self.augment and token_count > 2:
            assert rng is not None
            if self.config.token_dropout_probability > 0:
                dropped = rng.random(token_count) < self.config.token_dropout_probability
                if structurally_dropped_slots:
                    dropped[list(structurally_dropped_slots)] = True
            recoverable: list[tuple[int, int, int]] = []
            for source_global, target_global in sorted(recovery_edges):
                if source_global not in local_by_global or target_global not in local_by_global:
                    continue
                source = sequence.records[source_global]
                intermediate = [
                    global_index
                    for global_index in window.token_indices
                    if sequence.records[global_index].track_id == source.track_id
                    and sequence.records[global_index].frame_index == source.frame_index + 1
                    and sequence.records[global_index].identity_supervision_available
                ]
                if len(intermediate) == 1:
                    recoverable.append((source_global, intermediate[0], target_global))
            if recoverable and rng.random() < self.config.forced_gap_dropout_probability:
                source_global, middle_global, target_global = recoverable[
                    int(rng.integers(len(recoverable)))
                ]
                dropped[local_by_global[source_global]] = False
                dropped[local_by_global[target_global]] = False
                dropped[local_by_global[middle_global]] = True
            if dropped.all() or int((~dropped).sum()) < 2:
                dropped[:] = False
            for local_index in np.flatnonzero(dropped):
                token_mask[local_index] = 0.0
                association_supervision[local_index, :] = 0.0
                association_supervision[:, local_index] = 0.0
                division_supervision[local_index] = 0.0
                birth_supervision[local_index] = 0.0
                death_supervision[local_index] = 0.0
                teacher[local_index, :] = -1.0
                teacher[:, local_index] = -1.0

        # Activate a two-frame recovery row only when the known same-ID midpoint is absent
        # (natural annotation gap or simulated detector dropout). The positive can only be the
        # explicitly same identity; retained alternatives in the destination frame are negatives.
        for source_global, target_global in sorted(recovery_edges):
            if source_global not in local_by_global or target_global not in local_by_global:
                continue
            source_local = local_by_global[source_global]
            target_local = local_by_global[target_global]
            if token_mask[source_local] == 0.0 or token_mask[target_local] == 0.0:
                continue
            if (
                source_local in ambiguous_perturbation_slots
                or target_local in ambiguous_perturbation_slots
            ):
                continue
            source_record = sequence.records[source_global]
            target_record = sequence.records[target_global]
            midpoint_retained = any(
                token_mask[local] > 0.0
                and sequence.records[global_index].track_id == source_record.track_id
                and sequence.records[global_index].frame_index == source_record.frame_index + 1
                and sequence.records[global_index].identity_supervision_available
                for local, global_index in enumerate(window.token_indices)
            )
            adjacent_positive_retained = any(
                edge_source == source_global
                and edge_target in local_by_global
                and token_mask[local_by_global[edge_target]] > 0.0
                for edge_source, edge_target in sequence.all_edges
            )
            if midpoint_retained or adjacent_positive_retained:
                continue
            for candidate_local, candidate_global in enumerate(window.token_indices):
                candidate = sequence.records[candidate_global]
                if (
                    token_mask[candidate_local] > 0.0
                    and candidate_local not in ambiguous_perturbation_slots
                    and candidate.identity_supervision_available
                    and candidate.frame_index == target_record.frame_index
                ):
                    association_supervision[source_local, candidate_local] = 1.0
                    recovery_supervision[source_local, candidate_local] = 1.0
                    association[source_local, candidate_local] = float(
                        candidate_global == target_global
                    )
                    # Trackastra is intentionally adjacent-only; gap cells remain -1.
                    teacher[source_local, candidate_local] = -1.0

        return {
            "time_yx": torch.from_numpy(coordinates),
            "cell_features": torch.from_numpy(features),
            "cell_mask": torch.from_numpy(token_mask),
            "association_target": torch.from_numpy(association),
            "association_supervision": torch.from_numpy(association_supervision),
            "recovery_supervision": torch.from_numpy(recovery_supervision),
            "division_target": torch.from_numpy(division),
            "division_supervision": torch.from_numpy(division_supervision),
            "birth_target": torch.from_numpy(birth),
            "birth_supervision": torch.from_numpy(birth_supervision),
            "death_target": torch.from_numpy(death),
            "death_supervision": torch.from_numpy(death_supervision),
            "external_teacher_target": torch.from_numpy(teacher),
            "deployment_perturbation_code": torch.as_tensor(
                DEPLOYMENT_PERTURBATION_CODES[perturbation_kind], dtype=torch.int64
            ),
            "identity_ambiguous_token_mask": torch.from_numpy(
                np.asarray(
                    [index in ambiguous_perturbation_slots for index in range(maximum)],
                    dtype=np.float32,
                )
            ),
        }


class HierarchicalBalancedSampler(Sampler[int]):
    """Deterministically balance modality -> dataset -> acquisition -> window."""

    def __init__(
        self,
        dataset: TrackingWindowDataset,
        *,
        steps_per_epoch: int,
        batch_size: int,
        seed: int,
    ) -> None:
        if steps_per_epoch < 1 or batch_size < 1:
            raise ValueError("Balanced sampler needs positive steps and batch size")
        grouped: dict[str, dict[str, dict[str, list[int]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        for index, window in enumerate(dataset.windows):
            sequence = dataset.sequence_by_group[window.acquisition_group].sequence
            grouped[sequence.modality][sequence.dataset][window.acquisition_group].append(index)
        if not grouped:
            raise RuntimeError("Balanced tracking sampler received no windows")
        self.grouped = {
            modality: {
                source: {
                    acquisition: tuple(sorted(indices))
                    for acquisition, indices in acquisitions.items()
                }
                for source, acquisitions in datasets.items()
            }
            for modality, datasets in grouped.items()
        }
        self.steps_per_epoch = steps_per_epoch
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.policy_fingerprint_sha256 = _fingerprint(
            {
                "version": TRACKING_SAMPLER_VERSION,
                "steps_per_epoch": steps_per_epoch,
                "batch_size": batch_size,
                "seed": seed,
                "hierarchy": {
                    modality: {
                        source: {
                            acquisition: len(indices)
                            for acquisition, indices in acquisitions.items()
                        }
                        for source, acquisitions in datasets.items()
                    }
                    for modality, datasets in self.grouped.items()
                },
            }
        )

    def __len__(self) -> int:
        return self.steps_per_epoch * self.batch_size

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("Sampler epoch cannot be negative")
        self.epoch = epoch

    def state_dict(self) -> dict[str, object]:
        return {
            "version": TRACKING_SAMPLER_VERSION,
            "epoch": self.epoch,
            "policy_fingerprint_sha256": self.policy_fingerprint_sha256,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if (
            state.get("version") != TRACKING_SAMPLER_VERSION
            or state.get("policy_fingerprint_sha256")
            != self.policy_fingerprint_sha256
        ):
            raise RuntimeError("Balanced tracking sampler resume contract changed")
        self.set_epoch(int(state["epoch"]))

    def __iter__(self):
        rng = random.Random(
            int.from_bytes(
                hashlib.sha256(
                    f"{self.seed}:{self.epoch}:{self.policy_fingerprint_sha256}".encode()
                ).digest()[:8],
                "big",
            )
        )

        def shuffled(values: Sequence[Any]) -> list[Any]:
            result = sorted(values, key=str)
            rng.shuffle(result)
            return result

        modalities = shuffled(tuple(self.grouped))
        datasets = {
            modality: shuffled(tuple(self.grouped[modality])) for modality in modalities
        }
        acquisitions = {
            (modality, source): shuffled(tuple(self.grouped[modality][source]))
            for modality in modalities
            for source in datasets[modality]
        }
        windows = {
            (modality, source, acquisition): shuffled(
                self.grouped[modality][source][acquisition]
            )
            for modality in modalities
            for source in datasets[modality]
            for acquisition in acquisitions[(modality, source)]
        }
        modality_count = Counter()
        dataset_count = Counter()
        acquisition_count = Counter()
        for sample_index in range(len(self)):
            modality = modalities[sample_index % len(modalities)]
            source_options = datasets[modality]
            source = source_options[modality_count[modality] % len(source_options)]
            modality_count[modality] += 1
            acquisition_options = acquisitions[(modality, source)]
            acquisition = acquisition_options[
                dataset_count[(modality, source)] % len(acquisition_options)
            ]
            dataset_count[(modality, source)] += 1
            window_options = windows[(modality, source, acquisition)]
            window = window_options[
                acquisition_count[(modality, source, acquisition)] % len(window_options)
            ]
            acquisition_count[(modality, source, acquisition)] += 1
            yield window


def _move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _supervised_losses(output: Any, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    assignment = differentiable_assignment_loss(
        output,
        batch["association_target"],
        supervision_mask=batch["association_supervision"],
    )
    division = differentiable_division_loss(
        output,
        batch["division_target"],
        supervision_mask=batch["division_supervision"],
        association_supervision_mask=batch["association_supervision"],
    )
    birth = _masked_mean(
        F.binary_cross_entropy_with_logits(
            output.birth_logits, batch["birth_target"], reduction="none"
        ),
        batch["birth_supervision"] * output.token_mask,
    )
    death = _masked_mean(
        F.binary_cross_entropy_with_logits(
            output.death_logits, batch["death_target"], reduction="none"
        ),
        batch["death_supervision"] * output.token_mask,
    )
    link_error = (
        (
            torch.sigmoid(output.association_logits)
            - batch["association_target"]
        ).abs()
        * output.association_mask
        * batch["association_supervision"]
    ).sum(dim=-1)
    link_denominator = (
        output.association_mask * (batch["association_supervision"] > 0)
    ).sum(dim=-1).clamp_min(1)
    link_error = link_error / link_denominator
    event_error = (
        (output.division_probability - batch["division_target"]).abs()
        * batch["division_supervision"]
        + (output.birth_probability - batch["birth_target"]).abs()
        * batch["birth_supervision"]
        + (output.death_probability - batch["death_target"]).abs()
        * batch["death_supervision"]
    ) / (
        batch["division_supervision"]
        + batch["birth_supervision"]
        + batch["death_supervision"]
    ).clamp_min(1.0)
    uncertainty_target = ((link_error + event_error) / 2.0).detach()
    uncertainty = _masked_mean(
        F.binary_cross_entropy_with_logits(
            output.uncertainty_logits, uncertainty_target, reduction="none"
        ),
        output.token_mask,
    )
    total = assignment + division + birth + death + 0.25 * uncertainty
    return {
        "total": total,
        "assignment": assignment,
        "division": division,
        "birth": birth,
        "death": death,
        "uncertainty": uncertainty,
    }


def _external_teacher_loss(output: Any, target: torch.Tensor) -> torch.Tensor | None:
    valid = (target >= 0.0) & output.association_mask
    if not bool(valid.any()):
        return None
    return _masked_mean(
        F.binary_cross_entropy_with_logits(
            output.association_logits,
            target.clamp(0.0, 1.0),
            reduction="none",
        ),
        valid,
    )


class _QuietProgress:
    def __init__(self, iterable: Any = None, **_: Any) -> None:
        self.iterable = () if iterable is None else iterable

    def __iter__(self):
        return iter(self.iterable)


def _trackastra_sparse_weights(
    weights: Any,
    node_count: int,
) -> dict[tuple[int, int], float]:
    """Normalize the exact Trackastra 0.3 association contract without densifying it.

    ``predict_windows`` in the pinned commit returns ``(((source, target),
    probability), ...)``.  Accept scipy sparse arrays as a defensive compatibility
    path, but reject malformed, duplicate, out-of-range, or non-finite edges rather
    than silently changing teacher supervision.  A sequence can contain many
    thousands of detections, so constructing an N-by-N dense teacher matrix here
    would be both unnecessary and unsafe.
    """

    if node_count < 0:
        raise ValueError("node_count cannot be negative")
    if hasattr(weights, "toarray"):
        sparse = weights.tocoo()
        raw_edges = zip(sparse.row, sparse.col, sparse.data, strict=True)
    elif isinstance(weights, np.ndarray):
        dense = np.asarray(weights, dtype=np.float32)
        if dense.shape != (node_count, node_count):
            raise RuntimeError(
                "Trackastra node/association matrix shapes disagree: "
                f"{dense.shape} versus {(node_count, node_count)}"
            )
        rows, columns = np.nonzero(dense)
        raw_edges = zip(rows, columns, dense[rows, columns], strict=True)
    else:
        try:
            raw_edges = (
                (item[0][0], item[0][1], item[1])
                for item in iter(weights)
            )
        except TypeError as error:
            raise RuntimeError("Trackastra weights are not iterable") from error
    result: dict[tuple[int, int], float] = {}
    try:
        for edge_index, (raw_source, raw_target, raw_probability) in enumerate(raw_edges):
            source = int(raw_source)
            target = int(raw_target)
            probability = float(raw_probability)
            if not (0 <= source < node_count and 0 <= target < node_count):
                raise RuntimeError(
                    f"Trackastra association edge is outside 0..{node_count - 1}: "
                    f"{(source, target)!r}"
                )
            if (source, target) in result:
                raise RuntimeError(
                    f"Trackastra returned a duplicate association edge: {(source, target)!r}"
                )
            if not math.isfinite(probability):
                raise RuntimeError("Trackastra returned a non-finite association probability")
            result[(source, target)] = probability
    except (IndexError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"Malformed Trackastra association edge at index {locals().get('edge_index', 0)}"
        ) from error
    return result


def _trackastra_targets_for_sequence(
    sequence: SequenceTokens,
    teacher: Any,
) -> dict[tuple[str, str], float]:
    """Run adjacent-only Trackastra inference in 64-frame chunks with one-frame overlap."""
    frames = sequence.sequence.frames
    if not frames or any(
        (frame.height, frame.width) != (frames[0].height, frames[0].width)
        for frame in frames
    ):
        raise RuntimeError("Trackastra teacher requires a shape-consistent image sequence")
    if not hasattr(teacher, "_predict"):
        raise RuntimeError("Pinned Trackastra teacher has no expected _predict inference hook")
    records_by_frame: dict[int, list[TokenRecord]] = defaultdict(list)
    for record in sequence.records:
        records_by_frame[record.frame_index].append(record)
    targets: dict[tuple[str, str], float] = {}
    start = 0
    while start < len(frames):
        stop = min(start + 64, len(frames))
        chunk_frames = frames[start:stop]
        images = [_read_image(frame.image_path) for frame in chunk_frames]
        masks = [
            _read_instance_labels(frame, sequence.sequence.source_format)
            for frame in chunk_frames
        ]
        predictions = teacher._predict(  # noqa: SLF001 - pinned 0.3 contract
            np.stack(images),
            np.stack(masks),
            edge_threshold=1e-5,
            n_workers=0,
            progbar_class=_QuietProgress,
        )
        del images, masks
        if not isinstance(predictions, Mapping) or not {"nodes", "weights"} <= set(predictions):
            raise RuntimeError("Trackastra general_2d returned an unknown prediction contract")
        nodes = predictions["nodes"]
        sparse_weights = _trackastra_sparse_weights(predictions["weights"], len(nodes))
        token_by_node: list[str | None] = []
        global_position_by_node: list[int] = []
        for node in nodes:
            if not isinstance(node, Mapping):
                raise RuntimeError("Trackastra node metadata is not a mapping")
            local_time = int(node["time"])
            if not 0 <= local_time < len(chunk_frames):
                raise RuntimeError("Trackastra node time falls outside its inference chunk")
            global_position = start + local_time
            frame_index = frames[global_position].frame_index
            token = next(
                (
                    record.token_id
                    for record in records_by_frame[frame_index]
                    if record.component_id == int(node["label"])
                ),
                None,
            )
            token_by_node.append(token)
            global_position_by_node.append(global_position)
        for (source_index, target_index), raw_value in sparse_weights.items():
            source = token_by_node[source_index]
            target = token_by_node[target_index]
            if source is None or target is None:
                continue
            if global_position_by_node[target_index] != global_position_by_node[source_index] + 1:
                continue
            key = (source, target)
            value = float(np.clip(raw_value, 0.0, 1.0))
            if key in targets and not math.isclose(targets[key], value, abs_tol=1e-7):
                raise RuntimeError(f"Trackastra overlapping chunks disagree for edge {key}")
            targets[key] = value
        if stop == len(frames):
            break
        start = stop - 1
    return targets


def prepare_trackastra_teacher_targets(
    sequences: Sequence[SequenceTokens],
    teacher_directory: Path | None,
    *,
    device: torch.device,
    cache_directory: Path | None = None,
) -> tuple[dict[str, dict[tuple[str, str], float]], dict[str, object]]:
    artifact = teacher_artifact("general_2d")
    inherited_knowledge = {
        "dataset": "Ker et al. phase-contrast tracking dataset",
        "doi": "10.17605/OSF.IO/YSAQ2",
        "reported_scale": "49,919 frames across 48 sequences",
        "annotation": "human XML centroid/lineage tracks; sparse centroids, not full masks",
        "use": "association knowledge inherited only through Trackastra general_2d",
    }
    if teacher_directory is None:
        return {}, {
            "status": "not_available",
            "reason": "No verified local general_2d directory was supplied; no download attempted.",
            "artifact": asdict(artifact),
            "inherited_training_data": inherited_knowledge,
            "association_knowledge_inherited_this_run": False,
        }
    directory = teacher_directory.expanduser().resolve()
    if not directory.is_dir():
        raise FileNotFoundError(f"Configured Trackastra teacher directory is absent: {directory}")
    teacher = load_trackastra_teacher(directory, device=device.type)
    targets: dict[str, dict[tuple[str, str], float]] = {}
    cache = cache_directory.expanduser().resolve() if cache_directory is not None else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
    marker_path = directory / ".cellect_teacher_artifact.json"
    teacher_identity = (
        file_sha256(marker_path)
        if marker_path.is_file()
        else _fingerprint(asdict(artifact))
    )
    generated = 0
    reused = 0
    for sequence in sequences:
        if sequence.role == "train":
            cache_key = _fingerprint(
                {
                    "contract": TRACKASTRA_CHUNK_VERSION,
                    "sequence": sequence.sequence.sequence_fingerprint_sha256,
                    "teacher": teacher_identity,
                    "package_version": artifact.package_version,
                }
            )
            array_path = cache / f"{cache_key}.npz" if cache is not None else None
            manifest_path = cache / f"{cache_key}.json" if cache is not None else None
            if array_path is not None and manifest_path is not None and (
                array_path.exists() or manifest_path.exists()
            ):
                if not array_path.is_file() or not manifest_path.is_file():
                    raise RuntimeError("Trackastra target cache entry is incomplete")
                manifest = json.loads(manifest_path.read_text())
                if (
                    manifest.get("cache_key") != cache_key
                    or manifest.get("sequence_fingerprint_sha256")
                    != sequence.sequence.sequence_fingerprint_sha256
                    or manifest.get("npz_sha256") != file_sha256(array_path)
                ):
                    raise RuntimeError("Trackastra target cache failed identity/hash validation")
                with np.load(array_path, allow_pickle=False) as arrays:
                    sources = arrays["source"]
                    destinations = arrays["target"]
                    probabilities = arrays["probability"]
                if not (len(sources) == len(destinations) == len(probabilities)):
                    raise RuntimeError("Trackastra target cache arrays disagree")
                sequence_targets = {
                    (str(source), str(target)): float(probability)
                    for source, target, probability in zip(
                        sources, destinations, probabilities, strict=True
                    )
                }
                reused += 1
            else:
                sequence_targets = _trackastra_targets_for_sequence(sequence, teacher)
                sequence_targets = {
                    key: float(np.float32(value))
                    for key, value in sequence_targets.items()
                }
                generated += 1
                if array_path is not None and manifest_path is not None:
                    ordered = sorted(sequence_targets.items())
                    temporary = array_path.with_name(f".{array_path.name}.{os.getpid()}.tmp")
                    with temporary.open("wb") as handle:
                        np.savez_compressed(
                            handle,
                            source=np.asarray([pair[0][0] for pair in ordered], dtype=np.str_),
                            target=np.asarray([pair[0][1] for pair in ordered], dtype=np.str_),
                            probability=np.asarray([pair[1] for pair in ordered], dtype=np.float32),
                        )
                        handle.flush()
                        os.fsync(handle.fileno())
                    os.replace(temporary, array_path)
                    _atomic_json(
                        manifest_path,
                        {
                            "cache_key": cache_key,
                            "sequence_fingerprint_sha256": (
                                sequence.sequence.sequence_fingerprint_sha256
                            ),
                            "teacher_identity_sha256": teacher_identity,
                            "npz_sha256": file_sha256(array_path),
                            "association_targets": len(sequence_targets),
                        },
                    )
            targets[sequence.acquisition_group] = sequence_targets
    target_digest = hashlib.sha256()
    for acquisition_group in sorted(targets):
        target_digest.update(acquisition_group.encode("utf-8"))
        for (source, destination), probability in sorted(targets[acquisition_group].items()):
            target_digest.update(source.encode("utf-8"))
            target_digest.update(b"\0")
            target_digest.update(destination.encode("utf-8"))
            target_digest.update(np.float32(probability).tobytes())
    return targets, {
        "status": "used",
        "directory": str(directory),
        "artifact": asdict(artifact),
        "train_acquisitions_distilled": len(targets),
        "association_targets": sum(len(value) for value in targets.values()),
        "association_target_fingerprint_sha256": target_digest.hexdigest(),
        "target_cache": {
            "directory": str(cache) if cache is not None else None,
            "generated_acquisitions": generated,
            "reused_acquisitions": reused,
            "contract": TRACKASTRA_CHUNK_VERSION,
            "maximum_frames_per_inference": 64,
            "overlap_frames": 1,
        },
        "inherited_training_data": inherited_knowledge,
        "association_knowledge_inherited_this_run": True,
        "event_heads_distilled": False,
    }


def _ker_dataset_status(ker_root: Path | None) -> dict[str, object]:
    if ker_root is None:
        return {
            "status": "not_configured",
            "environment_variable": "CELLECT_KER_TRACKING_ROOT",
            "direct_supervision": False,
            "reason": (
                "The optional Ker source is centroid/XML lineage annotation rather than dense "
                "instance masks; association knowledge can be inherited through general_2d."
            ),
        }
    root = ker_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"CELLECT_KER_TRACKING_ROOT does not exist: {root}")
    return {
        "status": "explicitly_skipped",
        "root": str(root),
        "direct_supervision": False,
        "reason": (
            "No strict human-centroid/XML adapter is implemented in v4. Dense masks are absent, "
            "so the files are not converted or mixed with mask-derived tokens. Computer-generated "
            "tracks are never accepted as ground truth."
        ),
    }


def resolve_tracking_roots(
    data_root: Path,
    *,
    deepsea_root: Path | None = None,
    ctc_roots: Mapping[str, Path] | None = None,
    livecelltrack_root: Path | None = None,
    ctmc_root: Path | None = None,
    alfi_root: Path | None = None,
) -> tuple[Path, dict[str, Path], Path | None, Path | None, Path | None]:
    """Resolve already-local sources only; this function performs no downloads.

    A source named in ``CELLECT_OMIT_TRACKING_SOURCES`` resolves to ``None`` instead of being
    searched for, so an unreachable publisher can be excluded deliberately rather than by an
    accidental empty directory.
    """
    omitted = resolve_omitted_tracking_sources()
    data_root = data_root.expanduser().resolve()
    if deepsea_root is None:
        environment = os.environ.get("CELLECT_DEEPSEA_ROOT")
        if environment:
            deepsea_root = Path(environment)
        else:
            deepsea_root = data_root / "external" / "deepsea_phase"
    deepsea_root = deepsea_root.expanduser().resolve()
    if not deepsea_root.exists():
        raise FileNotFoundError(
            f"DeepSea tracking root was not found at {deepsea_root}; set CELLECT_DEEPSEA_ROOT"
        )

    resolved: dict[str, Path] = {}
    supplied = dict(ctc_roots or {})
    unknown = sorted(set(supplied) - set(CTC_TRACKING_DATASETS))
    if unknown:
        raise ValueError(f"Unknown CTC root keys: {unknown}")
    missing: list[str] = []
    for dataset in CTC_TRACKING_DATASETS:
        root = supplied.get(dataset)
        if root is None:
            environment = os.environ.get(CTC_ENVIRONMENT_NAMES[dataset])
            root = Path(environment) if environment else data_root / "external" / dataset
        root = root.expanduser().resolve()
        if not root.exists():
            missing.append(f"{dataset} ({root})")
        else:
            resolved[dataset] = root
    if missing:
        raise FileNotFoundError(
            "All five extracted transmitted-light CTC training roots are required; missing: "
            + ", ".join(missing)
        )
    if "livecelltrack_preview" in omitted:
        livecelltrack_root = None
    else:
        if livecelltrack_root is None:
            environment = os.environ.get(LIVECELLTRACK_ENVIRONMENT_NAME)
            livecelltrack_root = (
                Path(environment)
                if environment
                else data_root / "external" / "livecelltrack_preview"
            )
        livecelltrack_root = livecelltrack_root.expanduser().resolve()
        if not livecelltrack_root.exists():
            raise FileNotFoundError(
                f"LiveCellTrack preview root was not found at {livecelltrack_root}; run the "
                f"pipeline downloader or set {LIVECELLTRACK_ENVIRONMENT_NAME}"
            )
    if "ctmc_v1" in omitted:
        ctmc_root = None
    else:
        if ctmc_root is None:
            environment = os.environ.get("CELLECT_CTMC_ROOT")
            ctmc_root = (
                Path(environment) if environment else data_root / "external" / "ctmc_v1"
            )
        ctmc_root = ctmc_root.expanduser().resolve()
        if ctmc_root.is_file() and ctmc_root.suffix.casefold() == ".zip":
            previous = os.environ.get("CELLECT_CTMC_ROOT")
            try:
                os.environ["CELLECT_CTMC_ROOT"] = str(ctmc_root)
                ctmc_root = prepare_ctmc_v1(data_root, allow_network=False)
            finally:
                if previous is None:
                    os.environ.pop("CELLECT_CTMC_ROOT", None)
                else:
                    os.environ["CELLECT_CTMC_ROOT"] = previous
        if not ctmc_root.exists():
            raise FileNotFoundError(
                f"CTMC-v1 train root was not found at {ctmc_root}; set CELLECT_CTMC_ROOT"
            )
    if "alfi_task1" in omitted:
        alfi_root = None
    else:
        if alfi_root is None:
            environment = os.environ.get("CELLECT_ALFI_ROOT")
            alfi_root = (
                Path(environment) if environment else data_root / "external" / "alfi_task1"
            )
        alfi_root = alfi_root.expanduser().resolve()
        if alfi_root.is_file() and alfi_root.suffix.casefold() == ".zip":
            previous = os.environ.get("CELLECT_ALFI_ROOT")
            try:
                os.environ["CELLECT_ALFI_ROOT"] = str(alfi_root)
                alfi_root = prepare_alfi_task1(data_root, allow_network=False)
            finally:
                if previous is None:
                    os.environ.pop("CELLECT_ALFI_ROOT", None)
                else:
                    os.environ["CELLECT_ALFI_ROOT"] = previous
        if not alfi_root.exists():
            raise FileNotFoundError(
                f"ALFI Task-1 root was not found at {alfi_root}; set CELLECT_ALFI_ROOT"
            )
    return deepsea_root, resolved, livecelltrack_root, ctmc_root, alfi_root


def _preflight_sequences(
    sequences: Sequence[TrackingSequence],
    config: TrackingTrainingConfig,
    *,
    code_fingerprint: str,
    ker_status: Mapping[str, object],
) -> dict[str, object]:
    if not sequences:
        raise RuntimeError("No strict transmitted-light tracking acquisitions were discovered")
    roles = {role: 0 for role in TRACKING_ROLES}
    for sequence in sequences:
        roles[sequence.role] += 1
        if "test" in sequence.source_partition.casefold():
            raise RuntimeError(
                f"Tracking source partition is not development-only: {sequence.source_partition}"
            )
    empty = [role for role, count in roles.items() if count == 0]
    if empty:
        raise RuntimeError(
            "Acquisition-level role hashing produced empty scientific roles "
            f"{empty}; add acquisitions or revise a preregistered split before training"
        )
    summary = tracking_summary(sequences)
    if summary.get("official_test_labels_parsed") is not False:
        raise RuntimeError("Tracking source adapter did not preserve the official-test seal")
    ctmc_roles_by_cell_line: dict[str, set[str]] = defaultdict(set)
    for sequence in sequences:
        if sequence.source_format == CTMC_V1_SOURCE_FORMAT:
            ctmc_roles_by_cell_line[sequence.sequence_id.rsplit("-", 1)[0]].add(
                sequence.role
            )
    crossed = {
        cell_line: sorted(values)
        for cell_line, values in ctmc_roles_by_cell_line.items()
        if len(values) != 1
    }
    if crossed:
        raise RuntimeError(f"CTMC-v1 cell lines crossed scientific roles: {crossed}")
    data_fingerprint = _fingerprint(
        {
            "sequences": [
                {
                    "group": sequence.acquisition_group,
                    "role": sequence.role,
                    "fingerprint": sequence.sequence_fingerprint_sha256,
                    "source_partition": sequence.source_partition,
                }
                for sequence in sorted(sequences, key=lambda item: item.acquisition_group)
            ],
            "feature_version": TRACKING_FEATURE_VERSION,
            "window_version": TRACKING_WINDOW_VERSION,
            "sampler_version": TRACKING_SAMPLER_VERSION,
        }
    )
    runtime_contract = {
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "tifffile": tifffile.__version__,
    }
    stage_fingerprint = _fingerprint(
        {
            "run_version": TRACKING_RUN_VERSION,
            "data": data_fingerprint,
            "code": code_fingerprint,
            "config": asdict(config),
            "model_contract": CELLECT_TRACK_VERSION,
            "runtime": runtime_contract,
        }
    )
    return {
        "schema_version": TRACKING_TRAINING_SCHEMA_VERSION,
        "status": "PASS",
        "summary": summary,
        "roles": roles,
        "dataset_fingerprint_sha256": data_fingerprint,
        "code_fingerprint_sha256": code_fingerprint,
        "stage_fingerprint_sha256": stage_fingerprint,
        "runtime_contract": runtime_contract,
        "official_test_labels_parsed": False,
        "source_policy": (
            "strict DeepSea train, CTC training *_GT/TRA, LiveCellTrack human MOT gt.txt, "
            "CTMC-v1 train-only MOT+TRA, and ALFI MI01-MI08 DTLTruth; no hidden test labels, "
            "ALFI Task2, or ALFI semantic masks used as instance truth"
        ),
        "sampler": {
            "version": TRACKING_SAMPLER_VERSION,
            "hierarchy": "modality -> dataset -> acquisition -> window",
            "optimizer_steps_per_epoch": config.optimizer_steps_per_epoch,
            "checkpoint_steps_per_epoch": config.checkpoint_steps_per_epoch,
            "fixed_steps": True,
        },
        "gap_recovery": {
            "maximum_frame_gap": config.maximum_frame_gap,
            "positive_contract": "same explicitly known identity only",
            "teacher": "Trackastra adjacent-only",
        },
        "oof_cellect_token_cache_contract": {
            "version": OOF_CELLECT_TOKEN_CACHE_VERSION,
            "preferred_for_full_training": True,
            "status": "producer_not_yet_available_in_pretraining_pipeline",
            "required_manifest_fields": [
                "contract_version",
                "segmentation_run_fingerprint",
                "fold_assignment_fingerprint",
                "role_safe",
                "acquisitions",
            ],
            "per_acquisition_fields": [
                "tracking_sequence_fingerprint",
                "source_role",
                "held_out_segmentation_fold",
                "token_npz_path",
                "token_npz_sha256",
                "ground_truth_matching_policy",
            ],
            "fail_closed_rules": (
                "every acquisition must be inferred by a segmentation fold that did not train "
                "on that acquisition; cache roles/fingerprints/hashes must match; unmatched "
                "proposals remain explicit false positives and ambiguous merge/split matches are "
                "label-masked"
            ),
            "fallback_used_this_run": (
                "controlled image/mask-derived detector perturbations with a separately reported "
                "clean-versus-perturbed checkpoint ablation"
            ),
        },
        "scientific_grouping": {
            "ctmc_v1": "all runs sharing a cell-line prefix have one role",
            "ctmc_cell_lines": {
                key: next(iter(value)) for key, value in sorted(ctmc_roles_by_cell_line.items())
            },
            "alfi": (
                "whole-sequence roles; only three cell lines, so cell-line overlap across four "
                "roles is unavoidable and explicitly reported"
            ),
        },
        "ker_dataset": dict(ker_status),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _capture_rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def _restore_rng_state(state: Mapping[str, object]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise RuntimeError("Tracking checkpoint RNG state is incomplete")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = state["torch_cuda"]
    if torch.cuda.is_available():
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("Tracking checkpoint CUDA RNG device count changed")
        torch.cuda.set_rng_state_all(cuda_states)
    elif cuda_states:
        raise RuntimeError("CUDA RNG state exists but CUDA is unavailable")


def _atomic_checkpoint(path: Path, payload: Mapping[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(dict(payload), temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    digest = file_sha256(temporary)
    os.replace(temporary, path)
    _atomic_json(path.with_suffix(path.suffix + ".sha256.json"), {"sha256": digest})
    return digest


def _load_checkpoint(path: Path, expected_fingerprint: str) -> dict[str, object]:
    sidecar = path.with_suffix(path.suffix + ".sha256.json")
    if not path.is_file() or not sidecar.is_file():
        raise RuntimeError(f"Resume checkpoint or digest sidecar is incomplete: {path}")
    expected_digest = json.loads(sidecar.read_text()).get("sha256")
    observed_digest = file_sha256(path)
    if expected_digest != observed_digest:
        raise RuntimeError(f"Resume checkpoint SHA-256 mismatch: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"Resume checkpoint payload is invalid: {path}")
    if payload.get("stage_fingerprint") != expected_fingerprint:
        raise RuntimeError(f"Refusing incompatible tracking checkpoint resume: {path}")
    return payload


def _phase_epochs(config: TrackingTrainingConfig, phase: str) -> int:
    return {
        "heads": config.heads_epochs,
        "adapters": config.adapters_epochs,
        "full": config.full_epochs,
        "distillation": config.distillation_epochs,
    }[phase]


def _phase_learning_rate(config: TrackingTrainingConfig, phase: str) -> float:
    return {
        "heads": config.heads_learning_rate,
        "adapters": config.adapters_learning_rate,
        "full": config.full_learning_rate,
        "distillation": config.distillation_learning_rate,
    }[phase]


def _combined_loss(
    model: CellectTrack,
    batch: Mapping[str, torch.Tensor],
    *,
    internal_teacher: CellectTrack | None,
    external_teacher_weight: float,
    internal_teacher_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = model(batch["time_yx"], batch["cell_features"], batch["cell_mask"])
    supervised = _supervised_losses(output, batch)
    total = supervised["total"]
    components = {name: float(value.detach().cpu()) for name, value in supervised.items()}
    external = _external_teacher_loss(output, batch["external_teacher_target"])
    if external is not None and external_teacher_weight > 0:
        total = total + external_teacher_weight * external
        components["external_teacher"] = float(external.detach().cpu())
    if internal_teacher is not None and internal_teacher_weight > 0:
        with torch.no_grad():
            teacher_output = internal_teacher(
                batch["time_yx"], batch["cell_features"], batch["cell_mask"]
            )
        distillation = tracking_distillation_losses(output, teacher_output)
        total = total + internal_teacher_weight * distillation["total"]
        components.update(
            {
                f"internal_{name}": float(value.detach().cpu())
                for name, value in distillation.items()
            }
        )
    components["combined_total"] = float(total.detach().cpu())
    return total, components


def _validation_loss(
    model: CellectTrack,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    *,
    maximum_batches: int | None,
) -> tuple[float, dict[str, object]]:
    """Return checkpoint loss macro-averaged domain -> acquisition -> sampled windows."""
    sampler = loader.sampler
    dataset = loader.dataset
    if not isinstance(sampler, HierarchicalBalancedSampler) or not isinstance(
        dataset, TrackingWindowDataset
    ):
        raise RuntimeError("Macro tracking validation requires audited dataset and sampler")
    sampled_indices = list(iter(sampler))
    if maximum_batches is not None:
        batch_size = int(loader.batch_size or 1)
        sampled_indices = sampled_indices[: maximum_batches * batch_size]
    if not sampled_indices:
        raise RuntimeError("Checkpoint-role sampler selected no tracking windows")
    indices_by_acquisition: dict[str, list[int]] = defaultdict(list)
    for index in sampled_indices:
        indices_by_acquisition[dataset.windows[index].acquisition_group].append(index)
    model.eval()
    acquisition_losses: dict[str, float] = {}
    acquisition_counts: dict[str, int] = {}
    perturbation_counts: Counter[int] = Counter()
    with torch.no_grad():
        for acquisition_group, indices in sorted(indices_by_acquisition.items()):
            total = 0.0
            for index in indices:
                raw_item = dataset[index]
                perturbation_counts[int(raw_item["deployment_perturbation_code"])] += 1
                batch = _move_batch(
                    {name: value.unsqueeze(0) for name, value in raw_item.items()},
                    device,
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=device.type == "cuda",
                ):
                    output = model(
                        batch["time_yx"], batch["cell_features"], batch["cell_mask"]
                    )
                    loss = _supervised_losses(output, batch)["total"]
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Non-finite checkpoint-role tracking loss")
                total += float(loss.detach().cpu())
            acquisition_losses[acquisition_group] = total / len(indices)
            acquisition_counts[acquisition_group] = len(indices)
    losses_by_domain: dict[str, list[float]] = defaultdict(list)
    for acquisition_group, loss in acquisition_losses.items():
        losses_by_domain[dataset.sequence_by_group[acquisition_group].domain].append(loss)
    domain_losses = {
        domain: float(np.mean(values)) for domain, values in sorted(losses_by_domain.items())
    }
    macro_loss = float(np.mean(tuple(domain_losses.values())))
    return macro_loss, {
        "aggregation": "mean(window) per acquisition, mean(acquisition) per domain, mean(domain)",
        "macro_domain_acquisition_loss": macro_loss,
        "domain_losses": domain_losses,
        "acquisition_losses": dict(sorted(acquisition_losses.items())),
        "sampled_windows_by_acquisition": dict(sorted(acquisition_counts.items())),
        "fixed_sampled_windows": len(sampled_indices),
        "deployment_perturbation_counts": {
            name: perturbation_counts[code]
            for name, code in DEPLOYMENT_PERTURBATION_CODES.items()
        },
    }


def _train_phase(
    model: CellectTrack,
    *,
    model_name: str,
    phase: str,
    train_loader: DataLoader[dict[str, torch.Tensor]],
    checkpoint_loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    config: TrackingTrainingConfig,
    run_fingerprint: str,
    stage_directory: Path,
    maximum_batches: int | None,
    internal_teacher: CellectTrack | None,
    use_external_teacher: bool,
) -> dict[str, object]:
    train_sampler = train_loader.sampler
    checkpoint_sampler = checkpoint_loader.sampler
    if not isinstance(train_sampler, HierarchicalBalancedSampler) or not isinstance(
        checkpoint_sampler, HierarchicalBalancedSampler
    ):
        raise RuntimeError("Tracking phases require hierarchical balanced samplers")
    stage_fingerprint = _fingerprint(
        {
            "run": run_fingerprint,
            "model": model_name,
            "model_config": asdict(model.config),
            "phase": phase,
            "epochs": _phase_epochs(config, phase),
            "learning_rate": _phase_learning_rate(config, phase),
            "external_teacher": use_external_teacher,
            "internal_teacher": internal_teacher is not None,
            "train_sampler": train_sampler.policy_fingerprint_sha256,
            "checkpoint_sampler": checkpoint_sampler.policy_fingerprint_sha256,
        }
    )
    stage_directory.mkdir(parents=True, exist_ok=True)
    latest_path = stage_directory / "latest.pt"
    completed_path = stage_directory / "completed.pt"
    if completed_path.exists() or completed_path.with_suffix(".pt.sha256.json").exists():
        payload = _load_checkpoint(completed_path, stage_fingerprint)
        if payload.get("status") != "complete" or int(payload.get("optimizer_updates", 0)) < 1:
            raise RuntimeError(f"Completed stage lacks a real optimizer update: {completed_path}")
        model.load_state_dict(payload["model_state"], strict=True)
        train_sampler.load_state_dict(payload["train_sampler_state"])
        checkpoint_sampler.load_state_dict(payload["checkpoint_sampler_state"])
        _restore_rng_state(payload["rng_state_after_stage"])
        return dict(payload["report"])

    set_training_phase(model, phase)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError(f"Tracking phase {phase} has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=_phase_learning_rate(config, phase),
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=max(1, config.early_stopping_patience // 3), factor=0.5
    )
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    else:  # PyTorch <2.3 compatibility for the bounded local self-test.
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    start_epoch = 0
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0
    updates = 0
    history: list[dict[str, object]] = []
    if latest_path.exists() or latest_path.with_suffix(".pt.sha256.json").exists():
        payload = _load_checkpoint(latest_path, stage_fingerprint)
        model.load_state_dict(payload["model_state"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state"])
        scheduler.load_state_dict(payload["scheduler_state"])
        start_epoch = int(payload["next_epoch"])
        best_loss = float(payload["best_loss"])
        best_state = payload["best_state"]
        patience = int(payload["patience"])
        updates = int(payload["optimizer_updates"])
        history = list(payload["history"])
        scaler.load_state_dict(payload["grad_scaler_state"])
        train_sampler.load_state_dict(payload["train_sampler_state"])
        checkpoint_sampler.load_state_dict(payload["checkpoint_sampler_state"])
        _restore_rng_state(payload["rng_state"])

    if internal_teacher is not None:
        internal_teacher.eval()
        for parameter in internal_teacher.parameters():
            parameter.requires_grad = False
    model.to(device)
    snapshot = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    for epoch in range(start_epoch, _phase_epochs(config, phase)):
        train_sampler.set_epoch(epoch)
        checkpoint_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        epoch_batches = 0
        for batch_index, raw_batch in enumerate(train_loader):
            if maximum_batches is not None and batch_index >= maximum_batches:
                break
            batch = _move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                loss, _ = _combined_loss(
                    model,
                    batch,
                    internal_teacher=internal_teacher,
                    external_teacher_weight=(
                        config.external_teacher_weight if use_external_teacher else 0.0
                    ),
                    internal_teacher_weight=config.internal_teacher_weight,
                )
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite {model_name}/{phase} training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            audit = assert_gradient_audit(
                model,
                phase=phase,
                require_all_trainable_gradients=False,
            )
            if not audit["with_gradient"]:
                raise RuntimeError(f"{model_name}/{phase} produced no gradients")
            torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            updates += 1
            epoch_loss += float(loss.detach().cpu())
            epoch_batches += 1
        if not epoch_batches:
            raise RuntimeError(f"{model_name}/{phase} training loader yielded no batches")
        checkpoint_loss, checkpoint_aggregation = _validation_loss(
            model,
            checkpoint_loader,
            device,
            maximum_batches=maximum_batches,
        )
        scheduler.step(checkpoint_loss)
        improved = checkpoint_loss < best_loss - config.minimum_improvement
        if improved:
            best_loss = checkpoint_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / epoch_batches,
                "checkpoint_loss": checkpoint_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "optimizer_updates": updates,
                "checkpoint_aggregation": checkpoint_aggregation,
            }
        )
        _atomic_checkpoint(
            latest_path,
            {
                "schema_version": TRACKING_TRAINING_SCHEMA_VERSION,
                "stage_fingerprint": stage_fingerprint,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "next_epoch": epoch + 1,
                "best_loss": best_loss,
                "best_state": best_state,
                "patience": patience,
                "optimizer_updates": updates,
                "history": history,
                "grad_scaler_state": scaler.state_dict(),
                "train_sampler_state": train_sampler.state_dict(),
                "checkpoint_sampler_state": checkpoint_sampler.state_dict(),
                "rng_state": _capture_rng_state(),
            },
        )
        if patience >= config.early_stopping_patience:
            break
    if best_state is None:
        raise RuntimeError(f"{model_name}/{phase} never produced a finite checkpoint")
    model.load_state_dict(best_state, strict=True)
    changed_this_invocation = any(
        not torch.equal(snapshot[name], parameter.detach().cpu())
        for name, parameter in model.named_parameters()
        if name in snapshot
    )
    # A hash-verified latest checkpoint may already contain all epochs when a process stopped
    # between writing ``latest`` and the atomic ``completed`` marker. In that resume case the
    # recorded optimizer updates are the fail-closed evidence; on a fresh run we also compare
    # parameters byte-for-byte.
    changed = changed_this_invocation or (start_epoch > 0 and updates > 0)
    if not changed or updates < 1:
        raise RuntimeError(f"{model_name}/{phase} did not perform a real parameter update")
    report: dict[str, object] = {
        "model": model_name,
        "phase": phase,
        "status": "PASS",
        "stage_fingerprint": stage_fingerprint,
        "optimizer_updates": updates,
        "parameter_update_verified": True,
        "best_checkpoint_loss": best_loss,
        "epochs_completed": len(history),
        "history": history,
        "checkpoint_role_only": True,
        "sampler_policy": TRACKING_SAMPLER_VERSION,
        "optimizer_steps_per_epoch": train_sampler.steps_per_epoch,
        "checkpoint_steps_per_epoch": checkpoint_sampler.steps_per_epoch,
    }
    _atomic_checkpoint(
        completed_path,
        {
            "schema_version": TRACKING_TRAINING_SCHEMA_VERSION,
            "status": "complete",
            "stage_fingerprint": stage_fingerprint,
            "model_state": best_state,
            "optimizer_updates": updates,
            "report": report,
            "grad_scaler_state": scaler.state_dict(),
            "train_sampler_state": train_sampler.state_dict(),
            "checkpoint_sampler_state": checkpoint_sampler.state_dict(),
            "rng_state_after_stage": _capture_rng_state(),
        },
    )
    return report


def train_staged_models(
    heavy: CellectTrack,
    mobile: CellectTrack,
    *,
    datasets: Mapping[str, TrackingWindowDataset],
    device: torch.device,
    config: TrackingTrainingConfig,
    run_fingerprint: str,
    output_directory: Path,
    smoke: bool,
    external_teacher_available: bool,
) -> dict[str, object]:
    train_sampler = HierarchicalBalancedSampler(
        datasets["train"],
        steps_per_epoch=config.optimizer_steps_per_epoch,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    checkpoint_sampler = HierarchicalBalancedSampler(
        datasets["checkpoint"],
        steps_per_epoch=config.checkpoint_steps_per_epoch,
        batch_size=config.batch_size,
        seed=config.seed + 1,
    )
    train_loader = DataLoader(
        datasets["train"],
        batch_size=config.batch_size,
        sampler=train_sampler,
        drop_last=True,
        num_workers=0,
    )
    checkpoint_loader = DataLoader(
        datasets["checkpoint"],
        batch_size=config.batch_size,
        sampler=checkpoint_sampler,
        drop_last=True,
        num_workers=0,
    )
    maximum_batches = 1 if smoke else None
    reports: dict[str, list[dict[str, object]]] = {"heavy": [], "mobile": []}
    for phase in ("heads", "adapters", "full"):
        reports["heavy"].append(
            _train_phase(
                heavy,
                model_name="heavy",
                phase=phase,
                train_loader=train_loader,
                checkpoint_loader=checkpoint_loader,
                device=device,
                config=config,
                run_fingerprint=run_fingerprint,
                stage_directory=output_directory / "heavy" / phase,
                maximum_batches=maximum_batches,
                internal_teacher=None,
                use_external_teacher=external_teacher_available and phase == "full",
            )
        )
    for phase in ("heads", "adapters", "full", "distillation"):
        reports["mobile"].append(
            _train_phase(
                mobile,
                model_name="mobile",
                phase=phase,
                train_loader=train_loader,
                checkpoint_loader=checkpoint_loader,
                device=device,
                config=config,
                run_fingerprint=run_fingerprint,
                stage_directory=output_directory / "mobile" / phase,
                maximum_batches=maximum_batches,
                internal_teacher=heavy if phase == "distillation" else None,
                use_external_teacher=external_teacher_available and phase in {"full", "distillation"},
            )
        )
    return reports


def evaluate_deployment_perturbation_ablation(
    models: Mapping[str, CellectTrack],
    *,
    clean_dataset: TrackingWindowDataset,
    perturbed_dataset: TrackingWindowDataset,
    device: torch.device,
    config: TrackingTrainingConfig,
    smoke: bool,
) -> dict[str, object]:
    """Freeze clean-versus-detector-error checkpoint loss without selecting on it."""
    if clean_dataset.role != "checkpoint" or perturbed_dataset.role != "checkpoint":
        raise RuntimeError("Deployment perturbation ablation must use checkpoint acquisitions")
    reports: dict[str, object] = {}
    for model_name, model in sorted(models.items()):
        clean_sampler = HierarchicalBalancedSampler(
            clean_dataset,
            steps_per_epoch=config.checkpoint_steps_per_epoch,
            batch_size=config.batch_size,
            seed=config.seed + 10_001,
        )
        perturbed_sampler = HierarchicalBalancedSampler(
            perturbed_dataset,
            steps_per_epoch=config.checkpoint_steps_per_epoch,
            batch_size=config.batch_size,
            seed=config.seed + 10_001,
        )
        clean_loader = DataLoader(
            clean_dataset,
            batch_size=config.batch_size,
            sampler=clean_sampler,
            drop_last=True,
            num_workers=0,
        )
        perturbed_loader = DataLoader(
            perturbed_dataset,
            batch_size=config.batch_size,
            sampler=perturbed_sampler,
            drop_last=True,
            num_workers=0,
        )
        maximum_batches = 1 if smoke else None
        clean_loss, clean_report = _validation_loss(
            model, clean_loader, device, maximum_batches=maximum_batches
        )
        perturbed_loss, perturbed_report = _validation_loss(
            model, perturbed_loader, device, maximum_batches=maximum_batches
        )
        reports[model_name] = {
            "status": "PASS",
            "selection_role": "diagnostic checkpoint ablation only",
            "clean_macro_domain_acquisition_loss": clean_loss,
            "perturbed_macro_domain_acquisition_loss": perturbed_loss,
            "perturbed_minus_clean": perturbed_loss - clean_loss,
            "clean": clean_report,
            "perturbed": perturbed_report,
            "official_test_labels_used": False,
        }
    return {
        "contract": (
            "actual mask erosion/dilation plus image-derived merge, split, background false-"
            "positive, centroid/boundary jitter, and missed-detection perturbations; ambiguous "
            "synthetic identities are context-only and label-masked"
        ),
        "out_of_fold_cellect_tokens": {
            "status": "plug_in_supported_not_supplied",
            "preferred_full_training": True,
            "reason": (
                "role-safe frozen OOF Cellect proposal caches are the preferred deployment "
                "distribution, but cannot be generated before segmentation folds exist"
            ),
        },
        "models": reports,
    }


def predict_model_probabilities(
    model: CellectTrack,
    datasets: Mapping[str, TrackingWindowDataset],
    device: torch.device,
) -> ModelProbabilityOutput:
    model.eval().to(device)
    association_values: dict[str, dict[tuple[str, str], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    division_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    birth_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    death_values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with torch.no_grad():
        for role in TRACKING_ROLES:
            dataset = datasets[role]
            for item, window in enumerate(dataset.windows):
                batch = {
                    name: value.unsqueeze(0).to(device)
                    for name, value in dataset[item].items()
                    if name in {"time_yx", "cell_features", "cell_mask"}
                }
                output = model(
                    batch["time_yx"], batch["cell_features"], batch["cell_mask"]
                )
                sequence = dataset.sequence_by_group[window.acquisition_group]
                association = torch.sigmoid(output.association_logits[0]).cpu().numpy()
                division = output.division_probability[0].cpu().numpy()
                birth = output.birth_probability[0].cpu().numpy()
                death = output.death_probability[0].cpu().numpy()
                for source_local, source_global in enumerate(window.token_indices):
                    source = sequence.records[source_global]
                    division_values[sequence.acquisition_group][source.token_id].append(
                        float(division[source_local])
                    )
                    birth_values[sequence.acquisition_group][source.token_id].append(
                        float(birth[source_local])
                    )
                    death_values[sequence.acquisition_group][source.token_id].append(
                        float(death[source_local])
                    )
                    for target_local, target_global in enumerate(window.token_indices):
                        target = sequence.records[target_global]
                        frame_gap = target.frame_index - source.frame_index
                        if 1 <= frame_gap <= 2:
                            association_values[sequence.acquisition_group][
                                (source.token_id, target.token_id)
                            ].append(float(association[source_local, target_local]))

    def collapse_pair(
        values: Mapping[str, Mapping[tuple[str, str], Sequence[float]]]
    ) -> dict[str, dict[tuple[str, str], float]]:
        return {
            group: {key: float(np.mean(items)) for key, items in mapping.items()}
            for group, mapping in values.items()
        }

    def collapse_node(
        values: Mapping[str, Mapping[str, Sequence[float]]]
    ) -> dict[str, dict[str, float]]:
        return {
            group: {key: float(np.mean(items)) for key, items in mapping.items()}
            for group, mapping in values.items()
        }

    return ModelProbabilityOutput(
        association=collapse_pair(association_values),
        division=collapse_node(division_values),
        birth=collapse_node(birth_values),
        death=collapse_node(death_values),
    )


def ensemble_probabilities(
    members: Sequence[ModelProbabilityOutput],
) -> ModelProbabilityOutput:
    if not members:
        raise ValueError("A tracking ensemble needs at least one member")

    def average_pair(attribute: str) -> dict[str, dict[tuple[str, str], float]]:
        keys = {
            group: set().union(
                *(getattr(member, attribute).get(group, {}) for member in members)
            )
            for group in set().union(*(set(getattr(member, attribute)) for member in members))
        }
        return {
            group: {
                key: float(
                    np.mean(
                        [getattr(member, attribute).get(group, {}).get(key, 0.0) for member in members]
                    )
                )
                for key in group_keys
            }
            for group, group_keys in keys.items()
        }

    def average_node(attribute: str) -> dict[str, dict[str, float]]:
        groups = set().union(*(set(getattr(member, attribute)) for member in members))
        return {
            group: {
                key: float(
                    np.mean(
                        [getattr(member, attribute).get(group, {}).get(key, 0.0) for member in members]
                    )
                )
                for key in set().union(
                    *(set(getattr(member, attribute).get(group, {})) for member in members)
                )
            }
            for group in groups
        }

    return ModelProbabilityOutput(
        association=average_pair("association"),
        division=average_node("division"),
        birth=average_node("birth"),
        death=average_node("death"),
    )


@dataclass(frozen=True)
class TrackingThresholds:
    association: float
    division: float
    birth: float
    death: float
    decoder: str = "hungarian_global_v1"

    def __post_init__(self) -> None:
        if self.decoder not in {"hungarian_global_v1", "greedy_v1"}:
            raise ValueError(f"Unknown tracking decoder {self.decoder!r}")


def _event_gate_allows(
    source: str,
    target: str,
    *,
    birth_probability: Mapping[str, float],
    death_probability: Mapping[str, float],
    thresholds: TrackingThresholds,
) -> bool:
    return not (
        (
            thresholds.birth < 1.0
            and birth_probability.get(target, 0.0) >= thresholds.birth
        )
        or (
            thresholds.death < 1.0
            and death_probability.get(source, 0.0) >= thresholds.death
        )
    )


def _hungarian_frame_links(
    sources: Sequence[str],
    targets: Sequence[str],
    *,
    probability_by_pair: Mapping[tuple[str, str], float],
    division_probability: Mapping[str, float],
    birth_probability: Mapping[str, float],
    death_probability: Mapping[str, float],
    thresholds: TrackingThresholds,
    allow_division: bool,
) -> list[tuple[str, str, float]]:
    """Globally maximize valid links with explicit unmatched rows/columns."""
    ordered_sources = sorted(set(sources))
    ordered_targets = sorted(set(targets))
    if not ordered_sources or not ordered_targets:
        return []
    source_slots = [(source, 0) for source in ordered_sources]
    if allow_division:
        source_slots.extend(
            (source, 1)
            for source in ordered_sources
            if division_probability.get(source, 0.0) >= thresholds.division
        )
    # Real source slots and real targets share a square assignment with enough zero-valued
    # dummies on both sides to leave every detection unmatched. Invalid links are negative.
    real_rows = len(source_slots)
    real_columns = len(ordered_targets)
    size = real_rows + real_columns
    weights = np.zeros((size, size), dtype=np.float64)
    weights[:real_rows, :real_columns] = -1.0e9
    association_threshold = min(max(thresholds.association, 1e-6), 1.0 - 1e-6)
    baseline_log_odds = math.log(association_threshold / (1.0 - association_threshold))
    for row, (source, _) in enumerate(source_slots):
        for column, target in enumerate(ordered_targets):
            probability = float(probability_by_pair.get((source, target), -1.0))
            if probability < thresholds.association or not _event_gate_allows(
                source,
                target,
                birth_probability=birth_probability,
                death_probability=death_probability,
                thresholds=thresholds,
            ):
                continue
            clipped = min(max(probability, 1e-6), 1.0 - 1e-6)
            utility = math.log(clipped / (1.0 - clipped)) - baseline_log_odds
            # A threshold-equal edge has zero utility and is safely left unmatched.
            weights[row, column] = max(0.0, utility)
    assigned_rows, assigned_columns = linear_sum_assignment(weights, maximize=True)
    selected: list[tuple[str, str, float]] = []
    for row, column in zip(assigned_rows, assigned_columns, strict=True):
        if row >= real_rows or column >= real_columns or weights[row, column] <= 0.0:
            continue
        source = source_slots[row][0]
        target = ordered_targets[column]
        selected.append((source, target, float(probability_by_pair[(source, target)])))
    # A division duplicate may win the only child. Slot identity is not semantic; one selected
    # child remains a continuation and exactly two selected children become a division.
    return sorted(selected, key=lambda item: (item[0], item[1]))


def _hungarian_decode_links(
    records: Sequence[TokenRecord],
    probability_by_pair: Mapping[tuple[str, str], float],
    division_probability: Mapping[str, float],
    birth_probability: Mapping[str, float],
    death_probability: Mapping[str, float],
    thresholds: TrackingThresholds,
) -> list[tuple[str, str, float]]:
    records_by_frame: dict[int, list[str]] = defaultdict(list)
    for record in records:
        records_by_frame[record.frame_index].append(record.token_id)
    frames = sorted(records_by_frame)
    selected: list[tuple[str, str, float]] = []
    incoming: set[str] = set()
    outgoing: set[str] = set()
    # Adjacent evidence has precedence and supports explicit two-child divisions.
    for frame in frames:
        if frame + 1 not in records_by_frame:
            continue
        links = _hungarian_frame_links(
            records_by_frame[frame],
            records_by_frame[frame + 1],
            probability_by_pair=probability_by_pair,
            division_probability=division_probability,
            birth_probability=birth_probability,
            death_probability=death_probability,
            thresholds=thresholds,
            allow_division=True,
        )
        selected.extend(links)
        outgoing.update(source for source, _, _ in links)
        incoming.update(target for _, target, _ in links)
    # Gap-2 recovery is one-to-one and only considers endpoints still unmatched after every
    # adjacent-frame assignment; it can never displace adjacent truth or create a division.
    for frame in frames:
        if frame + 2 not in records_by_frame:
            continue
        sources = [item for item in records_by_frame[frame] if item not in outgoing]
        targets = [item for item in records_by_frame[frame + 2] if item not in incoming]
        links = _hungarian_frame_links(
            sources,
            targets,
            probability_by_pair=probability_by_pair,
            division_probability=division_probability,
            birth_probability=birth_probability,
            death_probability=death_probability,
            thresholds=thresholds,
            allow_division=False,
        )
        selected.extend(links)
        outgoing.update(source for source, _, _ in links)
        incoming.update(target for _, target, _ in links)
    return sorted(selected, key=lambda item: (item[0], item[1]))


def _tracking_case(
    sequence: SequenceTokens,
    probabilities: ModelProbabilityOutput,
    thresholds: TrackingThresholds,
) -> TrackingEvaluationCase:
    records = sequence.records
    record_by_id = {record.token_id: record for record in records}
    probability_by_pair = probabilities.association.get(sequence.acquisition_group, {})
    division_probability = probabilities.division.get(sequence.acquisition_group, {})
    birth_probability = probabilities.birth.get(sequence.acquisition_group, {})
    death_probability = probabilities.death.get(sequence.acquisition_group, {})
    if thresholds.decoder == "hungarian_global_v1":
        selected = _hungarian_decode_links(
            records,
            probability_by_pair,
            division_probability,
            birth_probability,
            death_probability,
            thresholds,
        )
        outgoing: dict[str, list[str]] = defaultdict(list)
        for source, target, _ in selected:
            outgoing[source].append(target)
    else:
        candidates = sorted(
            probability_by_pair.items(),
            key=lambda item: (
                record_by_id[item[0][1]].frame_index
                - record_by_id[item[0][0]].frame_index,
                -item[1],
                item[0][0],
                item[0][1],
            ),
        )
        incoming: set[str] = set()
        outgoing = defaultdict(list)
        selected = []
        for (source, target), probability in candidates:
            if probability < thresholds.association or target in incoming:
                continue
            if not _event_gate_allows(
                source,
                target,
                birth_probability=birth_probability,
                death_probability=death_probability,
                thresholds=thresholds,
            ):
                continue
            frame_gap = record_by_id[target].frame_index - record_by_id[source].frame_index
            if frame_gap > 1 and outgoing[source]:
                continue
            maximum_children = (
                2
                if frame_gap == 1
                and division_probability.get(source, 0.0) >= thresholds.division
                else 1
            )
            if len(outgoing[source]) >= maximum_children:
                continue
            incoming.add(target)
            outgoing[source].append(target)
            selected.append((source, target, probability))

    division_sources = {source for source, targets in outgoing.items() if len(targets) >= 2}
    continuation_pairs = {
        (source, target) for source, target, _ in selected if source not in division_sources
    }
    parent_pairs = {
        (source, target) for source, target, _ in selected if source in division_sources
    }

    predicted_track_by_token: dict[str, int] = {}
    next_track = 1
    records_by_frame: dict[int, list[TokenRecord]] = defaultdict(list)
    for record in records:
        records_by_frame[record.frame_index].append(record)
    continuation_parent = {target: source for source, target in continuation_pairs}
    for frame_index in sorted(records_by_frame):
        for record in sorted(records_by_frame[frame_index], key=lambda item: item.token_id):
            parent = continuation_parent.get(record.token_id)
            if parent is not None and parent in predicted_track_by_token:
                predicted_track_by_token[record.token_id] = predicted_track_by_token[parent]
            else:
                predicted_track_by_token[record.token_id] = next_track
                next_track += 1

    truth_detections = tuple(
        Detection(record.token_id, record.frame_index, record.track_id) for record in records
    )
    predicted_detections = tuple(
        Detection(
            f"pred:{record.token_id}",
            record.frame_index,
            predicted_track_by_token[record.token_id],
        )
        for record in records
    )
    pred_id = {record.token_id: f"pred:{record.token_id}" for record in records}
    truth_links = tuple(
        DirectedLink(records[source].token_id, records[target].token_id)
        for source, target in sequence.continuation_edges
    )
    truth_parents = tuple(
        DirectedLink(records[source].token_id, records[target].token_id)
        for source, target in sequence.parent_edges
    )
    # Non-selected continuation candidates retain probabilities for Brier/ECE. Candidates emitted
    # as division edges live only in parent_links so graph semantics remain unambiguous.
    prediction_links = tuple(
        DirectedLink(
            pred_id[source],
            pred_id[target],
            selected=(source, target) in continuation_pairs,
            probability=probability,
        )
        for (source, target), probability in sorted(probability_by_pair.items())
        if (source, target) not in parent_pairs
        and record_by_id[target].frame_index == record_by_id[source].frame_index + 1
    )
    prediction_parents = tuple(
        DirectedLink(
            pred_id[source],
            pred_id[target],
            selected=True,
            probability=probability_by_pair[(source, target)],
        )
        for source, target in sorted(parent_pairs)
    )
    matches = tuple(
        DetectionMatch(record.token_id, pred_id[record.token_id], 1.0) for record in records
    )
    return TrackingEvaluationCase(
        case_id=sequence.acquisition_group,
        domain=sequence.domain,
        truth=TrackingGraph(truth_detections, truth_links, truth_parents),
        prediction=TrackingGraph(
            predicted_detections, prediction_links, prediction_parents
        ),
        matches=matches,
        matching_protocol=(
            "tracking-by-detection exact instance correspondence; segmentation detections are "
            "held fixed while adjacent links and lineage are evaluated; selected two-frame "
            "recovery links affect identity reconstruction but are excluded from adjacent-link "
            "PRF/Brier/ECE"
        ),
        link_candidate_scope="provided_only",
        lineage_annotations_available=(
            sequence.sequence.source_format != LIVECELLTRACK_SOURCE_FORMAT
        ),
    )


def _cases_for_role(
    sequences: Sequence[SequenceTokens],
    role: str,
    probabilities: ModelProbabilityOutput,
    thresholds: TrackingThresholds,
) -> tuple[TrackingEvaluationCase, ...]:
    cases = tuple(
        _tracking_case(sequence, probabilities, thresholds)
        for sequence in sequences
        if sequence.role == role
    )
    if not cases:
        raise RuntimeError(f"No tracking cases exist for role {role}")
    return cases


def _diagnostic_score(evaluation: Any) -> float:
    macro = evaluation.macro_domain.metrics
    candidates = (
        ("temporal_links.f1", 0.35),
        ("division_events.f1", 0.20),
        ("division_parent_child_edges.f1", 0.10),
        ("identity.idf1", 0.25),
        ("identity.mostly_tracked_fraction", 0.10),
    )
    available = [(float(macro[name]), weight) for name, weight in candidates if macro.get(name) is not None]
    if not available:
        raise RuntimeError("Tracking evaluation produced no selection metrics")
    return sum(value * weight for value, weight in available) / sum(weight for _, weight in available)


def calibrate_tracking_thresholds(
    sequences: Sequence[SequenceTokens],
    probabilities: ModelProbabilityOutput,
    *,
    smoke: bool,
) -> tuple[TrackingThresholds, dict[str, object]]:
    association_values = (0.40, 0.60) if smoke else (0.25, 0.35, 0.45, 0.55, 0.65, 0.75)
    division_values = (0.40, 0.60) if smoke else (0.30, 0.40, 0.50, 0.60, 0.70)
    # The current sources contain no explicit positive birth/death annotations.  Do not tune
    # acquisition-censoring artifacts into biological event gates.
    event_values = (1.0,)
    decoder_values = ("hungarian_global_v1", "greedy_v1")
    rows: list[dict[str, object]] = []
    for association in association_values:
        for division in division_values:
            for birth in event_values:
                for death in event_values:
                    for decoder in decoder_values:
                        thresholds = TrackingThresholds(
                            association, division, birth, death, decoder
                        )
                        cases = _cases_for_role(
                            sequences, "calibration", probabilities, thresholds
                        )
                        evaluation = evaluate_tracking_dataset(cases)
                        rows.append(
                            {
                                "thresholds": asdict(thresholds),
                                "score": _diagnostic_score(evaluation),
                                "metrics": asdict(evaluation.macro_domain),
                            }
                        )
    rows.sort(
        key=lambda row: (
            -float(row["score"]),
            _canonical_json(row["thresholds"]),
        )
    )
    selected = rows[0]
    thresholds = TrackingThresholds(**selected["thresholds"])
    return thresholds, {
        "role": "calibration",
        "selected": selected,
        "trials": rows,
        "selection_metric": (
            "macro-domain weighted link/division/lineage-edge/IDF1/mostly-tracked diagnostic"
        ),
        "birth_death_policy": (
            "disabled (threshold=1.0): current source first/last observations are censored and "
            "no admitted source supplies explicit positive birth/death events"
        ),
        "decoder_policy": (
            "greedy and globally assigned Hungarian decoders are compared on calibration; "
            "the frozen winner is evaluated for ensemble membership"
        ),
        "test_labels_used": False,
    }


def select_tracking_ensemble(
    sequences: Sequence[SequenceTokens],
    candidates: Mapping[str, ModelProbabilityOutput],
    *,
    smoke: bool,
) -> tuple[str, ModelProbabilityOutput, TrackingThresholds, dict[str, object], tuple[TrackingEvaluationCase, ...]]:
    reports: dict[str, object] = {}
    best_name: str | None = None
    best_score = -math.inf
    best_probabilities: ModelProbabilityOutput | None = None
    best_thresholds: TrackingThresholds | None = None
    best_cases: tuple[TrackingEvaluationCase, ...] | None = None
    for name in sorted(candidates):
        probabilities = candidates[name]
        thresholds, calibration = calibrate_tracking_thresholds(
            sequences, probabilities, smoke=smoke
        )
        cases = _cases_for_role(
            sequences, "ensemble_selection", probabilities, thresholds
        )
        evaluation = evaluate_tracking_dataset(cases)
        score = _diagnostic_score(evaluation)
        reports[name] = {
            "calibration": calibration,
            "frozen_thresholds": asdict(thresholds),
            "ensemble_selection_role": "ensemble_selection",
            "ensemble_selection_score": score,
            "ensemble_selection_metrics": asdict(evaluation),
        }
        if score > best_score:
            best_name = name
            best_score = score
            best_probabilities = probabilities
            best_thresholds = thresholds
            best_cases = cases
    assert best_name is not None and best_probabilities is not None
    assert best_thresholds is not None and best_cases is not None
    return (
        best_name,
        best_probabilities,
        best_thresholds,
        {
            "role_separation": {
                "weight_selection": "checkpoint",
                "operating_thresholds": "calibration",
                "model_or_ensemble_membership": "ensemble_selection",
                "official_test": "sealed and never parsed",
            },
            "candidates": reports,
            "selected": best_name,
            "selected_score": best_score,
        },
        best_cases,
    )


TRACKING_OUTPUT_NAMES = (
    "association_logits",
    "association_mask",
    "division_probability",
    "birth_probability",
    "death_probability",
    "uncertainty",
)


def _deployment_inputs(
    dataset: TrackingWindowDataset,
    index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    item = dataset[index]
    return (
        item["time_yx"].unsqueeze(0),
        item["cell_features"].unsqueeze(0),
        item["cell_mask"].unsqueeze(0),
    )


def _maximum_output_difference(
    reference: Sequence[np.ndarray], candidate: Sequence[np.ndarray]
) -> float:
    if len(reference) != len(candidate):
        raise RuntimeError("Deployment output count changed")
    maximum = 0.0
    for expected, observed in zip(reference, candidate):
        if expected.shape != observed.shape:
            raise RuntimeError(
                f"Deployment output shape changed: {expected.shape} versus {observed.shape}"
            )
        maximum = max(maximum, float(np.max(np.abs(expected - observed))))
    return maximum


def export_tracking_model(
    model: CellectTrack,
    *,
    model_name: str,
    destination: Path,
    normalization: FeatureNormalization,
    calibration_dataset: TrackingWindowDataset,
    run_fingerprint: str,
    thresholds: TrackingThresholds,
    require_onnx: bool,
) -> dict[str, object]:
    destination.mkdir(parents=True, exist_ok=True)
    export_model = copy.deepcopy(model).cpu().eval()
    adapter = CellectTrackDeploymentAdapter(export_model).eval()
    example = _deployment_inputs(calibration_dataset, 0)
    state_path = destination / f"cellect_track_{model_name}.state.pt"
    state_digest = _atomic_checkpoint(
        state_path,
        {
            "schema_version": TRACKING_TRAINING_SCHEMA_VERSION,
            "stage_fingerprint": run_fingerprint,
            "model_name": model_name,
            "config": asdict(export_model.config),
            "model_state": export_model.state_dict(),
        },
    )

    traced = torch.jit.trace(adapter, example, check_trace=False, strict=False)
    torchscript_path = destination / f"cellect_track_{model_name}.torchscript.pt"
    torchscript_temporary = torchscript_path.with_name(f".{torchscript_path.name}.tmp")
    torch.jit.save(traced, torchscript_temporary)
    os.replace(torchscript_temporary, torchscript_path)
    torchscript_digest = file_sha256(torchscript_path)

    onnx_path = destination / f"cellect_track_{model_name}.onnx"
    onnx_status: dict[str, object]
    try:
        import onnx
        import onnxruntime as ort

        onnx_temporary = onnx_path.with_name(f".{onnx_path.name}.tmp")
        torch.onnx.export(
            adapter,
            example,
            onnx_temporary,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=("time_yx", "cell_features", "cell_mask"),
            output_names=TRACKING_OUTPUT_NAMES,
            dynamic_axes=None,
            dynamo=False,
        )
        checked = onnx.load(onnx_temporary)
        onnx.checker.check_model(checked)
        os.replace(onnx_temporary, onnx_path)
        onnx_digest = file_sha256(onnx_path)
        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        onnx_status = {
            "status": "PASS",
            "path": onnx_path.name,
            "sha256": onnx_digest,
            "opset": 17,
            "runtime": ort.__version__,
        }
    except Exception as error:
        if require_onnx:
            raise RuntimeError(f"Required ONNX tracking export failed: {error}") from error
        onnx_status = {
            "status": "SKIPPED_SELF_TEST_ENVIRONMENT",
            "reason": str(error),
        }
        session = None

    parity_rows: list[dict[str, float | int]] = []
    for index in range(min(2, len(calibration_dataset))):
        inputs = _deployment_inputs(calibration_dataset, index)
        with torch.no_grad():
            eager = tuple(value.detach().cpu().numpy() for value in adapter(*inputs))
            scripted = tuple(value.detach().cpu().numpy() for value in traced(*inputs))
        torchscript_difference = _maximum_output_difference(eager, scripted)
        if torchscript_difference > 1e-4:
            raise RuntimeError(
                f"TorchScript tracking parity failed: max abs {torchscript_difference}"
            )
        onnx_difference = 0.0
        if session is not None:
            onnx_outputs = session.run(
                list(TRACKING_OUTPUT_NAMES),
                {
                    "time_yx": inputs[0].numpy(),
                    "cell_features": inputs[1].numpy(),
                    "cell_mask": inputs[2].numpy(),
                },
            )
            onnx_difference = _maximum_output_difference(eager, onnx_outputs)
            if onnx_difference > 2e-3:
                raise RuntimeError(
                    f"ONNX tracking parity failed: max abs {onnx_difference}"
                )
        parity_rows.append(
            {
                "example": index,
                "torchscript_max_abs": torchscript_difference,
                "onnx_max_abs": onnx_difference,
            }
        )
    manifest: dict[str, object] = {
        "schema_version": TRACKING_TRAINING_SCHEMA_VERSION,
        "model_name": model_name,
        "model_contract": CELLECT_TRACK_VERSION,
        "run_fingerprint": run_fingerprint,
        "config": asdict(export_model.config),
        "fixed_input_contract": {
            "time_yx": {
                "shape": [1, export_model.config.max_tokens, 3],
                "dtype": "float32",
                "semantics": "frame index, normalized centroid y, normalized centroid x",
            },
            "cell_features": {
                "shape": [1, export_model.config.max_tokens, CELL_FEATURE_DIM],
                "dtype": "float32",
                "ordered_names": list(CELL_FEATURE_NAMES),
                "normalization_mean": list(normalization.mean),
                "normalization_standard_deviation": list(normalization.standard_deviation),
                "normalization_fingerprint": normalization.fingerprint_sha256,
            },
            "cell_mask": {
                "shape": [1, export_model.config.max_tokens],
                "dtype": "float32",
                "semantics": "1=real detection, 0=padding",
            },
        },
        "outputs": list(TRACKING_OUTPUT_NAMES),
        "segmentation_to_tracker_contract": {
            "instance_source": "Cellect segmentation labels before temporal association",
            "appearance": "robust 0.5/99.5 percentile grayscale scaling per frame",
            "shape_and_flow": TRACKING_FEATURE_VERSION,
            "centroids": "y/x mapped to [-1,1] by source image geometry",
            "motion": (
                "candidate-relative dt/dy/dx, absolute displacement, distance, and speed are "
                "computed inside CellectTrack without using ground-truth identities"
            ),
            "temporal_tiling": TRACKING_WINDOW_VERSION,
            "training_detection_noise": {
                "token_dropout_probability": calibration_dataset.config.token_dropout_probability,
                "centroid_jitter_std": calibration_dataset.config.coordinate_jitter_std,
                "standardized_area_shape_flow_and_appearance_jitter_std": (
                    calibration_dataset.config.feature_jitter_std
                ),
                "mask_morphology": (
                    "actual one-pixel erosion/dilation with features recomputed from pixels"
                ),
                "merge_split_false_positive": (
                    "image/mask-derived proposals; identity-ambiguous proposal labels and all "
                    "dependent link/event targets are masked"
                ),
                "proposal_probabilities": {
                    "morphology": calibration_dataset.config.morphology_perturbation_probability,
                    "merge": calibration_dataset.config.merge_proposal_probability,
                    "split": calibration_dataset.config.split_proposal_probability,
                    "false_positive": (
                        calibration_dataset.config.false_positive_proposal_probability
                    ),
                },
                "out_of_fold_cellect_proposals": (
                    "preferred future cache source; not claimed unless an audited role-safe "
                    "cache fingerprint is present in the run report"
                ),
            },
        },
        "frozen_thresholds": asdict(thresholds),
        "decoder_runtime_contract": {
            "selected": thresholds.decoder,
            "hungarian_global_v1": (
                "per adjacent frame pair, globally maximize threshold-relative log-odds with "
                "dummy unmatched rows/columns; duplicate rows only for division-positive "
                "sources; then recover still-unmatched endpoints at gap 2"
            ),
            "greedy_v1": (
                "deterministic adjacent-first descending-probability fallback retained only "
                "when it wins calibration"
            ),
            "tie_breaking": "lexicographically sorted token IDs and SciPy linear_sum_assignment",
        },
        "artifacts": {
            "state": {"path": state_path.name, "sha256": state_digest},
            "torchscript": {
                "path": torchscript_path.name,
                "sha256": torchscript_digest,
            },
            "onnx": onnx_status,
        },
        "parity": {
            "status": "PASS",
            "examples": parity_rows,
            "torchscript_tolerance": 1e-4,
            "onnx_tolerance": 2e-3,
            "coreml_swift_physical_iphone_required": True,
        },
    }
    manifest_path = destination / "manifest.json"
    _atomic_json(manifest_path, manifest)
    manifest["manifest_sha256"] = file_sha256(manifest_path)
    return manifest


def export_tracking_mean_ensemble(
    *,
    destination: Path,
    run_fingerprint: str,
    normalization: FeatureNormalization,
    thresholds: TrackingThresholds,
    member_exports: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    """Freeze the two-model probability-fusion runtime contract.

    This deliberately does not pretend that the mean ensemble is a third neural-network file.
    It is a composite deployment: run both individually parity-tested artifacts on identical
    tensors, fuse probabilities, and decode exactly once with the ensemble's own thresholds.
    """
    required_members = ("heavy", "mobile")
    if tuple(sorted(member_exports)) != tuple(sorted(required_members)):
        raise RuntimeError("The mean tracking ensemble requires heavy and mobile exports")
    member_contracts: dict[str, object] = {}
    for name in required_members:
        report = member_exports[name]
        parity = report.get("parity")
        if not isinstance(parity, Mapping) or parity.get("status") != "PASS":
            raise RuntimeError(f"Mean ensemble member {name} lacks passing export parity")
        member_contracts[name] = {
            "manifest_relative_path": f"../{name}/manifest.json",
            "manifest_sha256": report.get("manifest_sha256"),
            "model_contract": report.get("model_contract"),
            "member_frozen_thresholds_for_single_model_use": report.get(
                "frozen_thresholds"
            ),
        }
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "schema_version": TRACKING_TRAINING_SCHEMA_VERSION,
        "model_name": "mean_heavy_mobile",
        "artifact_type": "composite_probability_ensemble",
        "run_fingerprint": run_fingerprint,
        "normalization_fingerprint": normalization.fingerprint_sha256,
        "members": member_contracts,
        "runtime_contract": {
            "inputs": "identical normalized fixed-shape tensors are supplied to both members",
            "association": (
                "apply sigmoid to each member association_logits, then arithmetic-mean the "
                "probabilities wherever both association masks are true"
            ),
            "division_birth_death": (
                "arithmetic mean of the member probability outputs"
            ),
            "association_mask": "logical intersection of member association masks",
            "decoder": "decode fused probabilities once; never merge two decoded graphs",
            "frozen_decoder": thresholds.decoder,
            "gap_support_frames": [1, 2],
            "birth_death_gating": (
                "disabled while thresholds equal 1.0 because admitted sources contain no "
                "explicit positive event annotations"
            ),
        },
        "frozen_thresholds": asdict(thresholds),
        "standalone_model_artifact": False,
        "member_export_parity_required": True,
        "selection_evaluation": (
            "calibrated and evaluated as an independent third candidate before test access"
        ),
    }
    manifest_path = destination / "manifest.json"
    _atomic_json(manifest_path, manifest)
    manifest["manifest_sha256"] = file_sha256(manifest_path)
    return manifest


def _run_from_sequences(
    sequences: Sequence[TrackingSequence],
    *,
    output_root: Path,
    mode: str,
    config: TrackingTrainingConfig,
    device: torch.device,
    teacher_directory: Path | None,
    ker_root: Path | None,
    official_evaluator: OfficialTrackingEvaluator | None,
    heavy_config: CellectTrackConfig | None = None,
    mobile_config: CellectTrackConfig | None = None,
    require_smoke_marker: bool = True,
    require_onnx: bool = True,
) -> dict[str, object]:
    normalized_mode = mode.replace("_", "-")
    if normalized_mode not in {"best-smoke", "best"}:
        raise ValueError("Tracking mode must be 'best-smoke' or 'best'")
    smoke = normalized_mode == "best-smoke"
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    _seed_everything(config.seed)
    code_fingerprint = _code_fingerprint()
    ker_status = _ker_dataset_status(ker_root)
    preflight = _preflight_sequences(
        sequences,
        config,
        code_fingerprint=code_fingerprint,
        ker_status=ker_status,
    )
    _atomic_json(output_root / "tracking_preflight_v4.json", preflight)

    effective_config = (
        replace(
            config,
            batch_size=1,
            optimizer_steps_per_epoch=1,
            checkpoint_steps_per_epoch=1,
            heads_epochs=1,
            adapters_epochs=1,
            full_epochs=1,
            distillation_epochs=1,
            early_stopping_patience=1,
        )
        if smoke
        else config
    )
    sequence_tokens = tuple(extract_sequence_tokens(sequence) for sequence in sequences)
    normalization = fit_feature_normalization(sequence_tokens)
    _atomic_json(output_root / "tracking_feature_normalization_v4.json", asdict(normalization))

    teacher_targets, teacher_report = prepare_trackastra_teacher_targets(
        sequence_tokens,
        teacher_directory,
        device=device,
        cache_directory=output_root / "trackastra_target_cache_v4",
    )
    heavy_config = heavy_config or replace(
        CellectTrackHeavy(
            max_tokens=effective_config.max_tokens,
            cell_feature_dim=CELL_FEATURE_DIM,
        ).config,
        max_frame_gap=float(effective_config.maximum_frame_gap),
    )
    mobile_config = mobile_config or CellectTrackMobile(
        max_tokens=effective_config.max_tokens,
        cell_feature_dim=CELL_FEATURE_DIM,
    ).config
    if heavy_config.max_tokens != mobile_config.max_tokens:
        raise RuntimeError("Heavy/mobile distillation requires the same fixed max_tokens")
    if heavy_config.max_tokens != effective_config.max_tokens:
        raise RuntimeError("Model and temporal-window max_tokens contracts disagree")
    if (
        float(heavy_config.max_frame_gap) != float(effective_config.maximum_frame_gap)
        or float(mobile_config.max_frame_gap)
        != float(effective_config.maximum_frame_gap)
    ):
        raise RuntimeError(
            "Heavy/mobile max_frame_gap must exactly match the audited 1--2 frame target, "
            "evaluation, and deployment contract"
        )
    teacher_contract_identity = {
        key: teacher_report.get(key)
        for key in (
            "status",
            "artifact",
            "train_acquisitions_distilled",
            "association_targets",
            "association_target_fingerprint_sha256",
            "inherited_training_data",
            "association_knowledge_inherited_this_run",
            "event_heads_distilled",
        )
    }
    training_fingerprint = _fingerprint(
        {
            "stage": preflight["stage_fingerprint_sha256"],
            "teacher": teacher_contract_identity,
            "normalization": normalization.fingerprint_sha256,
            "heavy_config": asdict(heavy_config),
            "mobile_config": asdict(mobile_config),
        }
    )
    smoke_marker_path = output_root / "tracking_best_smoke_pass_v4.json"
    if not smoke and require_smoke_marker:
        if not smoke_marker_path.is_file():
            raise RuntimeError(
                "Run the v4 tracking best-smoke successfully before the full tracking run"
            )
        marker = json.loads(smoke_marker_path.read_text())
        if (
            marker.get("status") != "PASS"
            or marker.get("training_fingerprint_sha256") != training_fingerprint
        ):
            raise RuntimeError(
                "Tracking smoke marker does not match current code, data, model, teacher, and "
                "feature contracts; rerun best-smoke"
            )
    run_fingerprint = _fingerprint(
        {
            "training": training_fingerprint,
            "mode": normalized_mode,
            "effective_config": asdict(effective_config),
        }
    )
    experiment = (
        output_root
        / "experiments"
        / normalized_mode
        / run_fingerprint[:16]
    )
    experiment.mkdir(parents=True, exist_ok=True)
    datasets = {
        role: TrackingWindowDataset(
            sequence_tokens,
            normalization,
            effective_config,
            role=role,
            augment=role == "train",
            external_teacher=teacher_targets,
        )
        for role in TRACKING_ROLES
    }
    perturbed_checkpoint_dataset = TrackingWindowDataset(
        sequence_tokens,
        normalization,
        effective_config,
        role="checkpoint",
        augment=True,
        external_teacher=teacher_targets,
        deterministic_augmentation_seed=effective_config.seed + 20_001,
        force_deployment_perturbation=True,
    )
    heavy = CellectTrackHeavy(config=heavy_config).to(device)
    mobile = CellectTrackMobile(config=mobile_config).to(device)
    training_reports = train_staged_models(
        heavy,
        mobile,
        datasets=datasets,
        device=device,
        config=effective_config,
        run_fingerprint=run_fingerprint,
        output_directory=experiment / "stages",
        smoke=smoke,
        external_teacher_available=bool(teacher_targets),
    )
    deployment_perturbation_ablation = evaluate_deployment_perturbation_ablation(
        {"heavy": heavy, "mobile": mobile},
        clean_dataset=datasets["checkpoint"],
        perturbed_dataset=perturbed_checkpoint_dataset,
        device=device,
        config=effective_config,
        smoke=smoke,
    )
    _atomic_json(
        experiment / "tracking_deployment_perturbation_ablation_v4.json",
        deployment_perturbation_ablation,
    )
    heavy_probabilities = predict_model_probabilities(heavy, datasets, device)
    mobile_probabilities = predict_model_probabilities(mobile, datasets, device)
    candidates = {
        "heavy": heavy_probabilities,
        "mobile": mobile_probabilities,
        "mean_heavy_mobile": ensemble_probabilities(
            (heavy_probabilities, mobile_probabilities)
        ),
    }
    (
        selected_name,
        selected_probabilities,
        selected_thresholds,
        selection_report,
        selection_cases,
    ) = select_tracking_ensemble(sequence_tokens, candidates, smoke=smoke)
    del selected_probabilities
    _atomic_json(experiment / "tracking_selection_v4.json", selection_report)

    candidate_reports = selection_report["candidates"]
    if not isinstance(candidate_reports, Mapping) or set(candidate_reports) != set(candidates):
        raise RuntimeError("Every tracking candidate must retain an independent audit report")
    candidate_thresholds: dict[str, TrackingThresholds] = {}
    for candidate_name, candidate_report in candidate_reports.items():
        if not isinstance(candidate_report, Mapping):
            raise RuntimeError(f"Invalid selection report for {candidate_name}")
        frozen = candidate_report.get("frozen_thresholds")
        if not isinstance(frozen, Mapping):
            raise RuntimeError(f"Missing frozen thresholds for {candidate_name}")
        candidate_thresholds[str(candidate_name)] = TrackingThresholds(
            association=float(frozen["association"]),
            division=float(frozen["division"]),
            birth=float(frozen["birth"]),
            death=float(frozen["death"]),
            decoder=str(frozen["decoder"]),
        )

    export_reports = {
        "heavy": export_tracking_model(
            heavy,
            model_name="heavy",
            destination=experiment / "exports" / "heavy",
            normalization=normalization,
            calibration_dataset=datasets["calibration"],
            run_fingerprint=run_fingerprint,
            thresholds=candidate_thresholds["heavy"],
            require_onnx=require_onnx,
        ),
        "mobile": export_tracking_model(
            mobile,
            model_name="mobile",
            destination=experiment / "exports" / "mobile",
            normalization=normalization,
            calibration_dataset=datasets["calibration"],
            run_fingerprint=run_fingerprint,
            thresholds=candidate_thresholds["mobile"],
            require_onnx=require_onnx,
        ),
    }
    export_reports["mean_heavy_mobile"] = export_tracking_mean_ensemble(
        destination=experiment / "exports" / "mean_heavy_mobile",
        run_fingerprint=run_fingerprint,
        normalization=normalization,
        thresholds=candidate_thresholds["mean_heavy_mobile"],
        member_exports={name: export_reports[name] for name in ("heavy", "mobile")},
    )
    official_report: dict[str, object] | None = None
    if official_evaluator is not None:
        official = run_official_evaluator(
            selection_cases,
            experiment / "official_evaluator",
            official_evaluator,
        )
        official_report = asdict(official)
        _atomic_json(experiment / "official_evaluation.json", official_report)

    selected_members = (
        ["heavy", "mobile"]
        if selected_name == "mean_heavy_mobile"
        else [selected_name]
    )
    deployment_lock: dict[str, object] = {
        "schema_version": TRACKING_TRAINING_SCHEMA_VERSION,
        "status": "FROZEN_BEFORE_OFFICIAL_TEST",
        "run_fingerprint": run_fingerprint,
        "training_fingerprint_sha256": training_fingerprint,
        "selected_candidate": selected_name,
        "selected_members": selected_members,
        "merge_rule": (
            "arithmetic mean of association/division/birth/death probabilities"
            if len(selected_members) > 1
            else "single model"
        ),
        "thresholds": asdict(selected_thresholds),
        "feature_contract": asdict(normalization),
        "exports": export_reports,
        "selection_roles": {
            "weights": "checkpoint",
            "thresholds": "calibration",
            "membership": "ensemble_selection",
            "official_test": "not parsed or evaluated",
        },
        "coreml_swift_physical_iphone_gate": "required before app release",
    }
    _atomic_json(experiment / "deployment_lock_tracking_v4.json", deployment_lock)
    summary: dict[str, object] = {
        "schema_version": TRACKING_TRAINING_SCHEMA_VERSION,
        "status": "PASS",
        "mode": normalized_mode,
        "experiment_directory": str(experiment),
        "run_fingerprint": run_fingerprint,
        "training_fingerprint_sha256": training_fingerprint,
        "preflight": preflight,
        "normalization": asdict(normalization),
        "teacher": teacher_report,
        "ker_dataset": ker_status,
        "windows_by_role": {role: len(dataset) for role, dataset in datasets.items()},
        "training": training_reports,
        "deployment_perturbation_ablation": deployment_perturbation_ablation,
        "selection": selection_report,
        "selected_candidate": selected_name,
        "selected_thresholds": asdict(selected_thresholds),
        "exports": export_reports,
        "official_evaluator": official_report,
        "official_test_labels_parsed": False,
    }
    _atomic_json(experiment / "tracking_summary_v4.json", summary)
    if smoke:
        marker = {
            "schema_version": TRACKING_TRAINING_SCHEMA_VERSION,
            "status": "PASS",
            "training_fingerprint_sha256": training_fingerprint,
            "run_fingerprint": run_fingerprint,
            "real_optimizer_updates": {
                model_name: {
                    report["phase"]: report["optimizer_updates"]
                    for report in reports
                }
                for model_name, reports in training_reports.items()
            },
            "exports_and_parity": "PASS",
            "official_test_labels_parsed": False,
        }
        _atomic_json(smoke_marker_path, marker)
    return summary


def run_v4_tracking(
    data_root: Path | str = DEFAULT_DATA_ROOT,
    output_root: Path | str = DEFAULT_OUTPUT_ROOT,
    *,
    mode: str = "best",
    deepsea_root: Path | str | None = None,
    ctc_roots: Mapping[str, Path | str] | None = None,
    livecelltrack_root: Path | str | None = None,
    ctmc_root: Path | str | None = None,
    alfi_root: Path | str | None = None,
    teacher_directory: Path | str | None = None,
    ker_root: Path | str | None = None,
    config: TrackingTrainingConfig | None = None,
    device: str | torch.device | None = None,
    official_evaluator: OfficialTrackingEvaluator | None = None,
) -> dict[str, object]:
    """Public no-download entry for strict v4 tracking training and deployment export."""
    resolved_device = torch.device(device or "cuda")
    if resolved_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(
            "Workstation v4 tracking requires an available CUDA device; CPU is reserved for "
            "the bounded --self-test"
        )
    resolved_ctc = (
        {name: Path(path) for name, path in ctc_roots.items()}
        if ctc_roots is not None
        else None
    )
    deepsea, ctc, livecelltrack, ctmc, alfi = resolve_tracking_roots(
        Path(data_root),
        deepsea_root=Path(deepsea_root) if deepsea_root is not None else None,
        ctc_roots=resolved_ctc,
        livecelltrack_root=(
            Path(livecelltrack_root) if livecelltrack_root is not None else None
        ),
        ctmc_root=Path(ctmc_root) if ctmc_root is not None else None,
        alfi_root=Path(alfi_root) if alfi_root is not None else None,
    )
    sequences = discover_transmitted_light_tracking(
        deepsea_root=deepsea,
        ctc_roots=ctc,
        livecelltrack_root=livecelltrack,
        ctmc_root=ctmc,
        alfi_root=alfi,
    )
    configured_teacher = teacher_directory or os.environ.get("CELLECT_TRACKASTRA_TEACHER")
    configured_ker = ker_root or os.environ.get("CELLECT_KER_TRACKING_ROOT")
    return _run_from_sequences(
        sequences,
        output_root=Path(output_root),
        mode=mode,
        config=config or TrackingTrainingConfig(),
        device=resolved_device,
        teacher_directory=(Path(configured_teacher) if configured_teacher else None),
        ker_root=(Path(configured_ker) if configured_ker else None),
        official_evaluator=official_evaluator,
    )


def run_v4_tracking_training(*args: Any, **kwargs: Any) -> dict[str, object]:
    """Stable pipeline-facing alias for :func:`run_v4_tracking`."""
    return run_v4_tracking(*args, **kwargs)


def _synthetic_sequence(root: Path, sequence_id: str, role: str) -> TrackingSequence:
    sequence_root = root / sequence_id
    sequence_root.mkdir(parents=True)
    lineage_path = sequence_root / "man_track.txt"
    lineage_path.write_text("1 0 1 0\n2 0 2 0\n3 2 2 1\n4 2 2 1\n", encoding="utf-8")
    frame_ids = ((1, 2), (1, 2), (2, 3, 4))
    locations = {
        (0, 1): (12, 12),
        (0, 2): (45, 45),
        (1, 1): (15, 14),
        (1, 2): (43, 43),
        (2, 2): (41, 41),
        (2, 3): (12, 17),
        (2, 4): (22, 15),
    }
    frames: list[TrackingFrame] = []
    for frame_index, ids in enumerate(frame_ids):
        yy, xx = np.mgrid[:64, :64]
        image = (0.2 + 0.3 * xx / 63.0 + 0.2 * yy / 63.0).astype(np.float32)
        mask = np.zeros((64, 64), dtype=np.uint16)
        instances: list[FrameInstance] = []
        for component_id in ids:
            center_y, center_x = locations[(frame_index, component_id)]
            binary = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= 5**2
            mask[binary] = component_id
            image[binary] += 0.25 + 0.02 * component_id
            y_values, x_values = np.nonzero(binary)
            instances.append(
                FrameInstance(
                    track_id=component_id,
                    source_label=str(component_id),
                    component_id=component_id,
                    parent_track_id=(1 if component_id in {3, 4} else 0),
                    centroid_x=float(x_values.mean()),
                    centroid_y=float(y_values.mean()),
                    area_pixels=int(binary.sum()),
                )
            )
        image_path = sequence_root / f"t{frame_index:03d}.tif"
        mask_path = sequence_root / f"man_track{frame_index:03d}.tif"
        tifffile.imwrite(image_path, np.clip(image * 255, 0, 255).astype(np.uint8))
        tifffile.imwrite(mask_path, mask)
        frames.append(
            TrackingFrame(
                frame_index=frame_index,
                image_path=image_path,
                instance_path=mask_path,
                label_path=lineage_path,
                image_sha256=file_sha256(image_path),
                instance_sha256=file_sha256(mask_path),
                label_sha256=file_sha256(lineage_path),
                height=64,
                width=64,
                image_dtype="uint8",
                instance_dtype="uint16",
                instances=tuple(instances),
            )
        )
    tracks = (
        TrackLifetime(1, "1", 0, 1, 0, (0, 1)),
        TrackLifetime(2, "2", 0, 2, 0, (0, 1, 2)),
        TrackLifetime(3, "3", 2, 2, 1, (2,)),
        TrackLifetime(4, "4", 2, 2, 1, (2,)),
    )
    parents = (
        SourceParentLink(1, 3, 1, 2),
        SourceParentLink(1, 4, 1, 2),
    )
    group = f"synthetic/{sequence_id}".casefold()
    sequence_fingerprint = _fingerprint(
        {
            "group": group,
            "role": role,
            "images": [frame.image_sha256 for frame in frames],
            "masks": [frame.instance_sha256 for frame in frames],
        }
    )
    return TrackingSequence(
        schema_version=TRACKING_SCHEMA_VERSION,
        dataset="synthetic",
        sequence_id=sequence_id,
        acquisition_group=group,
        role=role,
        modality="phase contrast",
        organism="synthetic eukaryotic cells",
        source_format="ctc_tra_v1",
        source_partition="synthetic_development_ground_truth",
        root=sequence_root,
        frames=tuple(frames),
        tracks=tracks,
        parent_links=parents,
        lineage_sha256=file_sha256(lineage_path),
        sequence_fingerprint_sha256=sequence_fingerprint,
    )


def synthetic_self_test() -> dict[str, object]:
    """Run the complete no-download smoke path with tiny real optimizer stages."""
    if (
        CellectTrackHeavy(max_tokens=8).config.max_frame_gap != 2.0
        or CellectTrackMobile(max_tokens=8).config.max_frame_gap != 2.0
    ):
        raise AssertionError("Heavy/mobile factory gap contract diverged from audited gap-2")
    # Pin the real Trackastra 0.3 ``predict_windows`` return convention.  This
    # catches an API-shape drift before a multi-hour teacher-cache pass begins.
    teacher_weights = _trackastra_sparse_weights(
        (((0, 1), np.float32(0.75)), ((1, 0), np.float32(0.125))),
        2,
    )
    if teacher_weights != {(0, 1): 0.75, (1, 0): 0.125}:
        raise AssertionError("Trackastra sparse-edge contract conversion failed")
    adversarial_records = (
        TokenRecord("s1", 0, 1, 1, 0, (0.0, 0.0, 0.0), (0.0,) * CELL_FEATURE_DIM),
        TokenRecord("s2", 0, 2, 2, 0, (0.0, 0.0, 0.0), (0.0,) * CELL_FEATURE_DIM),
        TokenRecord("t1", 1, 3, 3, 0, (1.0, 0.0, 0.0), (0.0,) * CELL_FEATURE_DIM),
        TokenRecord("t2", 1, 4, 4, 0, (1.0, 0.0, 0.0), (0.0,) * CELL_FEATURE_DIM),
    )
    adversarial_probabilities = {
        ("s1", "t1"): 0.90,
        ("s1", "t2"): 0.80,
        ("s2", "t1"): 0.85,
        ("s2", "t2"): 0.10,
    }
    adversarial_links = _hungarian_decode_links(
        adversarial_records,
        adversarial_probabilities,
        {},
        {},
        {},
        TrackingThresholds(0.05, 1.0, 1.0, 1.0, "hungarian_global_v1"),
    )
    if {(source, target) for source, target, _ in adversarial_links} != {
        ("s1", "t2"),
        ("s2", "t1"),
    }:
        raise AssertionError("Hungarian decoder failed the adversarial 2x2 assignment")
    with tempfile.TemporaryDirectory(prefix="cellect-v4-tracking-training-") as temporary:
        root = Path(temporary)
        sequences = tuple(
            _synthetic_sequence(root / "data", f"sequence_{role}", role)
            for role in TRACKING_ROLES
        )
        config = TrackingTrainingConfig(
            max_tokens=8,
            window_frames=3,
            batch_size=1,
            heads_epochs=1,
            adapters_epochs=1,
            full_epochs=1,
            distillation_epochs=1,
            early_stopping_patience=1,
            token_dropout_probability=0.0,
        )
        live_source = sequences[0]
        live_sequence = replace(
            live_source,
            source_format=LIVECELLTRACK_SOURCE_FORMAT,
            parent_links=(),
            tracks=tuple(replace(track, parent_track_id=0) for track in live_source.tracks),
            frames=tuple(
                replace(
                    frame,
                    instances=tuple(
                        replace(instance, parent_track_id=0)
                        for instance in frame.instances
                    ),
                )
                for frame in live_source.frames
            ),
        )
        live_tokens = extract_sequence_tokens(live_sequence)
        live_normalization = fit_feature_normalization((live_tokens,))
        live_dataset = TrackingWindowDataset(
            (live_tokens,),
            live_normalization,
            config,
            role="train",
            augment=False,
        )
        if any(
            float(live_dataset[index]["division_supervision"].sum()) != 0.0
            for index in range(len(live_dataset))
        ):
            raise AssertionError(
                "Identity-only LiveCellTrack rows became negative division supervision"
            )
        train_tokens = extract_sequence_tokens(sequences[0])
        train_normalization = fit_feature_normalization((train_tokens,))
        perturbation_base = replace(
            config,
            token_dropout_probability=0.0,
            forced_gap_dropout_probability=0.0,
            morphology_perturbation_probability=0.0,
            merge_proposal_probability=0.0,
            split_proposal_probability=0.0,
            false_positive_proposal_probability=0.0,
        )
        for kind, code in DEPLOYMENT_PERTURBATION_CODES.items():
            if kind == "clean":
                continue
            configured = replace(
                perturbation_base,
                **{
                    {
                        "morphology": "morphology_perturbation_probability",
                        "merge": "merge_proposal_probability",
                        "split": "split_proposal_probability",
                        "false_positive": "false_positive_proposal_probability",
                    }[kind]: 1.0
                },
            )
            perturbed = TrackingWindowDataset(
                (train_tokens,),
                train_normalization,
                configured,
                role="train",
                augment=True,
                deterministic_augmentation_seed=7,
                force_deployment_perturbation=True,
            )[0]
            if int(perturbed["deployment_perturbation_code"]) != code:
                raise AssertionError(f"Synthetic {kind} perturbation did not execute")
            ambiguous = perturbed["identity_ambiguous_token_mask"] > 0
            if bool(ambiguous.any()) and (
                float(perturbed["association_supervision"][ambiguous].sum()) != 0.0
                or float(perturbed["association_supervision"][:, ambiguous].sum()) != 0.0
            ):
                raise AssertionError(f"Synthetic {kind} identity labels were not masked")
        gap_dataset = TrackingWindowDataset(
            (train_tokens,),
            train_normalization,
            replace(
                perturbation_base,
                forced_gap_dropout_probability=1.0,
            ),
            role="train",
            augment=True,
            deterministic_augmentation_seed=11,
        )
        gap_item = gap_dataset[0]
        if float(gap_item["recovery_supervision"].sum()) < 1.0:
            raise AssertionError("Known-ID two-frame dropout recovery was not supervised")
        if (
            float(gap_item["birth_target"].sum()) != 0.0
            or float(gap_item["death_target"].sum()) != 0.0
        ):
            raise AssertionError("Censored source endpoints became birth/death positives")
        sampler = HierarchicalBalancedSampler(
            gap_dataset, steps_per_epoch=3, batch_size=2, seed=13
        )
        sampled = list(iter(sampler))
        sampler_clone = HierarchicalBalancedSampler(
            gap_dataset, steps_per_epoch=3, batch_size=2, seed=13
        )
        sampler_clone.load_state_dict(sampler.state_dict())
        if len(sampled) != 6 or list(iter(sampler_clone)) != sampled:
            raise AssertionError("Balanced sampler state/policy replay failed")
        heavy_config = CellectTrackConfig(
            max_tokens=8,
            cell_feature_dim=CELL_FEATURE_DIM,
            model_dim=24,
            num_heads=4,
            num_layers=2,
            pair_hidden_dim=32,
            feedforward_multiplier=2,
            dropout=0.0,
            max_frame_gap=2.0,
        )
        mobile_config = CellectTrackConfig(
            max_tokens=8,
            cell_feature_dim=CELL_FEATURE_DIM,
            model_dim=16,
            num_heads=4,
            num_layers=1,
            pair_hidden_dim=24,
            feedforward_multiplier=2,
            dropout=0.0,
            max_frame_gap=2.0,
        )
        summary = _run_from_sequences(
            sequences,
            output_root=root / "output",
            mode="best-smoke",
            config=config,
            device=torch.device("cpu"),
            teacher_directory=None,
            ker_root=None,
            official_evaluator=None,
            heavy_config=heavy_config,
            mobile_config=mobile_config,
            require_smoke_marker=False,
            require_onnx=False,
        )
        phase_updates = {
            model_name: {
                str(report["phase"]): int(report["optimizer_updates"])
                for report in reports
            }
            for model_name, reports in summary["training"].items()
        }
        if set(phase_updates["heavy"]) != {"heads", "adapters", "full"}:
            raise AssertionError("Synthetic heavy training did not exercise every phase")
        if set(phase_updates["mobile"]) != {
            "heads",
            "adapters",
            "full",
            "distillation",
        }:
            raise AssertionError("Synthetic mobile training did not exercise every phase")
        if any(update < 1 for values in phase_updates.values() for update in values.values()):
            raise AssertionError("Synthetic best-smoke stage lacked a real optimizer update")
        for model_name in ("heavy", "mobile"):
            report = summary["exports"][model_name]
            artifact = report["artifacts"]["torchscript"]
            export_path = (
                Path(summary["experiment_directory"])
                / "exports"
                / model_name
                / artifact["path"]
            )
            if not export_path.is_file() or file_sha256(export_path) != artifact["sha256"]:
                raise AssertionError("Synthetic TorchScript artifact hash is invalid")
            expected_thresholds = summary["selection"]["candidates"][model_name][
                "frozen_thresholds"
            ]
            if report["frozen_thresholds"] != expected_thresholds:
                raise AssertionError("Single-model export received another candidate's thresholds")
        mean_report = summary["exports"]["mean_heavy_mobile"]
        if (
            mean_report["artifact_type"] != "composite_probability_ensemble"
            or mean_report["frozen_thresholds"]
            != summary["selection"]["candidates"]["mean_heavy_mobile"][
                "frozen_thresholds"
            ]
        ):
            raise AssertionError("Mean ensemble runtime/threshold contract was not frozen")
        if set(summary["selection"]["candidates"]) != {
            "heavy",
            "mobile",
            "mean_heavy_mobile",
        }:
            raise AssertionError("Not all three tracking candidates were evaluated")
        return {
            "status": "PASS",
            "real_optimizer_updates": phase_updates,
            "selected_candidate": summary["selected_candidate"],
            "roles": summary["preflight"]["roles"],
            "torchscript_parity": {
                name: summary["exports"][name]["parity"]["status"]
                for name in ("heavy", "mobile")
            },
            "onnx": {
                name: summary["exports"][name]["artifacts"]["onnx"]["status"]
                for name in ("heavy", "mobile")
            },
            "trackastra_sparse_edge_contract": "PASS",
            "hungarian_adversarial_assignment": "PASS",
            "deployment_detector_perturbations": "PASS",
            "gap_recovery_and_censoring": "PASS",
            "balanced_sampler_replay": "PASS",
            "candidate_specific_threshold_exports": "PASS",
            "identity_only_lineage_mask": "PASS",
            "downloads_performed": False,
            "official_test_labels_parsed": False,
        }


def _parse_ctc_roots(values: Sequence[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--ctc-root must be DATASET=PATH")
        dataset, path = value.split("=", 1)
        if dataset in roots:
            raise ValueError(f"Duplicate --ctc-root for {dataset}")
        roots[dataset] = Path(path)
    return roots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--mode", choices=("best-smoke", "best"), default="best-smoke")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--deepsea-root", type=Path)
    parser.add_argument(
        "--livecelltrack-root",
        type=Path,
        help="extracted pinned LiveCellTrack preview root (the direct API never downloads)",
    )
    parser.add_argument("--ctmc-root", type=Path, help="CTMC-v1 extracted root or ZIP")
    parser.add_argument("--alfi-root", type=Path, help="ALFI extracted root or pinned ZIP")
    parser.add_argument(
        "--ctc-root",
        action="append",
        default=[],
        metavar="DATASET=PATH",
        help="repeat for each of the five strict CTC dataset keys",
    )
    parser.add_argument("--teacher-directory", type=Path)
    parser.add_argument("--ker-root", type=Path)
    arguments = parser.parse_args()
    if arguments.self_test:
        print(json.dumps(synthetic_self_test(), indent=2, sort_keys=True))
        return
    report = run_v4_tracking(
        data_root=arguments.data_root,
        output_root=arguments.output_root,
        mode=arguments.mode,
        deepsea_root=arguments.deepsea_root,
        ctc_roots=_parse_ctc_roots(arguments.ctc_root) or None,
        livecelltrack_root=arguments.livecelltrack_root,
        ctmc_root=arguments.ctmc_root,
        alfi_root=arguments.alfi_root,
        teacher_directory=arguments.teacher_directory,
        ker_root=arguments.ker_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
