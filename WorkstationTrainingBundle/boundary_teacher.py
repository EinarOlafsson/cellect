#!/usr/bin/env python3
"""Ceb-inspired whole-boundary teacher for Cellect v4.

Ceb published a useful *mechanism* but no trained checkpoint.  This independent implementation
therefore learns its own keep-versus-merge policy from Cellect's acquisition-separated data.  It
extends the geometry-only Ceb signature with raw image evidence, Cellpose flow, region shape,
semantic probabilities, and model disagreement.  A graph teacher reasons jointly about candidate
boundaries sharing cells; a fixed-patch student can be exported or distilled into the dense v4
shape branch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch import Tensor, nn


BOUNDARY_TEACHER_VERSION = "cellect-shapeflow-boundary-graph-v2-frozen-proposals"
CANDIDATE_PATCH_CHANNELS = (
    "normalized_image",
    "foreground_probability",
    "context_boundary_probability",
    "cellpose_flow_y",
    "cellpose_flow_x",
    "flow_magnitude",
    "flow_divergence",
    "candidate_line",
    "region_a_mask",
    "region_b_mask",
    "region_a_signed_distance",
    "region_b_signed_distance",
    "union_outer_contour",
    "ensemble_uncertainty",
)


@dataclass(frozen=True)
class BoundaryTeacherSpec:
    patch_size: int = 128
    patch_channels: int = len(CANDIDATE_PATCH_CHANNELS)
    geometry_features: int = 12
    embedding_size: int = 192
    graph_layers: int = 4
    graph_heads: int = 6
    dropout: float = 0.10

    def __post_init__(self) -> None:
        if self.patch_size < 32 or self.patch_size % 16:
            raise ValueError("patch_size must be a multiple of 16 and at least 32")
        if self.patch_channels != len(CANDIDATE_PATCH_CHANNELS):
            raise ValueError("patch channel count does not match the documented contract")
        if self.embedding_size % self.graph_heads:
            raise ValueError("embedding_size must be divisible by graph_heads")


def _as_float_map(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}; expected {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


def _signed_distance(mask: np.ndarray) -> np.ndarray:
    inside = ndimage.distance_transform_edt(mask)
    outside = ndimage.distance_transform_edt(~mask)
    scale = max(float(inside.max(initial=0.0)), 1.0)
    return np.clip((inside - outside) / scale, -1.0, 1.0).astype(np.float32)


def _candidate_line(region_a: np.ndarray, region_b: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3), dtype=bool)
    line = ndimage.binary_dilation(region_a, structure=structure) & ndimage.binary_dilation(
        region_b, structure=structure
    )
    if not line.any():
        # A narrow annotation gap is common.  Grow symmetrically until the fronts meet.
        for iterations in (2, 3):
            line = ndimage.binary_dilation(
                region_a, structure=structure, iterations=iterations
            ) & ndimage.binary_dilation(region_b, structure=structure, iterations=iterations)
            if line.any():
                break
    return line


def _resize_channel(channel: np.ndarray, size: int, *, categorical: bool) -> np.ndarray:
    interpolation = cv2.INTER_NEAREST if categorical else cv2.INTER_LINEAR
    return cv2.resize(channel.astype(np.float32), (size, size), interpolation=interpolation)


def candidate_geometry_features(
    proposal_labels: np.ndarray,
    region_a_id: int,
    region_b_id: int,
) -> np.ndarray:
    """Return scale-normalized region and contact measurements for one candidate."""
    labels = np.asarray(proposal_labels)
    region_a = labels == int(region_a_id)
    region_b = labels == int(region_b_id)
    if not region_a.any() or not region_b.any():
        raise ValueError("candidate regions are missing from proposal_labels")
    union = region_a | region_b
    area_a, area_b = float(region_a.sum()), float(region_b.sum())
    ys_a, xs_a = np.nonzero(region_a)
    ys_b, xs_b = np.nonzero(region_b)
    height, width = labels.shape
    perimeter_a = float(np.count_nonzero(region_a ^ ndimage.binary_erosion(region_a)))
    perimeter_b = float(np.count_nonzero(region_b ^ ndimage.binary_erosion(region_b)))
    contact = float(_candidate_line(region_a, region_b).sum())
    bbox_y, bbox_x = np.nonzero(union)
    bbox_h = float(bbox_y.max() - bbox_y.min() + 1)
    bbox_w = float(bbox_x.max() - bbox_x.min() + 1)
    return np.asarray(
        [
            area_a / max(height * width, 1),
            area_b / max(height * width, 1),
            min(area_a, area_b) / max(area_a, area_b, 1.0),
            perimeter_a / max(np.sqrt(area_a), 1.0),
            perimeter_b / max(np.sqrt(area_b), 1.0),
            contact / max(perimeter_a + perimeter_b, 1.0),
            (float(ys_b.mean()) - float(ys_a.mean())) / max(height, 1),
            (float(xs_b.mean()) - float(xs_a.mean())) / max(width, 1),
            bbox_h / max(height, 1),
            bbox_w / max(width, 1),
            area_a + area_b,
            float(np.hypot(ys_b.mean() - ys_a.mean(), xs_b.mean() - xs_a.mean()))
            / max(np.hypot(height, width), 1.0),
        ],
        dtype=np.float32,
    )


def build_candidate_patch(
    image: np.ndarray,
    proposal_labels: np.ndarray,
    region_a_id: int,
    region_b_id: int,
    *,
    foreground_probability: np.ndarray,
    context_boundary_probability: np.ndarray,
    cellpose_flow_y: np.ndarray,
    cellpose_flow_x: np.ndarray,
    ensemble_uncertainty: np.ndarray | None = None,
    patch_size: int = 128,
    margin_fraction: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Render the documented 14-channel candidate signature and 12 geometry features."""
    labels = np.asarray(proposal_labels)
    if labels.ndim != 2:
        raise ValueError("proposal_labels must be a 2-D integer image")
    shape = labels.shape
    source = np.asarray(image, dtype=np.float32)
    if source.ndim == 3:
        source = source.mean(axis=2)
    source = _as_float_map(source, shape, "image")
    low, high = np.percentile(source, (1.0, 99.0))
    source = np.zeros_like(source) if high <= low else np.clip((source - low) / (high - low), 0, 1)
    foreground = _as_float_map(foreground_probability, shape, "foreground_probability")
    context = _as_float_map(
        context_boundary_probability, shape, "context_boundary_probability"
    )
    flow_y = _as_float_map(cellpose_flow_y, shape, "cellpose_flow_y")
    flow_x = _as_float_map(cellpose_flow_x, shape, "cellpose_flow_x")
    uncertainty = (
        np.zeros(shape, dtype=np.float32)
        if ensemble_uncertainty is None
        else _as_float_map(ensemble_uncertainty, shape, "ensemble_uncertainty")
    )
    region_a = labels == int(region_a_id)
    region_b = labels == int(region_b_id)
    if not region_a.any() or not region_b.any():
        raise ValueError("candidate regions are missing")
    union = region_a | region_b
    candidate = _candidate_line(region_a, region_b)
    outer = union ^ ndimage.binary_erosion(union)
    magnitude = np.hypot(flow_y, flow_x).astype(np.float32)
    divergence = (np.gradient(flow_y, axis=0) + np.gradient(flow_x, axis=1)).astype(np.float32)
    channels = (
        source,
        foreground,
        context,
        flow_y,
        flow_x,
        magnitude,
        divergence,
        candidate.astype(np.float32),
        region_a.astype(np.float32),
        region_b.astype(np.float32),
        _signed_distance(region_a),
        _signed_distance(region_b),
        outer.astype(np.float32),
        uncertainty,
    )
    ys, xs = np.nonzero(union | candidate)
    extent = max(int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1))
    margin = max(4, int(round(extent * margin_fraction)))
    center_y = int(round((int(ys.min()) + int(ys.max())) / 2))
    center_x = int(round((int(xs.min()) + int(xs.max())) / 2))
    radius = int(np.ceil(extent / 2)) + margin
    pad = radius + 2
    categorical_channels = {7, 8, 9, 12}
    rendered = []
    for index, channel in enumerate(channels):
        padded = np.pad(channel, pad, mode="reflect")
        cy, cx = center_y + pad, center_x + pad
        crop = padded[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1]
        rendered.append(
            _resize_channel(crop, patch_size, categorical=index in categorical_channels)
        )
    patch = np.ascontiguousarray(np.stack(rendered), dtype=np.float32)
    if patch.shape != (len(CANDIDATE_PATCH_CHANNELS), patch_size, patch_size):
        raise RuntimeError(f"candidate patch contract failed: {patch.shape}")
    return patch, candidate_geometry_features(labels, region_a_id, region_b_id)


class CandidatePatchEncoder(nn.Module):
    def __init__(self, spec: BoundaryTeacherSpec) -> None:
        super().__init__()
        widths = (48, 72, 112, spec.embedding_size)
        blocks: list[nn.Module] = []
        incoming = spec.patch_channels
        for width in widths:
            blocks.extend(
                [
                    nn.Conv2d(incoming, width, 3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(8 if width % 8 == 0 else 4, width),
                    nn.SiLU(),
                ]
            )
            incoming = width
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, patches: Tensor) -> Tensor:
        return self.pool(self.blocks(patches)).flatten(1)


class BoundaryGraphTeacher(nn.Module):
    """Workstation teacher: candidate image encoder plus neighborhood graph transformer."""

    def __init__(self, spec: BoundaryTeacherSpec = BoundaryTeacherSpec()) -> None:
        super().__init__()
        self.spec = spec
        self.patch_encoder = CandidatePatchEncoder(spec)
        self.geometry_projection = nn.Sequential(
            nn.Linear(spec.geometry_features, spec.embedding_size), nn.GELU()
        )
        layer = nn.TransformerEncoderLayer(
            d_model=spec.embedding_size,
            nhead=spec.graph_heads,
            dim_feedforward=spec.embedding_size * 4,
            dropout=spec.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.graph = nn.TransformerEncoder(layer, spec.graph_layers)
        self.keep_head = nn.Linear(spec.embedding_size, 1)
        self.utility_head = nn.Linear(spec.embedding_size, 1)
        self.uncertainty_head = nn.Linear(spec.embedding_size, 1)

    def forward(
        self,
        patches: Tensor,
        geometry: Tensor,
        valid: Tensor,
    ) -> dict[str, Tensor]:
        if patches.ndim != 5:
            raise ValueError("patches must have shape [B,K,C,H,W]")
        batch, candidates, channels, height, width = patches.shape
        if channels != self.spec.patch_channels or height != self.spec.patch_size or width != self.spec.patch_size:
            raise ValueError("candidate patch shape does not match BoundaryTeacherSpec")
        if geometry.shape != (batch, candidates, self.spec.geometry_features):
            raise ValueError("candidate geometry shape does not match BoundaryTeacherSpec")
        if valid.shape != (batch, candidates):
            raise ValueError("valid must have shape [B,K]")
        embedded = self.patch_encoder(patches.reshape(batch * candidates, channels, height, width))
        embedded = embedded.reshape(batch, candidates, -1)
        embedded = embedded + self.geometry_projection(geometry)
        tokens = self.graph(embedded, src_key_padding_mask=~valid.bool())
        return {
            "keep_logit": self.keep_head(tokens).squeeze(-1),
            "utility": torch.tanh(self.utility_head(tokens).squeeze(-1)),
            "log_variance": torch.clamp(self.uncertainty_head(tokens).squeeze(-1), -6, 6),
            "embedding": tokens,
        }


class BoundaryCandidateStudent(nn.Module):
    """Fixed-patch local student for Core ML or dense-student distillation."""

    def __init__(self, spec: BoundaryTeacherSpec = BoundaryTeacherSpec()) -> None:
        super().__init__()
        self.spec = spec
        self.encoder = CandidatePatchEncoder(spec)
        self.geometry_projection = nn.Sequential(
            nn.Linear(spec.geometry_features, spec.embedding_size), nn.GELU()
        )
        self.head = nn.Sequential(
            nn.Linear(spec.embedding_size, spec.embedding_size),
            nn.GELU(),
            nn.Linear(spec.embedding_size, 3),
        )

    def forward(self, patch: Tensor, geometry: Tensor) -> Tensor:
        embedding = self.encoder(patch) + self.geometry_projection(geometry)
        return self.head(embedding)  # keep logit, utility, log variance


@dataclass(frozen=True)
class BoundaryLossConfig:
    keep_weight: float = 1.0
    utility_weight: float = 0.5
    distillation_weight: float = 0.7
    uncertainty_weight: float = 0.05
    positive_weight: float = 1.5


def boundary_teacher_loss(
    outputs: Mapping[str, Tensor],
    labels: Tensor,
    utility: Tensor,
    valid: Tensor,
    *,
    teacher_keep_probability: Tensor | None = None,
    config: BoundaryLossConfig = BoundaryLossConfig(),
) -> dict[str, Tensor]:
    valid_float = valid.to(outputs["keep_logit"].dtype)
    denominator = valid_float.sum().clamp_min(1.0)
    keep_loss = F.binary_cross_entropy_with_logits(
        outputs["keep_logit"],
        labels.to(outputs["keep_logit"].dtype),
        pos_weight=torch.as_tensor(config.positive_weight, device=labels.device),
        reduction="none",
    )
    utility_loss = F.smooth_l1_loss(outputs["utility"], utility, reduction="none")
    supervised = (
        config.keep_weight * (keep_loss * valid_float).sum() / denominator
        + config.utility_weight * (utility_loss * valid_float).sum() / denominator
    )
    components = {"supervised": supervised}
    if teacher_keep_probability is not None:
        distillation = F.binary_cross_entropy_with_logits(
            outputs["keep_logit"], teacher_keep_probability, reduction="none"
        )
        components["distillation"] = (
            config.distillation_weight * (distillation * valid_float).sum() / denominator
        )
    log_variance = outputs["log_variance"]
    components["uncertainty"] = config.uncertainty_weight * (
        (torch.exp(-log_variance) * keep_loss + log_variance) * valid_float
    ).sum() / denominator
    components["total"] = torch.stack(tuple(components.values())).sum()
    return components


def run_contract_self_test() -> dict[str, object]:
    labels = np.zeros((64, 72), dtype=np.int32)
    labels[12:48, 8:30] = 1
    labels[12:48, 30:54] = 2
    image = np.linspace(0, 1, labels.size, dtype=np.float32).reshape(labels.shape)
    foreground = (labels > 0).astype(np.float32)
    boundary = _candidate_line(labels == 1, labels == 2).astype(np.float32)
    patch, geometry = build_candidate_patch(
        image,
        labels,
        1,
        2,
        foreground_probability=foreground,
        context_boundary_probability=boundary,
        cellpose_flow_y=np.zeros_like(image),
        cellpose_flow_x=np.zeros_like(image),
        patch_size=64,
    )
    spec = BoundaryTeacherSpec(patch_size=64, embedding_size=96, graph_layers=1, graph_heads=4)
    teacher = BoundaryGraphTeacher(spec)
    patches = torch.from_numpy(patch)[None, None].repeat(2, 3, 1, 1, 1)
    geometries = torch.from_numpy(geometry)[None, None].repeat(2, 3, 1)
    valid = torch.tensor([[True, True, False], [True, True, True]])
    outputs = teacher(patches, geometries, valid)
    labels_tensor = torch.ones(2, 3)
    utility_tensor = torch.full((2, 3), 0.5)
    loss = boundary_teacher_loss(outputs, labels_tensor, utility_tensor, valid)
    loss["total"].backward()
    if outputs["keep_logit"].shape != (2, 3) or not torch.isfinite(loss["total"]):
        raise AssertionError("boundary teacher contract failed")
    student = BoundaryCandidateStudent(spec)
    student_output = student(patches[:, 0], geometries[:, 0])
    if student_output.shape != (2, 3):
        raise AssertionError("boundary candidate student contract failed")
    return {
        "status": "PASS",
        "version": BOUNDARY_TEACHER_VERSION,
        "patch_shape": list(patch.shape),
        "geometry_shape": list(geometry.shape),
        "teacher_spec": asdict(spec),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_contract_self_test(), indent=2))
