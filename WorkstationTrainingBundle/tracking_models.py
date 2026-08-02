#!/usr/bin/env python3
"""CellectTrack models, losses, and optional Trackastra teacher hooks.

This is an independent tracking-by-detection design for Cellect, not Trackastra source code.  Each
sample contains a fixed, padded collection of detected-cell tokens.  A token combines ``time/y/x``
coordinates with appearance, shape, Cellpose-style flow, and neighborhood measurements derived
from the segmentation pipeline.  Alternating spatial and temporal relative-attention blocks build
contextual cell representations, while an explicit relative-motion scorer predicts directed links.

Trackastra is an optional workstation-only teacher.  It is never imported at module load, network
access is never implicit, and its v0.3.0 ``general_2d`` artifact is SHA-256 pinned.  The mobile
CellectTrack model and fixed-shape deployment adapter do not depend on Trackastra.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


CELLECT_TRACK_VERSION = "cellect-track-spatiotemporal-v1"
TRACKING_LOSS_VERSION = "assignment-division-distillation-v1"
TRACKASTRA_TEACHER_VERSION = "0.3.0"
INVALID_ASSOCIATION_LOGIT = -10_000.0

# This explicit order is the contract between segmentation feature extraction and CellectTrack.
# Coordinates are supplied separately as time/y/x and are not repeated here.
CELL_FEATURE_NAMES: tuple[str, ...] = (
    "intensity_mean",
    "intensity_std",
    "intensity_p10",
    "intensity_median",
    "intensity_p90",
    "gradient_mean",
    "texture_energy",
    "log_area",
    "log_perimeter",
    "compactness",
    "eccentricity",
    "solidity",
    "extent",
    "axis_ratio",
    "signed_distance_mean",
    "signed_distance_max",
    "flow_y_mean",
    "flow_x_mean",
    "flow_coherence",
    "flow_divergence",
    "neighbor_count",
    "nearest_neighbor_distance",
    "local_density",
    "contact_fraction",
)
CELL_FEATURE_DIM = len(CELL_FEATURE_NAMES)


@dataclass(frozen=True)
class TeacherArtifact:
    """Immutable metadata for an optional, externally maintained teacher model."""

    name: str
    package: str
    package_version: str
    url: str
    sha256: str
    source: str
    license_name: str


TRACKASTRA_TEACHER_REGISTRY: Mapping[str, TeacherArtifact] = {
    "general_2d": TeacherArtifact(
        name="general_2d",
        package="trackastra",
        package_version=TRACKASTRA_TEACHER_VERSION,
        url=(
            "https://github.com/weigertlab/trackastra-models/releases/download/"
            "v0.3.0/general_2d.zip"
        ),
        sha256="35cefd8634860d1dd43bcdcbdef7ae0caa24445f19bcfd35ec6b19039f2cd876",
        source="https://github.com/weigertlab/trackastra/tree/0.3.0",
        license_name="BSD-3-Clause",
    )
}


@dataclass(frozen=True)
class CellectTrackConfig:
    """Fixed-shape architecture and motion-window configuration."""

    max_tokens: int
    cell_feature_dim: int = CELL_FEATURE_DIM
    model_dim: int = 128
    num_heads: int = 8
    num_layers: int = 4
    pair_hidden_dim: int = 192
    feedforward_multiplier: int = 4
    dropout: float = 0.1
    max_frame_gap: float = 2.0

    def __post_init__(self) -> None:
        if self.max_tokens < 2:
            raise ValueError("max_tokens must be at least two")
        if self.cell_feature_dim < 1:
            raise ValueError("cell_feature_dim must be positive")
        if self.model_dim < 8 or self.model_dim % self.num_heads:
            raise ValueError("model_dim must be >= 8 and divisible by num_heads")
        if self.num_layers < 1 or self.pair_hidden_dim < 1:
            raise ValueError("num_layers and pair_hidden_dim must be positive")
        if self.feedforward_multiplier < 1:
            raise ValueError("feedforward_multiplier must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be within [0, 1)")
        if self.max_frame_gap <= 0:
            raise ValueError("max_frame_gap must be positive")


@dataclass
class CellectTrackOutput:
    """Dense fixed-shape outputs; invalid associations have a large negative logit."""

    association_logits: torch.Tensor
    association_mask: torch.Tensor
    division_logits: torch.Tensor
    division_probability: torch.Tensor
    birth_logits: torch.Tensor
    birth_probability: torch.Tensor
    death_logits: torch.Tensor
    death_probability: torch.Tensor
    uncertainty_logits: torch.Tensor
    uncertainty: torch.Tensor
    token_embeddings: torch.Tensor
    token_mask: torch.Tensor


class RelativeMultiheadAttention(nn.Module):
    """Self-attention with learned relative time, displacement, and distance bias."""

    def __init__(self, model_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.output = nn.Linear(model_dim, model_dim)
        bias_hidden = max(16, 4 * num_heads)
        self.relative_bias = nn.Sequential(
            nn.Linear(5, bias_hidden),
            nn.SiLU(),
            nn.Linear(bias_hidden, num_heads),
        )
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def relative_features(coordinates: torch.Tensor) -> torch.Tensor:
        # rel[b, i, j] describes candidate motion from token i to token j.
        relative = coordinates.unsqueeze(1) - coordinates.unsqueeze(2)
        absolute_time = relative[..., :1].abs()
        spatial_distance = torch.sqrt(
            relative[..., 1:2].square()
            + relative[..., 2:3].square()
            + 1e-8
        )
        return torch.cat([relative, absolute_time, spatial_distance], dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,
        coordinates: torch.Tensor,
        token_mask: torch.Tensor,
        allowed_pairs: torch.Tensor,
    ) -> torch.Tensor:
        batch, token_count, _ = tokens.shape
        qkv = self.qkv(tokens).reshape(
            batch, token_count, 3, self.num_heads, self.head_dim
        )
        queries = qkv[:, :, 0].permute(0, 2, 1, 3)
        keys = qkv[:, :, 1].permute(0, 2, 1, 3)
        values = qkv[:, :, 2].permute(0, 2, 1, 3)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        relative_bias = self.relative_bias(self.relative_features(coordinates))
        scores = scores + relative_bias.permute(0, 3, 1, 2)

        valid_pairs = (
            token_mask.unsqueeze(1)
            & token_mask.unsqueeze(2)
            & allowed_pairs
        )
        scores = torch.where(
            valid_pairs.unsqueeze(1),
            scores,
            torch.full_like(scores, INVALID_ASSOCIATION_LOGIT),
        )
        weights = torch.softmax(scores, dim=-1)
        weights = weights * valid_pairs.unsqueeze(1).to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        attended = torch.matmul(self.dropout(weights), values)
        attended = attended.permute(0, 2, 1, 3).reshape(
            batch, token_count, self.model_dim
        )
        attended = self.output(attended)
        return attended * token_mask.unsqueeze(-1).to(attended.dtype)


class SpatioTemporalTransformerBlock(nn.Module):
    """Same-frame spatial attention followed by cross-frame temporal attention."""

    def __init__(self, config: CellectTrackConfig) -> None:
        super().__init__()
        self.spatial_norm = nn.LayerNorm(config.model_dim)
        self.temporal_norm = nn.LayerNorm(config.model_dim)
        self.feedforward_norm = nn.LayerNorm(config.model_dim)
        self.spatial_attention = RelativeMultiheadAttention(
            config.model_dim, config.num_heads, config.dropout
        )
        self.temporal_attention = RelativeMultiheadAttention(
            config.model_dim, config.num_heads, config.dropout
        )
        hidden_dim = config.model_dim * config.feedforward_multiplier
        self.feedforward = nn.Sequential(
            nn.Linear(config.model_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.model_dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        coordinates: torch.Tensor,
        token_mask: torch.Tensor,
        spatial_pairs: torch.Tensor,
        temporal_pairs: torch.Tensor,
    ) -> torch.Tensor:
        tokens = tokens + self.spatial_attention(
            self.spatial_norm(tokens), coordinates, token_mask, spatial_pairs
        )
        tokens = tokens + self.temporal_attention(
            self.temporal_norm(tokens), coordinates, token_mask, temporal_pairs
        )
        tokens = tokens + self.feedforward(self.feedforward_norm(tokens))
        return tokens * token_mask.unsqueeze(-1).to(tokens.dtype)


class RelativeMotionPairScorer(nn.Module):
    """Directed association scorer with explicit relative displacement and velocity."""

    def __init__(self, model_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        # left, right, absolute feature difference, feature product, and eight motion values.
        pair_dim = 4 * model_dim + 8
        self.network = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(hidden_dim // 2, 16)),
            nn.GELU(),
            nn.Linear(max(hidden_dim // 2, 16), 1),
        )

    def forward(
        self, token_embeddings: torch.Tensor, coordinates: torch.Tensor
    ) -> torch.Tensor:
        left = token_embeddings.unsqueeze(2).expand(
            -1, -1, token_embeddings.shape[1], -1
        )
        right = token_embeddings.unsqueeze(1).expand(
            -1, token_embeddings.shape[1], -1, -1
        )
        relative = coordinates.unsqueeze(1) - coordinates.unsqueeze(2)
        absolute = relative.abs()
        time_delta = relative[..., :1]
        spatial_distance = torch.sqrt(
            relative[..., 1:2].square()
            + relative[..., 2:3].square()
            + 1e-8
        )
        speed = spatial_distance / time_delta.abs().clamp_min(1e-3)
        motion = torch.cat([relative, absolute, spatial_distance, speed], dim=-1)
        pair_features = torch.cat(
            [left, right, (right - left).abs(), right * left, motion], dim=-1
        )
        return self.network(pair_features).squeeze(-1)


class CellectTrack(nn.Module):
    """Contextual fixed-token cell linker shared by heavy and mobile configurations.

    ``time_yx`` has shape ``[B, N, 3]``.  Time is expressed in frame units; y/x should be
    normalized to approximately ``[-1, 1]`` using the source image dimensions. ``cell_features``
    has shape ``[B, N, config.cell_feature_dim]`` in ``CELL_FEATURE_NAMES`` order, and
    ``cell_mask`` has shape ``[B, N]`` with one for real cells and zero for padding.
    """

    def __init__(self, config: CellectTrackConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(config.cell_feature_dim),
            nn.Linear(config.cell_feature_dim, config.model_dim),
            nn.GELU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.coordinate_projection = nn.Sequential(
            nn.Linear(3, config.model_dim),
            nn.SiLU(),
            nn.Linear(config.model_dim, config.model_dim),
        )
        self.input_norm = nn.LayerNorm(config.model_dim)
        self.blocks = nn.ModuleList(
            SpatioTemporalTransformerBlock(config)
            for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.model_dim)
        self.pair_scorer = RelativeMotionPairScorer(
            config.model_dim, config.pair_hidden_dim, config.dropout
        )
        self.event_heads = nn.Linear(config.model_dim, 4)

    def _validate_inputs(
        self,
        time_yx: torch.Tensor,
        cell_features: torch.Tensor,
        cell_mask: torch.Tensor,
    ) -> None:
        if time_yx.ndim != 3 or time_yx.shape[-1] != 3:
            raise ValueError("time_yx must have shape [B, N, 3]")
        if cell_features.ndim != 3:
            raise ValueError("cell_features must have shape [B, N, F]")
        if cell_mask.ndim != 2:
            raise ValueError("cell_mask must have shape [B, N]")
        if time_yx.shape[:2] != cell_features.shape[:2] or (
            time_yx.shape[:2] != cell_mask.shape
        ):
            raise ValueError("time_yx, cell_features, and cell_mask disagree on B/N")
        if time_yx.shape[1] != self.config.max_tokens:
            raise ValueError(
                f"Expected exactly {self.config.max_tokens} padded tokens; "
                f"received {time_yx.shape[1]}"
            )
        if cell_features.shape[-1] != self.config.cell_feature_dim:
            raise ValueError(
                f"Expected {self.config.cell_feature_dim} cell features; "
                f"received {cell_features.shape[-1]}"
            )

    def forward(
        self,
        time_yx: torch.Tensor,
        cell_features: torch.Tensor,
        cell_mask: torch.Tensor,
    ) -> CellectTrackOutput:
        # The deployment adapter has a deliberately fixed signature.  Eager training receives clear
        # shape errors, while tracing avoids converting symbolic shape values to Python booleans.
        if not torch.jit.is_tracing() and not torch.jit.is_scripting():
            self._validate_inputs(time_yx, cell_features, cell_mask)
        if not time_yx.is_floating_point():
            time_yx = time_yx.float()
        if not cell_features.is_floating_point():
            cell_features = cell_features.float()
        token_mask = cell_mask > 0.5
        mask_float = token_mask.unsqueeze(-1).to(cell_features.dtype)

        # Centering makes absolute acquisition frame numbers irrelevant while preserving motion.
        valid_count = token_mask.sum(dim=1, keepdim=True).clamp_min(1)
        time_center = (
            time_yx[..., 0] * token_mask.to(time_yx.dtype)
        ).sum(dim=1, keepdim=True) / valid_count.to(time_yx.dtype)
        normalized_coordinates = torch.stack(
            [
                (time_yx[..., 0] - time_center) / self.config.max_frame_gap,
                time_yx[..., 1],
                time_yx[..., 2],
            ],
            dim=-1,
        )
        tokens = self.feature_projection(cell_features) + self.coordinate_projection(
            normalized_coordinates
        )
        tokens = self.input_norm(tokens) * mask_float

        time_difference = time_yx[:, None, :, 0] - time_yx[:, :, None, 0]
        same_frame = time_difference.abs() <= 1e-5
        token_count = time_yx.shape[1]
        identity = torch.eye(
            token_count, dtype=torch.bool, device=time_yx.device
        ).unsqueeze(0)
        temporal_pairs = (
            ((time_difference.abs() <= self.config.max_frame_gap) & ~same_frame)
            | identity
        )
        for block in self.blocks:
            tokens = block(
                tokens,
                normalized_coordinates,
                token_mask,
                same_frame,
                temporal_pairs,
            )
        embeddings = self.final_norm(tokens) * mask_float

        raw_association_logits = self.pair_scorer(
            embeddings, normalized_coordinates
        )
        association_mask = (
            token_mask.unsqueeze(2)
            & token_mask.unsqueeze(1)
            & (time_difference > 1e-5)
            & (time_difference <= self.config.max_frame_gap)
        )
        association_logits = torch.where(
            association_mask,
            raw_association_logits,
            torch.full_like(raw_association_logits, INVALID_ASSOCIATION_LOGIT),
        )

        event_logits = self.event_heads(embeddings)
        event_logits = event_logits * mask_float
        division_logits, birth_logits, death_logits, uncertainty_logits = (
            event_logits.unbind(dim=-1)
        )
        node_mask = token_mask.to(event_logits.dtype)
        return CellectTrackOutput(
            association_logits=association_logits,
            association_mask=association_mask,
            division_logits=division_logits,
            division_probability=torch.sigmoid(division_logits) * node_mask,
            birth_logits=birth_logits,
            birth_probability=torch.sigmoid(birth_logits) * node_mask,
            death_logits=death_logits,
            death_probability=torch.sigmoid(death_logits) * node_mask,
            uncertainty_logits=uncertainty_logits,
            uncertainty=torch.sigmoid(uncertainty_logits) * node_mask,
            token_embeddings=embeddings,
            token_mask=token_mask,
        )


def heavy_track_config(
    *, max_tokens: int = 256, cell_feature_dim: int = CELL_FEATURE_DIM
) -> CellectTrackConfig:
    """Return the workstation teacher/final-heavy configuration."""

    return CellectTrackConfig(
        max_tokens=max_tokens,
        cell_feature_dim=cell_feature_dim,
        model_dim=192,
        num_heads=8,
        num_layers=6,
        pair_hidden_dim=256,
        feedforward_multiplier=4,
        dropout=0.10,
        max_frame_gap=2.0,
    )


def mobile_track_config(
    *, max_tokens: int = 128, cell_feature_dim: int = CELL_FEATURE_DIM
) -> CellectTrackConfig:
    """Return the fixed-shape Core-ML deployment configuration."""

    return CellectTrackConfig(
        max_tokens=max_tokens,
        cell_feature_dim=cell_feature_dim,
        model_dim=64,
        num_heads=4,
        num_layers=2,
        pair_hidden_dim=96,
        feedforward_multiplier=2,
        dropout=0.0,
        max_frame_gap=2.0,
    )


class CellectTrackHeavy(CellectTrack):
    """High-capacity workstation CellectTrack model."""

    def __init__(
        self,
        *,
        max_tokens: int = 256,
        cell_feature_dim: int = CELL_FEATURE_DIM,
        config: CellectTrackConfig | None = None,
    ) -> None:
        super().__init__(
            config
            if config is not None
            else heavy_track_config(
                max_tokens=max_tokens, cell_feature_dim=cell_feature_dim
            )
        )


class CellectTrackMobile(CellectTrack):
    """Compact CellectTrack student intended for Core ML conversion."""

    def __init__(
        self,
        *,
        max_tokens: int = 128,
        cell_feature_dim: int = CELL_FEATURE_DIM,
        config: CellectTrackConfig | None = None,
    ) -> None:
        super().__init__(
            config
            if config is not None
            else mobile_track_config(
                max_tokens=max_tokens, cell_feature_dim=cell_feature_dim
            )
        )


class CellectTrackDeploymentAdapter(nn.Module):
    """Core-ML-friendly tuple adapter with fixed token and feature dimensions."""

    def __init__(self, model: CellectTrack) -> None:
        super().__init__()
        self.model = model

    @property
    def fixed_input_shapes(self) -> dict[str, tuple[int, ...]]:
        return {
            "time_yx": (1, self.model.config.max_tokens, 3),
            "cell_features": (
                1,
                self.model.config.max_tokens,
                self.model.config.cell_feature_dim,
            ),
            "cell_mask": (1, self.model.config.max_tokens),
        }

    def forward(
        self,
        time_yx: torch.Tensor,
        cell_features: torch.Tensor,
        cell_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        output = self.model(time_yx, cell_features, cell_mask)
        return (
            output.association_logits,
            output.association_mask.to(output.association_logits.dtype),
            output.division_probability,
            output.birth_probability,
            output.death_probability,
            output.uncertainty,
        )


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def masked_sinkhorn(
    logits: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    iterations: int = 8,
    temperature: float = 0.25,
) -> torch.Tensor:
    """Differentiable masked row/column normalization for soft assignment."""

    if logits.shape != valid_mask.shape or logits.ndim != 3:
        raise ValueError("logits and valid_mask must have matching [B, N, N] shapes")
    if iterations < 1 or temperature <= 0:
        raise ValueError("iterations and temperature must be positive")
    mask = valid_mask.to(logits.dtype)
    scaled = logits / temperature
    scaled = scaled - torch.where(
        valid_mask, scaled, torch.full_like(scaled, -torch.inf)
    ).amax(dim=(-2, -1), keepdim=True).clamp(min=-50.0, max=50.0)
    probabilities = torch.exp(scaled.clamp(min=-50.0, max=50.0)) * mask
    for _ in range(iterations):
        row_sum = probabilities.sum(dim=-1, keepdim=True)
        probabilities = torch.where(
            row_sum > 0, probabilities / row_sum.clamp_min(1e-8), probabilities
        )
        column_sum = probabilities.sum(dim=-2, keepdim=True)
        probabilities = torch.where(
            column_sum > 0,
            probabilities / column_sum.clamp_min(1e-8),
            probabilities,
        )
        probabilities = probabilities * mask
    return probabilities


def differentiable_assignment_loss(
    output: CellectTrackOutput,
    association_target: torch.Tensor,
    *,
    supervision_mask: torch.Tensor | None = None,
    positive_weight: float = 5.0,
    sinkhorn_weight: float = 0.25,
    sinkhorn_temperature: float = 0.25,
) -> torch.Tensor:
    """Combine calibrated link BCE with a differentiable soft-assignment objective."""

    if association_target.shape != output.association_logits.shape:
        raise ValueError("association_target must match association_logits")
    valid = output.association_mask
    if supervision_mask is not None:
        if supervision_mask.shape != valid.shape:
            raise ValueError("supervision_mask must match association_logits")
        valid = valid & (supervision_mask > 0)
    target = association_target.to(output.association_logits.dtype).clamp(0.0, 1.0)
    elementwise = F.binary_cross_entropy_with_logits(
        output.association_logits,
        target,
        reduction="none",
        pos_weight=torch.as_tensor(
            positive_weight,
            dtype=output.association_logits.dtype,
            device=output.association_logits.device,
        ),
    )
    binary_loss = _masked_mean(elementwise, valid)
    soft_assignment = masked_sinkhorn(
        output.association_logits,
        valid,
        temperature=sinkhorn_temperature,
    )
    target_rows = target * valid.to(target.dtype)
    target_rows = target_rows / target_rows.sum(dim=-1, keepdim=True).clamp_min(1.0)
    positive_rows = target_rows.sum(dim=-1) > 0
    assignment_nll = -(target_rows * soft_assignment.clamp_min(1e-8).log()).sum(
        dim=-1
    )
    assignment_loss = _masked_mean(assignment_nll, positive_rows)
    return binary_loss + sinkhorn_weight * assignment_loss


def differentiable_division_loss(
    output: CellectTrackOutput,
    division_target: torch.Tensor,
    *,
    supervision_mask: torch.Tensor | None = None,
    association_supervision_mask: torch.Tensor | None = None,
    consistency_weight: float = 0.25,
) -> torch.Tensor:
    """Supervise divisions and couple them to the differentiable outgoing-link count."""

    if division_target.shape != output.division_logits.shape:
        raise ValueError("division_target must match division logits")
    target = division_target.to(output.division_logits.dtype).clamp(0.0, 1.0)
    node_mask = output.token_mask
    if supervision_mask is not None:
        if supervision_mask.shape != output.division_logits.shape:
            raise ValueError("supervision_mask must match division logits")
        node_mask = node_mask & (supervision_mask > 0)
    classification = _masked_mean(
        F.binary_cross_entropy_with_logits(
            output.division_logits, target, reduction="none"
        ),
        node_mask,
    )
    link_probability = torch.sigmoid(output.association_logits)
    valid_associations = output.association_mask
    if association_supervision_mask is not None:
        if association_supervision_mask.shape != output.association_logits.shape:
            raise ValueError(
                "association_supervision_mask must match association logits"
            )
        valid_associations = valid_associations & (association_supervision_mask > 0)
    link_probability = link_probability * valid_associations.to(link_probability.dtype)
    expected_children = link_probability.sum(dim=-1)
    link_implied_division = torch.sigmoid(4.0 * (expected_children - 1.5))
    consistency = _masked_mean(
        (output.division_probability - link_implied_division).square(),
        node_mask,
    )
    return classification + consistency_weight * consistency


def tracking_supervised_losses(
    output: CellectTrackOutput,
    association_target: torch.Tensor,
    division_target: torch.Tensor,
    birth_target: torch.Tensor,
    death_target: torch.Tensor,
    *,
    uncertainty_target: torch.Tensor | None = None,
    supervision_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Return differentiable supervised components and their summed ``total`` loss."""

    node_shape = output.division_logits.shape
    for name, target in (
        ("division_target", division_target),
        ("birth_target", birth_target),
        ("death_target", death_target),
    ):
        if target.shape != node_shape:
            raise ValueError(f"{name} must have shape {node_shape}")
    assignment = differentiable_assignment_loss(
        output,
        association_target,
        supervision_mask=supervision_mask,
    )
    division = differentiable_division_loss(output, division_target)
    birth = _masked_mean(
        F.binary_cross_entropy_with_logits(
            output.birth_logits,
            birth_target.to(output.birth_logits.dtype).clamp(0.0, 1.0),
            reduction="none",
        ),
        output.token_mask,
    )
    death = _masked_mean(
        F.binary_cross_entropy_with_logits(
            output.death_logits,
            death_target.to(output.death_logits.dtype).clamp(0.0, 1.0),
            reduction="none",
        ),
        output.token_mask,
    )

    if uncertainty_target is None:
        link_errors = (
            torch.sigmoid(output.association_logits)
            - association_target.to(output.association_logits.dtype)
        ).abs() * output.association_mask.to(output.association_logits.dtype)
        outgoing_count = output.association_mask.sum(dim=-1).clamp_min(1)
        link_error = link_errors.sum(dim=-1) / outgoing_count.to(link_errors.dtype)
        event_error = (
            (output.division_probability - division_target.to(link_error.dtype)).abs()
            + (output.birth_probability - birth_target.to(link_error.dtype)).abs()
            + (output.death_probability - death_target.to(link_error.dtype)).abs()
        ) / 3.0
        calibrated_uncertainty = ((link_error + event_error) / 2.0).detach()
    else:
        if uncertainty_target.shape != node_shape:
            raise ValueError("uncertainty_target must match node outputs")
        calibrated_uncertainty = uncertainty_target.to(
            output.uncertainty_logits.dtype
        ).clamp(0.0, 1.0)
    uncertainty = _masked_mean(
        F.binary_cross_entropy_with_logits(
            output.uncertainty_logits,
            calibrated_uncertainty,
            reduction="none",
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


def tracking_distillation_losses(
    student: CellectTrackOutput,
    teacher: CellectTrackOutput,
    *,
    temperature: float = 2.0,
    association_weight: float = 1.0,
    event_weight: float = 0.5,
    uncertainty_weight: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Distill dense associations and calibrated biological-event probabilities."""

    if student.association_logits.shape != teacher.association_logits.shape:
        raise ValueError("Teacher and student association shapes differ")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    valid = student.association_mask & teacher.association_mask
    teacher_links = torch.sigmoid(teacher.association_logits.detach() / temperature)
    link_soft_cross_entropy = F.binary_cross_entropy_with_logits(
        student.association_logits / temperature,
        teacher_links,
        reduction="none",
    ) * temperature**2
    association = _masked_mean(link_soft_cross_entropy, valid)

    node_mask = student.token_mask & teacher.token_mask
    event = _masked_mean(
        (student.division_probability - teacher.division_probability.detach()).square()
        + (student.birth_probability - teacher.birth_probability.detach()).square()
        + (student.death_probability - teacher.death_probability.detach()).square(),
        node_mask,
    )
    uncertainty = _masked_mean(
        (student.uncertainty - teacher.uncertainty.detach()).square(), node_mask
    )
    total = (
        association_weight * association
        + event_weight * event
        + uncertainty_weight * uncertainty
    )
    return {
        "total": total,
        "association": association,
        "events": event,
        "uncertainty": uncertainty,
    }


TRAINING_PHASES = ("heads", "adapters", "full", "distillation", "frozen")


def set_training_phase(model: CellectTrack, phase: str) -> dict[str, Any]:
    """Apply an explicit freezing phase and return the resulting parameter audit."""

    if phase not in TRAINING_PHASES:
        raise ValueError(f"Unknown phase {phase!r}; choose from {TRAINING_PHASES}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    if phase in {"full", "distillation"}:
        for parameter in model.parameters():
            parameter.requires_grad = True
    elif phase == "heads":
        for module in (model.pair_scorer, model.event_heads):
            for parameter in module.parameters():
                parameter.requires_grad = True
    elif phase == "adapters":
        for module in (
            model.feature_projection,
            model.coordinate_projection,
            model.input_norm,
            model.final_norm,
            model.pair_scorer,
            model.event_heads,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = True
    return gradient_audit(model, phase=phase)


def gradient_audit(model: nn.Module, *, phase: str | None = None) -> dict[str, Any]:
    """Report frozen/trainable parameters plus missing or non-finite gradients."""

    trainable_names: list[str] = []
    frozen_names: list[str] = []
    missing_gradient: list[str] = []
    nonfinite_gradient: list[str] = []
    with_gradient: list[str] = []
    trainable_parameters = 0
    total_parameters = 0
    for name, parameter in model.named_parameters():
        total_parameters += parameter.numel()
        if parameter.requires_grad:
            trainable_names.append(name)
            trainable_parameters += parameter.numel()
            if parameter.grad is None:
                missing_gradient.append(name)
            else:
                with_gradient.append(name)
                if not bool(torch.isfinite(parameter.grad).all()):
                    nonfinite_gradient.append(name)
        else:
            frozen_names.append(name)
    return {
        "phase": phase,
        "total_parameter_count": total_parameters,
        "trainable_parameter_count": trainable_parameters,
        "frozen_parameter_count": total_parameters - trainable_parameters,
        "trainable_names": tuple(trainable_names),
        "frozen_names": tuple(frozen_names),
        "with_gradient": tuple(with_gradient),
        "missing_gradient": tuple(missing_gradient),
        "nonfinite_gradient": tuple(nonfinite_gradient),
    }


def assert_gradient_audit(
    model: nn.Module,
    *,
    phase: str | None = None,
    require_all_trainable_gradients: bool = True,
) -> dict[str, Any]:
    """Raise when a phase has non-finite or unexpectedly absent gradients."""

    audit = gradient_audit(model, phase=phase)
    if audit["nonfinite_gradient"]:
        raise RuntimeError(
            "Non-finite CellectTrack gradients: "
            + ", ".join(audit["nonfinite_gradient"])
        )
    if require_all_trainable_gradients and audit["missing_gradient"]:
        raise RuntimeError(
            "Trainable CellectTrack parameters without gradients: "
            + ", ".join(audit["missing_gradient"])
        )
    return audit


def teacher_artifact(name: str = "general_2d") -> TeacherArtifact:
    """Return a pinned teacher registry record without importing Trackastra."""

    try:
        return TRACKASTRA_TEACHER_REGISTRY[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown Trackastra teacher {name!r}; available: "
            f"{tuple(TRACKASTRA_TEACHER_REGISTRY)}"
        ) from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_extract_zip(
    archive: Path,
    destination: Path,
    *,
    maximum_files: int = 10_000,
    maximum_uncompressed_bytes: int = 4 * 1024**3,
) -> None:
    """Extract regular files only, rejecting traversal, links, and oversized archives."""

    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archive, "r") as bundle:
        entries = bundle.infolist()
        if len(entries) > maximum_files:
            raise RuntimeError("Teacher archive contains too many files")
        if sum(entry.file_size for entry in entries) > maximum_uncompressed_bytes:
            raise RuntimeError("Teacher archive expands beyond the safety limit")
        for entry in entries:
            relative = PurePosixPath(entry.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"Unsafe teacher archive path: {entry.filename!r}")
            mode = (entry.external_attr >> 16) & 0o170000
            if stat.S_ISLNK(mode):
                raise RuntimeError(f"Teacher archive contains a link: {entry.filename!r}")
            target = destination.joinpath(*relative.parts)
            target_resolved = target.resolve()
            if target_resolved != destination_resolved and (
                destination_resolved not in target_resolved.parents
            ):
                raise RuntimeError(f"Teacher archive escaped destination: {entry.filename!r}")
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(entry, "r") as source, target.open("xb") as sink:
                shutil.copyfileobj(source, sink, length=1024 * 1024)


def _artifact_manifest(model_directory: Path) -> dict[str, str]:
    manifest: dict[str, str] = {}
    for path in sorted(model_directory.rglob("*")):
        if path.is_file() and path.name != ".cellect_teacher_artifact.json":
            manifest[path.relative_to(model_directory).as_posix()] = _sha256(path)
    return manifest


def _write_artifact_marker(
    model_directory: Path, artifact: TeacherArtifact
) -> None:
    marker = {
        "name": artifact.name,
        "package": artifact.package,
        "package_version": artifact.package_version,
        "url": artifact.url,
        "archive_sha256": artifact.sha256,
        "files": _artifact_manifest(model_directory),
    }
    marker_path = model_directory / ".cellect_teacher_artifact.json"
    marker_path.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")


def verify_teacher_directory(
    model_directory: Path | str, name: str = "general_2d"
) -> bool:
    """Verify the downloader marker and every extracted file digest."""

    artifact = teacher_artifact(name)
    directory = Path(model_directory).expanduser().resolve()
    marker_path = directory / ".cellect_teacher_artifact.json"
    if not directory.is_dir() or not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    expected_metadata = {
        "name": artifact.name,
        "package": artifact.package,
        "package_version": artifact.package_version,
        "url": artifact.url,
        "archive_sha256": artifact.sha256,
    }
    if any(marker.get(key) != value for key, value in expected_metadata.items()):
        return False
    files = marker.get("files")
    if not isinstance(files, dict) or not files:
        return False
    observed_files = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name != ".cellect_teacher_artifact.json"
    }
    if observed_files != set(files):
        return False
    required = {"config.yaml", "model.pt", "train_config.yaml"}
    if not required <= set(files):
        return False
    for relative_name, expected_sha256 in files.items():
        relative = PurePosixPath(relative_name)
        if relative.is_absolute() or ".." in relative.parts:
            return False
        path = directory.joinpath(*relative.parts)
        if not path.is_file() or _sha256(path) != expected_sha256:
            return False
    return True


def download_trackastra_teacher(
    cache_directory: Path | str,
    *,
    name: str = "general_2d",
    allow_network: bool = False,
    timeout_seconds: float = 60.0,
    maximum_download_bytes: int = 2 * 1024**3,
) -> Path:
    """Download, hash-check, safely extract, and atomically install a teacher.

    Network access must be explicitly enabled.  Existing unverified directories are never replaced
    or deleted; callers must choose a different cache or inspect them manually.
    """

    if not allow_network:
        raise PermissionError(
            "Teacher download requires allow_network=True; no network request was made"
        )
    if timeout_seconds <= 0 or maximum_download_bytes <= 0:
        raise ValueError("timeout_seconds and maximum_download_bytes must be positive")
    artifact = teacher_artifact(name)
    cache = Path(cache_directory).expanduser().resolve()
    destination = cache / name
    if destination.exists():
        if verify_teacher_directory(destination, name):
            return destination
        raise FileExistsError(
            f"Refusing to replace unverified existing teacher directory: {destination}"
        )
    cache.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f".{name}-", dir=cache) as temp_name:
        temporary = Path(temp_name)
        archive = temporary / f"{name}.zip"
        request = urllib.request.Request(
            artifact.url,
            headers={"User-Agent": f"Cellect/{CELLECT_TRACK_VERSION}"},
        )
        digest = hashlib.sha256()
        bytes_written = 0
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            declared_length = response.headers.get("Content-Length")
            if declared_length is not None and int(declared_length) > maximum_download_bytes:
                raise RuntimeError("Teacher download exceeds the safety limit")
            with archive.open("xb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if bytes_written > maximum_download_bytes:
                        raise RuntimeError("Teacher download exceeded the safety limit")
                    digest.update(chunk)
                    handle.write(chunk)
        observed_digest = digest.hexdigest()
        if observed_digest != artifact.sha256:
            raise RuntimeError(
                "Trackastra teacher SHA-256 mismatch: "
                f"expected {artifact.sha256}, observed {observed_digest}"
            )

        extracted = temporary / "extracted"
        extracted.mkdir()
        _safe_extract_zip(archive, extracted)
        staged_model = extracted / name
        if not staged_model.is_dir():
            raise RuntimeError(
                f"Teacher archive did not contain the expected {name!r} directory"
            )
        required = {"config.yaml", "model.pt", "train_config.yaml"}
        if not required <= {path.name for path in staged_model.iterdir() if path.is_file()}:
            raise RuntimeError("Teacher archive is missing required model files")
        _write_artifact_marker(staged_model, artifact)
        if destination.exists():
            raise FileExistsError(
                f"Teacher directory appeared during download: {destination}"
            )
        os.replace(staged_model, destination)
    if not verify_teacher_directory(destination, name):
        raise RuntimeError("Installed teacher failed its extracted-file verification")
    return destination


def load_trackastra_teacher(
    model_directory: Path | str,
    *,
    name: str = "general_2d",
    device: str = "cpu",
    trust_local_checkpoint: bool = False,
    require_exact_package_version: bool = True,
) -> Any:
    """Lazily import Trackastra and load a verified or explicitly trusted checkpoint.

    A manually supplied directory may contain serialized model data, so it requires
    ``trust_local_checkpoint=True`` unless it was installed and manifested by
    ``download_trackastra_teacher``.
    """

    artifact = teacher_artifact(name)
    directory = Path(model_directory).expanduser().resolve()
    if not verify_teacher_directory(directory, name) and not trust_local_checkpoint:
        raise PermissionError(
            "Refusing to load an unverified serialized teacher. Use the safe downloader or set "
            "trust_local_checkpoint=True for a directory you independently trust."
        )
    try:
        installed_version = metadata.version(artifact.package)
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(
            f"Optional teacher requires {artifact.package}=={artifact.package_version}; "
            "install it only in the workstation training environment"
        ) from error
    if require_exact_package_version and installed_version != artifact.package_version:
        raise RuntimeError(
            f"Expected {artifact.package}=={artifact.package_version}, found "
            f"{installed_version}"
        )
    model_api = importlib.import_module("trackastra.model.model_api")
    trackastra_class = getattr(model_api, "Trackastra", None)
    if trackastra_class is None:
        raise RuntimeError("Installed Trackastra has no Trackastra model API")
    return trackastra_class.from_folder(directory, device=device)


def run_contract_self_test() -> None:
    """Exercise architecture, masks, losses, freezing, tracing, and registry on CPU."""

    torch.manual_seed(20260731)
    config = CellectTrackConfig(
        max_tokens=8,
        cell_feature_dim=6,
        model_dim=32,
        num_heads=4,
        num_layers=1,
        pair_hidden_dim=32,
        feedforward_multiplier=2,
        dropout=0.0,
        max_frame_gap=2.0,
    )
    model = CellectTrackMobile(config=config).cpu().eval()
    time_yx = torch.tensor(
        [
            [
                [0.0, -0.6, -0.4],
                [0.0, 0.5, 0.4],
                [1.0, -0.5, -0.3],
                [1.0, 0.4, 0.3],
                [2.0, -0.4, -0.2],
                [2.0, 0.3, 0.2],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    features = torch.linspace(-1.0, 1.0, 8 * 6, dtype=torch.float32).reshape(
        1, 8, 6
    )
    cell_mask = torch.tensor(
        [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    output = model(time_yx, features, cell_mask)
    assert output.association_logits.shape == (1, 8, 8)
    assert output.division_probability.shape == (1, 8)
    assert not bool(output.association_mask[:, 6:, :].any())
    assert not bool(output.association_mask[:, :, 6:].any())
    assert bool(output.association_mask[0, 0, 2])
    assert not bool(output.association_mask[0, 2, 0])
    assert torch.all(output.division_probability[:, 6:] == 0)
    assert torch.isfinite(output.association_logits).all()

    # Token order must not change predictions after applying the inverse permutation.
    permutation = torch.tensor([3, 0, 5, 2, 1, 4, 7, 6])
    inverse = torch.argsort(permutation)
    permuted = model(
        time_yx[:, permutation], features[:, permutation], cell_mask[:, permutation]
    )
    restored_logits = permuted.association_logits[:, inverse][:, :, inverse]
    restored_divisions = permuted.division_probability[:, inverse]
    assert torch.allclose(
        output.association_logits, restored_logits, atol=2e-5, rtol=2e-5
    )
    assert torch.allclose(
        output.division_probability, restored_divisions, atol=2e-5, rtol=2e-5
    )

    association_target = torch.zeros((1, 8, 8), dtype=torch.float32)
    for source, target in ((0, 2), (1, 3), (2, 4), (2, 5), (3, 5)):
        association_target[0, source, target] = 1.0
    division_target = torch.zeros((1, 8), dtype=torch.float32)
    division_target[0, 2] = 1.0
    birth_target = torch.zeros((1, 8), dtype=torch.float32)
    birth_target[0, :2] = 1.0
    death_target = torch.zeros((1, 8), dtype=torch.float32)
    death_target[0, 4:6] = 1.0

    set_training_phase(model, "heads")
    model.train()
    model.zero_grad(set_to_none=True)
    training_output = model(time_yx, features, cell_mask)
    losses = tracking_supervised_losses(
        training_output,
        association_target,
        division_target,
        birth_target,
        death_target,
    )
    assert all(torch.isfinite(value) for value in losses.values())
    losses["total"].backward()
    audit = assert_gradient_audit(model, phase="heads")
    assert audit["trainable_parameter_count"] < audit["total_parameter_count"]
    assert not audit["missing_gradient"] and not audit["nonfinite_gradient"]

    model.eval()
    teacher_config = CellectTrackConfig(
        max_tokens=8,
        cell_feature_dim=6,
        model_dim=48,
        num_heads=4,
        num_layers=2,
        pair_hidden_dim=48,
        feedforward_multiplier=2,
        dropout=0.0,
        max_frame_gap=2.0,
    )
    teacher = CellectTrackHeavy(config=teacher_config).cpu().eval()
    with torch.no_grad():
        teacher_output = teacher(time_yx, features, cell_mask)
        student_output = model(time_yx, features, cell_mask)
        distillation = tracking_distillation_losses(
            student_output, teacher_output
        )
    assert all(torch.isfinite(value) for value in distillation.values())

    adapter = CellectTrackDeploymentAdapter(model).eval()
    traced = torch.jit.trace(
        adapter,
        (time_yx, features, cell_mask),
        check_trace=False,
        strict=False,
    )
    traced_outputs = traced(time_yx, features, cell_mask)
    assert len(traced_outputs) == 6
    assert traced_outputs[0].shape == (1, 8, 8)
    assert adapter.fixed_input_shapes == {
        "time_yx": (1, 8, 3),
        "cell_features": (1, 8, 6),
        "cell_mask": (1, 8),
    }

    artifact = teacher_artifact()
    assert artifact.package_version == "0.3.0"
    assert artifact.url.endswith("/v0.3.0/general_2d.zip")
    assert artifact.sha256 == (
        "35cefd8634860d1dd43bcdcbdef7ae0caa24445f19bcfd35ec6b19039f2cd876"
    )
    try:
        download_trackastra_teacher(
            Path("unused-contract-cache"), allow_network=False
        )
    except PermissionError as error:
        assert "no network request" in str(error)
    else:
        raise AssertionError("Teacher downloader performed implicit network access")

    # Exercise the archive boundary without downloading or deserializing model data.
    with tempfile.TemporaryDirectory(prefix="cellect-track-contract-") as temp_name:
        temporary = Path(temp_name)
        valid_archive = temporary / "valid.zip"
        with zipfile.ZipFile(valid_archive, "w") as bundle:
            bundle.writestr("general_2d/config.yaml", "model: contract\n")
        valid_destination = temporary / "valid"
        valid_destination.mkdir()
        _safe_extract_zip(valid_archive, valid_destination)
        assert (valid_destination / "general_2d" / "config.yaml").read_text() == (
            "model: contract\n"
        )

        traversal_archive = temporary / "traversal.zip"
        with zipfile.ZipFile(traversal_archive, "w") as bundle:
            bundle.writestr("../escaped-checkpoint.pt", b"unsafe")
        traversal_destination = temporary / "traversal"
        traversal_destination.mkdir()
        try:
            _safe_extract_zip(traversal_archive, traversal_destination)
        except RuntimeError as error:
            assert "Unsafe teacher archive path" in str(error)
        else:
            raise AssertionError("Teacher archive path traversal was not rejected")
        assert not (temporary / "escaped-checkpoint.pt").exists()

    small_mobile = CellectTrackMobile(max_tokens=8, cell_feature_dim=6)
    small_heavy = CellectTrackHeavy(max_tokens=8, cell_feature_dim=6)
    mobile_parameters = sum(parameter.numel() for parameter in small_mobile.parameters())
    heavy_parameters = sum(parameter.numel() for parameter in small_heavy.parameters())
    assert mobile_parameters < heavy_parameters


__all__ = [
    "CELL_FEATURE_DIM",
    "CELL_FEATURE_NAMES",
    "CELLECT_TRACK_VERSION",
    "TRACKING_LOSS_VERSION",
    "INVALID_ASSOCIATION_LOGIT",
    "TRACKASTRA_TEACHER_REGISTRY",
    "TRACKASTRA_TEACHER_VERSION",
    "TRAINING_PHASES",
    "CellectTrack",
    "CellectTrackConfig",
    "CellectTrackDeploymentAdapter",
    "CellectTrackHeavy",
    "CellectTrackMobile",
    "CellectTrackOutput",
    "TeacherArtifact",
    "assert_gradient_audit",
    "differentiable_assignment_loss",
    "differentiable_division_loss",
    "download_trackastra_teacher",
    "gradient_audit",
    "heavy_track_config",
    "load_trackastra_teacher",
    "masked_sinkhorn",
    "mobile_track_config",
    "run_contract_self_test",
    "set_training_phase",
    "teacher_artifact",
    "tracking_distillation_losses",
    "tracking_supervised_losses",
    "verify_teacher_directory",
]


if __name__ == "__main__":
    run_contract_self_test()
    print(f"tracking model contract passed: {CELLECT_TRACK_VERSION}")
