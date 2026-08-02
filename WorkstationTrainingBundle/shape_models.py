#!/usr/bin/env python3
"""Deployable v4 shape-aware students and their training contracts.

The module deliberately keeps pretrained teachers outside the deployable student.  Cellpose flow,
Ceb-style whole-boundary decisions, and an earlier Cellect foreground/contact model can therefore
be used as immutable teachers during training without making the exported iPhone graph depend on
their Python runtimes.  The student retains five inspectable segmentation logits:

``foreground, fused boundary, context boundary, flow boundary, shape boundary``.

The legacy deployment adapter returns only foreground and fused-boundary logits so it remains
compatible with the current two-channel Swift postprocessor.  The extended adapter exposes all
five channels for v4 comparison and future on-device fusion.

``segmentation_models_pytorch`` is imported only when a production preset is constructed.  This
keeps documentation, checkpoint inspection, and the small CPU contract test independent of that
optional implementation detail.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


SHAPE_MODEL_CONTRACT_VERSION = "cellect-shape-student-v4"
SHAPE_TARGET_CONTRACT_VERSION = "foreground-contact-sdf-flow-affinity-v1"
LEGACY_DEPLOYMENT_OUTPUT_SEMANTICS = (
    "foreground_logit",
    "fused_boundary_logit",
)
EXTENDED_DEPLOYMENT_OUTPUT_SEMANTICS = (
    "foreground_logit",
    "fused_boundary_logit",
    "context_boundary_logit",
    "flow_boundary_logit",
    "shape_boundary_logit",
)
BOUNDARY_BRANCH_SEMANTICS = {
    "fused_boundary_logit": (
        "learned per-pixel fusion of context, flow, and shape boundary evidence"
    ),
    "context_boundary_logit": (
        "image/backbone context evidence for a true division between adjacent cells"
    ),
    "flow_boundary_logit": (
        "division evidence learned from center-directed Cellpose-style flow geometry"
    ),
    "shape_boundary_logit": (
        "division evidence learned from signed distance and same-instance affinities"
    ),
}
AFFINITY_SEMANTICS = (
    "same_instance_right",
    "same_instance_down",
    "same_instance_down_right",
    "same_instance_down_left",
)

__all__ = [
    "AFFINITY_SEMANTICS",
    "BOUNDARY_BRANCH_SEMANTICS",
    "DEFAULT_SHAPE_LOSS_CONFIG",
    "DeploymentAdapter",
    "EXTENDED_DEPLOYMENT_OUTPUT_SEMANTICS",
    "ExtendedDeploymentAdapter",
    "HIGH_ACCURACY_SHAPE_SPEC",
    "LEGACY_DEPLOYMENT_OUTPUT_SEMANTICS",
    "MOBILE_SHAPE_SPEC",
    "SHAPE_MODEL_CONTRACT_VERSION",
    "SHAPE_TARGET_CONTRACT_VERSION",
    "ShapeLossConfig",
    "ShapeModelSpec",
    "V4ShapeNet",
    "V4_SHAPE_MODEL_SPECS",
    "audit_frozen_teacher",
    "freeze_teacher",
    "gradient_audit",
    "initialize_from_v3",
    "module_state_sha256",
    "multitask_shape_loss",
    "run_contract_self_test",
    "set_training_phase",
]


@dataclass(frozen=True)
class ShapeModelSpec:
    """Serializable construction contract for one v4 student.

    ``architecture='contract'`` is reserved for the tiny built-in CPU test.  Production presets
    lazily construct their two-channel semantic backbone with segmentation-models-pytorch.
    """

    name: str
    tier: str
    encoder: str
    architecture: str
    image_size: int
    refinement_channels: int
    refinement_blocks: int
    context_grid: int = 0
    context_layers: int = 0
    context_heads: int = 1
    affinity_channels: int = len(AFFINITY_SEMANTICS)
    encoder_weights: str | None = "imagenet"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("ShapeModelSpec.name must not be empty")
        if self.image_size < 16:
            raise ValueError("ShapeModelSpec.image_size must be at least 16")
        if self.refinement_channels < 8:
            raise ValueError("refinement_channels must be at least 8")
        if self.refinement_blocks < 1:
            raise ValueError("refinement_blocks must be positive")
        if self.affinity_channels != len(AFFINITY_SEMANTICS):
            raise ValueError(
                f"v4 requires {len(AFFINITY_SEMANTICS)} affinity channels in the "
                f"documented order, got {self.affinity_channels}"
            )
        if self.context_layers < 0 or self.context_grid < 0:
            raise ValueError("context_layers/context_grid cannot be negative")
        if self.context_layers and self.context_grid < 2:
            raise ValueError("transformer context requires context_grid >= 2")
        if self.context_layers and self.refinement_channels % self.context_heads:
            raise ValueError("refinement_channels must be divisible by context_heads")


MOBILE_SHAPE_SPEC = ShapeModelSpec(
    name="shape_mobile_mobilenetv3_v4",
    tier="mobile",
    encoder="timm-mobilenetv3_small_100",
    architecture="Unet",
    image_size=512,
    refinement_channels=48,
    refinement_blocks=4,
)

HIGH_ACCURACY_SHAPE_SPEC = ShapeModelSpec(
    name="shape_context_segformer_b5_v4",
    tier="high_accuracy",
    encoder="mit_b5",
    architecture="Segformer",
    image_size=768,
    refinement_channels=128,
    refinement_blocks=6,
    context_grid=12,
    context_layers=2,
    context_heads=8,
)

V4_SHAPE_MODEL_SPECS = (MOBILE_SHAPE_SPEC, HIGH_ACCURACY_SHAPE_SPEC)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class _ConvNormAct(nn.Sequential):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        kernel_size: int = 3,
        *,
        dilation: int = 1,
        groups: int = 1,
    ) -> None:
        padding = dilation * (kernel_size // 2)
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.SiLU(inplace=True),
        )


class _DepthwiseShapeBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.depthwise = _ConvNormAct(
            channels,
            channels,
            3,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )

    def forward(self, features: Tensor) -> Tensor:
        return F.silu(features + self.pointwise(self.depthwise(features)))


class _PooledShapeTransformer(nn.Module):
    """Bounded global context: attention runs on a small fixed grid, never full pixels."""

    def __init__(
        self,
        channels: int,
        grid: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        self.grid = grid
        self.layers = layers
        if layers:
            layer = nn.TransformerEncoderLayer(
                d_model=channels,
                nhead=heads,
                dim_feedforward=channels * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.transformer: nn.Module = nn.TransformerEncoder(
                layer,
                layers,
                enable_nested_tensor=False,
            )
            self.position = nn.Parameter(torch.zeros(1, grid * grid, channels))
            nn.init.trunc_normal_(self.position, std=0.02)
            self.projection: nn.Module = nn.Sequential(
                nn.Conv2d(channels, channels, 1, bias=False),
                nn.GroupNorm(_group_count(channels), channels),
            )
        else:
            self.transformer = nn.Identity()
            self.register_parameter("position", None)
            self.projection = nn.Identity()

    def forward(self, features: Tensor) -> Tensor:
        if not self.layers:
            return features
        height, width = features.shape[-2:]
        pooled = F.adaptive_avg_pool2d(features, (self.grid, self.grid))
        tokens = pooled.flatten(2).transpose(1, 2)
        assert self.position is not None
        tokens = self.transformer(tokens + self.position)
        context = tokens.transpose(1, 2).reshape(
            features.shape[0], features.shape[1], self.grid, self.grid
        )
        context = F.interpolate(
            context,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        return F.silu(features + self.projection(context))


def _zero_last_convolution(module: nn.Module) -> None:
    convolutions = [child for child in module.modules() if isinstance(child, nn.Conv2d)]
    if not convolutions:
        raise ValueError("Expected at least one convolution to zero-initialize")
    nn.init.zeros_(convolutions[-1].weight)
    if convolutions[-1].bias is not None:
        nn.init.zeros_(convolutions[-1].bias)


class _ShapeRefinementFusion(nn.Module):
    """Predict complementary geometry and fuse three explicit boundary hypotheses."""

    def __init__(self, spec: ShapeModelSpec) -> None:
        super().__init__()
        channels = spec.refinement_channels
        self.affinity_channels = spec.affinity_channels
        # Three grayscale-like image channels plus foreground/contact backbone probabilities.
        self.input_projection = _ConvNormAct(5, channels)
        dilations = (1, 2, 4, 2, 1, 4)
        self.local_blocks = nn.Sequential(
            *[
                _DepthwiseShapeBlock(channels, dilations[index % len(dilations)])
                for index in range(spec.refinement_blocks)
            ]
        )
        self.global_context = _PooledShapeTransformer(
            channels,
            spec.context_grid,
            spec.context_layers,
            spec.context_heads,
        )

        self.foreground_delta = nn.Conv2d(channels, 1, 1)
        self.context_boundary_delta = nn.Conv2d(channels, 1, 1)
        self.sdf_head = nn.Conv2d(channels, 1, 1)
        self.flow_head = nn.Conv2d(channels, 2, 1)
        self.affinity_head = nn.Conv2d(channels, spec.affinity_channels, 1)
        self.uncertainty_head = nn.Conv2d(channels, 2, 1)

        self.flow_boundary_features = _ConvNormAct(channels + 2, channels)
        self.flow_boundary_delta = nn.Conv2d(channels, 1, 1)
        self.shape_boundary_features = _ConvNormAct(
            channels + 1 + spec.affinity_channels,
            channels,
        )
        self.shape_boundary_delta = nn.Conv2d(channels, 1, 1)
        self.boundary_fusion_weights = nn.Conv2d(channels, 3, 1)
        self.boundary_fusion_residual = nn.Sequential(
            _ConvNormAct(channels + 3 + 2, channels),
            nn.Conv2d(channels, 1, 1),
        )

        # With an imported v3 backbone the initial deployed logits are exactly the old logits.
        # Training then learns branch-specific residuals and their fusion without destroying the
        # useful pretrained decision surface on the first optimizer step.
        for module in (
            self.foreground_delta,
            self.context_boundary_delta,
            self.flow_boundary_delta,
            self.shape_boundary_delta,
            self.boundary_fusion_weights,
            self.boundary_fusion_residual,
            self.uncertainty_head,
        ):
            _zero_last_convolution(module)

    def forward(self, image: Tensor, base_logits: Tensor) -> dict[str, Tensor]:
        base_probabilities = torch.sigmoid(base_logits.float()).to(base_logits.dtype)
        features = self.input_projection(torch.cat((image, base_probabilities), dim=1))
        features = self.global_context(self.local_blocks(features))

        foreground_logit = base_logits[:, 0:1] + self.foreground_delta(features)
        context_boundary_logit = (
            base_logits[:, 1:2] + self.context_boundary_delta(features)
        )
        signed_distance = torch.tanh(self.sdf_head(features))
        flow_yx = torch.tanh(self.flow_head(features))
        affinity_logits = self.affinity_head(features)
        log_variance = torch.clamp(self.uncertainty_head(features), -6.0, 6.0)

        flow_boundary_logit = base_logits[:, 1:2] + self.flow_boundary_delta(
            self.flow_boundary_features(torch.cat((features, flow_yx), dim=1))
        )
        shape_boundary_logit = base_logits[:, 1:2] + self.shape_boundary_delta(
            self.shape_boundary_features(
                torch.cat(
                    (features, signed_distance, torch.sigmoid(affinity_logits)),
                    dim=1,
                )
            )
        )
        boundary_components = torch.cat(
            (
                context_boundary_logit,
                flow_boundary_logit,
                shape_boundary_logit,
            ),
            dim=1,
        )
        fusion_weights = torch.softmax(self.boundary_fusion_weights(features), dim=1)
        fused_boundary_logit = (fusion_weights * boundary_components).sum(
            dim=1, keepdim=True
        )
        fused_boundary_logit = fused_boundary_logit + self.boundary_fusion_residual(
            torch.cat((features, boundary_components, log_variance), dim=1)
        )

        segmentation_logits = torch.cat(
            (foreground_logit, fused_boundary_logit), dim=1
        )
        extended_segmentation_logits = torch.cat(
            (
                foreground_logit,
                fused_boundary_logit,
                context_boundary_logit,
                flow_boundary_logit,
                shape_boundary_logit,
            ),
            dim=1,
        )
        return {
            "segmentation_logits": segmentation_logits,
            "extended_segmentation_logits": extended_segmentation_logits,
            "foreground_logit": foreground_logit,
            "fused_boundary_logit": fused_boundary_logit,
            "context_boundary_logit": context_boundary_logit,
            "flow_boundary_logit": flow_boundary_logit,
            "shape_boundary_logit": shape_boundary_logit,
            "signed_distance": signed_distance,
            "flow_yx": flow_yx,
            "affinity_logits": affinity_logits,
            "log_variance": log_variance,
            "boundary_fusion_weights": fusion_weights,
            "base_segmentation_logits": base_logits,
        }


class _ContractBackbone(nn.Module):
    """Small SMP-shaped backbone used only by ``run_contract_self_test``."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 8, 3, padding=1),
            nn.SiLU(),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1),
            nn.SiLU(),
        )
        self.segmentation_head = nn.Conv2d(8, 2, 1)

    def forward(self, image: Tensor) -> Tensor:
        return self.segmentation_head(self.decoder(self.encoder(image)))


def _build_semantic_backbone(spec: ShapeModelSpec) -> nn.Module:
    if spec.architecture.casefold() == "contract":
        return _ContractBackbone()
    try:
        import segmentation_models_pytorch as smp
    except ImportError as error:  # pragma: no cover - exercised by workstation setup
        raise RuntimeError(
            "Production v4 shape models require segmentation-models-pytorch. "
            "Install WorkstationTrainingBundle/requirements.txt first."
        ) from error
    constructors = {
        "unet": smp.Unet,
        "unetplusplus": smp.UnetPlusPlus,
        "deeplabv3plus": smp.DeepLabV3Plus,
        "segformer": smp.Segformer,
    }
    architecture_key = spec.architecture.replace("+", "plus").casefold()
    try:
        constructor = constructors[architecture_key]
    except KeyError as error:
        raise ValueError(
            f"Unsupported semantic backbone architecture: {spec.architecture}"
        ) from error
    return constructor(
        encoder_name=spec.encoder,
        encoder_weights=spec.encoder_weights,
        in_channels=3,
        classes=2,
        activation=None,
    )


class V4ShapeNet(nn.Module):
    """Shape-aware student retaining a compatible two-channel semantic backbone."""

    output_contract_version = SHAPE_MODEL_CONTRACT_VERSION
    legacy_output_semantics = LEGACY_DEPLOYMENT_OUTPUT_SEMANTICS
    extended_output_semantics = EXTENDED_DEPLOYMENT_OUTPUT_SEMANTICS

    def __init__(
        self,
        spec: ShapeModelSpec,
        *,
        semantic_backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.semantic_backbone = semantic_backbone or _build_semantic_backbone(spec)
        self.refinement = _ShapeRefinementFusion(spec)
        self.training_phase = "full"

    def forward(self, image: Tensor) -> dict[str, Tensor]:
        if not torch.jit.is_tracing() and (image.ndim != 4 or image.shape[1] != 3):
            raise ValueError(
                f"V4ShapeNet input must be NCHW with three channels, got {tuple(image.shape)}"
            )
        base_logits = self.semantic_backbone(image)
        if not torch.jit.is_tracing() and not isinstance(base_logits, Tensor):
            raise TypeError("The semantic backbone must return one logits tensor")
        expected_shape = (image.shape[0], 2, image.shape[2], image.shape[3])
        if not torch.jit.is_tracing() and tuple(base_logits.shape) != expected_shape:
            raise RuntimeError(
                f"Semantic backbone returned {tuple(base_logits.shape)}, expected "
                f"{expected_shape}"
            )
        return self.refinement(image, base_logits)

    def train(self, mode: bool = True) -> V4ShapeNet:
        super().train(mode)
        if mode and self.training_phase == "heads":
            self.semantic_backbone.eval()
        elif mode and self.training_phase == "decoder":
            encoder = getattr(self.semantic_backbone, "encoder", None)
            if isinstance(encoder, nn.Module):
                encoder.eval()
        return self


class DeploymentAdapter(nn.Module):
    """Legacy iPhone contract: ``[foreground, fused boundary]`` logits."""

    output_name = "segmentation_logits"
    output_semantics = LEGACY_DEPLOYMENT_OUTPUT_SEMANTICS

    def __init__(self, model: V4ShapeNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: Tensor) -> Tensor:
        return self.model(image)["segmentation_logits"]


class ExtendedDeploymentAdapter(nn.Module):
    """V4 comparison contract exposing all learned boundary hypotheses."""

    output_name = "extended_segmentation_logits"
    output_semantics = EXTENDED_DEPLOYMENT_OUTPUT_SEMANTICS

    def __init__(self, model: V4ShapeNet) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: Tensor) -> Tensor:
        return self.model(image)["extended_segmentation_logits"]


@dataclass(frozen=True)
class ShapeLossConfig:
    foreground_bce_weight: float = 1.0
    foreground_dice_weight: float = 1.0
    fused_boundary_weight: float = 1.0
    context_boundary_weight: float = 0.35
    flow_boundary_weight: float = 0.35
    shape_boundary_weight: float = 0.35
    boundary_bce_weight: float = 1.0
    boundary_dice_weight: float = 1.0
    boundary_positive_weight: float = 5.0
    signed_distance_weight: float = 0.45
    flow_regression_weight: float = 0.55
    flow_direction_weight: float = 0.25
    affinity_weight: float = 0.40
    uncertainty_weight: float = 0.05
    base_semantic_weight: float = 0.10
    cellpose_probability_weight: float = 0.20
    cellpose_flow_weight: float = 0.40
    context_teacher_weight: float = 0.20
    flow_boundary_teacher_weight: float = 0.25
    shape_teacher_weight: float = 0.35
    fused_teacher_weight: float = 0.20
    require_auxiliary_targets: bool = True


DEFAULT_SHAPE_LOSS_CONFIG = ShapeLossConfig()


def _target(targets: Mapping[str, Tensor], *names: str) -> Tensor | None:
    for name in names:
        value = targets.get(name)
        if value is not None:
            return value
    return None


def _bchw(value: Tensor, channels: int, name: str) -> Tensor:
    if value.ndim == 3 and channels == 1:
        value = value[:, None]
    if value.ndim != 4 or value.shape[1] != channels:
        raise ValueError(
            f"{name} must have shape [N,{channels},H,W], got {tuple(value.shape)}"
        )
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return value


def _same_grid(reference: Tensor, value: Tensor, name: str) -> Tensor:
    if value.shape[0] != reference.shape[0] or value.shape[-2:] != reference.shape[-2:]:
        raise ValueError(
            f"{name} grid {tuple(value.shape)} does not match student grid "
            f"{tuple(reference.shape)}; transform cached teacher maps with the sample"
        )
    return value.to(device=reference.device, dtype=reference.dtype)


def _masked_mean(values: Tensor, valid: Tensor) -> Tensor:
    expanded = valid.expand_as(values)
    return (values * expanded).sum() / expanded.sum().clamp_min(1.0)


def _soft_dice_loss(logits: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    probability = torch.sigmoid(logits)
    intersection = (probability * target * valid).sum(dim=(1, 2, 3))
    denominator = (probability * valid).sum(dim=(1, 2, 3)) + (
        target * valid
    ).sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def _semantic_loss(
    logits: Tensor,
    target: Tensor,
    valid: Tensor,
    *,
    positive_weight: float,
    bce_weight: float,
    dice_weight: float,
) -> tuple[Tensor, Tensor]:
    pos_weight = torch.as_tensor(
        positive_weight,
        device=logits.device,
        dtype=logits.dtype,
    )
    elementwise = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=pos_weight,
        reduction="none",
    )
    bce = _masked_mean(elementwise, valid)
    dice = _soft_dice_loss(logits, target, valid)
    return bce_weight * bce + dice_weight * dice, elementwise


def _probability_teacher(
    teacher: Mapping[str, Any],
    reference: Tensor,
    *names: str,
) -> Tensor | None:
    value = _target(teacher, *names)
    if value is None:
        return None
    value = _same_grid(reference, _bchw(value, 1, names[0]), names[0])
    minimum = float(value.detach().min())
    maximum = float(value.detach().max())
    if minimum < -1e-5 or maximum > 1.0 + 1e-5:
        raise ValueError(
            f"{names[0]} must contain probabilities in [0,1], got [{minimum}, {maximum}]"
        )
    return value.clamp(0.0, 1.0)


def _soft_teacher_bce(student_logits: Tensor, teacher_probability: Tensor, valid: Tensor) -> Tensor:
    return _masked_mean(
        F.binary_cross_entropy_with_logits(
            student_logits,
            teacher_probability,
            reduction="none",
        ),
        valid,
    )


def multitask_shape_loss(
    outputs: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    teacher: Mapping[str, Any] | None = None,
    config: ShapeLossConfig = DEFAULT_SHAPE_LOSS_CONFIG,
) -> dict[str, object]:
    """Compute supervised geometry and optional immutable-teacher distillation losses.

    Target maps must already share the student's augmented pixel grid.  The function refuses to
    resize them silently because doing so is particularly unsafe for direction-valued flow maps.
    ``teacher`` probabilities use explicit ``*_probability`` names; logits are never guessed.
    """

    foreground_logit = _bchw(outputs["foreground_logit"], 1, "foreground_logit")
    foreground = _target(targets, "foreground", "foreground_target")
    contact = _target(
        targets,
        "contact",
        "boundary",
        "contact_boundary",
        "internal_contact",
    )
    if foreground is None or contact is None:
        raise KeyError("targets require foreground and contact/contact_boundary maps")
    foreground = _same_grid(
        foreground_logit,
        _bchw(foreground, 1, "foreground"),
        "foreground",
    )
    contact = _same_grid(
        foreground_logit,
        _bchw(contact, 1, "contact"),
        "contact",
    )
    valid_value = _target(targets, "valid", "valid_mask")
    valid = (
        torch.ones_like(foreground)
        if valid_value is None
        else _same_grid(
            foreground_logit,
            _bchw(valid_value, 1, "valid"),
            "valid",
        )
    )
    valid = (valid > 0.5).to(foreground.dtype)

    components: dict[str, Tensor] = {}
    foreground_loss, foreground_elementwise = _semantic_loss(
        foreground_logit,
        foreground,
        valid,
        positive_weight=1.0,
        bce_weight=config.foreground_bce_weight,
        dice_weight=config.foreground_dice_weight,
    )
    components["foreground_supervised"] = foreground_loss

    boundary_weights = {
        "fused_boundary_logit": config.fused_boundary_weight,
        "context_boundary_logit": config.context_boundary_weight,
        "flow_boundary_logit": config.flow_boundary_weight,
        "shape_boundary_logit": config.shape_boundary_weight,
    }
    boundary_elementwise: Tensor | None = None
    for output_name, weight in boundary_weights.items():
        logits = _same_grid(
            foreground_logit,
            _bchw(outputs[output_name], 1, output_name),
            output_name,
        )
        branch_loss, elementwise = _semantic_loss(
            logits,
            contact,
            valid,
            positive_weight=config.boundary_positive_weight,
            bce_weight=config.boundary_bce_weight,
            dice_weight=config.boundary_dice_weight,
        )
        components[f"{output_name}_supervised"] = weight * branch_loss
        if output_name == "fused_boundary_logit":
            boundary_elementwise = elementwise

    signed_distance_target = _target(targets, "signed_distance", "sdf")
    flow_target = _target(targets, "flow_yx", "flow")
    if flow_target is None:
        centroid_flow_y = _target(targets, "centroid_flow_y")
        centroid_flow_x = _target(targets, "centroid_flow_x")
        if (centroid_flow_y is None) != (centroid_flow_x is None):
            raise KeyError(
                "centroid_flow_y and centroid_flow_x must be supplied together"
            )
        if centroid_flow_y is not None and centroid_flow_x is not None:
            centroid_flow_y = _bchw(centroid_flow_y, 1, "centroid_flow_y")
            centroid_flow_x = _bchw(centroid_flow_x, 1, "centroid_flow_x")
            flow_target = torch.cat((centroid_flow_y, centroid_flow_x), dim=1)
    affinity_target = _target(targets, "affinities", "affinity")
    if config.require_auxiliary_targets and (
        signed_distance_target is None or flow_target is None or affinity_target is None
    ):
        raise KeyError(
            "v4 shape training requires signed_distance, flow_yx, and affinities targets"
        )

    if signed_distance_target is not None:
        signed_distance = _same_grid(
            foreground_logit,
            _bchw(outputs["signed_distance"], 1, "signed_distance output"),
            "signed_distance output",
        )
        signed_distance_target = _same_grid(
            foreground_logit,
            _bchw(signed_distance_target, 1, "signed_distance target"),
            "signed_distance target",
        )
        components["signed_distance"] = config.signed_distance_weight * _masked_mean(
            F.smooth_l1_loss(
                signed_distance,
                signed_distance_target,
                reduction="none",
            ),
            valid,
        )

    if flow_target is not None:
        flow = _same_grid(
            foreground_logit,
            _bchw(outputs["flow_yx"], 2, "flow_yx output"),
            "flow_yx output",
        )
        flow_target = _same_grid(
            foreground_logit,
            _bchw(flow_target, 2, "flow_yx target"),
            "flow_yx target",
        )
        flow_valid_value = _target(targets, "flow_valid")
        flow_valid = foreground * valid
        if flow_valid_value is not None:
            flow_valid = flow_valid * (
                _same_grid(
                    foreground_logit,
                    _bchw(flow_valid_value, 1, "flow_valid"),
                    "flow_valid",
                )
                > 0.5
            ).to(flow_valid.dtype)
        components["flow_regression"] = config.flow_regression_weight * _masked_mean(
            F.smooth_l1_loss(flow, flow_target, reduction="none"),
            flow_valid,
        )
        cosine = F.cosine_similarity(flow, flow_target, dim=1, eps=1e-6)[:, None]
        components["flow_direction"] = config.flow_direction_weight * _masked_mean(
            1.0 - cosine,
            flow_valid,
        )

    if affinity_target is not None:
        affinity_logits = _bchw(
            outputs["affinity_logits"],
            len(AFFINITY_SEMANTICS),
            "affinity_logits",
        )
        affinity_logits = _same_grid(
            foreground_logit, affinity_logits, "affinity_logits"
        )
        affinity_target = _bchw(
            affinity_target,
            affinity_target.shape[1],
            "affinities target",
        )
        if affinity_target.shape[1] < len(AFFINITY_SEMANTICS):
            raise ValueError(
                "affinities target has fewer channels than the four deployed v4 "
                f"offsets: {affinity_target.shape[1]}"
            )
        # ``shape_targets.geometry_targets`` provides eight offsets by default.  The mobile and
        # high-accuracy students deliberately use its first four local offsets, whose order is
        # exactly AFFINITY_SEMANTICS, to bound inference memory on an iPhone.
        affinity_target = affinity_target[:, : len(AFFINITY_SEMANTICS)]
        affinity_target = _same_grid(
            foreground_logit,
            affinity_target,
            "affinities target",
        )
        affinity_valid_value = _target(targets, "affinity_valid")
        affinity_valid = valid
        if affinity_valid_value is not None:
            if affinity_valid_value.ndim == 3:
                affinity_valid_value = affinity_valid_value[:, None]
            if affinity_valid_value.ndim != 4 or (
                affinity_valid_value.shape[1] != 1
                and affinity_valid_value.shape[1] < len(AFFINITY_SEMANTICS)
            ):
                raise ValueError(
                    "affinity_valid must have one channel or at least the four local "
                    f"affinity channels, got {tuple(affinity_valid_value.shape)}"
                )
            if affinity_valid_value.shape[1] > len(AFFINITY_SEMANTICS):
                affinity_valid_value = affinity_valid_value[
                    :, : len(AFFINITY_SEMANTICS)
                ]
            affinity_valid_value = _same_grid(
                foreground_logit,
                affinity_valid_value,
                "affinity_valid",
            )
            affinity_valid = affinity_valid * (
                affinity_valid_value > 0.5
            ).to(valid.dtype)
        components["affinities"] = config.affinity_weight * _masked_mean(
            F.binary_cross_entropy_with_logits(
                affinity_logits,
                affinity_target,
                reduction="none",
            ),
            affinity_valid,
        )

    base_logits = _bchw(outputs["base_segmentation_logits"], 2, "base logits")
    base_target = torch.cat((foreground, contact), dim=1)
    components["base_semantic"] = config.base_semantic_weight * _masked_mean(
        F.binary_cross_entropy_with_logits(base_logits, base_target, reduction="none"),
        valid,
    )

    assert boundary_elementwise is not None
    log_variance = _same_grid(
        foreground_logit,
        _bchw(outputs["log_variance"], 2, "log_variance"),
        "log_variance",
    )
    semantic_elementwise = torch.cat(
        (foreground_elementwise, boundary_elementwise), dim=1
    )
    components["uncertainty"] = config.uncertainty_weight * _masked_mean(
        torch.exp(-log_variance) * semantic_elementwise + log_variance,
        valid,
    )

    teacher = teacher or {}
    teacher_valid_value = _target(teacher, "valid", "teacher_valid")
    teacher_valid = valid
    if teacher_valid_value is not None:
        teacher_valid = valid * (
            _same_grid(
                foreground_logit,
                _bchw(teacher_valid_value, 1, "teacher_valid"),
                "teacher_valid",
            )
            > 0.5
        ).to(valid.dtype)

    cellpose_probability = _probability_teacher(
        teacher,
        foreground_logit,
        "cellpose_cell_probability",
    )
    if cellpose_probability is None:
        # Direct compatibility with cellpose_teacher.py's lossless raw cache contract.
        cellprob_logit_value = teacher.get("cellprob_logit")
        if isinstance(cellprob_logit_value, Tensor):
            cellprob_logit = _same_grid(
                foreground_logit,
                _bchw(cellprob_logit_value, 1, "cellprob_logit"),
                "cellprob_logit",
            )
            cellpose_probability = torch.sigmoid(cellprob_logit)
    if cellpose_probability is not None:
        components["cellpose_probability_distillation"] = (
            config.cellpose_probability_weight
            * _soft_teacher_bce(
                foreground_logit,
                cellpose_probability,
                teacher_valid,
            )
        )

    teacher_flow_value = _target(teacher, "cellpose_flow_yx")
    if teacher_flow_value is None:
        flow_y_raw = teacher.get("flow_y_raw")
        flow_x_raw = teacher.get("flow_x_raw")
        if (flow_y_raw is None) != (flow_x_raw is None):
            raise KeyError("flow_y_raw and flow_x_raw must be supplied together")
        if isinstance(flow_y_raw, Tensor) and isinstance(flow_x_raw, Tensor):
            scale = teacher.get("flow_scale_divisor", 5.0)
            if isinstance(scale, Tensor):
                if scale.numel() != 1:
                    raise ValueError("flow_scale_divisor must be scalar")
                scale = float(scale.detach().cpu())
            scale = float(scale)
            if not torch.isfinite(torch.tensor(scale)) or scale <= 0:
                raise ValueError("flow_scale_divisor must be one positive finite value")
            teacher_flow_value = torch.cat(
                (
                    _bchw(flow_y_raw, 1, "flow_y_raw"),
                    _bchw(flow_x_raw, 1, "flow_x_raw"),
                ),
                dim=1,
            ) / scale
    if teacher_flow_value is not None:
        teacher_flow = _same_grid(
            foreground_logit,
            _bchw(teacher_flow_value, 2, "cellpose_flow_yx"),
            "cellpose_flow_yx",
        )
        student_flow = _same_grid(
            foreground_logit,
            _bchw(outputs["flow_yx"], 2, "flow_yx output"),
            "flow_yx output",
        )
        flow_valid = foreground * teacher_valid
        regression = _masked_mean(
            F.smooth_l1_loss(student_flow, teacher_flow, reduction="none"),
            flow_valid,
        )
        cosine = F.cosine_similarity(
            student_flow, teacher_flow, dim=1, eps=1e-6
        )[:, None]
        components["cellpose_flow_distillation"] = config.cellpose_flow_weight * (
            regression + 0.5 * _masked_mean(1.0 - cosine, flow_valid)
        )

    teacher_branches = (
        (
            "context_boundary_logit",
            ("context_boundary_probability", "v3_boundary_probability"),
            config.context_teacher_weight,
        ),
        (
            "flow_boundary_logit",
            ("flow_boundary_probability", "cellpose_boundary_probability"),
            config.flow_boundary_teacher_weight,
        ),
        (
            "shape_boundary_logit",
            ("shape_boundary_probability", "ceb_boundary_probability"),
            config.shape_teacher_weight,
        ),
        (
            "fused_boundary_logit",
            ("fused_boundary_probability",),
            config.fused_teacher_weight,
        ),
    )
    for output_name, teacher_names, weight in teacher_branches:
        probability = _probability_teacher(
            teacher,
            foreground_logit,
            *teacher_names,
        )
        if probability is not None:
            components[f"{output_name}_distillation"] = weight * _soft_teacher_bce(
                outputs[output_name], probability, teacher_valid
            )

    total = torch.stack(tuple(components.values())).sum()
    return {
        "total": total,
        "components": components,
        "semantics": {
            "target_contract": SHAPE_TARGET_CONTRACT_VERSION,
            "legacy_deployment": list(LEGACY_DEPLOYMENT_OUTPUT_SEMANTICS),
            "extended_deployment": list(EXTENDED_DEPLOYMENT_OUTPUT_SEMANTICS),
            "boundary_branches": dict(BOUNDARY_BRANCH_SEMANTICS),
            "flow_channels": ["center_flow_y", "center_flow_x"],
            "affinity_channels": list(AFFINITY_SEMANTICS),
        },
    }


def _checkpoint_state(checkpoint: Any) -> Mapping[str, Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Checkpoint must be a state mapping or contain model/state_dict")
    for key in ("model", "state_dict", "model_state_dict"):
        candidate = checkpoint.get(key)
        if isinstance(candidate, Mapping):
            checkpoint = candidate
            break
    return {
        str(key): value
        for key, value in checkpoint.items()
        if isinstance(value, Tensor)
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def initialize_from_v3(
    model: V4ShapeNet,
    checkpoint: str | Path | Mapping[str, Any],
    *,
    minimum_backbone_fraction: float = 0.99,
) -> dict[str, object]:
    """Load only exact, shape-compatible v3 semantic-backbone tensors.

    V3 SMP checkpoints use keys such as ``encoder.*`` and ``decoder.*``.  V4 retains that model
    under ``semantic_backbone.*``; prefixing those exact keys transfers learned segmentation
    knowledge while leaving every new geometry/fusion head independently initialized.
    """

    if not 0.0 <= minimum_backbone_fraction <= 1.0:
        raise ValueError("minimum_backbone_fraction must be in [0,1]")
    source_path: Path | None = None
    if isinstance(checkpoint, (str, Path)):
        source_path = Path(checkpoint).resolve()
        checkpoint_payload = torch.load(
            source_path,
            map_location="cpu",
            weights_only=False,
        )
    else:
        checkpoint_payload = checkpoint
    source_state = _checkpoint_state(checkpoint_payload)
    destination_state = model.state_dict()
    backbone_keys = {
        key for key in destination_state if key.startswith("semantic_backbone.")
    }
    backbone_numel = sum(destination_state[key].numel() for key in backbone_keys)
    if backbone_numel == 0:
        raise RuntimeError("V4ShapeNet has no semantic_backbone parameters or buffers")

    loaded: dict[str, Tensor] = {}
    shape_mismatches: list[dict[str, object]] = []
    unmatched: list[str] = []
    collisions: list[str] = []
    removable_prefixes = ("module.", "model.", "network.")
    for source_key, value in source_state.items():
        normalized = source_key
        changed = True
        while changed:
            changed = False
            for prefix in removable_prefixes:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix) :]
                    changed = True
        candidates = [normalized]
        if not normalized.startswith("semantic_backbone."):
            candidates.insert(0, f"semantic_backbone.{normalized}")
        destination_key = next(
            (key for key in candidates if key in backbone_keys),
            None,
        )
        if destination_key is None:
            unmatched.append(source_key)
            continue
        if tuple(value.shape) != tuple(destination_state[destination_key].shape):
            shape_mismatches.append(
                {
                    "source": source_key,
                    "destination": destination_key,
                    "source_shape": list(value.shape),
                    "destination_shape": list(destination_state[destination_key].shape),
                }
            )
            continue
        if destination_key in loaded:
            collisions.append(destination_key)
            continue
        loaded[destination_key] = value.detach().to(
            dtype=destination_state[destination_key].dtype
        )
    if collisions:
        raise RuntimeError(
            "Multiple v3 tensors mapped to the same v4 destination: "
            + ", ".join(sorted(set(collisions))[:10])
        )
    loaded_numel = sum(value.numel() for value in loaded.values())
    loaded_fraction = loaded_numel / backbone_numel
    if loaded_fraction < minimum_backbone_fraction:
        raise RuntimeError(
            "Compatible v3 checkpoint coverage is too low: "
            f"{loaded_fraction:.3%} < {minimum_backbone_fraction:.3%}. "
            "Check the selected v3 architecture and encoder."
        )
    model.load_state_dict(loaded, strict=False)
    return {
        "contract": SHAPE_MODEL_CONTRACT_VERSION,
        "source_path": str(source_path) if source_path is not None else None,
        "source_sha256": _file_sha256(source_path) if source_path is not None else None,
        "loaded_tensor_count": len(loaded),
        "loaded_parameter_and_buffer_elements": loaded_numel,
        "semantic_backbone_elements": backbone_numel,
        "semantic_backbone_loaded_fraction": loaded_fraction,
        "loaded_keys": sorted(loaded),
        "shape_mismatches": shape_mismatches,
        "unmatched_source_key_count": len(unmatched),
        "new_v4_heads_loaded": False,
    }


_PHASE_ALIASES = {
    "heads": "heads",
    "refiner": "heads",
    "shape_heads": "heads",
    "decoder": "decoder",
    "partial": "decoder",
    "full": "full",
    "end_to_end": "full",
    "frozen": "frozen",
}


def _matches_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return "*" in prefixes or any(name.startswith(prefix) for prefix in prefixes)


def set_training_phase(model: V4ShapeNet, phase: str) -> dict[str, object]:
    """Apply the auditable three-stage fine-tuning schedule.

    - ``heads``: retain the imported v3 backbone exactly and learn all new v4 mechanisms.
    - ``decoder``: additionally fine-tune the old decoder/segmentation head.
    - ``full``: end-to-end low-learning-rate fine-tuning.
    - ``frozen``: inference/teacher mode with no trainable parameters.
    """

    try:
        canonical = _PHASE_ALIASES[phase.casefold()]
    except KeyError as error:
        raise ValueError(
            f"Unknown training phase {phase!r}; choose heads, decoder, full, or frozen"
        ) from error
    prefixes_by_phase = {
        "heads": ("refinement.",),
        "decoder": (
            "refinement.",
            "semantic_backbone.decoder.",
            "semantic_backbone.segmentation_head.",
        ),
        "full": ("*",),
        "frozen": (),
    }
    prefixes = prefixes_by_phase[canonical]
    trainable_names: list[str] = []
    frozen_names: list[str] = []
    trainable_elements = 0
    total_elements = 0
    for name, parameter in model.named_parameters():
        trainable = _matches_prefix(name, prefixes)
        parameter.requires_grad_(trainable)
        if not trainable:
            parameter.grad = None
            frozen_names.append(name)
        else:
            trainable_names.append(name)
            trainable_elements += parameter.numel()
        total_elements += parameter.numel()
    if canonical != "frozen" and not trainable_names:
        raise RuntimeError(f"Training phase {canonical} selected no parameters")
    model.training_phase = canonical
    model.train(model.training)
    return {
        "phase": canonical,
        "allowed_prefixes": list(prefixes),
        "trainable_tensor_count": len(trainable_names),
        "frozen_tensor_count": len(frozen_names),
        "trainable_elements": trainable_elements,
        "total_elements": total_elements,
        "trainable_fraction": (
            trainable_elements / total_elements if total_elements else 0.0
        ),
        "trainable_names": trainable_names,
    }


def gradient_audit(
    model: V4ShapeNet,
    *,
    phase: str | None = None,
    require_nonzero_trainable_gradient: bool = True,
) -> dict[str, object]:
    """Fail when gradients escape a freeze phase or contain non-finite values."""

    canonical = model.training_phase if phase is None else _PHASE_ALIASES[phase.casefold()]
    prefixes_by_phase = {
        "heads": ("refinement.",),
        "decoder": (
            "refinement.",
            "semantic_backbone.decoder.",
            "semantic_backbone.segmentation_head.",
        ),
        "full": ("*",),
        "frozen": (),
    }
    allowed = prefixes_by_phase[canonical]
    nonzero: list[str] = []
    missing: list[str] = []
    escaped: list[str] = []
    nonfinite: list[str] = []
    for name, parameter in model.named_parameters():
        permitted = _matches_prefix(name, allowed)
        gradient = parameter.grad
        if gradient is None:
            if permitted and parameter.requires_grad:
                missing.append(name)
            continue
        if not torch.isfinite(gradient).all():
            nonfinite.append(name)
        has_nonzero = bool(torch.count_nonzero(gradient.detach()).item())
        if has_nonzero:
            nonzero.append(name)
            if not permitted or not parameter.requires_grad:
                escaped.append(name)
    if nonfinite:
        raise RuntimeError("Non-finite gradients: " + ", ".join(nonfinite[:10]))
    if escaped:
        raise RuntimeError("Gradients escaped freeze schedule: " + ", ".join(escaped[:10]))
    if require_nonzero_trainable_gradient and canonical != "frozen" and not nonzero:
        raise RuntimeError("No trainable v4 parameter received a non-zero gradient")
    return {
        "phase": canonical,
        "nonzero_gradient_tensor_count": len(nonzero),
        "missing_gradient_tensor_count": len(missing),
        "escaped_gradient_tensor_count": len(escaped),
        "nonfinite_gradient_tensor_count": len(nonfinite),
        "nonzero_gradient_names": nonzero,
    }


def module_state_sha256(module: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        # Reshape first because viewing a scalar buffer (for example BatchNorm's
        # ``num_batches_tracked``) directly as bytes is unsupported by some Torch releases.
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def freeze_teacher(module: nn.Module) -> str:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    return module_state_sha256(module)


def audit_frozen_teacher(module: nn.Module, expected_sha256: str) -> dict[str, object]:
    changed = module_state_sha256(module) != expected_sha256
    trainable = [name for name, value in module.named_parameters() if value.requires_grad]
    gradients = [
        name
        for name, value in module.named_parameters()
        if value.grad is not None and bool(torch.count_nonzero(value.grad).item())
    ]
    if changed or trainable or gradients:
        raise RuntimeError(
            "Frozen teacher audit failed: "
            f"changed={changed}, trainable={trainable[:5]}, gradients={gradients[:5]}"
        )
    return {
        "sha256": expected_sha256,
        "unchanged": True,
        "trainable_parameter_count": 0,
        "gradient_parameter_count": 0,
    }


def _contract_spec(name: str, *, transformer: bool) -> ShapeModelSpec:
    return ShapeModelSpec(
        name=name,
        tier="contract",
        encoder="contract",
        architecture="contract",
        image_size=32,
        refinement_channels=16,
        refinement_blocks=2,
        context_grid=4 if transformer else 0,
        context_layers=1 if transformer else 0,
        context_heads=4 if transformer else 1,
        encoder_weights=None,
    )


def run_contract_self_test() -> dict[str, object]:
    """Exercise shapes, loss, gradients, v3 transfer, adapters, and global context on CPU."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1701)
        model = V4ShapeNet(_contract_spec("mobile_contract", transformer=False))
        set_training_phase(model, "heads")
        model.train()
        image = torch.rand(2, 3, 32, 40)
        outputs = model(image)
        expected_shapes = {
            "segmentation_logits": (2, 2, 32, 40),
            "extended_segmentation_logits": (2, 5, 32, 40),
            "foreground_logit": (2, 1, 32, 40),
            "fused_boundary_logit": (2, 1, 32, 40),
            "context_boundary_logit": (2, 1, 32, 40),
            "flow_boundary_logit": (2, 1, 32, 40),
            "shape_boundary_logit": (2, 1, 32, 40),
            "signed_distance": (2, 1, 32, 40),
            "flow_yx": (2, 2, 32, 40),
            "affinity_logits": (2, 4, 32, 40),
            "log_variance": (2, 2, 32, 40),
            "boundary_fusion_weights": (2, 3, 32, 40),
        }
        for name, shape in expected_shapes.items():
            if tuple(outputs[name].shape) != shape:
                raise AssertionError(f"{name}: {tuple(outputs[name].shape)} != {shape}")
            if not torch.isfinite(outputs[name]).all():
                raise AssertionError(f"{name} contains non-finite values")
        if not torch.allclose(
            outputs["boundary_fusion_weights"].sum(dim=1),
            torch.ones(2, 32, 40),
            atol=1e-6,
        ):
            raise AssertionError("Boundary fusion weights do not sum to one")

        foreground = torch.zeros(2, 1, 32, 40)
        foreground[:, :, 7:25, 5:35] = 1
        contact = torch.zeros_like(foreground)
        contact[:, :, 7:25, 19:21] = 1
        signed_distance = foreground * 0.8 - (1.0 - foreground) * 0.8
        flow = torch.zeros(2, 2, 32, 40)
        flow[:, 0, 7:25, 5:35] = 0.6
        flow[:, 1, 7:25, 5:35] = 0.8
        # Match shape_targets.geometry_targets: eight affinity maps/valid maps and separate
        # centroid-flow components.  The loss selects the first four local affinity offsets.
        affinities = foreground.repeat(1, 8, 1, 1)
        targets = {
            "foreground": foreground,
            "internal_contact": contact,
            "signed_distance": signed_distance,
            "centroid_flow_y": flow[:, 0],
            "centroid_flow_x": flow[:, 1],
            "affinities": affinities,
            "affinity_valid": torch.ones_like(affinities),
            "valid": torch.ones_like(foreground),
        }
        teacher = {
            "cellprob_logit": torch.logit(foreground * 0.9 + 0.05),
            "flow_y_raw": flow[:, 0] * 5.0,
            "flow_x_raw": flow[:, 1] * 5.0,
            "flow_scale_divisor": 5.0,
            "cellpose_boundary_probability": contact * 0.9 + 0.05,
            "ceb_boundary_probability": contact * 0.9 + 0.05,
            "context_boundary_probability": contact * 0.85 + 0.075,
            "fused_boundary_probability": contact * 0.9 + 0.05,
        }
        loss_report = multitask_shape_loss(outputs, targets, teacher)
        total = loss_report["total"]
        assert isinstance(total, Tensor)
        if total.ndim or not torch.isfinite(total):
            raise AssertionError("Shape loss is not one finite scalar")
        total.backward()
        gradient_report = gradient_audit(model, phase="heads")

        model.eval()
        legacy_adapter = DeploymentAdapter(model).eval()
        extended_adapter = ExtendedDeploymentAdapter(model).eval()
        with torch.no_grad():
            legacy = legacy_adapter(image)
            extended = extended_adapter(image)
        if tuple(legacy.shape) != (2, 2, 32, 40):
            raise AssertionError("Legacy deployment adapter contract changed")
        if tuple(extended.shape) != (2, 5, 32, 40):
            raise AssertionError("Extended deployment adapter contract changed")
        if not torch.allclose(legacy, extended[:, :2], atol=0.0, rtol=0.0):
            raise AssertionError("Legacy logits differ from the first two extended logits")
        traced = torch.jit.trace(legacy_adapter, image, strict=False)
        with torch.no_grad():
            traced_legacy = traced(image)
        if not torch.allclose(legacy, traced_legacy, atol=1e-5, rtol=1e-5):
            raise AssertionError("Traced legacy adapter differs from eager execution")

        source_state = {
            key.removeprefix("semantic_backbone."): value.detach().clone()
            for key, value in model.state_dict().items()
            if key.startswith("semantic_backbone.")
        }
        torch.manual_seed(1702)
        initialized = V4ShapeNet(_contract_spec("init_contract", transformer=False))
        initialization_report = initialize_from_v3(
            initialized,
            {"model": source_state},
            minimum_backbone_fraction=0.99,
        )
        initialized.eval()
        with torch.no_grad():
            initialized_outputs = initialized(image)
            initialized_base = initialized.semantic_backbone(image)
        if not torch.allclose(
            initialized_outputs["segmentation_logits"],
            initialized_base,
            atol=1e-6,
            rtol=1e-6,
        ):
            raise AssertionError("Zero-initialized fusion did not preserve v3 logits")

        transformer_model = V4ShapeNet(
            _contract_spec("transformer_contract", transformer=True)
        ).eval()
        with torch.no_grad():
            transformer_output = transformer_model(image[:1])
        if tuple(transformer_output["extended_segmentation_logits"].shape) != (
            1,
            5,
            32,
            40,
        ):
            raise AssertionError("High-accuracy transformer context contract failed")

        teacher_module = _ContractBackbone()
        teacher_digest = freeze_teacher(teacher_module)
        frozen_teacher_report = audit_frozen_teacher(
            teacher_module, teacher_digest
        )
        return {
            "status": "PASS",
            "model_contract": SHAPE_MODEL_CONTRACT_VERSION,
            "target_contract": SHAPE_TARGET_CONTRACT_VERSION,
            "legacy_output_semantics": list(LEGACY_DEPLOYMENT_OUTPUT_SEMANTICS),
            "extended_output_semantics": list(EXTENDED_DEPLOYMENT_OUTPUT_SEMANTICS),
            "boundary_branch_semantics": dict(BOUNDARY_BRANCH_SEMANTICS),
            "output_shapes": {name: list(shape) for name, shape in expected_shapes.items()},
            "loss": float(total.detach()),
            "loss_components": sorted(loss_report["components"]),
            "gradient_audit": gradient_report,
            "v3_initialization": initialization_report,
            "frozen_teacher": frozen_teacher_report,
            "torchscript_legacy_adapter": "passed",
            "transformer_context": "passed",
            "mobile_spec": asdict(MOBILE_SHAPE_SPEC),
            "high_accuracy_spec": asdict(HIGH_ACCURACY_SHAPE_SPEC),
        }


if __name__ == "__main__":
    print(json.dumps(run_contract_self_test(), indent=2))
