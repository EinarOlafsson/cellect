#!/usr/bin/env python3
"""TorchScript/ONNX/output-contract parity gates for Cellect deployment artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from deployment_runtime import (
    POSTPROCESS_VERSION,
    TILED_INFERENCE_VERSION,
    PostprocessSettings,
    deployment_tensor,
    predict_tiled,
    reconstruct_iphone_instances,
    tiled_constant_model_seam_report,
)
from scientific_splits import BUNDLE_VERSION


PARITY_VERSION = "torchscript-onnx-postprocess-fusion-v2"


PostprocessCallback = Callable[
    [np.ndarray, np.ndarray, PostprocessSettings], np.ndarray
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _difference(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    return {
        "mean_absolute_error": float(difference.mean()),
        "p99_absolute_error": float(np.percentile(difference, 99)),
        "maximum_absolute_error": float(difference.max()),
    }


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits.astype(np.float32), -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def _config_payload(config: PostprocessSettings) -> dict[str, object]:
    required = (
        "foreground_threshold",
        "boundary_threshold",
        "min_area_fraction",
    )
    for name in required:
        if not hasattr(config, name):
            raise TypeError(f"Postprocess config is missing {name}")
        value = float(getattr(config, name))
        if not np.isfinite(value):
            raise ValueError(f"Postprocess config {name} is not finite")

    if is_dataclass(config) and not isinstance(config, type):
        raw = asdict(config)
    elif hasattr(config, "__dict__"):
        raw = dict(vars(config))
    else:
        raw = {name: getattr(config, name) for name in required}

    payload: dict[str, object] = {}
    for name, value in raw.items():
        if str(name).startswith("_"):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if value is None or isinstance(value, (bool, int, float, str)):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not np.isfinite(float(value)):
                    raise ValueError(f"Postprocess config {name} is not finite")
            payload[str(name)] = value
    for name in required:
        payload[name] = float(getattr(config, name))
    return payload


def _validated_labels(
    callback: PostprocessCallback,
    probabilities: np.ndarray,
    config: PostprocessSettings,
    description: str,
) -> np.ndarray:
    if probabilities.ndim != 3 or probabilities.shape[0] != 2:
        raise RuntimeError(
            f"{description} probabilities must have shape (2,H,W), got "
            f"{probabilities.shape}"
        )
    labels = np.asarray(callback(probabilities[0], probabilities[1], config))
    if labels.shape != tuple(probabilities.shape[1:]):
        raise RuntimeError(
            f"{description} labels have shape {labels.shape}, expected "
            f"{probabilities.shape[1:]}"
        )
    if not np.isfinite(labels).all() or np.any(labels < 0):
        raise RuntimeError(f"{description} labels contain invalid values")
    if not np.array_equal(labels, np.rint(labels)):
        raise RuntimeError(f"{description} labels must be integer-valued")
    return labels.astype(np.int32)


def _resize_probabilities(probabilities: np.ndarray, size: int) -> np.ndarray:
    if probabilities.shape == (2, size, size):
        return probabilities.astype(np.float32, copy=False)
    tensor = torch.from_numpy(np.ascontiguousarray(probabilities))[None].float()
    return (
        torch.nn.functional.interpolate(
            tensor,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )[0]
        .numpy()
        .astype(np.float32, copy=False)
    )


@torch.inference_mode()
def verify_deployment_artifacts(
    model: nn.Module,
    image_paths: list[Path],
    image_size: int,
    torchscript_path: Path,
    onnx_path: Path,
    destination: Path,
    device: torch.device,
    *,
    postprocess_config: PostprocessSettings,
    postprocess_callback: PostprocessCallback | None = None,
) -> dict[str, object]:
    """Export Linux references while leaving Core ML/Swift approval to the Mac gate."""
    if not image_paths:
        raise RuntimeError(
            "Deployment parity requires at least one real calibration image"
        )
    callback = postprocess_callback or reconstruct_iphone_instances
    config_payload = _config_payload(postprocess_config)
    # Loading a second large SegFormer as TorchScript beside the eager model on the RTX 3090 can
    # exceed memory even though inference for either artifact fits independently.
    traced = torch.jit.load(str(torchscript_path), map_location="cpu").eval()
    # The export comparison below runs the eager model on the CPU as well.  CUDA and CPU kernels
    # for the same weights disagree by ~0.4% of the logit scale on this hardware, which is device
    # arithmetic rather than export error and would swamp a 1e-3 tolerance meant to detect a wrong
    # export.  Comparing all three on one device isolates artifact fidelity, and the CPU path is
    # also the closer analogue of the on-device deployment target.
    reference_device = torch.device("cpu")
    model = model.eval().to(reference_device)
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    rows: list[dict[str, object]] = []
    golden_input: np.ndarray | None = None
    golden_logits: np.ndarray | None = None
    golden_image: np.ndarray | None = None
    for path in image_paths[:3]:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Could not decode deployment parity image {path}")
        tensor_cpu = deployment_tensor(image, image_size)
        eager = model(tensor_cpu).float().cpu().numpy()
        torchscript = traced(tensor_cpu).float().cpu().numpy()
        onnx = session.run(None, {session.get_inputs()[0].name: tensor_cpu.numpy()})[0]
        expected_shape = (1, 2, image_size, image_size)
        for name, output in (
            ("eager", eager),
            ("torchscript", torchscript),
            ("onnx", onnx),
        ):
            if output.shape != expected_shape:
                raise RuntimeError(
                    f"{name} output shape {output.shape}, expected {expected_shape}"
                )
            if not np.isfinite(output).all():
                raise RuntimeError(f"{name} emitted NaN or infinity")
        rows.append(
            {
                "image": str(path),
                "torchscript": _difference(eager, torchscript),
                "onnx": _difference(eager, onnx),
            }
        )
        if golden_input is None:
            golden_input = tensor_cpu.numpy()
            golden_logits = eager
            golden_image = image.copy()

    maximum_torchscript = max(
        float(row["torchscript"]["maximum_absolute_error"]) for row in rows
    )
    maximum_onnx = max(float(row["onnx"]["maximum_absolute_error"]) for row in rows)
    p99_onnx = max(float(row["onnx"]["p99_absolute_error"]) for row in rows)
    export_passed = (
        maximum_torchscript <= 1e-3 and maximum_onnx <= 1e-2 and p99_onnx <= 1e-3
    )

    assert (
        golden_input is not None
        and golden_logits is not None
        and golden_image is not None
    )
    eager_probabilities = _sigmoid(golden_logits)[0]
    expected_fifo_labels = _validated_labels(
        reconstruct_iphone_instances,
        eager_probabilities,
        postprocess_config,
        "eager FIFO",
    )
    expected_postprocessed_labels = _validated_labels(
        callback,
        eager_probabilities,
        postprocess_config,
        "eager postprocessed",
    )

    # Two inference paths on the same non-square source provide distinct member maps for the
    # equal-weight fusion fixture without requiring a second checkpoint during per-model export.
    seam_image = cv2.resize(
        golden_image,
        (image_size + 73, image_size + 17),
        interpolation=cv2.INTER_LINEAR,
    )
    whole_frame_tensor = deployment_tensor(seam_image, image_size).to(reference_device)
    whole_frame_logits = model(whole_frame_tensor).float().cpu().numpy()
    expected_whole_frame_shape = (1, 2, image_size, image_size)
    if whole_frame_logits.shape != expected_whole_frame_shape:
        raise RuntimeError(
            "Fusion whole-frame output shape "
            f"{whole_frame_logits.shape}, expected {expected_whole_frame_shape}"
        )
    whole_frame_probabilities = _sigmoid(whole_frame_logits)[0]
    tiled = predict_tiled(
        model,
        seam_image,
        tile_size=image_size,
        device=reference_device,
        overlap_fraction=0.25,
        tile_batch_size=1,
    )
    if tiled.shape != (2, *seam_image.shape) or not np.isfinite(tiled).all():
        raise RuntimeError("Tiled inference failed its non-square finite-output gate")

    fusion_size = max(whole_frame_probabilities.shape[2], tiled.shape[2])
    fusion_members = np.stack(
        [
            _resize_probabilities(whole_frame_probabilities, fusion_size),
            _resize_probabilities(tiled, fusion_size),
        ]
    ).astype(np.float32, copy=False)
    expected_equal_weight_fusion = (fusion_members[0] + fusion_members[1]) * np.float32(
        0.5
    )
    expected_fused_fifo_labels = _validated_labels(
        reconstruct_iphone_instances,
        expected_equal_weight_fusion,
        postprocess_config,
        "equal-weight fused FIFO",
    )
    expected_fused_postprocessed_labels = _validated_labels(
        callback,
        expected_equal_weight_fusion,
        postprocess_config,
        "equal-weight fused postprocessed",
    )

    constant_seam_report = tiled_constant_model_seam_report()
    passed = export_passed and bool(constant_seam_report["passed"])

    numeric_config = sorted(
        (name, float(value))
        for name, value in config_payload.items()
        if isinstance(value, (bool, int, float))
    )
    callback_name = (
        f"{getattr(callback, '__module__', '<unknown>')}."
        f"{getattr(callback, '__qualname__', getattr(callback, '__name__', '<callable>'))}"
    )

    destination.mkdir(parents=True, exist_ok=True)
    golden_path = destination / "deployment_golden_v4.npz"
    np.savez_compressed(
        golden_path,
        input_nchw_float32=golden_input.astype(np.float32),
        eager_logits_nchw_float32=golden_logits.astype(np.float32),
        eager_probabilities_chw_float32=eager_probabilities.astype(np.float32),
        postprocess_config_json=np.asarray(
            json.dumps(config_payload, sort_keys=True), dtype=np.str_
        ),
        postprocess_config_names=np.asarray(
            [name for name, _ in numeric_config], dtype=np.str_
        ),
        postprocess_config_values_float64=np.asarray(
            [value for _, value in numeric_config], dtype=np.float64
        ),
        expected_fifo_labels_hw_int32=expected_fifo_labels,
        expected_postprocessed_labels_hw_int32=expected_postprocessed_labels,
        fusion_source_grayscale_hw_uint8=seam_image.astype(np.uint8),
        fusion_member_names=np.asarray(
            ["whole_frame_resize", "native_tiled"], dtype=np.str_
        ),
        fusion_member_0_probabilities_chw_float32=whole_frame_probabilities,
        fusion_member_1_probabilities_chw_float32=tiled.astype(np.float32),
        fusion_member_aligned_probabilities_mchw_float32=fusion_members,
        fusion_equal_weights_float32=np.asarray([0.5, 0.5], dtype=np.float32),
        expected_equal_weight_fusion_chw_float32=expected_equal_weight_fusion,
        expected_fused_fifo_labels_hw_int32=expected_fused_fifo_labels,
        expected_fused_labels_hw_int32=expected_fused_postprocessed_labels,
        expected_fused_postprocessed_labels_hw_int32=(
            expected_fused_postprocessed_labels
        ),
        tiled_constant_seam_maximum_absolute_error_float64=np.asarray(
            constant_seam_report["seam_maximum_absolute_error"], dtype=np.float64
        ),
        tiled_constant_maximum_absolute_error_float64=np.asarray(
            constant_seam_report["maximum_absolute_error"], dtype=np.float64
        ),
    )
    report: dict[str, object] = {
        "schema_version": 4,
        "bundle_version": BUNDLE_VERSION,
        "parity_version": PARITY_VERSION,
        # Eager reference, TorchScript and ONNX are all evaluated on this device, so the recorded
        # errors measure export fidelity rather than CUDA-versus-CPU arithmetic.
        "reference_device": str(reference_device),
        "postprocess_contract": POSTPROCESS_VERSION,
        "tiled_inference_contract": TILED_INFERENCE_VERSION,
        "input_contract": {
            "shape": [1, 3, image_size, image_size],
            "dtype": "float32",
            "range": "[0,1]",
            "channels": "rendered grayscale repeated three times",
            "whole_frame_resize_reference": "OpenCV INTER_AREA on Linux; Mac fixture must compare CoreGraphics rendering",
        },
        "output_contract": {
            "shape": [1, 2, image_size, image_size],
            "semantics": ["foreground_logit", "contact_boundary_logit"],
        },
        "torchscript_sha256": _sha256(torchscript_path),
        "onnx_sha256": _sha256(onnx_path),
        "real_image_comparisons": rows,
        "thresholds": {
            "torchscript_max_absolute_error": 1e-3,
            "onnx_max_absolute_error": 1e-2,
            "onnx_p99_absolute_error": 1e-3,
        },
        "tiled_non_square_shape": list(tiled.shape),
        "tiled_constant_model_seam": constant_seam_report,
        "postprocess_reference": {
            "config": config_payload,
            "callback": callback_name,
            "fifo_instance_count": int(expected_fifo_labels.max()),
            "postprocessed_instance_count": int(expected_postprocessed_labels.max()),
        },
        "fusion_reference": {
            "member_names": ["whole_frame_resize", "native_tiled"],
            "raw_member_shapes": [
                list(whole_frame_probabilities.shape),
                list(tiled.shape),
            ],
            "alignment": "bilinear pixel-center (PyTorch align_corners=False)",
            "aligned_member_shape": list(fusion_members.shape),
            "weights": [0.5, 0.5],
            "expected_probability_shape": list(expected_equal_weight_fusion.shape),
            "fifo_instance_count": int(expected_fused_fifo_labels.max()),
            "postprocessed_instance_count": int(
                expected_fused_postprocessed_labels.max()
            ),
        },
        "golden_fixture": golden_path.name,
        "golden_fixture_sha256": _sha256(golden_path),
        "linux_export_parity_passed": export_passed,
        "linux_reference_fixture_generated": True,
        "linux_reference_checks_passed": passed,
        "coreml_swift_execution_performed": False,
        "coreml_swift_parity_passed": None,
        "coreml_swift_parity_required_on_mac": True,
        "deployment_approval_status": "requires Core ML/Swift parity on a Mac and physical-iPhone validation",
    }
    report_path = destination / "deployment_parity_v4.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise RuntimeError("Deployment artifact parity failed. See " + str(report_path))
    return report
