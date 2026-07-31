#!/usr/bin/env python3
"""Memory-bounded inference and the Cellect iPhone post-processing reference.

The deployment reconstruction intentionally mirrors ``CoreMLCellCounter.swift``: 3x3 opening and
closing with a cleared one-pixel border, 8-connected interior markers, then raster-ordered FIFO
multi-source expansion.  SciPy watershed is retained by the main pipeline only as a separately
labelled research diagnostic; it must not select settings shipped to the app.
"""

from __future__ import annotations

from collections import deque
from typing import Protocol

import cv2
import numpy as np
import torch
from torch import nn


POSTPROCESS_VERSION = "iphone-fifo-8connected-v1"
TILED_INFERENCE_VERSION = "overlap-hann-logit-v1"


class PostprocessSettings(Protocol):
    foreground_threshold: float
    boundary_threshold: float
    min_area_fraction: float


def deployment_tensor(image: np.ndarray, size: int) -> torch.Tensor:
    """Create the Linux reference for the app's grayscale RGB, float32 [0,1] input."""
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(np.ascontiguousarray(resized)).float().div_(255.0)
    return tensor[None, None].repeat(1, 3, 1, 1)


@torch.inference_mode()
def predict_deployment(
    model: nn.Module,
    image: np.ndarray,
    size: int,
    device: torch.device,
) -> np.ndarray:
    tensor = deployment_tensor(image, size).to(device, non_blocking=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda",
    ):
        logits = model(tensor)
    return torch.sigmoid(logits.float())[0].cpu().numpy()


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


@torch.inference_mode()
def predict_tiled(
    model: nn.Module,
    image: np.ndarray,
    tile_size: int,
    device: torch.device,
    overlap_fraction: float = 0.25,
    tile_batch_size: int = 1,
) -> np.ndarray:
    """Stitch logits first and post-process the complete native-resolution probability map."""
    if not 0 <= overlap_fraction < 0.75:
        raise ValueError("overlap_fraction must be in [0, 0.75)")
    if tile_batch_size < 1:
        raise ValueError("tile_batch_size must be positive")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original_height, original_width = image.shape
    pad_height = max(0, tile_size - original_height)
    pad_width = max(0, tile_size - original_width)
    padded = cv2.copyMakeBorder(
        image,
        0,
        pad_height,
        0,
        pad_width,
        cv2.BORDER_REFLECT_101,
    )
    height, width = padded.shape
    stride = max(1, int(round(tile_size * (1.0 - overlap_fraction))))
    starts = [
        (y, x)
        for y in _tile_starts(height, tile_size, stride)
        for x in _tile_starts(width, tile_size, stride)
    ]
    hann = np.outer(np.hanning(tile_size), np.hanning(tile_size)).astype(np.float32)
    hann = np.maximum(hann, 1e-3)
    logit_sum = np.zeros((2, height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    for offset in range(0, len(starts), tile_batch_size):
        batch_starts = starts[offset : offset + tile_batch_size]
        tiles = np.stack(
            [padded[y : y + tile_size, x : x + tile_size] for y, x in batch_starts]
        )
        tensor = torch.from_numpy(tiles).to(device=device, dtype=torch.float32)
        tensor = tensor.div_(255.0)[:, None].repeat(1, 3, 1, 1)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            batch_logits = model(tensor)
        batch_logits = batch_logits.float().cpu().numpy()
        for index, (y, x) in enumerate(batch_starts):
            logit_sum[:, y : y + tile_size, x : x + tile_size] += (
                batch_logits[index] * hann
            )
            weight_sum[y : y + tile_size, x : x + tile_size] += hann
    logits = logit_sum / np.maximum(weight_sum[None], 1e-8)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -80, 80)))
    return probabilities[:, :original_height, :original_width]


def _erode_like_iphone(pixels: np.ndarray) -> np.ndarray:
    output = np.zeros(pixels.shape, dtype=bool)
    if pixels.shape[0] < 3 or pixels.shape[1] < 3:
        return pixels.astype(bool, copy=True)
    eroded = cv2.erode(
        pixels.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    output[1:-1, 1:-1] = eroded[1:-1, 1:-1]
    return output


def _dilate_like_iphone(pixels: np.ndarray) -> np.ndarray:
    output = np.zeros(pixels.shape, dtype=bool)
    if pixels.shape[0] < 3 or pixels.shape[1] < 3:
        return pixels.astype(bool, copy=True)
    dilated = cv2.dilate(
        pixels.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    output[1:-1, 1:-1] = dilated[1:-1, 1:-1]
    return output


def _open_like_iphone(pixels: np.ndarray) -> np.ndarray:
    return _dilate_like_iphone(_erode_like_iphone(pixels))


def _close_like_iphone(pixels: np.ndarray) -> np.ndarray:
    return _erode_like_iphone(_dilate_like_iphone(pixels))


def _relabel(labels: np.ndarray) -> np.ndarray:
    values = np.unique(labels)
    values = values[values > 0]
    output = np.zeros(labels.shape, dtype=np.int32)
    for new_label, old_label in enumerate(values, start=1):
        output[labels == old_label] = new_label
    return output


def reconstruct_iphone_instances(
    foreground_probability: np.ndarray,
    boundary_probability: np.ndarray,
    config: PostprocessSettings,
) -> np.ndarray:
    foreground = foreground_probability >= config.foreground_threshold
    foreground = _open_like_iphone(foreground)
    foreground = _close_like_iphone(foreground)
    interiors = foreground & (boundary_probability < config.boundary_threshold)
    marker_count, labels = cv2.connectedComponents(
        interiors.astype(np.uint8), connectivity=8
    )
    marker_count -= 1
    labels = labels.astype(np.int32)
    if marker_count == 0:
        _, labels = cv2.connectedComponents(foreground.astype(np.uint8), connectivity=8)
        return _relabel(labels)

    height, width = foreground.shape
    queue: deque[int] = deque(np.flatnonzero(labels.ravel() > 0).tolist())
    flat_labels = labels.ravel()
    flat_foreground = foreground.ravel()
    while queue:
        index = queue.popleft()
        x = index % width
        y = index // width
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue
                neighbor = ny * width + nx
                if flat_foreground[neighbor] and flat_labels[neighbor] == 0:
                    flat_labels[neighbor] = flat_labels[index]
                    queue.append(neighbor)
    orphan_mask = foreground & (labels == 0)
    orphan_count, orphans = cv2.connectedComponents(
        orphan_mask.astype(np.uint8), connectivity=8
    )
    if orphan_count > 1:
        labels[orphans > 0] = marker_count + orphans[orphans > 0]

    minimum_area = max(
        1,
        int(round(config.min_area_fraction * foreground_probability.size)),
    )
    if minimum_area > 1 and labels.max() > 0:
        areas = np.bincount(labels.ravel())
        remove = np.flatnonzero(areas < minimum_area)
        remove = remove[remove > 0]
        if remove.size:
            labels[np.isin(labels, remove)] = 0
    return _relabel(labels)


def tiled_constant_model_seam_report() -> dict[str, object]:
    """Quantify stitching error for a constant model at overlap and padding seams."""

    class ConstantModel(nn.Module):
        def forward(self, image: torch.Tensor) -> torch.Tensor:
            shape = (image.shape[0], 2, image.shape[2], image.shape[3])
            output = torch.empty(shape, dtype=image.dtype, device=image.device)
            output[:, 0] = 1.25
            output[:, 1] = -0.75
            return output

    image = np.arange(173 * 257, dtype=np.uint8).reshape(173, 257)
    probabilities = predict_tiled(
        ConstantModel(),
        image,
        tile_size=96,
        device=torch.device("cpu"),
        overlap_fraction=0.25,
        tile_batch_size=3,
    )
    expected = 1.0 / (1.0 + np.exp(-np.asarray([1.25, -0.75], dtype=np.float64)))
    absolute_error = np.abs(probabilities.astype(np.float64) - expected[:, None, None])

    tile_size = 96
    stride = int(round(tile_size * (1.0 - 0.25)))
    y_starts = _tile_starts(image.shape[0], tile_size, stride)
    x_starts = _tile_starts(image.shape[1], tile_size, stride)
    seam_mask = np.zeros(image.shape, dtype=bool)
    y_edges = sorted(
        set(y_starts[1:])
        | {
            start + tile_size
            for start in y_starts[:-1]
            if start + tile_size < image.shape[0]
        }
    )
    x_edges = sorted(
        set(x_starts[1:])
        | {
            start + tile_size
            for start in x_starts[:-1]
            if start + tile_size < image.shape[1]
        }
    )
    for edge in y_edges:
        seam_mask[max(0, edge - 1) : min(image.shape[0], edge + 2), :] = True
    for edge in x_edges:
        seam_mask[:, max(0, edge - 1) : min(image.shape[1], edge + 2)] = True
    seam_error = absolute_error[:, seam_mask]
    channel_ranges = [float(channel.max() - channel.min()) for channel in probabilities]
    maximum_absolute_error = float(absolute_error.max())
    seam_maximum_absolute_error = float(seam_error.max())
    tolerance = 1e-6
    passed = (
        probabilities.shape == (2, *image.shape)
        and maximum_absolute_error <= tolerance
        and seam_maximum_absolute_error <= tolerance
        and max(channel_ranges) <= tolerance
    )
    return {
        "contract": TILED_INFERENCE_VERSION,
        "image_shape": list(image.shape),
        "tile_size": tile_size,
        "overlap_fraction": 0.25,
        "tile_batch_size": 3,
        "stride": stride,
        "seam_pixel_count": int(seam_mask.sum()),
        "expected_probabilities": expected.tolist(),
        "channel_probability_ranges": channel_ranges,
        "mean_absolute_error": float(absolute_error.mean()),
        "maximum_absolute_error": maximum_absolute_error,
        "seam_mean_absolute_error": float(seam_error.mean()),
        "seam_maximum_absolute_error": seam_maximum_absolute_error,
        "absolute_error_tolerance": tolerance,
        "passed": passed,
    }


def tiled_stitch_self_check() -> dict[str, object]:
    """Fail closed when a constant model changes across tile seams or padding."""
    report = tiled_constant_model_seam_report()
    if not report["passed"]:
        raise AssertionError(f"Tiled constant-model seam check failed: {report}")
    return report
