#!/usr/bin/env python3
"""Shared instance-to-target conversion for all Cellect semantic datasets."""

from __future__ import annotations

import cv2
import numpy as np
from skimage.segmentation import expand_labels


BOUNDARY_TARGET_VERSION = "internal-contact-plus-deepsea-wmap-v2"


def internal_contact_boundary(
    labels: np.ndarray,
    explicit_contact_map: np.ndarray | None = None,
) -> np.ndarray:
    """Return only cell-cell contact ridges, never ordinary cell/background contours.

    The returned ridge is projected onto cell interiors because Cellect removes high-boundary
    pixels from the foreground to seed its instance expansion.  DeepSea's official U-Net weight
    map often occupies the narrow background gap between cells, so it is dilated onto the two
    adjacent cell interiors before being combined with label-label interfaces.
    """
    labels = np.asarray(labels, dtype=np.int64)
    foreground = labels > 0
    boundary = np.zeros(labels.shape, dtype=bool)

    def label_interfaces(label_image: np.ndarray) -> np.ndarray:
        interfaces = np.zeros(label_image.shape, dtype=bool)
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            source_y = slice(max(0, -dy), labels.shape[0] - max(0, dy))
            source_x = slice(max(0, -dx), labels.shape[1] - max(0, dx))
            neighbor_y = slice(max(0, dy), labels.shape[0] - max(0, -dy))
            neighbor_x = slice(max(0, dx), labels.shape[1] - max(0, -dx))
            source = label_image[source_y, source_x]
            neighbor = label_image[neighbor_y, neighbor_x]
            interfaces[source_y, source_x] |= (
                (source > 0) & (neighbor > 0) & (source != neighbor)
            )
        return interfaces

    direct_contact = cv2.dilate(
        label_interfaces(labels).astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    boundary |= direct_contact & foreground
    # Many annotation formats leave a one- or two-pixel zero gap between touching cells.  Expand
    # labels only one pixel per side to find those near contacts, then project the interface onto the
    # original foreground.  Truly isolated cells remain boundary-negative.
    expanded = expand_labels(labels, distance=1)
    near_contact = label_interfaces(expanded)
    near_contact = cv2.dilate(
        near_contact.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    boundary |= near_contact & foreground
    if explicit_contact_map is not None:
        explicit = np.asarray(explicit_contact_map, dtype=bool)
        if explicit.shape != labels.shape:
            raise RuntimeError(
                "Explicit contact map and labels have different dimensions: "
                f"{explicit.shape} versus {labels.shape}"
            )
        _, components = cv2.connectedComponents(
            explicit.astype(np.uint8), connectivity=8
        )
        retained = np.zeros(explicit.shape, dtype=bool)
        for component_id in range(1, int(components.max()) + 1):
            component = components == component_id
            neighborhood = cv2.dilate(
                component.astype(np.uint8),
                np.ones((3, 3), dtype=np.uint8),
                iterations=2,
            ).astype(bool)
            adjacent_labels = np.unique(labels[neighborhood & foreground])
            adjacent_labels = adjacent_labels[adjacent_labels > 0]
            if len(adjacent_labels) >= 2:
                retained |= component
        projected = cv2.dilate(
            retained.astype(np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            iterations=2,
        ).astype(bool)
        boundary |= projected
    return boundary & foreground
