#!/usr/bin/env python3
"""Deterministic v4 instance-geometry and boundary-decision targets.

This module is an independent implementation for Cellect.  It converts an integer instance-label
image into bounded, deployment-friendly shape targets and scores candidate region boundaries by
comparing the two possible local decisions: keep the boundary or merge the regions.

All public target arrays are NumPy arrays generated on the CPU.  The implementation deliberately
uses only NumPy, SciPy, and (in the contract test) PyTorch, which are already pinned by the
workstation bundle.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt, label as component_label
from scipy.optimize import linear_sum_assignment


GEOMETRY_TARGET_VERSION = "cellect-instance-geometry-v4"
CANDIDATE_UTILITY_VERSION = "cellect-keep-merge-matched-iou-v1"
SHAPE_TARGET_VERSION = f"{GEOMETRY_TARGET_VERSION}+{CANDIDATE_UTILITY_VERSION}"

# Forward offsets avoid duplicate supervision while covering local diagonals and longer-range
# relationships.  The longer offsets help distinguish two touching instances whose immediately
# adjacent pixels may be occupied by an uncertain or annotated boundary ridge.
DEFAULT_AFFINITY_OFFSETS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 0),
    (1, 1),
    (1, -1),
    (0, 3),
    (3, 0),
    (0, 5),
    (5, 0),
)


@dataclass(frozen=True)
class CandidateKeepMergeTarget:
    """Ground-truth utility of retaining one candidate region-region boundary.

    ``utility`` is ``keep_score - merge_score``.  Therefore ``label == 1`` means keep/split,
    ``label == 0`` means remove/merge, and ``label == -1`` means that the candidate is ambiguous or
    insufficiently supported by annotated foreground.
    """

    region_a: int
    region_b: int
    keep_score: float
    merge_score: float
    utility: float
    label: int
    minimum_ground_truth_fraction: float
    ground_truth_ids: tuple[int, ...]


def _validated_label_image(labels: np.ndarray, name: str = "labels") -> np.ndarray:
    array = np.asarray(labels)
    if array.ndim != 2 or 0 in array.shape:
        raise ValueError(f"{name} must be a non-empty 2-D array; received {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must contain numeric instance IDs")
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"{name} contains a non-finite instance ID")
    if bool((array < 0).any()):
        raise ValueError(f"{name} contains a negative instance ID")
    rounded = np.rint(array)
    if not bool(np.equal(array, rounded).all()):
        raise ValueError(f"{name} contains a non-integer instance ID")
    if float(rounded.max(initial=0)) > float(np.iinfo(np.int64).max):
        raise ValueError(f"{name} contains an instance ID larger than int64")
    return rounded.astype(np.int64, copy=False)


def _normalized_offsets(
    offsets: Iterable[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    normalized: list[tuple[int, int]] = []
    observed: set[tuple[int, int]] = set()
    for offset in offsets:
        if len(offset) != 2:
            raise ValueError(f"Affinity offset must have two values; received {offset!r}")
        dy_float, dx_float = float(offset[0]), float(offset[1])
        if not np.isfinite(dy_float) or not np.isfinite(dx_float):
            raise ValueError(f"Affinity offset must be finite; received {offset!r}")
        dy, dx = int(round(dy_float)), int(round(dx_float))
        if dy_float != dy or dx_float != dx:
            raise ValueError(f"Affinity offset must be integral; received {offset!r}")
        if dy == 0 and dx == 0:
            raise ValueError("Affinity offset (0, 0) has no instance-separation meaning")
        pair = (dy, dx)
        if pair in observed:
            raise ValueError(f"Duplicate affinity offset: {pair}")
        observed.add(pair)
        normalized.append(pair)
    return tuple(normalized)


def _overlap_slices(
    shape: tuple[int, int], dy: int, dx: int
) -> tuple[tuple[slice, slice], tuple[slice, slice]] | None:
    height, width = shape
    if abs(dy) >= height or abs(dx) >= width:
        return None
    source_y = slice(max(0, -dy), height - max(0, dy))
    source_x = slice(max(0, -dx), width - max(0, dx))
    neighbor_y = slice(max(0, dy), height - max(0, -dy))
    neighbor_x = slice(max(0, dx), width - max(0, -dx))
    return (source_y, source_x), (neighbor_y, neighbor_x)


def _half_plane_offsets(radius: int) -> tuple[tuple[int, int], ...]:
    """Return one of each pair of opposite offsets within a Chebyshev radius."""

    offsets: list[tuple[int, int]] = []
    for dy in range(0, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx <= 0:
                continue
            offsets.append((dy, dx))
    return tuple(offsets)


def _instance_perimeter(labels: np.ndarray) -> np.ndarray:
    """Return the one-pixel perimeter of each label, including label-label interfaces."""

    foreground = labels > 0
    perimeter = np.zeros(labels.shape, dtype=bool)
    padded = np.pad(labels, 1, mode="constant", constant_values=0)
    height, width = labels.shape
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            neighbor = padded[
                1 + dy : 1 + dy + height,
                1 + dx : 1 + dx + width,
            ]
            perimeter |= foreground & (neighbor != labels)
    return perimeter


def _unblocked_different_instances(
    labels: np.ndarray,
    source_slice: tuple[slice, slice],
    neighbor_slice: tuple[slice, slice],
    dy: int,
    dx: int,
) -> np.ndarray:
    """Compare endpoints while rejecting pairs whose connecting segment crosses a third cell."""

    source = labels[source_slice]
    neighbor = labels[neighbor_slice]
    different = (source > 0) & (neighbor > 0) & (source != neighbor)
    steps = max(abs(dy), abs(dx))
    if steps <= 1 or not bool(different.any()):
        return different

    source_y, source_x = source_slice
    assert source_y.start is not None and source_y.stop is not None
    assert source_x.start is not None and source_x.stop is not None
    clear = np.ones(source.shape, dtype=bool)
    visited_offsets: set[tuple[int, int]] = set()
    for step in range(1, steps):
        intermediate_dy = int(np.rint(dy * step / steps))
        intermediate_dx = int(np.rint(dx * step / steps))
        intermediate_offset = (intermediate_dy, intermediate_dx)
        if intermediate_offset in visited_offsets:
            continue
        visited_offsets.add(intermediate_offset)
        intermediate = labels[
            source_y.start + intermediate_dy : source_y.stop + intermediate_dy,
            source_x.start + intermediate_dx : source_x.stop + intermediate_dx,
        ]
        clear &= (intermediate == 0) | (intermediate == source) | (
            intermediate == neighbor
        )
    return different & clear


def internal_contact_target(
    labels: np.ndarray,
    *,
    max_background_gap: int = 2,
    boundary_radius: int = 1,
    explicit_contact_map: np.ndarray | None = None,
) -> np.ndarray:
    """Return cell-interior pixels close to another instance, excluding outer contours.

    Two instances are considered near-contact neighbors when their closest labeled pixels are at
    most ``max_background_gap + 1`` pixels apart in Chebyshev distance.  This includes annotations
    with a narrow zero-valued ridge between touching cells.  The result is always projected onto
    foreground pixels, so ordinary cell/background contours cannot become positive targets.
    """

    instance_labels = _validated_label_image(labels)
    if max_background_gap < 0:
        raise ValueError("max_background_gap must be non-negative")
    if boundary_radius < 0:
        raise ValueError("boundary_radius must be non-negative")
    foreground = instance_labels > 0
    contact_endpoints = np.zeros(instance_labels.shape, dtype=bool)
    instance_perimeter = _instance_perimeter(instance_labels)
    comparison_radius = int(max_background_gap) + 1
    for dy, dx in _half_plane_offsets(comparison_radius):
        slices = _overlap_slices(instance_labels.shape, dy, dx)
        if slices is None:
            continue
        source_slice, neighbor_slice = slices
        different_instances = _unblocked_different_instances(
            instance_labels, source_slice, neighbor_slice, dy, dx
        )
        contact_endpoints[source_slice] |= different_instances
        contact_endpoints[neighbor_slice] |= different_instances

    # Offset comparisons find nearby cells, but only their facing surface pixels should seed the
    # contact target.  Without this projection, a two-pixel allowed gap would also make several
    # layers of otherwise interior pixels positive.
    contact_endpoints &= instance_perimeter

    if boundary_radius:
        contact_endpoints = binary_dilation(
            contact_endpoints,
            structure=np.ones((3, 3), dtype=bool),
            iterations=int(boundary_radius),
        )

    if explicit_contact_map is not None:
        explicit = np.asarray(explicit_contact_map, dtype=bool)
        if explicit.shape != instance_labels.shape:
            raise ValueError(
                "explicit_contact_map and labels have different shapes: "
                f"{explicit.shape} versus {instance_labels.shape}"
            )
        components, component_count = component_label(
            explicit, structure=np.ones((3, 3), dtype=np.uint8)
        )
        retained = np.zeros(explicit.shape, dtype=bool)
        inspection_radius = max(1, comparison_radius + int(boundary_radius))
        for component_id in range(1, int(component_count) + 1):
            component = components == component_id
            neighborhood = binary_dilation(
                component,
                structure=np.ones((3, 3), dtype=bool),
                iterations=inspection_radius,
            )
            neighboring_ids = np.unique(instance_labels[neighborhood & foreground])
            if int(np.count_nonzero(neighboring_ids > 0)) >= 2:
                retained |= component
        retained = binary_dilation(
            retained,
            structure=np.ones((3, 3), dtype=bool),
            iterations=inspection_radius,
        )
        contact_endpoints |= retained

    return np.ascontiguousarray(contact_endpoints & foreground)


def normalized_signed_distance(labels: np.ndarray) -> np.ndarray:
    """Return a cell-scale signed-distance target bounded to ``[-1, 1]``.

    Foreground distance is normalized independently by each instance's maximum interior distance.
    Background distance is negative and normalized using the nearest instance's corresponding
    scale, then clipped at ``-1``.  Consequently small and large cells share the same target range
    while the zero crossing remains at the cell/background contour.
    """

    instance_labels = _validated_label_image(labels)
    foreground = instance_labels > 0
    result = np.full(instance_labels.shape, -1.0, dtype=np.float32)
    instance_ids = np.unique(instance_labels[foreground])
    if len(instance_ids) == 0:
        return result

    normalizers: dict[int, float] = {}
    for instance_id_value in instance_ids:
        instance_id = int(instance_id_value)
        mask = instance_labels == instance_id
        # Padding gives border-touching and full-frame objects a real exterior zero set.
        padded = np.pad(mask, 1, mode="constant", constant_values=False)
        interior_distance = distance_transform_edt(padded)[1:-1, 1:-1]
        normalizer = max(float(interior_distance[mask].max(initial=0.0)), 1.0)
        normalizers[instance_id] = normalizer
        result[mask] = np.asarray(
            interior_distance[mask] / normalizer, dtype=np.float32
        )

    background = ~foreground
    if bool(background.any()):
        exterior_distance, nearest_indices = distance_transform_edt(
            background, return_indices=True
        )
        nearest_ids = instance_labels[tuple(nearest_indices)]
        for instance_id, normalizer in normalizers.items():
            selected = background & (nearest_ids == instance_id)
            result[selected] = -np.minimum(
                exterior_distance[selected] / normalizer, 1.0
            ).astype(np.float32)
    return np.ascontiguousarray(np.clip(result, -1.0, 1.0), dtype=np.float32)


def centroid_flow(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-instance normalized ``(dy, dx)`` displacement toward each centroid.

    Each cell's vectors are divided by its largest centroid-to-pixel distance, bounding their
    magnitude by one without making the target depend on the cell's absolute pixel diameter.
    Background vectors are zero.
    """

    instance_labels = _validated_label_image(labels)
    flow_y = np.zeros(instance_labels.shape, dtype=np.float32)
    flow_x = np.zeros(instance_labels.shape, dtype=np.float32)
    instance_ids = np.unique(instance_labels[instance_labels > 0])
    for instance_id_value in instance_ids:
        instance_id = int(instance_id_value)
        ys, xs = np.nonzero(instance_labels == instance_id)
        centroid_y = float(ys.mean())
        centroid_x = float(xs.mean())
        displacement_y = centroid_y - ys.astype(np.float64)
        displacement_x = centroid_x - xs.astype(np.float64)
        scale = max(
            float(np.hypot(displacement_y, displacement_x).max(initial=0.0)), 1.0
        )
        flow_y[ys, xs] = np.asarray(displacement_y / scale, dtype=np.float32)
        flow_x[ys, xs] = np.asarray(displacement_x / scale, dtype=np.float32)
    return np.ascontiguousarray(flow_y), np.ascontiguousarray(flow_x)


def same_instance_affinities(
    labels: np.ndarray,
    affinity_offsets: Iterable[tuple[int, int]] = DEFAULT_AFFINITY_OFFSETS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return same-cell affinities, their valid mask, and normalized offsets.

    An affinity is one only when both offset endpoints are foreground pixels with the same positive
    instance ID.  ``valid`` is one whenever both endpoints are foreground, including negative
    examples spanning two cells.  Background and out-of-frame comparisons are intentionally
    invalid rather than overwhelming the shape loss with easy background negatives.
    """

    instance_labels = _validated_label_image(labels)
    offsets = _normalized_offsets(affinity_offsets)
    affinities = np.zeros(
        (len(offsets), *instance_labels.shape), dtype=np.float32
    )
    valid = np.zeros_like(affinities)
    for channel, (dy, dx) in enumerate(offsets):
        slices = _overlap_slices(instance_labels.shape, dy, dx)
        if slices is None:
            continue
        source_slice, neighbor_slice = slices
        source = instance_labels[source_slice]
        neighbor = instance_labels[neighbor_slice]
        comparison_valid = (source > 0) & (neighbor > 0)
        same = comparison_valid & (source == neighbor)
        valid[channel][source_slice] = comparison_valid.astype(np.float32)
        affinities[channel][source_slice] = same.astype(np.float32)
    return (
        np.ascontiguousarray(affinities),
        np.ascontiguousarray(valid),
        np.asarray(offsets, dtype=np.int32).reshape((-1, 2)),
    )


def geometry_targets(
    labels: np.ndarray,
    affinity_offsets: Iterable[tuple[int, int]] = DEFAULT_AFFINITY_OFFSETS,
    *,
    explicit_contact_map: np.ndarray | None = None,
    max_background_gap: int = 2,
    boundary_radius: int = 1,
) -> dict[str, np.ndarray]:
    """Build all v4 geometry targets from one integer instance-label image.

    Spatial scalar fields have shape ``[H, W]``.  Affinities and their valid mask have shape
    ``[K, H, W]``; ``affinity_offsets`` has shape ``[K, 2]``.  Every returned value is a C-contiguous
    NumPy array, making the dictionary safe to cache in ``npz`` files or convert with
    ``torch.from_numpy``.
    """

    instance_labels = _validated_label_image(labels)
    flow_y, flow_x = centroid_flow(instance_labels)
    affinities, affinity_valid, normalized_offsets = same_instance_affinities(
        instance_labels, affinity_offsets
    )
    return {
        "foreground": np.ascontiguousarray(
            (instance_labels > 0).astype(np.float32)
        ),
        "internal_contact": internal_contact_target(
            instance_labels,
            max_background_gap=max_background_gap,
            boundary_radius=boundary_radius,
            explicit_contact_map=explicit_contact_map,
        ).astype(np.float32),
        "signed_distance": normalized_signed_distance(instance_labels),
        "centroid_flow_y": flow_y,
        "centroid_flow_x": flow_x,
        "affinities": affinities,
        "affinity_valid": affinity_valid,
        "affinity_offsets": normalized_offsets,
    }


def generate_candidate_pairs(
    proposal_labels: np.ndarray,
    *,
    max_background_gap: int = 1,
) -> tuple[tuple[int, int], ...]:
    """Generate deterministic candidate pairs from touching or narrowly separated regions."""

    labels = _validated_label_image(proposal_labels, "proposal_labels")
    if max_background_gap < 0:
        raise ValueError("max_background_gap must be non-negative")
    comparison_radius = int(max_background_gap) + 1
    pairs: set[tuple[int, int]] = set()
    for dy, dx in _half_plane_offsets(comparison_radius):
        slices = _overlap_slices(labels.shape, dy, dx)
        if slices is None:
            continue
        source_slice, neighbor_slice = slices
        source = labels[source_slice]
        neighbor = labels[neighbor_slice]
        different = _unblocked_different_instances(
            labels, source_slice, neighbor_slice, dy, dx
        )
        if not bool(different.any()):
            continue
        observed = np.unique(
            np.stack([source[different], neighbor[different]], axis=1), axis=0
        )
        for left_value, right_value in observed:
            left, right = sorted((int(left_value), int(right_value)))
            pairs.add((left, right))
    return tuple(sorted(pairs))


def _matched_mean_iou(
    predicted_masks: Sequence[np.ndarray],
    ground_truth_labels: np.ndarray,
    ground_truth_ids: Sequence[int],
) -> float:
    if not predicted_masks or not ground_truth_ids:
        return 0.0
    ious = np.zeros((len(predicted_masks), len(ground_truth_ids)), dtype=np.float64)
    for prediction_index, prediction in enumerate(predicted_masks):
        prediction_area = int(np.count_nonzero(prediction))
        for truth_index, ground_truth_id in enumerate(ground_truth_ids):
            truth = ground_truth_labels == int(ground_truth_id)
            intersection = int(np.count_nonzero(prediction & truth))
            union = prediction_area + int(np.count_nonzero(truth)) - intersection
            if union:
                ious[prediction_index, truth_index] = intersection / union
    predicted_indices, truth_indices = linear_sum_assignment(-ious)
    matched_sum = float(ious[predicted_indices, truth_indices].sum())
    return matched_sum / max(len(predicted_masks), len(ground_truth_ids), 1)


def candidate_keep_merge_utility(
    proposal_labels: np.ndarray,
    ground_truth_labels: np.ndarray,
    region_a: int,
    region_b: int,
    *,
    decision_margin: float = 0.05,
    minimum_ground_truth_fraction: float = 0.50,
) -> CandidateKeepMergeTarget:
    """Score one candidate using optimal matched IoU for keep and merge hypotheses."""

    proposals = _validated_label_image(proposal_labels, "proposal_labels")
    ground_truth = _validated_label_image(ground_truth_labels, "ground_truth_labels")
    if proposals.shape != ground_truth.shape:
        raise ValueError(
            "proposal_labels and ground_truth_labels have different shapes: "
            f"{proposals.shape} versus {ground_truth.shape}"
        )
    left, right = sorted((int(region_a), int(region_b)))
    if left <= 0 or left == right:
        raise ValueError("Candidate region IDs must be distinct positive integers")
    region_left = proposals == left
    region_right = proposals == right
    if not bool(region_left.any()) or not bool(region_right.any()):
        raise ValueError(f"Candidate regions {(left, right)} are not both present")
    if decision_margin < 0:
        raise ValueError("decision_margin must be non-negative")
    if not 0.0 <= minimum_ground_truth_fraction <= 1.0:
        raise ValueError("minimum_ground_truth_fraction must be within [0, 1]")

    region_fractions = (
        float(np.count_nonzero(region_left & (ground_truth > 0)))
        / float(np.count_nonzero(region_left)),
        float(np.count_nonzero(region_right & (ground_truth > 0)))
        / float(np.count_nonzero(region_right)),
    )
    observed_fraction = min(region_fractions)
    union = region_left | region_right
    relevant_ids = tuple(
        int(value) for value in np.unique(ground_truth[union]) if int(value) > 0
    )
    keep_score = _matched_mean_iou(
        (region_left, region_right), ground_truth, relevant_ids
    )
    merge_score = _matched_mean_iou((union,), ground_truth, relevant_ids)
    utility = float(keep_score - merge_score)
    if observed_fraction < minimum_ground_truth_fraction or not relevant_ids:
        target = -1
    elif utility > decision_margin:
        target = 1
    elif utility < -decision_margin:
        target = 0
    else:
        target = -1
    return CandidateKeepMergeTarget(
        region_a=left,
        region_b=right,
        keep_score=float(keep_score),
        merge_score=float(merge_score),
        utility=utility,
        label=target,
        minimum_ground_truth_fraction=observed_fraction,
        ground_truth_ids=relevant_ids,
    )


def candidate_keep_merge_targets(
    proposal_labels: np.ndarray,
    ground_truth_labels: np.ndarray,
    candidate_pairs: Iterable[tuple[int, int]] | None = None,
    *,
    max_background_gap: int = 1,
    decision_margin: float = 0.05,
    minimum_ground_truth_fraction: float = 0.50,
) -> tuple[CandidateKeepMergeTarget, ...]:
    """Generate or consume candidate pairs and return stable keep/merge utility targets."""

    proposals = _validated_label_image(proposal_labels, "proposal_labels")
    ground_truth = _validated_label_image(ground_truth_labels, "ground_truth_labels")
    if proposals.shape != ground_truth.shape:
        raise ValueError(
            "proposal_labels and ground_truth_labels have different shapes: "
            f"{proposals.shape} versus {ground_truth.shape}"
        )
    if candidate_pairs is None:
        normalized_pairs = generate_candidate_pairs(
            proposals, max_background_gap=max_background_gap
        )
    else:
        pair_set: set[tuple[int, int]] = set()
        for pair in candidate_pairs:
            if len(pair) != 2:
                raise ValueError(f"Candidate pair must contain two IDs; received {pair!r}")
            left, right = sorted((int(pair[0]), int(pair[1])))
            if left <= 0 or left == right:
                raise ValueError(
                    f"Candidate pair must contain distinct positive IDs; received {pair!r}"
                )
            pair_set.add((left, right))
        normalized_pairs = tuple(sorted(pair_set))
    return tuple(
        candidate_keep_merge_utility(
            proposals,
            ground_truth,
            left,
            right,
            decision_margin=decision_margin,
            minimum_ground_truth_fraction=minimum_ground_truth_fraction,
        )
        for left, right in normalized_pairs
    )


def run_contract_self_test() -> None:
    """Run deterministic synthetic CPU assertions for every public target family."""

    empty = np.zeros((24, 28), dtype=np.int32)
    empty_targets = geometry_targets(empty, ((0, 1), (1, 0)))
    assert not bool(empty_targets["foreground"].any())
    assert not bool(empty_targets["internal_contact"].any())
    assert np.array_equal(
        empty_targets["signed_distance"], np.full(empty.shape, -1, np.float32)
    )
    assert not bool(empty_targets["centroid_flow_y"].any())
    assert not bool(empty_targets["centroid_flow_x"].any())
    assert empty_targets["affinities"].shape == (2, *empty.shape)
    assert not bool(empty_targets["affinity_valid"].any())

    single = np.zeros((33, 35), dtype=np.int32)
    yy, xx = np.ogrid[:33, :35]
    single[(yy - 16) ** 2 + (xx - 17) ** 2 <= 9**2] = 1
    first = geometry_targets(single, ((0, 1), (1, 0), (0, 3)))
    second = geometry_targets(single, ((0, 1), (1, 0), (0, 3)))
    for key in first:
        assert np.array_equal(first[key], second[key]), f"Non-deterministic {key}"
        assert first[key].flags.c_contiguous, f"Non-contiguous {key}"
    assert not bool(first["internal_contact"].any())
    assert float(first["signed_distance"][16, 17]) == 1.0
    assert float(first["signed_distance"][0, 0]) == -1.0
    assert first["centroid_flow_x"][16, 10] > 0
    assert first["centroid_flow_x"][16, 24] < 0
    assert first["centroid_flow_y"][9, 17] > 0
    assert first["centroid_flow_y"][23, 17] < 0
    assert first["affinities"][0, 16, 16] == 1
    stacked = np.stack(
        [
            first["foreground"],
            first["internal_contact"],
            first["signed_distance"],
            first["centroid_flow_y"],
            first["centroid_flow_x"],
        ]
    )
    assert stacked.dtype == np.float32 and stacked.flags.c_contiguous

    touching = np.zeros((32, 36), dtype=np.int32)
    touching[8:24, 4:14] = 1
    touching[8:24, 14:24] = 2
    touching_targets = geometry_targets(touching, ((0, 1), (1, 0)))
    contact = touching_targets["internal_contact"].astype(bool)
    assert bool(contact[:, 11:17].any())
    assert not bool(contact[:, :10].any()), "Outer contour became an internal contact"
    assert not bool(contact[:, 18:].any()), "Outer contour became an internal contact"
    assert touching_targets["affinity_valid"][0, 12, 13] == 1
    assert touching_targets["affinities"][0, 12, 13] == 0

    narrow_gap = np.zeros((24, 30), dtype=np.int32)
    narrow_gap[5:19, 3:11] = 1
    narrow_gap[5:19, 12:20] = 2
    assert generate_candidate_pairs(narrow_gap, max_background_gap=1) == ((1, 2),)
    assert bool(
        internal_contact_target(narrow_gap, max_background_gap=1)[:, 8:15].any()
    )

    proposals = np.zeros((20, 24), dtype=np.int32)
    proposals[4:16, 4:10] = 1
    proposals[4:16, 10:16] = 2
    one_truth = np.zeros_like(proposals)
    one_truth[4:16, 4:16] = 1
    merge_target = candidate_keep_merge_utility(proposals, one_truth, 2, 1)
    assert merge_target.label == 0
    assert np.isclose(merge_target.keep_score, 0.25)
    assert np.isclose(merge_target.merge_score, 1.0)
    assert np.isclose(merge_target.utility, -0.75)

    two_truth = proposals.copy()
    keep_target = candidate_keep_merge_targets(proposals, two_truth)
    assert len(keep_target) == 1 and keep_target[0].label == 1
    assert np.isclose(keep_target[0].keep_score, 1.0)
    assert np.isclose(keep_target[0].merge_score, 0.25)
    assert np.isclose(keep_target[0].utility, 0.75)

    no_truth = np.zeros_like(proposals)
    ignored = candidate_keep_merge_utility(proposals, no_truth, 1, 2)
    assert ignored.label == -1 and ignored.ground_truth_ids == ()

    try:
        geometry_targets(np.array([[0, -1]], dtype=np.int32))
    except ValueError as error:
        assert "negative" in str(error)
    else:
        raise AssertionError("Negative instance IDs were accepted")


__all__ = [
    "CANDIDATE_UTILITY_VERSION",
    "DEFAULT_AFFINITY_OFFSETS",
    "GEOMETRY_TARGET_VERSION",
    "SHAPE_TARGET_VERSION",
    "CandidateKeepMergeTarget",
    "candidate_keep_merge_targets",
    "candidate_keep_merge_utility",
    "centroid_flow",
    "generate_candidate_pairs",
    "geometry_targets",
    "internal_contact_target",
    "normalized_signed_distance",
    "run_contract_self_test",
    "same_instance_affinities",
]


if __name__ == "__main__":
    run_contract_self_test()
    print(f"shape target contract passed: {SHAPE_TARGET_VERSION}")
