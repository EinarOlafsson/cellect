#!/usr/bin/env python3
"""Transparent diagnostic metrics for cell tracking and lineage predictions.

These metrics are intended for model development, ablations, calibration, and domain-wise
validation.  They are deliberately **not** implementations of the Cell Tracking Challenge TRA
score, AOGM, or HOTA.  Those scores have evaluator-specific matching, graph, and weighting
rules.  Use :func:`run_official_evaluator` with an adapter around the official evaluator when an
official score is required.

The native evaluator consumes two explicit tracking graphs and one-to-one, same-frame detection
matches.  Requiring matches makes the segmentation/object-matching policy auditable instead of
silently choosing an IoU or centroid threshold in this module.  It reports:

* detection precision/recall/F1;
* all-pairs within-track association precision/recall/F1;
* adjacent-frame continuation-link precision/recall/F1;
* strict complete-division-event and parent/child-edge precision/recall/F1;
* adjacent-frame ID switches, track fragmentations, mostly-tracked/mostly-lost fractions, and
  globally assigned IDF1 when every detection has a track identity;
* per-frame count agreement; and
* Brier score and expected calibration error (ECE) for supplied link candidates.

Per-domain summaries pool sufficient statistics within a domain.  The macro-domain summary then
weights every domain equally, preventing a large acquisition domain from dominating the result.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Mapping, Protocol, Sequence, runtime_checkable


Identifier = str | int
LINK_CANDIDATE_SCOPES = ("complete", "provided_only")
NATIVE_METRIC_WARNING = (
    "Cellect tracking diagnostics are not official CTC TRA/AOGM or HOTA scores. "
    "Use the official evaluator adapter interface for those names."
)


def _valid_identifier(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    return isinstance(value, str) and bool(value.strip())


def _identifier_key(value: Identifier) -> tuple[str, str]:
    return type(value).__name__, str(value)


def _finite_probability(value: float | None, label: str) -> None:
    if value is None:
        return
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{label} must be finite and in [0, 1]")


@dataclass(frozen=True)
class Detection:
    """One object observation; ``track_id=None`` denotes unavailable identity output."""

    detection_id: Identifier
    frame_index: int
    track_id: Identifier | None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if not _valid_identifier(self.detection_id):
            raise ValueError("Detection ID must be a non-empty string or integer")
        if self.frame_index < 0:
            raise ValueError("Frame index must be non-negative")
        if self.track_id is not None and not _valid_identifier(self.track_id):
            raise ValueError("Track ID must be None, a non-empty string, or an integer")
        _finite_probability(self.confidence, "Detection confidence")


@dataclass(frozen=True)
class DirectedLink:
    """A candidate adjacent-frame link.

    ``selected`` identifies links used in the discrete tracking graph.  Supplying probabilities
    for both selected and rejected candidates gives meaningful unconditional calibration.  If
    only selected links are supplied, declare the case's candidate scope as ``provided_only``.
    """

    source_detection_id: Identifier
    target_detection_id: Identifier
    selected: bool = True
    probability: float | None = None

    def __post_init__(self) -> None:
        if not _valid_identifier(self.source_detection_id):
            raise ValueError("Link source detection ID is invalid")
        if not _valid_identifier(self.target_detection_id):
            raise ValueError("Link target detection ID is invalid")
        if self.source_detection_id == self.target_detection_id:
            raise ValueError("A link cannot point from a detection to itself")
        if not isinstance(self.selected, bool):
            raise ValueError("Link selected flag must be Boolean")
        _finite_probability(self.probability, "Link probability")


@dataclass(frozen=True)
class TrackingGraph:
    """Detections, continuation candidates, and division parent/child candidates.

    ``links`` are ordinary continuation links. ``parent_links`` encode lineage edges from the
    last parent observation to first daughter observations.  A complete division event contains
    at least two selected parent links sharing one parent detection.
    """

    detections: tuple[Detection, ...]
    links: tuple[DirectedLink, ...] = ()
    parent_links: tuple[DirectedLink, ...] = ()

    def __post_init__(self) -> None:
        detection_by_id: dict[Identifier, Detection] = {}
        for detection in self.detections:
            if detection.detection_id in detection_by_id:
                raise ValueError(f"Duplicate detection ID: {detection.detection_id!r}")
            detection_by_id[detection.detection_id] = detection

        identities_by_frame: set[tuple[int, Identifier]] = set()
        for detection in self.detections:
            if detection.track_id is None:
                continue
            key = (detection.frame_index, detection.track_id)
            if key in identities_by_frame:
                raise ValueError(
                    f"Track {detection.track_id!r} has multiple detections in frame "
                    f"{detection.frame_index}"
                )
            identities_by_frame.add(key)

        self._validate_edges(detection_by_id, self.links, "continuation")
        self._validate_edges(detection_by_id, self.parent_links, "parent")

        selected_incoming: dict[Identifier, Identifier] = {}
        selected_outgoing: dict[Identifier, Identifier] = {}
        for link in self.links:
            if not link.selected:
                continue
            source = detection_by_id[link.source_detection_id]
            target = detection_by_id[link.target_detection_id]
            if (
                source.track_id is not None
                and target.track_id is not None
                and source.track_id != target.track_id
            ):
                raise ValueError("Selected continuation link changes track identity")
            if link.target_detection_id in selected_incoming:
                raise ValueError("A detection has multiple selected continuation predecessors")
            if link.source_detection_id in selected_outgoing:
                raise ValueError("A detection has multiple selected continuation successors")
            selected_incoming[link.target_detection_id] = link.source_detection_id
            selected_outgoing[link.source_detection_id] = link.target_detection_id

        selected_parent_for_child: dict[Identifier, Identifier] = {}
        selected_division_parents: set[Identifier] = set()
        for link in self.parent_links:
            if not link.selected:
                continue
            source = detection_by_id[link.source_detection_id]
            target = detection_by_id[link.target_detection_id]
            if (
                source.track_id is not None
                and target.track_id is not None
                and source.track_id == target.track_id
            ):
                raise ValueError("Selected parent link must connect different track identities")
            if link.target_detection_id in selected_parent_for_child:
                raise ValueError("A daughter detection has multiple selected parents")
            selected_parent_for_child[link.target_detection_id] = link.source_detection_id
            selected_division_parents.add(link.source_detection_id)

        if set(selected_incoming) & set(selected_parent_for_child):
            raise ValueError("A detection cannot have both continuation and division predecessors")
        if set(selected_outgoing) & selected_division_parents:
            raise ValueError("A parent cannot both continue and divide in the same next frame")

    @staticmethod
    def _validate_edges(
        detection_by_id: Mapping[Identifier, Detection],
        edges: tuple[DirectedLink, ...],
        label: str,
    ) -> None:
        pairs: set[tuple[Identifier, Identifier]] = set()
        for edge in edges:
            if edge.source_detection_id not in detection_by_id:
                raise ValueError(f"Unknown {label} source: {edge.source_detection_id!r}")
            if edge.target_detection_id not in detection_by_id:
                raise ValueError(f"Unknown {label} target: {edge.target_detection_id!r}")
            pair = (edge.source_detection_id, edge.target_detection_id)
            if pair in pairs:
                raise ValueError(f"Duplicate {label} candidate: {pair!r}")
            pairs.add(pair)
            source = detection_by_id[edge.source_detection_id]
            target = detection_by_id[edge.target_detection_id]
            if target.frame_index != source.frame_index + 1:
                raise ValueError(
                    f"{label.capitalize()} links must join immediately adjacent frames: "
                    f"{source.frame_index} -> {target.frame_index}"
                )


@dataclass(frozen=True)
class DetectionMatch:
    """A caller-declared one-to-one object match within one frame."""

    truth_detection_id: Identifier
    predicted_detection_id: Identifier
    similarity: float | None = None

    def __post_init__(self) -> None:
        if not _valid_identifier(self.truth_detection_id):
            raise ValueError("Truth match ID is invalid")
        if not _valid_identifier(self.predicted_detection_id):
            raise ValueError("Prediction match ID is invalid")
        _finite_probability(self.similarity, "Detection-match similarity")


@dataclass(frozen=True)
class TrackingEvaluationCase:
    """One independently acquired sequence and its explicit evaluation correspondence."""

    case_id: str
    domain: str
    truth: TrackingGraph
    prediction: TrackingGraph
    matches: tuple[DetectionMatch, ...]
    matching_protocol: str
    link_candidate_scope: str = "provided_only"
    lineage_annotations_available: bool = True

    def __post_init__(self) -> None:
        if not self.case_id.strip() or not self.domain.strip():
            raise ValueError("Case ID and domain must be non-empty")
        if not self.matching_protocol.strip():
            raise ValueError("Matching protocol must document the object correspondence rule")
        if self.link_candidate_scope not in LINK_CANDIDATE_SCOPES:
            raise ValueError(
                f"Link candidate scope must be one of {LINK_CANDIDATE_SCOPES}"
            )
        if not isinstance(self.lineage_annotations_available, bool):
            raise ValueError("Lineage-annotation availability must be Boolean")
        truth = {item.detection_id: item for item in self.truth.detections}
        prediction = {item.detection_id: item for item in self.prediction.detections}
        seen_truth: set[Identifier] = set()
        seen_prediction: set[Identifier] = set()
        for match in self.matches:
            if match.truth_detection_id not in truth:
                raise ValueError(f"Match references unknown truth detection: {match}")
            if match.predicted_detection_id not in prediction:
                raise ValueError(f"Match references unknown prediction detection: {match}")
            if match.truth_detection_id in seen_truth:
                raise ValueError("Truth detections may appear in at most one match")
            if match.predicted_detection_id in seen_prediction:
                raise ValueError("Predicted detections may appear in at most one match")
            seen_truth.add(match.truth_detection_id)
            seen_prediction.add(match.predicted_detection_id)
            if (
                truth[match.truth_detection_id].frame_index
                != prediction[match.predicted_detection_id].frame_index
            ):
                raise ValueError("Matched detections must occupy the same frame")


@dataclass(frozen=True)
class PrecisionRecallF1:
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float | None
    recall: float | None
    f1: float | None


@dataclass(frozen=True)
class CalibrationBin:
    lower: float
    upper: float
    includes_upper: bool
    count: int
    confidence_sum: float
    positive_count: int
    mean_confidence: float | None
    observed_frequency: float | None
    ece_contribution: float


@dataclass(frozen=True)
class CalibrationMetrics:
    candidate_count: int
    scored_candidate_count: int
    positive_count: int
    probability_coverage: float | None
    squared_error_sum: float
    brier_score: float | None
    expected_calibration_error: float | None
    bins: tuple[CalibrationBin, ...]
    candidate_scope: str
    interpretation: str


@dataclass(frozen=True)
class CountConsistency:
    frame_count: int
    truth_total: int
    predicted_total: int
    absolute_error_sum: int
    squared_error_sum: int
    signed_error_sum: int
    exact_frame_count: int
    relative_absolute_error_sum: float
    count_normalizer_sum: int
    transition_count: int
    delta_absolute_error_sum: int
    mean_absolute_error: float | None
    root_mean_squared_error: float | None
    mean_signed_error: float | None
    exact_frame_fraction: float | None
    mean_absolute_percentage_error: float | None
    normalized_count_agreement: float | None
    mean_absolute_count_delta_error: float | None


@dataclass(frozen=True)
class IdentityMetrics:
    truth_track_count: int
    predicted_track_count: int
    identity_true_positives: int
    identity_false_positives: int
    identity_false_negatives: int
    identity_precision: float | None
    identity_recall: float | None
    idf1: float | None
    id_switches: int
    fragmentations: int
    mostly_tracked: int
    partially_tracked: int
    mostly_lost: int
    mostly_tracked_fraction: float | None
    mostly_lost_fraction: float | None
    mean_truth_track_coverage: float | None
    id_switches_per_100_truth_detections: float | None
    fragmentations_per_truth_track: float | None
    definition: str


@dataclass(frozen=True)
class TrackingMetricSummary:
    case_count: int
    detection: PrecisionRecallF1
    association_pairs: PrecisionRecallF1 | None
    temporal_links: PrecisionRecallF1
    division_events: PrecisionRecallF1
    division_parent_child_edges: PrecisionRecallF1
    identity: IdentityMetrics | None
    count_consistency: CountConsistency
    link_calibration: CalibrationMetrics
    identity_unavailable_reason: str | None
    lineage_annotated_case_count: int
    lineage_unavailable_reason: str | None


@dataclass(frozen=True)
class CaseTrackingEvaluation:
    case_id: str
    domain: str
    matching_protocol: str
    metrics: TrackingMetricSummary


@dataclass(frozen=True)
class DomainTrackingEvaluation:
    domain: str
    case_ids: tuple[str, ...]
    metrics: TrackingMetricSummary


@dataclass(frozen=True)
class MacroDomainSummary:
    domain_count: int
    metrics: Mapping[str, float | None]
    definition: str


@dataclass(frozen=True)
class DatasetTrackingEvaluation:
    cases: tuple[CaseTrackingEvaluation, ...]
    domains: tuple[DomainTrackingEvaluation, ...]
    macro_domain: MacroDomainSummary
    native_metric_warning: str
    definitions: tuple[str, ...]


def _prf(true_positives: int, false_positives: int, false_negatives: int) -> PrecisionRecallF1:
    if min(true_positives, false_positives, false_negatives) < 0:
        raise ValueError("PRF counts cannot be negative")
    predicted = true_positives + false_positives
    actual = true_positives + false_negatives
    precision = true_positives / predicted if predicted else None
    recall = true_positives / actual if actual else None
    denominator = 2 * true_positives + false_positives + false_negatives
    f1 = (2 * true_positives / denominator) if denominator else None
    return PrecisionRecallF1(
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _selected_pairs(edges: Sequence[DirectedLink]) -> set[tuple[Identifier, Identifier]]:
    return {
        (edge.source_detection_id, edge.target_detection_id)
        for edge in edges
        if edge.selected
    }


def _mapped_edge_prf(
    truth_edges: Sequence[DirectedLink],
    predicted_edges: Sequence[DirectedLink],
    prediction_to_truth: Mapping[Identifier, Identifier],
) -> PrecisionRecallF1:
    truth_pairs = _selected_pairs(truth_edges)
    selected_predictions = [edge for edge in predicted_edges if edge.selected]
    true_positives = 0
    for edge in selected_predictions:
        source = prediction_to_truth.get(edge.source_detection_id)
        target = prediction_to_truth.get(edge.target_detection_id)
        if source is not None and target is not None and (source, target) in truth_pairs:
            true_positives += 1
    return _prf(
        true_positives,
        len(selected_predictions) - true_positives,
        len(truth_pairs) - true_positives,
    )


def _division_events(edges: Sequence[DirectedLink]) -> set[tuple[Identifier, frozenset[Identifier]]]:
    children_by_parent: dict[Identifier, set[Identifier]] = {}
    for edge in edges:
        if edge.selected:
            children_by_parent.setdefault(edge.source_detection_id, set()).add(
                edge.target_detection_id
            )
    return {
        (parent, frozenset(children))
        for parent, children in children_by_parent.items()
        if len(children) >= 2
    }


def _division_event_prf(
    truth_edges: Sequence[DirectedLink],
    predicted_edges: Sequence[DirectedLink],
    prediction_to_truth: Mapping[Identifier, Identifier],
) -> PrecisionRecallF1:
    truth_events = _division_events(truth_edges)
    predicted_events = _division_events(predicted_edges)
    mapped_events: set[tuple[Identifier, frozenset[Identifier]]] = set()
    invalid_events = 0
    for parent, children in predicted_events:
        mapped_parent = prediction_to_truth.get(parent)
        mapped_children = [prediction_to_truth.get(child) for child in children]
        if mapped_parent is None or any(child is None for child in mapped_children):
            invalid_events += 1
            continue
        mapped_events.add(
            (mapped_parent, frozenset(child for child in mapped_children if child is not None))
        )
    true_positives = len(mapped_events & truth_events)
    false_positives = len(mapped_events - truth_events) + invalid_events
    false_negatives = len(truth_events) - true_positives
    return _prf(true_positives, false_positives, false_negatives)


def _association_prf(
    truth: TrackingGraph,
    prediction: TrackingGraph,
    matches: Sequence[DetectionMatch],
) -> PrecisionRecallF1 | None:
    if any(item.track_id is None for item in truth.detections):
        return None
    if any(item.track_id is None for item in prediction.detections):
        return None
    truth_by_id = {item.detection_id: item for item in truth.detections}
    prediction_by_id = {item.detection_id: item for item in prediction.detections}
    truth_track_sizes: dict[Identifier, int] = {}
    predicted_track_sizes: dict[Identifier, int] = {}
    contingency: dict[tuple[Identifier, Identifier], int] = {}
    for item in truth.detections:
        assert item.track_id is not None
        truth_track_sizes[item.track_id] = truth_track_sizes.get(item.track_id, 0) + 1
    for item in prediction.detections:
        assert item.track_id is not None
        predicted_track_sizes[item.track_id] = predicted_track_sizes.get(item.track_id, 0) + 1
    for match in matches:
        truth_track = truth_by_id[match.truth_detection_id].track_id
        predicted_track = prediction_by_id[match.predicted_detection_id].track_id
        assert truth_track is not None and predicted_track is not None
        pair = (truth_track, predicted_track)
        contingency[pair] = contingency.get(pair, 0) + 1
    actual_pairs = sum(size * (size - 1) // 2 for size in truth_track_sizes.values())
    predicted_pairs = sum(
        size * (size - 1) // 2 for size in predicted_track_sizes.values()
    )
    true_positives = sum(size * (size - 1) // 2 for size in contingency.values())
    return _prf(
        true_positives,
        predicted_pairs - true_positives,
        actual_pairs - true_positives,
    )


def _hungarian_minimum(cost: list[list[int]]) -> list[tuple[int, int]]:
    """Rectangular O(n^3) Hungarian assignment without an optional SciPy dependency."""
    if not cost or not cost[0]:
        return []
    rows = len(cost)
    columns = len(cost[0])
    if any(len(row) != columns for row in cost):
        raise ValueError("Assignment cost matrix is ragged")
    transposed = rows > columns
    matrix = [list(row) for row in cost]
    if transposed:
        matrix = [list(row) for row in zip(*matrix)]
        rows, columns = columns, rows

    u = [0] * (rows + 1)
    v = [0] * (columns + 1)
    p = [0] * (columns + 1)
    way = [0] * (columns + 1)
    for row_index in range(1, rows + 1):
        p[0] = row_index
        column0 = 0
        minimum = [math.inf] * (columns + 1)
        used = [False] * (columns + 1)
        while True:
            used[column0] = True
            active_row = p[column0]
            delta = math.inf
            next_column = 0
            for column in range(1, columns + 1):
                if used[column]:
                    continue
                current = matrix[active_row - 1][column - 1] - u[active_row] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    way[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    next_column = column
            for column in range(columns + 1):
                if used[column]:
                    u[p[column]] += int(delta)
                    v[column] -= int(delta)
                else:
                    minimum[column] -= delta
            column0 = next_column
            if p[column0] == 0:
                break
        while True:
            next_column = way[column0]
            p[column0] = p[next_column]
            column0 = next_column
            if column0 == 0:
                break

    pairs = [(p[column] - 1, column - 1) for column in range(1, columns + 1) if p[column]]
    if transposed:
        return [(column, row) for row, column in pairs]
    return pairs


def _maximum_assignment_value(weights: list[list[int]]) -> int:
    if not weights or not weights[0]:
        return 0
    maximum = max(max(row) for row in weights)
    costs = [[maximum - value for value in row] for row in weights]
    return sum(weights[row][column] for row, column in _hungarian_minimum(costs))


def _identity_metrics(
    truth: TrackingGraph,
    prediction: TrackingGraph,
    matches: Sequence[DetectionMatch],
) -> IdentityMetrics | None:
    if any(item.track_id is None for item in truth.detections):
        return None
    if any(item.track_id is None for item in prediction.detections):
        return None

    truth_by_id = {item.detection_id: item for item in truth.detections}
    prediction_by_id = {item.detection_id: item for item in prediction.detections}
    truth_tracks = sorted(
        {item.track_id for item in truth.detections if item.track_id is not None},
        key=_identifier_key,
    )
    predicted_tracks = sorted(
        {item.track_id for item in prediction.detections if item.track_id is not None},
        key=_identifier_key,
    )
    truth_index = {track_id: index for index, track_id in enumerate(truth_tracks)}
    predicted_index = {track_id: index for index, track_id in enumerate(predicted_tracks)}
    weights = [[0] * len(predicted_tracks) for _ in truth_tracks]
    prediction_for_truth: dict[Identifier, Detection] = {}
    for match in matches:
        truth_detection = truth_by_id[match.truth_detection_id]
        predicted_detection = prediction_by_id[match.predicted_detection_id]
        assert truth_detection.track_id is not None
        assert predicted_detection.track_id is not None
        weights[truth_index[truth_detection.track_id]][
            predicted_index[predicted_detection.track_id]
        ] += 1
        prediction_for_truth[truth_detection.detection_id] = predicted_detection

    identity_true_positives = _maximum_assignment_value(weights)
    identity_false_positives = len(prediction.detections) - identity_true_positives
    identity_false_negatives = len(truth.detections) - identity_true_positives
    identity_prf = _prf(
        identity_true_positives,
        identity_false_positives,
        identity_false_negatives,
    )

    detections_by_truth_track: dict[Identifier, list[Detection]] = {
        track_id: [] for track_id in truth_tracks
    }
    for detection in truth.detections:
        assert detection.track_id is not None
        detections_by_truth_track[detection.track_id].append(detection)

    id_switches = 0
    fragmentations = 0
    coverage_values: list[float] = []
    mostly_tracked = 0
    mostly_lost = 0
    partially_tracked = 0
    for track_id in truth_tracks:
        ordered = sorted(
            detections_by_truth_track[track_id],
            key=lambda item: (item.frame_index, _identifier_key(item.detection_id)),
        )
        matched_flags = [item.detection_id in prediction_for_truth for item in ordered]
        matched_count = sum(matched_flags)
        coverage = matched_count / len(ordered)
        coverage_values.append(coverage)
        if coverage >= 0.8:
            mostly_tracked += 1
        elif coverage <= 0.2:
            mostly_lost += 1
        else:
            partially_tracked += 1

        matched_runs = 0
        inside_run = False
        for matched in matched_flags:
            if matched and not inside_run:
                matched_runs += 1
                inside_run = True
            elif not matched:
                inside_run = False
        fragmentations += max(0, matched_runs - 1)

        for previous, current in zip(ordered, ordered[1:]):
            if current.frame_index != previous.frame_index + 1:
                continue
            previous_prediction = prediction_for_truth.get(previous.detection_id)
            current_prediction = prediction_for_truth.get(current.detection_id)
            if previous_prediction is None or current_prediction is None:
                continue
            if previous_prediction.track_id != current_prediction.track_id:
                id_switches += 1

    truth_detection_count = len(truth.detections)
    truth_track_count = len(truth_tracks)
    return IdentityMetrics(
        truth_track_count=truth_track_count,
        predicted_track_count=len(predicted_tracks),
        identity_true_positives=identity_true_positives,
        identity_false_positives=identity_false_positives,
        identity_false_negatives=identity_false_negatives,
        identity_precision=identity_prf.precision,
        identity_recall=identity_prf.recall,
        idf1=identity_prf.f1,
        id_switches=id_switches,
        fragmentations=fragmentations,
        mostly_tracked=mostly_tracked,
        partially_tracked=partially_tracked,
        mostly_lost=mostly_lost,
        mostly_tracked_fraction=(mostly_tracked / truth_track_count if truth_track_count else None),
        mostly_lost_fraction=(mostly_lost / truth_track_count if truth_track_count else None),
        mean_truth_track_coverage=(
            sum(coverage_values) / truth_track_count if truth_track_count else None
        ),
        id_switches_per_100_truth_detections=(
            100.0 * id_switches / truth_detection_count if truth_detection_count else None
        ),
        fragmentations_per_truth_track=(
            fragmentations / truth_track_count if truth_track_count else None
        ),
        definition=(
            "IDF1 uses a maximum-overlap one-to-one assignment between complete truth and "
            "predicted track IDs. ID switches require adjacent matched truth observations to "
            "change predicted identity. Fragmentation is a resumed matched run after an "
            "unmatched truth observation. Mostly tracked is >=80%; mostly lost is <=20%."
        ),
    )


def _count_consistency(cases: Sequence[TrackingEvaluationCase]) -> CountConsistency:
    frame_count = 0
    truth_total = 0
    predicted_total = 0
    absolute_error_sum = 0
    squared_error_sum = 0
    signed_error_sum = 0
    exact_frame_count = 0
    relative_absolute_error_sum = 0.0
    count_normalizer_sum = 0
    transition_count = 0
    delta_absolute_error_sum = 0
    for case in cases:
        truth_counts: dict[int, int] = {}
        predicted_counts: dict[int, int] = {}
        for detection in case.truth.detections:
            truth_counts[detection.frame_index] = truth_counts.get(detection.frame_index, 0) + 1
        for detection in case.prediction.detections:
            predicted_counts[detection.frame_index] = (
                predicted_counts.get(detection.frame_index, 0) + 1
            )
        frames = sorted(set(truth_counts) | set(predicted_counts))
        truth_sequence: list[int] = []
        predicted_sequence: list[int] = []
        for frame in frames:
            truth_count = truth_counts.get(frame, 0)
            predicted_count = predicted_counts.get(frame, 0)
            truth_sequence.append(truth_count)
            predicted_sequence.append(predicted_count)
            error = predicted_count - truth_count
            frame_count += 1
            truth_total += truth_count
            predicted_total += predicted_count
            absolute_error_sum += abs(error)
            squared_error_sum += error * error
            signed_error_sum += error
            exact_frame_count += int(error == 0)
            relative_absolute_error_sum += abs(error) / max(1, truth_count)
            count_normalizer_sum += max(truth_count, predicted_count)
        for index in range(1, len(frames)):
            if frames[index] != frames[index - 1] + 1:
                continue
            truth_delta = truth_sequence[index] - truth_sequence[index - 1]
            prediction_delta = predicted_sequence[index] - predicted_sequence[index - 1]
            delta_absolute_error_sum += abs(prediction_delta - truth_delta)
            transition_count += 1
    return CountConsistency(
        frame_count=frame_count,
        truth_total=truth_total,
        predicted_total=predicted_total,
        absolute_error_sum=absolute_error_sum,
        squared_error_sum=squared_error_sum,
        signed_error_sum=signed_error_sum,
        exact_frame_count=exact_frame_count,
        relative_absolute_error_sum=relative_absolute_error_sum,
        count_normalizer_sum=count_normalizer_sum,
        transition_count=transition_count,
        delta_absolute_error_sum=delta_absolute_error_sum,
        mean_absolute_error=(absolute_error_sum / frame_count if frame_count else None),
        root_mean_squared_error=(
            math.sqrt(squared_error_sum / frame_count) if frame_count else None
        ),
        mean_signed_error=(signed_error_sum / frame_count if frame_count else None),
        exact_frame_fraction=(exact_frame_count / frame_count if frame_count else None),
        mean_absolute_percentage_error=(
            relative_absolute_error_sum / frame_count if frame_count else None
        ),
        normalized_count_agreement=(
            1.0 - absolute_error_sum / count_normalizer_sum
            if count_normalizer_sum
            else (1.0 if frame_count else None)
        ),
        mean_absolute_count_delta_error=(
            delta_absolute_error_sum / transition_count if transition_count else None
        ),
    )


def _calibration(
    cases: Sequence[TrackingEvaluationCase],
    bin_count: int,
) -> CalibrationMetrics:
    if bin_count <= 0:
        raise ValueError("ECE bin count must be positive")
    candidate_count = sum(len(case.prediction.links) for case in cases)
    probability_label_pairs: list[tuple[float, int]] = []
    scopes = {case.link_candidate_scope for case in cases}
    for case in cases:
        truth_pairs = _selected_pairs(case.truth.links)
        prediction_to_truth = {
            match.predicted_detection_id: match.truth_detection_id for match in case.matches
        }
        for edge in case.prediction.links:
            if edge.probability is None:
                continue
            source = prediction_to_truth.get(edge.source_detection_id)
            target = prediction_to_truth.get(edge.target_detection_id)
            label = int(
                source is not None and target is not None and (source, target) in truth_pairs
            )
            probability_label_pairs.append((edge.probability, label))
    scored_count = len(probability_label_pairs)
    positive_count = sum(label for _, label in probability_label_pairs)
    squared_error_sum = sum(
        (probability - label) ** 2 for probability, label in probability_label_pairs
    )
    bin_values: list[list[tuple[float, int]]] = [[] for _ in range(bin_count)]
    for probability, label in probability_label_pairs:
        index = min(int(probability * bin_count), bin_count - 1)
        bin_values[index].append((probability, label))
    bins: list[CalibrationBin] = []
    ece = 0.0
    for index, values in enumerate(bin_values):
        confidence_sum = sum(probability for probability, _ in values)
        positives = sum(label for _, label in values)
        count = len(values)
        mean_confidence = confidence_sum / count if count else None
        observed_frequency = positives / count if count else None
        contribution = (
            count / scored_count * abs(mean_confidence - observed_frequency)
            if count and scored_count and mean_confidence is not None and observed_frequency is not None
            else 0.0
        )
        ece += contribution
        bins.append(
            CalibrationBin(
                lower=index / bin_count,
                upper=(index + 1) / bin_count,
                includes_upper=index == bin_count - 1,
                count=count,
                confidence_sum=confidence_sum,
                positive_count=positives,
                mean_confidence=mean_confidence,
                observed_frequency=observed_frequency,
                ece_contribution=contribution,
            )
        )
    if not scopes:
        scope = "provided_only"
    elif scopes == {"complete"}:
        scope = "complete"
    elif scopes == {"provided_only"}:
        scope = "provided_only"
    else:
        scope = "mixed"
    interpretation = (
        "Calibration covers the caller-declared complete candidate set."
        if scope == "complete"
        else (
            "Calibration is conditional on supplied candidates and must not be interpreted as "
            "full link-candidate calibration."
        )
    )
    return CalibrationMetrics(
        candidate_count=candidate_count,
        scored_candidate_count=scored_count,
        positive_count=positive_count,
        probability_coverage=(scored_count / candidate_count if candidate_count else None),
        squared_error_sum=squared_error_sum,
        brier_score=(squared_error_sum / scored_count if scored_count else None),
        expected_calibration_error=(ece if scored_count else None),
        bins=tuple(bins),
        candidate_scope=scope,
        interpretation=interpretation,
    )


def _sum_prf(values: Sequence[PrecisionRecallF1]) -> PrecisionRecallF1:
    return _prf(
        sum(value.true_positives for value in values),
        sum(value.false_positives for value in values),
        sum(value.false_negatives for value in values),
    )


def _aggregate_identity(values: Sequence[IdentityMetrics | None]) -> IdentityMetrics | None:
    if any(value is None for value in values):
        return None
    identities = [value for value in values if value is not None]
    truth_tracks = sum(value.truth_track_count for value in identities)
    predicted_tracks = sum(value.predicted_track_count for value in identities)
    idtp = sum(value.identity_true_positives for value in identities)
    idfp = sum(value.identity_false_positives for value in identities)
    idfn = sum(value.identity_false_negatives for value in identities)
    prf = _prf(idtp, idfp, idfn)
    truth_detections = idtp + idfn
    id_switches = sum(value.id_switches for value in identities)
    fragmentations = sum(value.fragmentations for value in identities)
    mostly_tracked = sum(value.mostly_tracked for value in identities)
    partially_tracked = sum(value.partially_tracked for value in identities)
    mostly_lost = sum(value.mostly_lost for value in identities)
    coverage_sum = sum(
        (value.mean_truth_track_coverage or 0.0) * value.truth_track_count
        for value in identities
    )
    return IdentityMetrics(
        truth_track_count=truth_tracks,
        predicted_track_count=predicted_tracks,
        identity_true_positives=idtp,
        identity_false_positives=idfp,
        identity_false_negatives=idfn,
        identity_precision=prf.precision,
        identity_recall=prf.recall,
        idf1=prf.f1,
        id_switches=id_switches,
        fragmentations=fragmentations,
        mostly_tracked=mostly_tracked,
        partially_tracked=partially_tracked,
        mostly_lost=mostly_lost,
        mostly_tracked_fraction=(mostly_tracked / truth_tracks if truth_tracks else None),
        mostly_lost_fraction=(mostly_lost / truth_tracks if truth_tracks else None),
        mean_truth_track_coverage=(coverage_sum / truth_tracks if truth_tracks else None),
        id_switches_per_100_truth_detections=(
            100.0 * id_switches / truth_detections if truth_detections else None
        ),
        fragmentations_per_truth_track=(
            fragmentations / truth_tracks if truth_tracks else None
        ),
        definition=(
            "Identity assignments are optimized independently within each acquisition, then "
            "their sufficient statistics are pooled. Switch and fragmentation definitions "
            "match the native per-case diagnostics, not an official challenge evaluator."
        ),
    )


def _evaluate_cases(
    cases: Sequence[TrackingEvaluationCase],
    ece_bins: int,
) -> TrackingMetricSummary:
    per_detection: list[PrecisionRecallF1] = []
    per_association: list[PrecisionRecallF1 | None] = []
    per_link: list[PrecisionRecallF1] = []
    per_division: list[PrecisionRecallF1] = []
    per_parent_edge: list[PrecisionRecallF1] = []
    per_identity: list[IdentityMetrics | None] = []
    for case in cases:
        prediction_to_truth = {
            match.predicted_detection_id: match.truth_detection_id for match in case.matches
        }
        matched = len(case.matches)
        per_detection.append(
            _prf(
                matched,
                len(case.prediction.detections) - matched,
                len(case.truth.detections) - matched,
            )
        )
        per_association.append(_association_prf(case.truth, case.prediction, case.matches))
        per_link.append(
            _mapped_edge_prf(case.truth.links, case.prediction.links, prediction_to_truth)
        )
        if case.lineage_annotations_available:
            per_division.append(
                _division_event_prf(
                    case.truth.parent_links,
                    case.prediction.parent_links,
                    prediction_to_truth,
                )
            )
            per_parent_edge.append(
                _mapped_edge_prf(
                    case.truth.parent_links,
                    case.prediction.parent_links,
                    prediction_to_truth,
                )
            )
        per_identity.append(_identity_metrics(case.truth, case.prediction, case.matches))
    identity_available = all(value is not None for value in per_identity)
    association_available = all(value is not None for value in per_association)
    unavailable_reason = None
    if not identity_available:
        unavailable_reason = (
            "At least one truth or predicted detection lacks a track_id; association pairs, "
            "IDF1, switches, fragmentations, and track-coverage categories are unavailable."
        )
    return TrackingMetricSummary(
        case_count=len(cases),
        detection=_sum_prf(per_detection),
        association_pairs=(
            _sum_prf([value for value in per_association if value is not None])
            if association_available
            else None
        ),
        temporal_links=_sum_prf(per_link),
        division_events=_sum_prf(per_division),
        division_parent_child_edges=_sum_prf(per_parent_edge),
        identity=_aggregate_identity(per_identity),
        count_consistency=_count_consistency(cases),
        link_calibration=_calibration(cases, ece_bins),
        identity_unavailable_reason=unavailable_reason,
        lineage_annotated_case_count=sum(
            case.lineage_annotations_available for case in cases
        ),
        lineage_unavailable_reason=(
            None
            if all(case.lineage_annotations_available for case in cases)
            else (
                "Division-event and parent/child metrics omit cases whose annotation format "
                "contains identities but no parent lineage."
            )
        ),
    )


def evaluate_tracking_case(
    case: TrackingEvaluationCase,
    *,
    ece_bins: int = 10,
) -> CaseTrackingEvaluation:
    """Evaluate one sequence with the documented native diagnostic definitions."""
    return CaseTrackingEvaluation(
        case_id=case.case_id,
        domain=case.domain,
        matching_protocol=case.matching_protocol,
        metrics=_evaluate_cases((case,), ece_bins),
    )


def _metric_value(summary: TrackingMetricSummary, name: str) -> float | None:
    path = name.split(".")
    value: object = summary
    for element in path:
        if value is None:
            return None
        value = getattr(value, element)
    if value is None:
        return None
    return float(value)


def _macro_domain(domains: Sequence[DomainTrackingEvaluation]) -> MacroDomainSummary:
    metric_names = (
        "detection.precision",
        "detection.recall",
        "detection.f1",
        "association_pairs.precision",
        "association_pairs.recall",
        "association_pairs.f1",
        "temporal_links.precision",
        "temporal_links.recall",
        "temporal_links.f1",
        "division_events.precision",
        "division_events.recall",
        "division_events.f1",
        "division_parent_child_edges.f1",
        "identity.identity_precision",
        "identity.identity_recall",
        "identity.idf1",
        "identity.id_switches_per_100_truth_detections",
        "identity.fragmentations_per_truth_track",
        "identity.mostly_tracked_fraction",
        "identity.mostly_lost_fraction",
        "count_consistency.mean_absolute_error",
        "count_consistency.normalized_count_agreement",
        "count_consistency.mean_absolute_count_delta_error",
        "link_calibration.brier_score",
        "link_calibration.expected_calibration_error",
    )
    values: dict[str, float | None] = {}
    for name in metric_names:
        domain_values = [
            value
            for domain in domains
            if (value := _metric_value(domain.metrics, name)) is not None
        ]
        values[name] = sum(domain_values) / len(domain_values) if domain_values else None
    return MacroDomainSummary(
        domain_count=len(domains),
        metrics=values,
        definition=(
            "Arithmetic mean of each available domain-level rate/score; every domain receives "
            "equal weight. Undefined domain metrics are omitted for that metric."
        ),
    )


def evaluate_tracking_dataset(
    cases: Sequence[TrackingEvaluationCase],
    *,
    ece_bins: int = 10,
) -> DatasetTrackingEvaluation:
    """Evaluate acquisitions, pool within domains, and compute equal-domain macro means."""
    case_tuple = tuple(cases)
    if not case_tuple:
        raise ValueError("At least one tracking evaluation case is required")
    case_ids = [case.case_id for case in case_tuple]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Tracking evaluation case IDs must be unique")
    ordered_cases = tuple(sorted(case_tuple, key=lambda item: item.case_id.casefold()))
    case_results = tuple(
        evaluate_tracking_case(case, ece_bins=ece_bins) for case in ordered_cases
    )
    grouped: dict[str, list[TrackingEvaluationCase]] = {}
    for case in ordered_cases:
        grouped.setdefault(case.domain, []).append(case)
    domains = tuple(
        DomainTrackingEvaluation(
            domain=domain,
            case_ids=tuple(case.case_id for case in grouped[domain]),
            metrics=_evaluate_cases(grouped[domain], ece_bins),
        )
        for domain in sorted(grouped, key=str.casefold)
    )
    return DatasetTrackingEvaluation(
        cases=case_results,
        domains=domains,
        macro_domain=_macro_domain(domains),
        native_metric_warning=NATIVE_METRIC_WARNING,
        definitions=(
            "Association positives are all unordered pairs of observations sharing a track ID; "
            "this is a transparent pairwise diagnostic and can weight long tracks strongly.",
            "Temporal-link correctness requires both predicted endpoints to be object-matched "
            "to the exact endpoints of a selected truth continuation link.",
            "A division event is correct only when the matched parent and the complete set of at "
            "least two matched daughters exactly equal one truth event. Parent/child edge PRF "
            "is also reported for partial-credit diagnosis.",
            "Division metrics omit identity-only cases whose source annotations contain no "
            "parent lineage; absent parent fields are not interpreted as negative divisions.",
            "Brier/ECE labels are computed only for supplied predicted link candidates; they are "
            "unconditional only when link_candidate_scope='complete'.",
            NATIVE_METRIC_WARNING,
        ),
    )


@dataclass(frozen=True)
class OfficialEvaluationResult:
    """Opaque result returned by an adapter that invokes an external official evaluator."""

    evaluator_name: str
    evaluator_version: str
    scores: Mapping[str, float]
    artifacts: Mapping[str, str]
    provenance: str

    def __post_init__(self) -> None:
        if not self.evaluator_name.strip() or not self.evaluator_version.strip():
            raise ValueError("Official evaluator name and version are required")
        if not self.provenance.strip():
            raise ValueError("Official evaluator provenance is required")
        for name, value in self.scores.items():
            if not name.strip() or not math.isfinite(value):
                raise ValueError("Official evaluator scores must have names and finite values")


@runtime_checkable
class OfficialTrackingEvaluator(Protocol):
    """Format-specific exporter and official-binary invoker supplied by an integration.

    Implementations must export the exact files required by the named evaluator, invoke that
    evaluator without substituting these native diagnostics, and return version/provenance data.
    The opaque export receipt may contain paths or identifiers needed by ``invoke``.
    """

    def export(
        self,
        cases: tuple[TrackingEvaluationCase, ...],
        destination: Path,
    ) -> object:
        ...

    def invoke(self, export_receipt: object) -> OfficialEvaluationResult:
        ...


def run_official_evaluator(
    cases: Sequence[TrackingEvaluationCase],
    destination: Path,
    evaluator: OfficialTrackingEvaluator,
) -> OfficialEvaluationResult:
    """Delegate export and scoring to an official tool; no TRA/HOTA formula lives here."""
    case_tuple = tuple(cases)
    if not case_tuple:
        raise ValueError("Official evaluation requires at least one case")
    destination = destination.expanduser().resolve()
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"Official evaluation destination is not a directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    receipt = evaluator.export(case_tuple, destination)
    result = evaluator.invoke(receipt)
    if not isinstance(result, OfficialEvaluationResult):
        raise TypeError("Official evaluator adapter returned an invalid result type")
    return result


def _perfect_case(case_id: str, domain: str) -> TrackingEvaluationCase:
    truth_detections = (
        Detection("g_a0", 0, "A"),
        Detection("g_b0", 0, "B"),
        Detection("g_a1", 1, "A"),
        Detection("g_b1", 1, "B"),
        Detection("g_c2", 2, "C"),
        Detection("g_d2", 2, "D"),
        Detection("g_b2", 2, "B"),
    )
    prediction_detections = tuple(
        Detection(f"p_{item.detection_id[2:]}", item.frame_index, f"P-{item.track_id}")
        for item in truth_detections
    )
    truth_links = (
        DirectedLink("g_a0", "g_a1"),
        DirectedLink("g_b0", "g_b1"),
        DirectedLink("g_b1", "g_b2"),
    )
    prediction_links = (
        DirectedLink("p_a0", "p_a1", probability=0.9),
        DirectedLink("p_b0", "p_b1", probability=0.8),
        DirectedLink("p_b1", "p_b2", probability=0.95),
        DirectedLink("p_a0", "p_b1", selected=False, probability=0.1),
    )
    truth_parents = (
        DirectedLink("g_a1", "g_c2"),
        DirectedLink("g_a1", "g_d2"),
    )
    prediction_parents = (
        DirectedLink("p_a1", "p_c2", probability=0.92),
        DirectedLink("p_a1", "p_d2", probability=0.91),
    )
    matches = tuple(
        DetectionMatch(item.detection_id, f"p_{item.detection_id[2:]}", 1.0)
        for item in truth_detections
    )
    return TrackingEvaluationCase(
        case_id=case_id,
        domain=domain,
        truth=TrackingGraph(truth_detections, truth_links, truth_parents),
        prediction=TrackingGraph(
            prediction_detections,
            prediction_links,
            prediction_parents,
        ),
        matches=matches,
        matching_protocol="synthetic exact correspondence",
        link_candidate_scope="complete",
    )


def _imperfect_identity_case() -> TrackingEvaluationCase:
    truth = TrackingGraph(
        detections=tuple(Detection(f"g{frame}", frame, "G") for frame in range(5)),
        links=tuple(DirectedLink(f"g{frame}", f"g{frame + 1}") for frame in range(4)),
    )
    prediction = TrackingGraph(
        detections=(
            Detection("p0", 0, "P"),
            Detection("false1", 1, "FALSE"),
            Detection("p2", 2, "P"),
            Detection("q3", 3, "Q"),
            Detection("q4", 4, "Q"),
            Detection("extra4", 4, "EXTRA"),
        ),
        links=(
            DirectedLink("p0", "false1", selected=False, probability=0.2),
            DirectedLink("false1", "p2", selected=False, probability=0.3),
            DirectedLink("p2", "q3", selected=False, probability=0.7),
            DirectedLink("q3", "q4", selected=True, probability=0.8),
        ),
    )
    return TrackingEvaluationCase(
        case_id="imperfect",
        domain="phase-contrast",
        truth=truth,
        prediction=prediction,
        matches=(
            DetectionMatch("g0", "p0"),
            DetectionMatch("g2", "p2"),
            DetectionMatch("g3", "q3"),
            DetectionMatch("g4", "q4"),
        ),
        matching_protocol="synthetic exact correspondence",
        link_candidate_scope="complete",
    )


def synthetic_self_test() -> dict[str, object]:
    """Exercise perfect topology, calibration, identity failure modes, and macro domains."""
    perfect = _perfect_case("perfect", "brightfield")
    perfect_result = evaluate_tracking_case(perfect, ece_bins=5)
    metrics = perfect_result.metrics
    if metrics.detection.f1 != 1.0 or metrics.temporal_links.f1 != 1.0:
        raise AssertionError("Perfect tracking graph did not receive perfect link/detection F1")
    if metrics.division_events.f1 != 1.0 or metrics.division_parent_child_edges.f1 != 1.0:
        raise AssertionError("Perfect division graph did not receive perfect division F1")
    if metrics.association_pairs is None or metrics.association_pairs.f1 != 1.0:
        raise AssertionError("Perfect identities did not receive perfect association F1")
    if metrics.identity is None or metrics.identity.idf1 != 1.0:
        raise AssertionError("Perfect identities did not receive IDF1=1")
    if metrics.count_consistency.normalized_count_agreement != 1.0:
        raise AssertionError("Perfect frame counts did not agree")
    expected_brier = (0.1**2 + 0.2**2 + 0.05**2 + 0.1**2) / 4
    if metrics.link_calibration.brier_score is None or not math.isclose(
        metrics.link_calibration.brier_score,
        expected_brier,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("Synthetic link Brier score is incorrect")

    identity_only = replace(
        perfect,
        case_id="identity-only-no-lineage",
        lineage_annotations_available=False,
    )
    identity_only_metrics = evaluate_tracking_case(identity_only, ece_bins=5).metrics
    if (
        identity_only_metrics.division_events.f1 is not None
        or identity_only_metrics.division_parent_child_edges.f1 is not None
        or identity_only_metrics.lineage_annotated_case_count != 0
        or identity_only_metrics.lineage_unavailable_reason is None
    ):
        raise AssertionError("Identity-only annotations were scored as division ground truth")

    imperfect = _imperfect_identity_case()
    imperfect_metrics = evaluate_tracking_case(imperfect, ece_bins=5).metrics
    if imperfect_metrics.identity is None:
        raise AssertionError("Synthetic identity metrics unexpectedly unavailable")
    if imperfect_metrics.identity.id_switches != 1:
        raise AssertionError("Adjacent-frame identity switch was not counted")
    if imperfect_metrics.identity.fragmentations != 1:
        raise AssertionError("Interrupted/resumed truth track was not counted as fragmented")
    if imperfect_metrics.detection.true_positives != 4:
        raise AssertionError("Synthetic object matching count is incorrect")

    dataset = evaluate_tracking_dataset((imperfect, perfect), ece_bins=5)
    if dataset.macro_domain.domain_count != 2 or len(dataset.domains) != 2:
        raise AssertionError("Per-domain or macro-domain evaluation is incomplete")
    first = json.dumps(asdict(dataset), sort_keys=True, allow_nan=False)
    second = json.dumps(
        asdict(evaluate_tracking_dataset((perfect, imperfect), ece_bins=5)),
        sort_keys=True,
        allow_nan=False,
    )
    if first != second:
        raise AssertionError("Tracking metrics are not deterministic under case ordering")

    class _SyntheticOfficialAdapter:
        def export(
            self,
            cases: tuple[TrackingEvaluationCase, ...],
            destination: Path,
        ) -> object:
            return {"case_count": len(cases), "destination": str(destination)}

        def invoke(self, export_receipt: object) -> OfficialEvaluationResult:
            if not isinstance(export_receipt, dict) or export_receipt["case_count"] != 1:
                raise AssertionError("Official evaluator export receipt was not preserved")
            return OfficialEvaluationResult(
                evaluator_name="synthetic external evaluator",
                evaluator_version="test-only",
                scores={"synthetic": 0.5},
                artifacts={"directory": str(export_receipt["destination"])},
                provenance="Deterministic interface self-test; not an official CTC score.",
            )

    with tempfile.TemporaryDirectory(prefix="cellect-tracking-metrics-") as temporary:
        official = run_official_evaluator(
            (perfect,), Path(temporary) / "official", _SyntheticOfficialAdapter()
        )
    if official.scores != {"synthetic": 0.5}:
        raise AssertionError("Official evaluator adapter interface failed")

    return {
        "status": "PASS",
        "perfect_detection_f1": metrics.detection.f1,
        "perfect_link_f1": metrics.temporal_links.f1,
        "perfect_division_f1": metrics.division_events.f1,
        "perfect_idf1": metrics.identity.idf1,
        "perfect_link_brier": metrics.link_calibration.brier_score,
        "identity_only_lineage_mask": "PASS",
        "imperfect_id_switches": imperfect_metrics.identity.id_switches,
        "imperfect_fragmentations": imperfect_metrics.identity.fragmentations,
        "domains": [domain.domain for domain in dataset.domains],
        "official_interface": "PASS",
        "official_metric_claimed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run deterministic synthetic tracking-metric checks",
    )
    arguments = parser.parse_args()
    if not arguments.self_test:
        parser.error("No action selected; use --self-test")
    print(json.dumps(synthetic_self_test(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
