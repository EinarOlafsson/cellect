#!/usr/bin/env python3
"""Fast fixture checks for dataset adapters, sparse supervision, and sampling policy."""

from __future__ import annotations

import io
import hashlib
import os
import tempfile
import unittest
import urllib.error
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch
from cellpose import models as cellpose_models

from accuracy_data import (
    ExternalCellDataset,
    ExternalSample,
    ExternalSourceFileExclusion,
    discover_cellpose_pairs,
    discover_ctc_gold,
    discover_deepsea,
    discover_suffix_pairs,
    deepsea_instance_labels,
    deepsea_subfolders,
    download,
    find_local_deepsea_root,
    labels_to_targets as external_labels_to_targets,
    materialize_cellpose_pairs,
    normalize_uint8,
    source_split,
)
from data_preflight import _seal_external_source_file_exclusions
from deployment_runtime import tiled_stitch_self_check
from pipeline import (
    ACCURACY_CONFIG,
    DEFAULT_POSTPROCESS_CONFIG,
    PostprocessConfig,
    channel_metrics,
    combined_loss,
    instance_labels,
    modality_balanced_sampler,
    segmentation_metrics,
    training_transform,
    tune_postprocessing_records,
)
from scientific_splits import (
    VALIDATION_ROLES,
    livecell_role,
    scientific_group,
    validation_role,
)


def expect_runtime_error(
    action: Callable[[], object],
    expected: str | tuple[str, ...],
) -> None:
    fragments = (expected,) if isinstance(expected, str) else expected
    try:
        action()
    except RuntimeError as error:
        for fragment in fragments:
            assert fragment in str(error), f"Expected {fragment!r} in {error!r}"
    else:
        raise AssertionError(f"Expected RuntimeError containing {fragments!r}")


def check_resumable_download_retry() -> None:
    payload = b"retry-contract" * 256

    class FixtureResponse(io.BytesIO):
        status = 200

        def __init__(self, body: bytes):
            super().__init__(body)
            self.headers = {"Content-Length": str(len(body))}

        def __enter__(self) -> "FixtureResponse":
            return self

        def __exit__(self, *_: object) -> None:
            self.close()

    with tempfile.TemporaryDirectory(prefix="cellect-download-retry-") as temporary:
        destination = Path(temporary) / "fixture.bin"
        with (
            patch(
                "accuracy_data.urllib.request.urlopen",
                side_effect=[
                    urllib.error.URLError("temporary name-resolution failure"),
                    FixtureResponse(payload),
                ],
            ) as mocked_open,
            patch("accuracy_data.time.sleep") as mocked_sleep,
        ):
            download("https://fixture.invalid/data", destination, attempts=2)
        assert destination.read_bytes() == payload
        assert mocked_open.call_count == 2
        mocked_sleep.assert_called_once_with(1)


def check_external_exclusion_provenance() -> None:
    with tempfile.TemporaryDirectory(prefix="cellect-exclusion-provenance-") as temporary:
        root = Path(temporary)
        image = root / "image.tif"
        mask = root / "mask.tif"
        orphan = root / "wmaps" / "frame195(1).tif"
        orphan.parent.mkdir()
        image.write_bytes(b"image")
        mask.write_bytes(b"mask")
        orphan.write_bytes(b"orphan-boundary-map")
        exclusion = ExternalSourceFileExclusion(
            path=orphan,
            relative_path="train/wmaps/frame195(1).tif",
            role="wmap",
            reason="fixture image-less companion",
        )
        sample = ExternalSample(
            "deepsea_phase",
            "train",
            image,
            mask,
            source_file_exclusions=(exclusion,),
        )
        rows, paths, fingerprint_rows, errors = _seal_external_source_file_exclusions(
            [sample]
        )
        assert not errors
        assert paths == {orphan.resolve()}
        assert len(rows) == 1 and len(fingerprint_rows) == 1
        assert rows[0]["relative_path"] == exclusion.relative_path
        assert rows[0]["sha256"] == hashlib.sha256(orphan.read_bytes()).hexdigest()
        assert str(rows[0]["sha256"]) in fingerprint_rows[0]


def check_split_grouping() -> None:
    low = Path("/dataset/wtF12BF_1_crop_1_im.tif")
    high = Path("/dataset/wtF12BF_20_crop_2_im.tif")
    assert source_split("yeaz_brightfield", low) == source_split(
        "yeaz_brightfield", high
    ), "YeaZ exposures/crops from one field crossed data splits"

    deepsea_early = Path("/dataset/train/images/A11_z003_c001.png")
    deepsea_late = Path("/dataset/train/images/A11_z016_c001.png")
    assert scientific_group("deepsea_phase", deepsea_early) == (
        "deepsea_phase/A11"
    )
    assert source_split("deepsea_phase", deepsea_early) == source_split(
        "deepsea_phase", deepsea_late
    ), "DeepSea variants from one annotated source crossed data splits"


def check_validation_role_disjointness() -> None:
    samples: list[ExternalSample] = []
    observed_roles: set[str] = set()
    for index in range(256):
        image_path = Path("/validation") / f"field_{index:03d}_img.tif"
        samples.append(
            ExternalSample(
                "validation_fixture",
                "val",
                image_path,
                image_path.with_name(f"field_{index:03d}_masks.tif"),
            )
        )
        observed_roles.add(validation_role("validation_fixture", image_path))
        if observed_roles == set(VALIDATION_ROLES) and len(samples) >= 12:
            break
    assert observed_roles == set(VALIDATION_ROLES)

    paths_by_role: dict[str, set[Path]] = {}
    for role in VALIDATION_ROLES:
        dataset = ExternalCellDataset(
            samples,
            split="val",
            image_size=64,
            train=False,
            transform=None,
            validation_role=role,
        )
        paths_by_role[role] = {sample.image_path for sample in dataset.samples}
        assert paths_by_role[role], f"Validation role {role} received no fixture samples"
    all_paths = {sample.image_path for sample in samples}
    assert set().union(*paths_by_role.values()) == all_paths
    for left_index, left in enumerate(VALIDATION_ROLES):
        for right in VALIDATION_ROLES[left_index + 1 :]:
            assert paths_by_role[left].isdisjoint(paths_by_role[right])

    crop_one = Path("/validation/acquisition_7_crop_1_img.tif")
    crop_two = Path("/validation/acquisition_7_crop_2_image.tif")
    assert validation_role("validation_fixture", crop_one) == validation_role(
        "validation_fixture", crop_two
    )
    moved_crop = Path(
        "/different/mount/validation_fixture/images/acquisition_7_crop_3_img.tif"
    )
    original_crop = Path(
        "/first/mount/validation_fixture/images/acquisition_7_crop_1_img.tif"
    )
    assert scientific_group("validation_fixture", moved_crop) == scientific_group(
        "validation_fixture", original_crop
    ), "Moving an unchanged dataset altered its scientific acquisition identity"
    assert validation_role("validation_fixture", moved_crop) == validation_role(
        "validation_fixture", original_crop
    )

    livecell_lineage = (
        "train/path/BT474_Phase_B3_1_00d12h00m_1.tif",
        "val/path/BT474_Phase_B3_1_00d16h00m_2.tif",
        "BT474_Phase_B3_1_02d16h00m_7.tif",
    )
    assert {
        scientific_group("livecell", path) for path in livecell_lineage
    } == {"livecell/BT474_Phase_B3_1"}
    assert len({livecell_role(path) for path in livecell_lineage}) == 1, (
        "LIVECell timepoints/crops from one acquisition crossed scientific roles"
    )


def check_normalization_contract() -> None:
    native = np.arange(8 * 9, dtype=np.uint8).reshape(8, 9)
    assert np.array_equal(normalize_uint8(native), native)
    color = np.stack([native, np.flipud(native), np.fliplr(native)], axis=-1)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    assert gray.dtype == np.uint8
    assert np.array_equal(normalize_uint8(gray), gray)

    high_bit_depth = np.linspace(100, 60_000, 64 * 64, dtype=np.uint16).reshape(64, 64)
    scaled = normalize_uint8(high_bit_depth)
    assert scaled.dtype == np.uint8 and scaled.shape == high_bit_depth.shape
    assert int(scaled.min()) == 0 and int(scaled.max()) == 255
    assert np.all(np.diff(scaled[32].astype(np.int16)) >= 0)


def check_internal_boundary_targets() -> None:
    isolated = np.zeros((48, 48), dtype=np.int32)
    isolated[10:38, 10:38] = 1
    isolated_targets, count = external_labels_to_targets(isolated)
    assert count == 1 and not bool(isolated_targets[..., 1].any())

    border_isolated = np.zeros((48, 48), dtype=np.int32)
    border_isolated[:28, :28] = 1
    border_targets, _ = external_labels_to_targets(border_isolated)
    assert not bool(border_targets[..., 1].any())

    for gap in (0, 1, 2):
        labels = np.zeros((48, 48), dtype=np.int32)
        labels[12:36, 4:16] = 1
        right_start = 16 + gap
        labels[12:36, right_start : right_start + 12] = 2
        targets, count = external_labels_to_targets(labels)
        boundary = targets[..., 1].astype(bool)
        assert count == 2 and bool(boundary.any()), f"No boundary for {gap}-pixel gap"
        assert not bool(boundary[:, :12].any()), "Left outer contour became a boundary"
        assert not bool(boundary[:, right_start + 4 :].any()), (
            "Right outer contour became a boundary"
        )


def check_foundation_models() -> None:
    required = {"cpsam_v2", "cpdino", "cpdino-vitb"}
    assert required <= set(cellpose_models.MODEL_NAMES)
    from cellpose import vit

    assert hasattr(vit, "dinov3_vitl16") and hasattr(vit, "dinov3_vitb16"), (
        "DINOv3 did not import; CPDINO models cannot be trained"
    )


def check_masked_loss() -> None:
    targets = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
    targets[:, 0, 2:5, 1:3] = 1
    targets[:, 1, 2:5, 1] = 1
    targets[:, 2, :, :4] = 1
    baseline = torch.zeros((1, 2, 8, 8), dtype=torch.float32)
    changed_only_outside_valid = baseline.clone()
    changed_only_outside_valid[:, :, :, 4:] = 50
    assert torch.allclose(
        combined_loss(baseline, targets, ACCURACY_CONFIG),
        combined_loss(changed_only_outside_valid, targets, ACCURACY_CONFIG),
    ), "ignored pixels changed the loss"
    assert channel_metrics(
        baseline[:, :1], targets[:, :1], targets[:, 2:3]
    ) == channel_metrics(
        changed_only_outside_valid[:, :1], targets[:, :1], targets[:, 2:3]
    ), "ignored pixels changed a metric"


def check_yeast_ceiling() -> None:
    samples = (
        [SimpleNamespace(dataset="yeast_microstructures") for _ in range(300)]
        + [SimpleNamespace(dataset="yeaz_phase") for _ in range(300)]
        + [SimpleNamespace(dataset="yeaz_brightfield") for _ in range(300)]
        + [SimpleNamespace(dataset="qpi_adherent") for _ in range(100)]
    )
    _, report = modality_balanced_sampler(1000, samples)
    assert report["expected_domain_probabilities"]["yeast"] <= 0.1000001


def check_object_confidence_filter() -> None:
    foreground = np.zeros((96, 96), dtype=np.float32)
    boundary = np.zeros_like(foreground)
    truth = np.zeros((96, 96), dtype=np.int32)
    truth[12:36, 12:36] = 1
    truth[54:82, 54:82] = 2
    foreground[12:36, 12:36] = 0.92
    foreground[54:82, 54:82] = 0.88
    # A low-confidence foreground island imitates an inter-cell/background artifact.
    foreground[15:42, 58:85] = 0.53
    for region in (truth == 1, truth == 2):
        perimeter = region & ~cv2.erode(region.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        boundary[perimeter] = 0.9

    baseline = instance_labels(foreground, boundary, DEFAULT_POSTPROCESS_CONFIG)
    filtered = instance_labels(
        foreground,
        boundary,
        PostprocessConfig(min_mean_foreground_probability=0.65),
    )
    assert int(baseline.max()) == 3
    assert int(filtered.max()) == 2
    baseline_metrics = segmentation_metrics(baseline, truth)
    filtered_metrics = segmentation_metrics(filtered, truth)
    assert (
        filtered_metrics["false_positive_area_fraction"]
        < baseline_metrics["false_positive_area_fraction"]
    )
    assert filtered_metrics["foreground_precision"] > baseline_metrics["foreground_precision"]
    search = tune_postprocessing_records(
        [(foreground, boundary, truth, "synthetic")],
        "synthetic_artifact",
    )
    # V4 evaluates 153 reconstruction settings, then 19 additional object-filter variants for
    # each of the four AUC-selected reconstruction families.
    assert search["candidate_count"] == 153 + 4 * 19
    assert len(search["boundary_cutoff_curves"]) == 9
    assert all(
        len(curve["boundary_cutoffs"]) == 17
        for curve in search["boundary_cutoff_curves"]
    )
    assert (
        search["selected"]["metrics"]["false_positive_area_fraction"]
        < search["baseline"]["metrics"]["false_positive_area_fraction"]
    )


def check_dataset_adapters() -> None:
    with tempfile.TemporaryDirectory(prefix="cellect-self-check-") as temporary:
        root = Path(temporary)
        dataset_root = root / "DIC-C2DH-HeLa"
        raw = dataset_root / "01"
        gold = dataset_root / "01_GT" / "SEG"
        raw.mkdir(parents=True)
        gold.mkdir(parents=True)
        image = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
        labels = np.zeros((64, 64), dtype=np.uint16)
        labels[20:40, 18:38] = 1
        assert cv2.imwrite(str(raw / "t000.tif"), image)
        assert cv2.imwrite(str(gold / "man_seg000.tif"), labels)

        samples = discover_ctc_gold(root, "ctc_dic_hela")
        assert len(samples) == 1 and samples[0].split == "train"
        dataset = ExternalCellDataset(
            samples,
            split="train",
            image_size=64,
            train=True,
            transform=training_transform(64, accuracy=True),
        )
        # Repeated calls exercise stochastic phone and geometric transforms on a two-channel mask.
        for _ in range(8):
            item = dataset[0]
            assert tuple(item["mask"].shape) == (3, 64, 64)
            assert 0 < item["mask"][2].sum().item() < 64 * 64
            assert bool((item["mask"][0] <= item["mask"][2]).all())

        counts = materialize_cellpose_pairs(samples, root / "cellpose")
        assert counts["omitted_ctc_sparse_gold"] == 1
        assert counts["final_test_excluded"] == 0
        assert sum(counts[role] for role in ("train", *VALIDATION_ROLES)) == 0

        curated_root = root / "curated"
        curated_files = (
            ("train", "field"),
            ("val", "validation_field"),
            ("test", "test_field"),
        )
        for split, stem in curated_files:
            directory = curated_root / split
            directory.mkdir(parents=True)
            assert cv2.imwrite(str(directory / f"{stem}.tif"), image)
            assert cv2.imwrite(str(directory / f"{stem}_masks.tif"), labels)
        discovered = discover_cellpose_pairs(
            curated_root, "cellpose_transmitted_light"
        )
        assert len(discovered) == 3
        assert {sample.split for sample in discovered} == {"train", "val", "test"}
        curated_counts = materialize_cellpose_pairs(
            discovered,
            root / "cellpose_curated",
        )
        assert curated_counts["train"] == 1
        assert curated_counts["final_test_excluded"] == 1
        assert sum(curated_counts[role] for role in VALIDATION_ROLES) == 1

        deepsea = root / "deepsea"
        # Reproduce the actual multi-ZIP Google Drive layout, including both wrapper levels.
        tracking_branch = deepsea / "track"
        segment_branch = deepsea / "segment"
        final_branch = deepsea / "final"
        tracking_root = tracking_branch / "tracking_dataset"
        segment_root = segment_branch / "segmentation_dataset"
        final_root = final_branch / "final_dataset"
        for directory in (tracking_root, segment_root, final_root):
            directory.mkdir(parents=True)

        deepsea_binary = np.zeros((64, 64), dtype=np.uint8)
        deepsea_binary[12:52, 8:31] = 255
        deepsea_binary[12:52, 33:56] = 255
        deepsea_touching = np.zeros_like(deepsea_binary)
        # Official DeepSea maps mostly occupy the narrow background gap between cells.
        deepsea_touching[16:48, 31:33] = 255

        def write_triplet(
            split_root: Path,
            stem: str,
            extensions: tuple[str, str, str] = (".tif", ".tif", ".tif"),
        ) -> tuple[Path, Path, Path]:
            image_directory = split_root / "images"
            mask_directory = split_root / "masks"
            wmap_directory = split_root / "wmaps"
            for directory in (image_directory, mask_directory, wmap_directory):
                directory.mkdir(parents=True, exist_ok=True)
            image_path = image_directory / f"{stem}{extensions[0]}"
            mask_path = mask_directory / f"{stem}{extensions[1]}"
            wmap_path = wmap_directory / f"{stem}{extensions[2]}"
            assert cv2.imwrite(str(image_path), image)
            assert cv2.imwrite(str(mask_path), deepsea_binary)
            assert cv2.imwrite(str(wmap_path), deepsea_touching)
            return image_path, mask_path, wmap_path

        train_two = write_triplet(segment_root / "train", "frame002")
        train_ten = write_triplet(segment_root / "train", "frame010")
        train_mixed_extensions = write_triplet(
            segment_root / "train",
            "frame003",
            (".png", ".tif", ".bmp"),
        )
        test_one = write_triplet(segment_root / "test", "frame001")
        extra_wmap = segment_root / "train" / "wmaps" / "frame195(1).tif"
        assert cv2.imwrite(str(extra_wmap), deepsea_touching)

        # Same-name and malformed decoys must never be ingested from tracking/final.
        for decoy_root in (tracking_root, final_root):
            decoy_images = decoy_root / "train" / "images"
            decoy_images.mkdir(parents=True)
            assert cv2.imwrite(str(decoy_images / "frame001.tif"), image)

        assert set(deepsea_subfolders(deepsea)) == {"track", "segment", "final"}
        assert deepsea_subfolders(deepsea)["segment"] == segment_branch
        # Explicit roots must not fall through to a real DeepSea directory beside the bundle.
        discovered_root = find_local_deepsea_root((deepsea,))
        assert discovered_root is not None
        assert discovered_root[0] == deepsea.resolve()
        discovered_parent = find_local_deepsea_root((root,))
        assert discovered_parent is not None
        assert discovered_parent[0] == deepsea.resolve()
        deepsea_samples = discover_deepsea(deepsea)
        repeated_samples = discover_deepsea(deepsea)
        assert len(deepsea_samples) == 4
        deepsea_exclusions = [
            exclusion
            for sample in deepsea_samples
            for exclusion in sample.source_file_exclusions
        ]
        assert len(deepsea_exclusions) == 1
        assert deepsea_exclusions[0].path == extra_wmap.resolve()
        assert deepsea_exclusions[0].role == "wmap"
        assert deepsea_exclusions[0].relative_path.endswith(
            "train/wmaps/frame195(1).tif"
        )
        assert [sample.image_path for sample in repeated_samples] == [
            sample.image_path for sample in deepsea_samples
        ]
        # Keep compatibility with archives that omit the outer track/segment/final wrappers.
        direct_deepsea = root / "deepsea_direct_collection"
        direct_tracking = direct_deepsea / "tracking_dataset"
        direct_segment = direct_deepsea / "segmentation_dataset"
        direct_final = direct_deepsea / "final_dataset"
        for directory in (direct_tracking, direct_segment, direct_final):
            directory.mkdir(parents=True)
        direct_triplet = write_triplet(direct_segment / "train", "direct001")
        direct_samples = discover_deepsea(direct_deepsea)
        assert [sample.image_path for sample in direct_samples] == [direct_triplet[0]]

        partial_deepsea = root / "deepsea_partial_collection"
        (partial_deepsea / "segmentation_dataset").mkdir(parents=True)
        expect_runtime_error(
            lambda: discover_deepsea(partial_deepsea),
            ("Incomplete DeepSea collection", "final", "track"),
        )
        with patch.dict(os.environ, {"CELLECT_DEEPSEA_ROOT": str(deepsea)}):
            configured = find_local_deepsea_root()
            assert configured is not None and configured[0] == deepsea.resolve()
        with patch.dict(os.environ, {"CELLECT_DEEPSEA_ROOT": str(partial_deepsea)}):
            expect_runtime_error(
                find_local_deepsea_root,
                ("CELLECT_DEEPSEA_ROOT", "found roles", "segment"),
            )
        assert all(sample.image_path.is_relative_to(segment_root) for sample in deepsea_samples)
        assert not any(
            sample.image_path.is_relative_to(tracking_root)
            for sample in deepsea_samples
        )
        assert not any(
            sample.image_path.is_relative_to(final_root)
            for sample in deepsea_samples
        )
        development_names = [
            sample.image_path.name
            for sample in deepsea_samples
            if sample.split in {"train", "val"}
        ]
        assert development_names == ["frame002.tif", "frame003.png", "frame010.tif"]
        assert [sample.image_path for sample in deepsea_samples if sample.split == "test"] == [
            test_one[0]
        ]
        paired_paths = {
            (sample.image_path, sample.instance_path, sample.boundary_path)
            for sample in deepsea_samples
        }
        assert paired_paths == {train_two, train_ten, train_mixed_extensions, test_one}

        recovered, explicit_boundary = deepsea_instance_labels(
            deepsea_binary,
            deepsea_touching,
        )
        assert int(recovered.max()) == 2, (
            "DeepSea touching-edge map did not split the binary mask into two instances"
        )
        assert explicit_boundary is not None and bool(explicit_boundary.any())
        assert bool((explicit_boundary & ~(deepsea_binary > 0)).any()), (
            "DeepSea gap-centered wmap was incorrectly clipped to foreground"
        )
        recovered_targets, recovered_count = external_labels_to_targets(
            recovered,
            explicit_boundary=explicit_boundary,
        )
        recovered_boundary = recovered_targets[..., 1].astype(bool)
        assert recovered_count == 2 and bool(recovered_boundary.any())
        assert not bool(recovered_boundary[:, :24].any())
        assert not bool(recovered_boundary[:, 40:].any())

        missing_wmap_root = root / "deepsea_missing_wmap"
        for role in ("images", "masks"):
            (missing_wmap_root / "train" / role).mkdir(parents=True)
        assert cv2.imwrite(
            str(missing_wmap_root / "train" / "images" / "frame.tif"), image
        )
        assert cv2.imwrite(
            str(missing_wmap_root / "train" / "masks" / "frame.tif"),
            deepsea_binary,
        )
        expect_runtime_error(
            lambda: discover_deepsea(missing_wmap_root),
            "missing=['wmaps']",
        )

        orphan_root = root / "deepsea_orphans" / "train"
        write_triplet(orphan_root, "complete")
        assert cv2.imwrite(
            str(orphan_root / "masks" / "mask_only.tif"), deepsea_binary
        )
        assert cv2.imwrite(
            str(orphan_root / "wmaps" / "wmap_only.tif"), deepsea_touching
        )
        companion_samples = discover_deepsea(orphan_root.parent)
        assert len(companion_samples) == 1
        companion_exclusions = companion_samples[0].source_file_exclusions
        assert len(companion_exclusions) == 2
        assert {value.role for value in companion_exclusions} == {"mask", "wmap"}
        assert cv2.imwrite(str(orphan_root / "images" / "image_only.tif"), image)
        expect_runtime_error(
            lambda: discover_deepsea(orphan_root.parent),
            ("missing masks=['image_only']", "missing wmaps=['image_only']"),
        )

        ambiguous_directory_root = root / "deepsea_ambiguous_directory" / "train"
        write_triplet(ambiguous_directory_root, "frame")
        duplicate_wmap_directory = ambiguous_directory_root / "unetwmaps"
        duplicate_wmap_directory.mkdir()
        assert cv2.imwrite(
            str(duplicate_wmap_directory / "frame.tif"), deepsea_touching
        )
        expect_runtime_error(
            lambda: discover_deepsea(ambiguous_directory_root.parent),
            "ambiguous={'wmaps':",
        )

        ambiguous_stem_root = root / "deepsea_ambiguous_stem" / "train"
        write_triplet(ambiguous_stem_root, "frame")
        assert cv2.imwrite(str(ambiguous_stem_root / "images" / "frame.png"), image)
        expect_runtime_error(
            lambda: discover_deepsea(ambiguous_stem_root.parent),
            "Ambiguous DeepSea image stem",
        )


def check_duplicate_and_leakage_exclusions() -> None:
    """A duplicated download and a shared acquisition must leave the supervised sample list."""
    image = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    binary = np.zeros((64, 64), dtype=np.uint8)
    binary[12:52, 8:31] = 255
    binary[12:52, 33:56] = 255
    touching = np.zeros_like(binary)
    touching[16:48, 31:33] = 255

    with tempfile.TemporaryDirectory(prefix="cellect-duplicate-exclusions-") as temporary:
        root = Path(temporary)
        segment_root = root / "deepsea"

        def write_triplet(split: str, stem: str) -> Path:
            image_path = segment_root / split / "images" / f"{stem}.png"
            mask_path = segment_root / split / "masks" / f"{stem}.png"
            wmap_path = segment_root / split / "wmaps" / f"{stem}.png"
            for path in (image_path, mask_path, wmap_path):
                path.parent.mkdir(parents=True, exist_ok=True)
            assert cv2.imwrite(str(image_path), image)
            assert cv2.imwrite(str(mask_path), binary)
            assert cv2.imwrite(str(wmap_path), touching)
            return image_path

        kept = write_triplet("train", "B04_z001_c001")
        write_triplet("train", "A11_z002_c001")
        write_triplet("test", "A11_z007_c001")
        for role, source in (
            ("images", kept),
            ("masks", segment_root / "train" / "masks" / "B04_z001_c001.png"),
            ("wmaps", segment_root / "train" / "wmaps" / "B04_z001_c001.png"),
        ):
            copy = segment_root / "train" / role / "B04_z001_c001(1).png"
            copy.write_bytes(source.read_bytes())

        samples = discover_deepsea(root / "deepsea")
        selected = {sample.image_path.name for sample in samples}
        assert selected == {"B04_z001_c001.png", "A11_z007_c001.png"}, selected
        exclusions = [
            exclusion
            for sample in samples
            for exclusion in sample.source_file_exclusions
        ]
        roles = sorted(exclusion.role for exclusion in exclusions)
        assert roles == [
            "duplicate-image",
            "duplicate-mask",
            "duplicate-wmap",
            "image",
            "mask",
            "wmap",
        ], roles
        withheld = next(
            exclusion for exclusion in exclusions if exclusion.role == "image"
        )
        assert withheld.relative_path.endswith("train/images/A11_z002_c001.png")
        assert "official DeepSea test split" in withheld.reason

        # A published archive that repeats one frame under two numbers must contribute it once.
        qpi_root = root / "qpi_adherent" / "PC3"
        qpi_root.mkdir(parents=True)
        assert cv2.imwrite(str(qpi_root / "00127_PC3_img.tif"), image)
        assert cv2.imwrite(str(qpi_root / "00127_PC3_mask.tif"), binary)
        (qpi_root / "00128_PC3_img.tif").write_bytes(
            (qpi_root / "00127_PC3_img.tif").read_bytes()
        )
        (qpi_root / "00128_PC3_mask.tif").write_bytes(
            (qpi_root / "00127_PC3_mask.tif").read_bytes()
        )
        assert cv2.imwrite(str(qpi_root / "00200_PC3_img.tif"), image[::-1])
        assert cv2.imwrite(str(qpi_root / "00200_PC3_mask.tif"), binary)
        qpi_samples = discover_suffix_pairs(root / "qpi_adherent", "qpi_adherent")
        assert [sample.image_path.name for sample in qpi_samples] == [
            "00127_PC3_img.tif",
            "00200_PC3_img.tif",
        ]
        qpi_exclusions = [
            exclusion
            for sample in qpi_samples
            for exclusion in sample.source_file_exclusions
        ]
        assert {exclusion.role for exclusion in qpi_exclusions} == {
            "duplicate-image",
            "duplicate-mask",
        }


def check_v4_contracts() -> None:
    """Run bounded orchestration contracts once, without network or external data.

    The shape orchestration test already covers shape targets/models, teacher cache handoff,
    three-phase optimization, frozen five-channel boundary search, exports, and resume.  The
    tracking orchestration test already covers both tracking model tiers, all seven optimizer
    phases, calibration/ensemble selection, and TorchScript parity.  Only the strict tracking
    adapters and detailed identity metrics are tested separately because those are not duplicated
    by the orchestration fixture.
    """
    from tracking_data import synthetic_self_test as tracking_data_self_test
    from tracking_metrics import synthetic_self_test as tracking_metrics_self_test
    from train_cellpose_sam import run_resume_contract_self_test
    from v4_shape_training import run_contract_self_test as shape_self_test
    from v4_tracking_training import synthetic_self_test as tracking_training_self_test

    shape = shape_self_test()
    assert shape["status"] == "PASS"
    assert shape["boundary_search"] == "passed"
    assert shape["resume"] == "passed"

    tracking_data = tracking_data_self_test()
    assert tracking_data["status"] == "PASS"
    assert tracking_data["test_label_seal"] == "PASS"
    assert tracking_data["ctmc_v1_train_only_adapter"] == "PASS"
    assert tracking_data["alfi_duplicate_event_quarantine"] == "PASS"

    tracking_metrics = tracking_metrics_self_test()
    assert tracking_metrics["status"] == "PASS"
    assert tracking_metrics["official_metric_claimed"] is False

    cellpose_resume = run_resume_contract_self_test()
    assert cellpose_resume["status"] == "PASS"
    assert cellpose_resume["model_optimizer_rng_exact"] is True

    tracking_training = tracking_training_self_test()
    assert tracking_training["status"] == "PASS"
    assert tracking_training["downloads_performed"] is False
    assert tracking_training["official_test_labels_parsed"] is False
    assert tracking_training["hungarian_adversarial_assignment"] == "PASS"
    assert tracking_training["deployment_detector_perturbations"] == "PASS"
    assert tracking_training["gap_recovery_and_censoring"] == "PASS"
    assert tracking_training["candidate_specific_threshold_exports"] == "PASS"
    assert all(
        updates >= 1
        for phases in tracking_training["real_optimizer_updates"].values()
        for updates in phases.values()
    )


def check_coreml_artifact_bridge() -> None:
    """Run portable hash/path/fixture tests without importing Core ML or PyTorch there."""

    from coreml_bridge_contract_test import CoreMLBridgeContractTests

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CoreMLBridgeContractTests)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    assert result.wasSuccessful(), "Core ML portable artifact contract checks failed"


def main() -> None:
    check_resumable_download_retry()
    check_external_exclusion_provenance()
    check_foundation_models()
    check_split_grouping()
    check_validation_role_disjointness()
    check_normalization_contract()
    check_internal_boundary_targets()
    check_masked_loss()
    check_yeast_ceiling()
    check_object_confidence_filter()
    check_dataset_adapters()
    check_duplicate_and_leakage_exclusions()
    tiled_stitch_self_check()
    check_v4_contracts()
    check_coreml_artifact_bridge()
    print("Cellect workstation bundle self-check passed.")


if __name__ == "__main__":
    main()
