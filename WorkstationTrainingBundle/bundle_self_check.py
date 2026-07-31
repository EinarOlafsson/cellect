#!/usr/bin/env python3
"""Fast fixture checks for dataset adapters, sparse supervision, and sampling policy."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from cellpose import models as cellpose_models

from accuracy_data import (
    ExternalCellDataset,
    ExternalSample,
    discover_cellpose_pairs,
    discover_ctc_gold,
    discover_deepsea,
    deepsea_instance_labels,
    deepsea_subfolders,
    find_local_deepsea_root,
    labels_to_targets as external_labels_to_targets,
    materialize_cellpose_pairs,
    normalize_uint8,
    source_split,
)
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
    # V3 evaluates 153 reconstruction settings, then 19 additional object-filter variants for
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
        tracking_root = deepsea / "tracking_dataset"
        segment_root = deepsea / "segmentation_dataset"
        final_root = deepsea / "final_dataset"
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

        # Same-name and malformed decoys must never be ingested from tracking/final.
        for decoy_root in (tracking_root, final_root):
            decoy_images = decoy_root / "train" / "images"
            decoy_images.mkdir(parents=True)
            assert cv2.imwrite(str(decoy_images / "frame001.tif"), image)

        assert set(deepsea_subfolders(deepsea)) == {"track", "segment", "final"}
        assert deepsea_subfolders(deepsea)["segment"] == segment_root
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
        assert [sample.image_path for sample in repeated_samples] == [
            sample.image_path for sample in deepsea_samples
        ]
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
        assert cv2.imwrite(str(orphan_root / "images" / "image_only.tif"), image)
        assert cv2.imwrite(
            str(orphan_root / "masks" / "mask_only.tif"), deepsea_binary
        )
        assert cv2.imwrite(
            str(orphan_root / "wmaps" / "wmap_only.tif"), deepsea_touching
        )
        expect_runtime_error(
            lambda: discover_deepsea(orphan_root.parent),
            ("orphan images=", "orphan masks=", "orphan wmaps="),
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


def main() -> None:
    check_foundation_models()
    check_split_grouping()
    check_validation_role_disjointness()
    check_normalization_contract()
    check_internal_boundary_targets()
    check_masked_loss()
    check_yeast_ceiling()
    check_object_confidence_filter()
    check_dataset_adapters()
    tiled_stitch_self_check()
    print("Cellect workstation bundle self-check passed.")


if __name__ == "__main__":
    main()
