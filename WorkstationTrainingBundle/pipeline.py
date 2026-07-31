#!/usr/bin/env python3
"""One-command LIVECell training, evaluation, export, and return packaging."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, replace
from itertools import combinations
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from cellpose import metrics as cellpose_metrics
from pycocotools.coco import COCO
from scipy import ndimage
from skimage.segmentation import watershed
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from deployment_runtime import (
    POSTPROCESS_VERSION,
    TILED_INFERENCE_VERSION,
    predict_deployment,
    reconstruct_iphone_instances,
)
from mask_targets import BOUNDARY_TARGET_VERSION, internal_contact_boundary
from scientific_splits import (
    BUNDLE_VERSION,
    livecell_role,
    validation_role,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
RUNS = OUTPUT / "runs"
DOWNLOADS = DATA / "downloads"
IMAGES_ZIP = DOWNLOADS / "images.zip"
IMAGES_ROOT = DATA / "images"
ANNOTATIONS = DATA / "annotations"

BASE_URL = "https://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021"
URLS = {
    "images": f"{BASE_URL}/images.zip",
    "train": f"{BASE_URL}/annotations/LIVECell/livecell_coco_train.json",
    "val": f"{BASE_URL}/annotations/LIVECell/livecell_coco_val.json",
    "test": f"{BASE_URL}/annotations/LIVECell/livecell_coco_test.json",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    encoder: str
    architecture: str = "Unet"
    image_size: int = 512
    batch_size: int = 8


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    early_stopping_patience: int
    minimum_improvement: float
    decoder_learning_rate: float
    encoder_learning_rate: float
    weight_decay: float
    l1_lambda: float
    gradient_clip_norm: float
    accumulation_steps: int
    scheduler_patience: int
    scheduler_factor: float
    minimum_learning_rate: float
    ema_decay: float
    boundary_loss_weight: float
    boundary_positive_weight: float
    focal_gamma: float
    tversky_loss_weight: float
    tversky_alpha: float
    tversky_beta: float
    validation_instance_samples: int
    use_accuracy_augmentations: bool


@dataclass(frozen=True)
class PostprocessConfig:
    """Parameters that turn foreground/boundary probabilities into cell instances."""

    foreground_threshold: float = 0.50
    boundary_threshold: float = 0.45
    min_area_fraction: float = 20.0 / (512.0 * 512.0)
    min_mean_foreground_probability: float = 0.0
    core_probability_threshold: float = 0.70
    min_core_fraction: float = 0.0
    min_boundary_support: float = 0.0


DEFAULT_POSTPROCESS_CONFIG = PostprocessConfig()
BOUNDARY_CUTOFF_SWEEP = tuple(round(0.10 + 0.05 * index, 2) for index in range(17))


MODEL_SPECS = [
    ModelSpec(
        "mobilenetv3_small_unet", "timm-mobilenetv3_small_100", "Unet", 384, 24
    ),
    ModelSpec(
        "mobilenetv3_large_deeplab", "timm-mobilenetv3_large_100", "DeepLabV3Plus", 512, 12
    ),
    ModelSpec("efficientnet_b0_unet", "efficientnet-b0", "Unet", 512, 16),
    ModelSpec("resnet18_unet", "resnet18", "Unet", 512, 16),
    ModelSpec("efficientnet_b3_unetpp", "efficientnet-b3", "UnetPlusPlus", 512, 8),
    ModelSpec("resnet50_deeplab", "resnet50", "DeepLabV3Plus", 512, 8),
    ModelSpec("resnet101_unetpp", "resnet101", "UnetPlusPlus", 512, 4),
    ModelSpec("segformer_b2", "mit_b2", "Segformer", 512, 8),
    ModelSpec("segformer_b5", "mit_b5", "Segformer", 512, 2),
]

ACCURACY_MODEL_SPECS = [
    ModelSpec("mobilenetv3_small_unet_accuracy_v2", "timm-mobilenetv3_small_100", "Unet", 512, 16),
    ModelSpec("mobilenetv3_large_deeplab_accuracy_v2", "timm-mobilenetv3_large_100", "DeepLabV3Plus", 640, 8),
    ModelSpec("efficientnet_b0_unet_accuracy_v2", "efficientnet-b0", "Unet", 640, 8),
    ModelSpec("resnet18_unet_accuracy_v2", "resnet18", "Unet", 640, 8),
    ModelSpec("efficientnet_b3_unetpp_accuracy_v2", "efficientnet-b3", "UnetPlusPlus", 768, 4),
    ModelSpec("resnet50_deeplab_accuracy_v2", "resnet50", "DeepLabV3Plus", 768, 3),
    ModelSpec("resnet101_unetpp_accuracy_v2", "resnet101", "UnetPlusPlus", 768, 2),
    ModelSpec("segformer_b2_accuracy_v2", "mit_b2", "Segformer", 768, 3),
    ModelSpec("segformer_b5_accuracy_v2", "mit_b5", "Segformer", 768, 1),
]

ALL_MODEL_SPECS = MODEL_SPECS + ACCURACY_MODEL_SPECS
FOUNDATION_MODELS = ("cpsam_v2", "cpdino", "cpdino-vitb")

STANDARD_CONFIG = TrainingConfig(
    epochs=35,
    early_stopping_patience=6,
    minimum_improvement=0.0,
    decoder_learning_rate=3e-4,
    encoder_learning_rate=3e-4,
    weight_decay=1e-4,
    l1_lambda=0.0,
    gradient_clip_norm=0.0,
    accumulation_steps=1,
    scheduler_patience=2,
    scheduler_factor=0.5,
    minimum_learning_rate=0.0,
    ema_decay=0.0,
    boundary_loss_weight=0.5,
    boundary_positive_weight=1.0,
    focal_gamma=0.0,
    tversky_loss_weight=0.0,
    tversky_alpha=0.5,
    tversky_beta=0.5,
    validation_instance_samples=64,
    use_accuracy_augmentations=False,
)

ACCURACY_CONFIG = TrainingConfig(
    epochs=300,
    early_stopping_patience=30,
    minimum_improvement=1e-4,
    decoder_learning_rate=1e-4,
    encoder_learning_rate=3e-5,
    weight_decay=1e-4,
    l1_lambda=0.0,
    gradient_clip_norm=1.0,
    accumulation_steps=8,
    scheduler_patience=8,
    scheduler_factor=0.5,
    minimum_learning_rate=1e-7,
    ema_decay=0.999,
    boundary_loss_weight=1.0,
    boundary_positive_weight=5.0,
    focal_gamma=2.0,
    tversky_loss_weight=0.35,
    tversky_alpha=0.3,
    tversky_beta=0.7,
    validation_instance_samples=24,
    use_accuracy_augmentations=True,
)


def seed_everything(seed: int = 1337) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 1024:
        print(f"Reusing {destination}")
        return
    partial = destination.with_suffix(destination.suffix + ".partial")
    existing = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url)
    if existing:
        request.add_header("Range", f"bytes={existing}-")
    print(f"Downloading {url}")
    with urllib.request.urlopen(request) as response:
        resumed = existing > 0 and getattr(response, "status", None) == 206
        if existing and not resumed:
            existing = 0
        mode = "ab" if resumed else "wb"
        total = response.headers.get("Content-Length")
        remaining = int(total) if total else None
        with partial.open(mode) as target:
            with tqdm(
                total=(existing + remaining) if remaining else None,
                initial=existing,
                unit="B",
                unit_scale=True,
                desc=destination.name,
            ) as progress:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
                    progress.update(len(chunk))
    partial.replace(destination)


def prepare_data() -> None:
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    ANNOTATIONS.mkdir(parents=True, exist_ok=True)
    download(URLS["images"], IMAGES_ZIP)
    for split in ("train", "val", "test"):
        download(URLS[split], ANNOTATIONS / f"livecell_coco_{split}.json")
    marker = IMAGES_ROOT / ".extracted"
    if not marker.exists():
        print("Extracting LIVECell images")
        IMAGES_ROOT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(IMAGES_ZIP) as archive:
            archive.extractall(IMAGES_ROOT)
        marker.touch()


def materialize_livecell_for_cellpose(
    destination: Path,
    limit_per_role: int | None = None,
) -> dict[str, int]:
    counts = {
        "train": 0,
        "checkpoint": 0,
        "calibration": 0,
        "ensemble_selection": 0,
    }
    candidates: list[tuple[str, str, COCO, int, dict[str, object]]] = []
    for split in ("train", "val"):
        coco = COCO(str(ANNOTATIONS / f"livecell_coco_{split}.json"))
        for image_id in sorted(coco.getImgIds()):
            metadata = coco.loadImgs([image_id])[0]
            role = livecell_role(str(metadata["file_name"]))
            if role not in counts:
                raise RuntimeError(f"Unexpected LIVECell acquisition role: {role}")
            candidates.append((role, split, coco, image_id, metadata))
    # Hash ordering avoids letting the upstream file boundary consume a best-smoke role cap before
    # the other COCO source is considered, while remaining deterministic across machines.
    candidates.sort(
        key=lambda item: (
            item[0],
            hashlib.sha256(
                f"{item[1]}:{item[4]['file_name']}".encode()
            ).hexdigest(),
        )
    )
    for output_split, split, coco, image_id, metadata in tqdm(
        candidates,
        desc="materialize combined LIVECell train+val for Cellpose-SAM",
    ):
        if limit_per_role is not None and counts[output_split] >= limit_per_role:
            continue
        output_directory = destination / output_split
        output_directory.mkdir(parents=True, exist_ok=True)
        source_image = locate_image(str(metadata["file_name"]))
        stem = f"livecell_{split}_{image_id}"
        image_destination = output_directory / f"{stem}_img{source_image.suffix}"
        mask_destination = output_directory / f"{stem}_masks.tif"
        if not image_destination.exists():
            try:
                os.link(source_image, image_destination)
            except OSError:
                shutil.copy2(source_image, image_destination)
        if not mask_destination.exists():
            labels = np.zeros(
                (int(metadata["height"]), int(metadata["width"])),
                dtype=np.uint16,
            )
            annotation_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=None)
            for label, annotation in enumerate(
                coco.loadAnns(annotation_ids),
                start=1,
            ):
                labels[coco.annToMask(annotation).astype(bool)] = label
            if not cv2.imwrite(str(mask_destination), labels):
                raise RuntimeError(f"Could not write {mask_destination}")
        counts[output_split] += 1
    return counts


def locate_image(file_name: str) -> Path:
    direct_candidates = [
        IMAGES_ROOT / file_name,
        IMAGES_ROOT / "images" / file_name,
        IMAGES_ROOT / "livecell_train_val_images" / file_name,
        IMAGES_ROOT / "livecell_test_images" / file_name,
    ]
    for candidate in direct_candidates:
        if candidate.exists():
            return candidate
    matches = list(IMAGES_ROOT.rglob(Path(file_name).name))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Could not find LIVECell image {file_name}")
    for match in matches:
        if str(match).endswith(file_name):
            return match
    return matches[0]


def phone_sensor_artifacts(image: np.ndarray, **_: object) -> np.ndarray:
    """Approximate phone-through-eyepiece illumination, tone mapping, and shot noise."""
    source = image.astype(np.float32) / 255.0
    height, width = source.shape[:2]
    yy, xx = np.mgrid[-1:1:complex(height), -1:1:complex(width)]
    center_x = random.uniform(-0.25, 0.25)
    center_y = random.uniform(-0.25, 0.25)
    radius = (xx - center_x) ** 2 + (yy - center_y) ** 2
    vignette = np.clip(1.0 - random.uniform(0.0, 0.55) * radius, 0.35, 1.2)
    direction = random.uniform(0, 2 * np.pi)
    gradient = 1.0 + random.uniform(-0.3, 0.3) * (
        np.cos(direction) * xx + np.sin(direction) * yy
    )
    source *= (vignette * gradient)[..., None]
    exposure = 2.0 ** random.uniform(-0.75, 0.75)
    source = np.clip(source * exposure, 0, 1)
    gamma = random.uniform(0.7, 1.5)
    source = source ** gamma
    if random.random() < 0.6:
        photons = random.uniform(30, 220)
        source = np.random.poisson(source * photons).astype(np.float32) / photons
    if random.random() < 0.4:
        source += np.random.normal(0, random.uniform(0.002, 0.025), source.shape)
    return np.clip(source * 255.0, 0, 255).astype(np.uint8)


def training_transform(image_size: int, accuracy: bool) -> A.Compose:
    transforms: list[object] = [
        A.PadIfNeeded(
            min_height=image_size,
            min_width=image_size,
            border_mode=cv2.BORDER_REFLECT_101,
        ),
        A.RandomCrop(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ]
    if accuracy:
        transforms.extend(
            [
                A.Affine(
                    scale=(0.8, 1.25),
                    translate_percent=(-0.08, 0.08),
                    rotate=(-25, 25),
                    shear=(-8, 8),
                    border_mode=cv2.BORDER_REFLECT_101,
                    p=0.65,
                ),
                A.OneOf(
                    [
                        A.RandomBrightnessContrast(0.25, 0.25),
                        A.RandomGamma(gamma_limit=(70, 140)),
                        A.CLAHE(clip_limit=(1.0, 3.0)),
                    ],
                    p=0.75,
                ),
                A.OneOf(
                    [
                        A.GaussNoise(std_range=(0.005, 0.06)),
                        A.GaussianBlur(blur_limit=(3, 7)),
                        A.MotionBlur(blur_limit=(3, 7)),
                    ],
                    p=0.35,
                ),
                A.Compose(
                    [
                        A.Lambda(image=phone_sensor_artifacts, p=0.8),
                        A.OneOf(
                            [
                                A.Downscale(
                                    scale_range=(0.35, 0.9),
                                    interpolation_pair={
                                        "downscale": cv2.INTER_AREA,
                                        "upscale": cv2.INTER_LINEAR,
                                    },
                                ),
                                A.ImageCompression(quality_range=(40, 95)),
                                A.Sharpen(alpha=(0.05, 0.35), lightness=(0.8, 1.2)),
                            ],
                            p=0.75,
                        ),
                    ],
                    p=0.65,
                ),
                A.OneOf(
                    [
                        A.Perspective(scale=(0.01, 0.05)),
                        A.OpticalDistortion(
                            distort_limit=(-0.08, 0.08),
                            keypoint_remapping_method="mask",
                        ),
                        A.ElasticTransform(
                            alpha=8,
                            sigma=5,
                            interpolation=cv2.INTER_LINEAR,
                            mask_interpolation=cv2.INTER_NEAREST,
                        ),
                    ],
                    p=0.2,
                ),
                A.CoarseDropout(
                    num_holes_range=(1, 5),
                    hole_height_range=(0.01, 0.06),
                    hole_width_range=(0.01, 0.06),
                    fill="random_uniform",
                    fill_mask=None,
                    p=0.12,
                ),
            ]
        )
    else:
        transforms.extend(
            [
                A.RandomBrightnessContrast(0.15, 0.15, p=0.5),
                A.GaussNoise(std_range=(0.01, 0.04), p=0.2),
            ]
        )
    return A.Compose(transforms)


def robust_uint8(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    low, high = np.percentile(image, (0.5, 99.5))
    if high <= low:
        return np.zeros(image.shape, dtype=np.uint8)
    return np.clip((image - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)


def labels_to_targets(labels: np.ndarray) -> tuple[np.ndarray, int]:
    labels = labels.astype(np.int32)
    foreground = labels > 0
    boundary = internal_contact_boundary(labels)
    count = int(np.unique(labels[foreground]).size)
    valid = np.ones(labels.shape, dtype=np.uint8)
    return np.stack([foreground, boundary, valid], axis=-1).astype(np.uint8), count


class LiveCellDataset(Dataset):
    def __init__(
        self,
        annotation_path: Path | tuple[Path, ...] | list[Path],
        image_size: int,
        train: bool,
        limit: int | None = None,
        accuracy_augmentations: bool = False,
        role_name: str | None = None,
    ) -> None:
        annotation_paths = (
            [annotation_path]
            if isinstance(annotation_path, Path)
            else list(annotation_path)
        )
        self.cocos: dict[str, COCO] = {}
        records: list[tuple[str, int, str]] = []
        for path in annotation_paths:
            source_key = path.stem.removeprefix("livecell_coco_")
            if source_key in self.cocos:
                raise ValueError(f"Duplicate LIVECell COCO source key: {source_key}")
            coco = COCO(str(path))
            self.cocos[source_key] = coco
            for image_id in sorted(coco.getImgIds()):
                metadata = coco.loadImgs([image_id])[0]
                file_name = str(metadata["file_name"])
                if role_name is not None and livecell_role(file_name) != role_name:
                    continue
                records.append((source_key, image_id, file_name))
        records.sort(
            key=lambda record: hashlib.sha256(
                f"{record[0]}:{record[2]}".encode()
            ).hexdigest()
        )
        if limit:
            records = records[:limit]
        self.records = records
        self.image_size = image_size
        self.train = train
        self.role_name = role_name
        self.transform = (
            training_transform(image_size, accuracy_augmentations)
            if train
            else None
        )

    def __len__(self) -> int:
        return len(self.records)

    def _mask(
        self,
        coco: COCO,
        image_id: int,
        height: int,
        width: int,
    ) -> np.ndarray:
        instances = np.zeros((height, width), dtype=np.int32)
        annotation_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=None)
        annotations = coco.loadAnns(annotation_ids)
        for label, annotation in enumerate(annotations, start=1):
            instance = coco.annToMask(annotation).astype(np.uint8)
            instances[instance.astype(bool)] = label
        return instances

    def image_paths(self, limit: int | None = None) -> list[Path]:
        records = self.records if limit is None else self.records[:limit]
        return [locate_image(file_name) for _, _, file_name in records]

    def __getitem__(self, index: int) -> dict[str, object]:
        source_key, image_id, file_name = self.records[index]
        coco = self.cocos[source_key]
        metadata = coco.loadImgs([image_id])[0]
        path = locate_image(metadata["file_name"])
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise RuntimeError(f"Could not decode {path}")
        if self.train and self.transform is not None and random.random() < 0.35:
            gray = robust_uint8(gray)
        instances = self._mask(coco, image_id, gray.shape[0], gray.shape[1])
        image = np.repeat(gray[..., None], 3, axis=2)
        if self.transform:
            transformed = self.transform(image=image, mask=instances)
            image, instances = transformed["image"], transformed["mask"]
        else:
            image = cv2.resize(
                image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA
            )
            instances = cv2.resize(
                instances,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            )
        targets, count = labels_to_targets(instances)
        image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        mask_tensor = torch.from_numpy(targets.transpose(2, 0, 1).astype(np.float32))
        instance_tensor = (
            torch.from_numpy(instances.astype(np.int32))
            if not self.train
            else torch.empty(0, dtype=torch.int32)
        )
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "instances": instance_tensor,
            "count": count,
            "name": f"{source_key}_{Path(file_name).name}",
            "dataset": "livecell",
            "coco_source": source_key,
        }


def modality_balanced_sampler(
    livecell_count: int,
    external_samples: list[object],
) -> tuple[WeightedRandomSampler, dict[str, object]]:
    yeast_datasets = {"yeast_microstructures", "yeaz_phase", "yeaz_brightfield"}
    yeast_probability_ceiling = 0.10

    def sampling_domain(dataset: str) -> str:
        # Treat the three yeast releases as one morphology domain. Balancing them separately
        # would give round/oval budding cells three independent allocations.
        return "yeast" if dataset in yeast_datasets else dataset

    datasets = ["livecell"] * livecell_count + [
        str(getattr(sample, "dataset")) for sample in external_samples
    ]
    dataset_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    for dataset in datasets:
        domain = sampling_domain(dataset)
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    # Per-image 1/sqrt(N) gives each domain total mass sqrt(N): large domains retain more
    # influence, while small modalities are not drowned out. Yeast receives an additional hard
    # ceiling because all three yeast sources share similar size and round/oval morphology.
    weights = np.asarray(
        [1.0 / np.sqrt(domain_counts[sampling_domain(dataset)]) for dataset in datasets],
        dtype=np.float64,
    )
    domains = np.asarray([sampling_domain(dataset) for dataset in datasets])
    yeast_mask = domains == "yeast"
    yeast_mass = float(weights[yeast_mask].sum())
    non_yeast_mass = float(weights[~yeast_mask].sum())
    if yeast_mass and non_yeast_mass:
        permitted_yeast_mass = (
            yeast_probability_ceiling
            / (1.0 - yeast_probability_ceiling)
            * non_yeast_mass
        )
        if yeast_mass > permitted_yeast_mass:
            weights[yeast_mask] *= permitted_yeast_mass / yeast_mass

    total_mass = float(weights.sum())
    expected_domain_probabilities = {
        domain: float(weights[domains == domain].sum() / total_mass)
        for domain in sorted(domain_counts)
    }
    generator = torch.Generator().manual_seed(1337)
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )
    return sampler, {
        "unit": "image",
        "dataset_image_counts": dataset_counts,
        "sampling_domain_image_counts": domain_counts,
        "expected_domain_probabilities": expected_domain_probabilities,
        "yeast_probability_ceiling": yeast_probability_ceiling,
    }


def build_model(spec: ModelSpec) -> nn.Module:
    architectures = {
        "Unet": smp.Unet,
        "UnetPlusPlus": smp.UnetPlusPlus,
        "DeepLabV3Plus": smp.DeepLabV3Plus,
        "Segformer": smp.Segformer,
    }
    try:
        constructor = architectures[spec.architecture]
    except KeyError as error:
        raise ValueError(spec.architecture) from error
    return constructor(
        encoder_name=spec.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        classes=2,
        activation=None,
    )


def probe_training_microbatch(
    spec: ModelSpec,
    requested_batch_size: int,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[int, TrainingConfig, dict[str, object]]:
    """Allocate EMA, gradients and Adam state before committing to a long run."""
    effective_batch = requested_batch_size * config.accumulation_steps
    candidate = requested_batch_size
    failures: list[dict[str, object]] = []
    while candidate >= 1:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model = None
        ema_model = None
        optimizer = None
        images = None
        targets = None
        loss = None
        try:
            model = build_model(spec).to(device)
            if config.ema_decay > 0:
                ema_model = copy.deepcopy(model).eval()
                for parameter in ema_model.parameters():
                    parameter.requires_grad_(False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            images = torch.zeros(
                candidate,
                3,
                spec.image_size,
                spec.image_size,
                device=device,
            )
            targets = torch.zeros(
                candidate,
                3,
                spec.image_size,
                spec.image_size,
                device=device,
            )
            targets[:, 2] = 1
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = combined_loss(model(images), targets, config)
            loss.backward()
            optimizer.step()
            if ema_model is not None:
                update_ema(ema_model, model, config.ema_decay)
            chosen_config = replace(
                config,
                accumulation_steps=max(1, int(np.ceil(effective_batch / candidate))),
            )
            return candidate, chosen_config, {
                "requested_microbatch": requested_batch_size,
                "selected_microbatch": candidate,
                "requested_effective_batch": effective_batch,
                "selected_accumulation_steps": chosen_config.accumulation_steps,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device)
                / (1024**3),
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device)
                / (1024**3),
                "oom_retries": failures,
            }
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            if "out of memory" not in str(error).lower():
                raise
            failures.append({"microbatch": candidate, "error": str(error)})
            candidate //= 2
        finally:
            del loss, targets, images, optimizer, ema_model, model
            torch.cuda.empty_cache()
    raise RuntimeError(
        f"{spec.name} cannot complete one production-size optimizer step even at batch 1"
    )


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    if valid is None:
        valid = torch.ones_like(targets)
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets * valid).sum(dim=(1, 2, 3))
    denominator = (probabilities * valid).sum(dim=(1, 2, 3)) + (
        targets * valid
    ).sum(dim=(1, 2, 3))
    return 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()


def focal_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    positive_weight: float = 1.0,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    pos_weight = torch.tensor(
        positive_weight,
        dtype=logits.dtype,
        device=logits.device,
    )
    losses = nn.functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    if gamma > 0:
        probabilities = torch.sigmoid(logits)
        correct_probability = targets * probabilities + (1 - targets) * (1 - probabilities)
        losses = ((1 - correct_probability) ** gamma) * losses
    if valid is None:
        return losses.mean()
    return (losses * valid).sum() / valid.sum().clamp_min(1.0)


def tversky_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    beta: float,
    valid: torch.Tensor | None = None,
) -> torch.Tensor:
    if valid is None:
        valid = torch.ones_like(targets)
    probabilities = torch.sigmoid(logits)
    true_positive = (probabilities * targets * valid).sum(dim=(1, 2, 3))
    false_positive = (probabilities * (1 - targets) * valid).sum(dim=(1, 2, 3))
    false_negative = ((1 - probabilities) * targets * valid).sum(dim=(1, 2, 3))
    score = (true_positive + 1.0) / (
        true_positive + alpha * false_positive + beta * false_negative + 1.0
    )
    return 1.0 - score.mean()


def combined_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    config: TrainingConfig,
) -> torch.Tensor:
    valid = targets[:, 2:3] if targets.shape[1] >= 3 else None
    foreground_loss = focal_bce_loss(
        logits[:, :1],
        targets[:, :1],
        gamma=config.focal_gamma,
        valid=valid,
    ) + dice_loss(logits[:, :1], targets[:, :1], valid)
    if config.tversky_loss_weight > 0:
        foreground_loss = foreground_loss + config.tversky_loss_weight * tversky_loss(
            logits[:, :1],
            targets[:, :1],
            alpha=config.tversky_alpha,
            beta=config.tversky_beta,
            valid=valid,
        )
    boundary_loss = focal_bce_loss(
        logits[:, 1:2],
        targets[:, 1:2],
        gamma=config.focal_gamma,
        positive_weight=config.boundary_positive_weight,
        valid=valid,
    ) + dice_loss(logits[:, 1:2], targets[:, 1:2], valid)
    return foreground_loss + config.boundary_loss_weight * boundary_loss


def pixel_metrics(logits: torch.Tensor, targets: torch.Tensor) -> tuple[float, float]:
    valid = targets[:, 2:3] if targets.shape[1] >= 3 else None
    return channel_metrics(logits[:, :1], targets[:, :1], valid)


def channel_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid: torch.Tensor | None = None,
) -> tuple[float, float]:
    if valid is None:
        valid = torch.ones_like(targets, dtype=torch.bool)
    else:
        valid = valid >= 0.5
    predictions = torch.sigmoid(logits) >= 0.5
    truth = targets >= 0.5
    intersection = (predictions & truth & valid).sum(dim=(1, 2, 3)).float()
    union = ((predictions | truth) & valid).sum(dim=(1, 2, 3)).float()
    pred_sum = (predictions & valid).sum(dim=(1, 2, 3)).float()
    truth_sum = (truth & valid).sum(dim=(1, 2, 3)).float()
    iou = ((intersection + 1.0) / (union + 1.0)).mean().item()
    dice = ((2.0 * intersection + 1.0) / (pred_sum + truth_sum + 1.0)).mean().item()
    return iou, dice


def train_epoch(
    model: nn.Module,
    ema_model: nn.Module | None,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: TrainingConfig,
) -> float:
    model.train()
    losses = []
    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = combined_loss(model(images), masks, config)
            if config.l1_lambda > 0:
                l1_penalty = sum(
                    parameter.abs().sum() for parameter in model.parameters()
                )
                loss = loss + config.l1_lambda * l1_penalty
            remainder = len(loader) % config.accumulation_steps
            final_group_start = len(loader) - remainder if remainder else len(loader)
            group_size = (
                remainder
                if remainder and batch_index >= final_group_start
                else config.accumulation_steps
            )
            scaled_loss = loss / group_size
        scaler.scale(scaled_loss).backward()
        should_step = (
            (batch_index + 1) % config.accumulation_steps == 0
            or batch_index + 1 == len(loader)
        )
        if should_step:
            if config.gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip_norm,
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema_model is not None:
                update_ema(ema_model, model, config.ema_decay)
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    for ema_parameter, parameter in zip(
        ema_model.parameters(),
        model.parameters(),
        strict=True,
    ):
        ema_parameter.mul_(decay).add_(parameter, alpha=1 - decay)
    for ema_buffer, buffer in zip(
        ema_model.buffers(),
        model.buffers(),
        strict=True,
    ):
        ema_buffer.copy_(buffer)


def macro_average_domain_metric_rows(
    domain_rows: dict[str, list[dict[str, float]]],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Average images within each dataset, then give every dataset equal weight."""
    populated = {domain: rows for domain, rows in domain_rows.items() if rows}
    if not populated:
        raise ValueError("Cannot macro-average empty domain metric rows")
    first_domain_rows = next(iter(populated.values()))
    expected_keys = set(first_domain_rows[0])
    per_domain: dict[str, dict[str, float]] = {}
    for domain, rows in sorted(populated.items()):
        if any(set(row) != expected_keys for row in rows):
            raise ValueError(f"Inconsistent metric keys within validation domain {domain}")
        per_domain[domain] = {
            key: float(np.mean([row[key] for row in rows]))
            for key in sorted(expected_keys)
        }
    macro = {
        key: float(np.mean([metrics[key] for metrics in per_domain.values()]))
        for key in sorted(expected_keys)
    }
    return macro, per_domain


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
) -> dict[str, object]:
    model.eval()
    domain_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    domain_instance_ap: dict[str, list[dict[str, float]]] = defaultdict(list)
    instance_samples: dict[str, int] = defaultdict(int)
    for batch in tqdm(loader, desc="validate", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(images)
        probabilities = torch.sigmoid(logits).detach().cpu().numpy()
        for index in range(images.shape[0]):
            source_name = str(batch["dataset"][index])
            sample_logits = logits[index : index + 1]
            sample_masks = masks[index : index + 1]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                validation_loss = combined_loss(sample_logits, sample_masks, config)
            iou, dice = pixel_metrics(sample_logits, sample_masks)
            valid = sample_masks[:, 2:3] if sample_masks.shape[1] >= 3 else None
            _, boundary_dice = channel_metrics(
                sample_logits[:, 1:2],
                sample_masks[:, 1:2],
                valid,
            )
            predicted_instances = instance_labels(
                probabilities[index, 0],
                probabilities[index, 1],
            )
            domain_rows[source_name].append(
                {
                    "loss": float(validation_loss.item()),
                    "iou": iou,
                    "dice": dice,
                    "boundary_dice": boundary_dice,
                    "count_mae": float(
                        abs(
                            int(predicted_instances.max())
                            - int(batch["count"][index])
                        )
                    ),
                }
            )
            if instance_samples[source_name] < config.validation_instance_samples:
                truth_labels = batch["instances"][index].numpy().astype(np.int32)
                ap, _, _, _ = cellpose_metrics.average_precision(
                    [truth_labels],
                    [predicted_instances],
                    threshold=[0.5, 0.75],
                )
                ap_values = np.asarray(ap)[0]
                domain_instance_ap[source_name].append(
                    {
                        "instance_ap50": float(ap_values[0]),
                        "instance_ap75": float(ap_values[1]),
                    }
                )
                instance_samples[source_name] += 1

    macro, per_domain = macro_average_domain_metric_rows(domain_rows)
    for domain, metrics in per_domain.items():
        ap_rows = domain_instance_ap.get(domain, [])
        if ap_rows:
            ap_metrics, _ = macro_average_domain_metric_rows({domain: ap_rows})
        else:
            ap_metrics = {"instance_ap50": 0.0, "instance_ap75": 0.0}
        metrics.update(ap_metrics)
        metrics["selection_score"] = (
            0.35 * metrics["dice"]
            + 0.15 * metrics["boundary_dice"]
            + 0.30 * metrics["instance_ap50"]
            + 0.20 * metrics["instance_ap75"]
        )
        metrics["evaluation_images"] = len(domain_rows[domain])
        metrics["instance_ap_samples"] = instance_samples[domain]

    ap50 = float(
        np.mean([metrics["instance_ap50"] for metrics in per_domain.values()])
    )
    ap75 = float(
        np.mean([metrics["instance_ap75"] for metrics in per_domain.values()])
    )
    selection_score = (
        0.35 * macro["dice"]
        + 0.15 * macro["boundary_dice"]
        + 0.30 * ap50
        + 0.20 * ap75
    )
    return {
        "loss": macro["loss"],
        "iou": macro["iou"],
        "dice": macro["dice"],
        "boundary_dice": macro["boundary_dice"],
        "instance_ap50": ap50,
        "instance_ap75": ap75,
        "count_mae": macro["count_mae"],
        "selection_score": selection_score,
        "domain_count": len(per_domain),
        "domain_aggregation": (
            "per-image metrics averaged within dataset, then equally across datasets"
        ),
        "per_domain_metrics": per_domain,
    }


def atomic_torch_save(payload: dict[str, object], destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    previous = destination.with_suffix(destination.suffix + ".prev")
    torch.save(payload, temporary)
    if destination.is_file():
        shutil.copy2(destination, previous)
    temporary.replace(destination)


def train_model(
    spec: ModelSpec,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
    stage_fingerprint: str,
) -> tuple[Path, list[dict[str, float]]]:
    run_dir = RUNS / spec.name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = run_dir / "best.pt", run_dir / "last.pt"
    model = build_model(spec).to(device)
    encoder_parameters = list(model.encoder.parameters())
    encoder_parameter_ids = {id(parameter) for parameter in encoder_parameters}
    decoder_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in encoder_parameter_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": config.encoder_learning_rate,
            },
            {
                "params": decoder_parameters,
                "lr": config.decoder_learning_rate,
            },
        ],
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.minimum_learning_rate,
    )
    scaler = torch.amp.GradScaler("cuda")
    ema_model = None
    if config.ema_decay > 0:
        ema_model = copy.deepcopy(model).eval()
        for parameter in ema_model.parameters():
            parameter.requires_grad_(False)
    history: list[dict[str, float]] = []
    start_epoch, best_score, stale = 0, -1.0, 0
    if last_path.exists():
        try:
            checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
        except Exception:
            previous_path = last_path.with_suffix(last_path.suffix + ".prev")
            if not previous_path.is_file():
                raise
            print(f"Recovering {spec.name} from {previous_path.name}")
            checkpoint = torch.load(previous_path, map_location="cpu", weights_only=False)
        if checkpoint.get("stage_fingerprint") != stage_fingerprint:
            raise RuntimeError(
                f"Refusing to resume {spec.name}: checkpoint fingerprint does not match v3 run"
            )
        model.load_state_dict(checkpoint["model"])
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except ValueError:
            print(
                f"Optimizer layout changed for {spec.name}; "
                "resuming weights with fresh optimizer state."
            )
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        if ema_model is not None and checkpoint.get("ema_model") is not None:
            ema_model.load_state_dict(checkpoint["ema_model"])
        if "python_random_state" in checkpoint:
            random.setstate(checkpoint["python_random_state"])
        if "numpy_random_state" in checkpoint:
            np.random.set_state(checkpoint["numpy_random_state"])
        if "torch_random_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_random_state"])
        if "cuda_random_state" in checkpoint:
            torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint.get("best_score", checkpoint.get("best_dice", -1.0))
        stale = checkpoint["stale"]
        print(f"Resuming {spec.name} at epoch {start_epoch + 1}")
    for epoch in range(start_epoch, config.epochs):
        train_loss = train_epoch(
            model,
            ema_model,
            train_loader,
            optimizer,
            scaler,
            device,
            config,
        )
        evaluation_model = ema_model if ema_model is not None else model
        metrics = validate(evaluation_model, val_loader, device, config)
        scheduler.step(metrics["selection_score"])
        row = {
            "epoch": float(epoch + 1),
            "train_loss": train_loss,
            "val_loss": metrics["loss"],
            "val_iou": metrics["iou"],
            "val_dice": metrics["dice"],
            "val_boundary_dice": metrics["boundary_dice"],
            "val_instance_ap50": metrics["instance_ap50"],
            "val_instance_ap75": metrics["instance_ap75"],
            "val_count_mae": metrics["count_mae"],
            "val_selection_score": metrics["selection_score"],
            "encoder_learning_rate": optimizer.param_groups[0]["lr"],
            "decoder_learning_rate": optimizer.param_groups[1]["lr"],
            "learning_rate": optimizer.param_groups[1]["lr"],
        }
        history.append(row)
        print(json.dumps({"model": spec.name, **row}))
        improved = metrics["selection_score"] > best_score + config.minimum_improvement
        if improved:
            best_score, stale = metrics["selection_score"], 0
            atomic_torch_save(
                {
                    "stage_fingerprint": stage_fingerprint,
                    "spec": asdict(spec),
                    "training_config": asdict(config),
                    "model": evaluation_model.state_dict(),
                    "metrics": metrics,
                },
                best_path,
            )
        else:
            stale += 1
        atomic_torch_save(
            {
                "stage_fingerprint": stage_fingerprint,
                "epoch": epoch,
                "spec": asdict(spec),
                "model": model.state_dict(),
                "ema_model": ema_model.state_dict() if ema_model is not None else None,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "history": history,
                "validation_metrics": metrics,
                "best_score": best_score,
                "best_dice": max(row["val_dice"] for row in history),
                "stale": stale,
                "python_random_state": random.getstate(),
                "numpy_random_state": np.random.get_state(),
                "torch_random_state": torch.get_rng_state(),
                "cuda_random_state": torch.cuda.get_rng_state_all(),
            },
            last_path,
        )
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        if stale >= config.early_stopping_patience:
            print(f"Early stopping {spec.name}")
            break
    return best_path, history


def instance_labels(
    foreground_probability: np.ndarray,
    boundary_probability: np.ndarray,
    config: PostprocessConfig = DEFAULT_POSTPROCESS_CONFIG,
) -> np.ndarray:
    labels = reconstruct_instance_labels(
        foreground_probability,
        boundary_probability,
        config,
    )
    return filter_instance_labels(
        labels,
        foreground_probability,
        boundary_probability,
        config,
    )


def reconstruct_instance_labels(
    foreground_probability: np.ndarray,
    boundary_probability: np.ndarray,
    config: PostprocessConfig,
) -> np.ndarray:
    """Run the iPhone's 8-connected FIFO expansion without confidence filtering."""
    return reconstruct_iphone_instances(
        foreground_probability,
        boundary_probability,
        config,
    )


def research_watershed_instance_labels(
    foreground_probability: np.ndarray,
    boundary_probability: np.ndarray,
    config: PostprocessConfig,
) -> np.ndarray:
    """Research-only distance watershed; never used to select deployable settings."""
    clean = ndimage.binary_opening(
        foreground_probability >= config.foreground_threshold,
        iterations=1,
    )
    clean = ndimage.binary_closing(clean, iterations=1)
    interiors = clean & (boundary_probability < config.boundary_threshold)
    markers, _ = ndimage.label(interiors)
    if markers.max() == 0:
        markers, _ = ndimage.label(clean)
    distance = ndimage.distance_transform_edt(clean)
    labels = watershed(-distance, markers, mask=clean)
    minimum_area = max(
        1,
        int(round(config.min_area_fraction * foreground_probability.size)),
    )
    if labels.max() > 0 and minimum_area > 1:
        areas = np.bincount(labels.ravel())
        small = np.flatnonzero(areas < minimum_area)
        small = small[small != 0]
        if small.size:
            labels[np.isin(labels, small)] = 0
    return relabel_instances(labels)


def relabel_instances(labels: np.ndarray) -> np.ndarray:
    if labels.max() == 0:
        return np.zeros(labels.shape, dtype=np.int32)
    unique = np.unique(labels)
    unique = unique[unique != 0]
    remap = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    remap[unique] = np.arange(1, len(unique) + 1)
    return remap[labels]


def filter_instance_labels(
    labels: np.ndarray,
    foreground_probability: np.ndarray,
    boundary_probability: np.ndarray,
    config: PostprocessConfig,
) -> np.ndarray:
    """Reject reconstructed objects with weak probability, core, or boundary evidence."""
    if labels.max() == 0:
        return labels.astype(np.int32, copy=False)
    if not any(
        (
            config.min_mean_foreground_probability,
            config.min_core_fraction,
            config.min_boundary_support,
        )
    ):
        return labels.astype(np.int32, copy=False)
    label_count = int(labels.max()) + 1
    areas = np.bincount(labels.ravel(), minlength=label_count).astype(np.float64)
    safe_areas = np.maximum(areas, 1.0)
    foreground_sums = np.bincount(
        labels.ravel(),
        weights=foreground_probability.ravel(),
        minlength=label_count,
    )
    core_counts = np.bincount(
        labels.ravel(),
        weights=(
            foreground_probability >= config.core_probability_threshold
        ).ravel(),
        minlength=label_count,
    )
    perimeter = np.zeros(labels.shape, dtype=bool)
    foreground = labels > 0
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted = np.roll(labels, shift=(dy, dx), axis=(0, 1))
        perimeter |= foreground & (shifted != labels)
    perimeter[[0, -1], :] |= foreground[[0, -1], :]
    perimeter[:, [0, -1]] |= foreground[:, [0, -1]]
    perimeter_counts = np.bincount(
        labels[perimeter],
        minlength=label_count,
    ).astype(np.float64)
    boundary_sums = np.bincount(
        labels[perimeter],
        weights=boundary_probability[perimeter],
        minlength=label_count,
    )
    mean_foreground = foreground_sums / safe_areas
    core_fraction = core_counts / safe_areas
    boundary_support = boundary_sums / np.maximum(perimeter_counts, 1.0)
    reject = (
        (mean_foreground < config.min_mean_foreground_probability)
        | (core_fraction < config.min_core_fraction)
        | (boundary_support < config.min_boundary_support)
    )
    reject[0] = False
    filtered = labels.copy()
    filtered[reject[labels]] = 0
    return relabel_instances(filtered)


def segmentation_metrics(
    prediction_labels: np.ndarray,
    truth_labels: np.ndarray,
) -> dict[str, float]:
    """Measure foreground, reconstructed boundaries, instances, and counting together."""
    truth_foreground = truth_labels > 0
    prediction_foreground = prediction_labels > 0
    true_positive = int((prediction_foreground & truth_foreground).sum())
    false_positive = int((prediction_foreground & ~truth_foreground).sum())
    false_negative = int((~prediction_foreground & truth_foreground).sum())
    foreground_denominator = int(
        prediction_foreground.sum() + truth_foreground.sum()
    )
    foreground_union = true_positive + false_positive + false_negative

    prediction_targets, _ = labels_to_targets(prediction_labels)
    truth_targets, _ = labels_to_targets(truth_labels)
    prediction_boundary = prediction_targets[..., 1] > 0
    truth_boundary = truth_targets[..., 1] > 0
    boundary_true_positive = int((prediction_boundary & truth_boundary).sum())
    boundary_false_positive = int((prediction_boundary & ~truth_boundary).sum())
    boundary_false_negative = int((~prediction_boundary & truth_boundary).sum())
    boundary_denominator = int(prediction_boundary.sum() + truth_boundary.sum())

    thresholds = (0.5, 0.75, 0.9)
    average_precision, _, _, _ = cellpose_metrics.average_precision(
        [truth_labels],
        [prediction_labels],
        threshold=list(thresholds),
    )
    ap = np.asarray(average_precision)[0]
    return {
        "foreground_dice": (2 * true_positive + 1) / (foreground_denominator + 1),
        "foreground_iou": (true_positive + 1) / (foreground_union + 1),
        "foreground_precision": (true_positive + 1) / (
            true_positive + false_positive + 1
        ),
        "foreground_recall": (true_positive + 1) / (
            true_positive + false_negative + 1
        ),
        "false_positive_area_fraction": false_positive / prediction_labels.size,
        "false_negative_area_fraction": false_negative / prediction_labels.size,
        "boundary_dice": (2 * boundary_true_positive + 1) / (
            boundary_denominator + 1
        ),
        "boundary_precision": (boundary_true_positive + 1) / (
            boundary_true_positive + boundary_false_positive + 1
        ),
        "boundary_recall": (boundary_true_positive + 1) / (
            boundary_true_positive + boundary_false_negative + 1
        ),
        "instance_ap50": float(ap[0]),
        "instance_ap75": float(ap[1]),
        "instance_ap90": float(ap[2]),
        "count_absolute_error": abs(
            int(prediction_labels.max()) - int(truth_labels.max())
        ),
    }


def postprocessing_selection_score(metrics: dict[str, float]) -> float:
    return (
        0.25 * metrics["foreground_dice"]
        + 0.15 * metrics["boundary_dice"]
        + 0.35 * metrics["instance_ap50"]
        + 0.25 * metrics["instance_ap75"]
    )


def evaluate_postprocessing_configs(
    records: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
    configs: list[PostprocessConfig],
) -> list[dict[str, object]]:
    """Evaluate configs while sharing each expensive iPhone-FIFO reconstruction."""
    metric_rows: dict[PostprocessConfig, dict[str, list[dict[str, float]]]] = {
        config: defaultdict(list) for config in configs
    }
    for foreground, boundary, truth, domain in tqdm(
        records,
        desc=f"postprocess grid ({len(configs)} settings)",
        leave=False,
    ):
        reconstruction_cache: dict[tuple[float, float, float], np.ndarray] = {}
        for config in configs:
            reconstruction_key = (
                config.foreground_threshold,
                config.boundary_threshold,
                config.min_area_fraction,
            )
            if reconstruction_key not in reconstruction_cache:
                reconstruction_cache[reconstruction_key] = reconstruct_instance_labels(
                    foreground,
                    boundary,
                    config,
                )
            labels = filter_instance_labels(
                reconstruction_cache[reconstruction_key],
                foreground,
                boundary,
                config,
            )
            metric_rows[config][domain].append(segmentation_metrics(labels, truth))
    results = []
    for config, domain_rows in metric_rows.items():
        per_domain_metrics = {
            domain: average_metric_rows(rows)
            for domain, rows in sorted(domain_rows.items())
        }
        metrics = average_metric_rows(list(per_domain_metrics.values()))
        results.append(
            {
                "config": asdict(config),
                "metrics": metrics,
                "per_domain_metrics": per_domain_metrics,
                "selection_score": postprocessing_selection_score(metrics),
            }
        )
    return sorted(results, key=lambda row: row["selection_score"], reverse=True)


def postprocessing_filter_grid() -> list[tuple[float, float, float]]:
    filters = {(0.0, 0.0, 0.0)}
    mean_thresholds = (0.50, 0.55, 0.60, 0.65)
    core_fractions = (0.05, 0.15, 0.30)
    filters.update((value, 0.0, 0.0) for value in mean_thresholds)
    filters.update((0.0, value, 0.0) for value in core_fractions)
    filters.update(
        (mean, core, 0.0)
        for mean in mean_thresholds
        for core in core_fractions
    )
    return sorted(filters)


def boundary_cutoff_curves(
    reconstruction_results: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Integrate downstream validation metrics across the boundary cutoff sweep."""
    grouped: dict[tuple[float, float], list[dict[str, object]]] = defaultdict(list)
    for row in reconstruction_results:
        config = row["config"]
        grouped[
            (
                float(config["foreground_threshold"]),
                float(config["min_area_fraction"]),
            )
        ].append(row)
    curves = []
    for (foreground_threshold, min_area_fraction), rows in grouped.items():
        rows.sort(key=lambda row: float(row["config"]["boundary_threshold"]))
        cutoffs = np.asarray(
            [float(row["config"]["boundary_threshold"]) for row in rows],
            dtype=np.float64,
        )
        denominator = float(cutoffs[-1] - cutoffs[0])
        metric_auc = {
            metric: float(
                np.trapz(
                    [float(row["metrics"][metric]) for row in rows],
                    cutoffs,
                )
                / denominator
            )
            for metric in rows[0]["metrics"]
        }
        selection_auc = float(
            np.trapz(
                [float(row["selection_score"]) for row in rows],
                cutoffs,
            )
            / denominator
        )
        best = max(rows, key=lambda row: row["selection_score"])
        curves.append(
            {
                "foreground_threshold": foreground_threshold,
                "min_area_fraction": min_area_fraction,
                "boundary_cutoffs": cutoffs.tolist(),
                "normalized_metric_auc": metric_auc,
                "normalized_selection_score_auc": selection_auc,
                "best_operating_point": best,
                "operating_points": rows,
            }
        )
    return sorted(
        curves,
        key=lambda curve: curve["normalized_selection_score_auc"],
        reverse=True,
    )


def tune_postprocessing_records(
    records: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
    context: str,
) -> dict[str, object]:
    """Two-stage validation search: reconstruction first, then object filters."""
    if not records:
        raise ValueError("Post-processing tuning requires at least one validation image.")
    area_fractions = (0.0, 20.0 / (512.0 * 512.0), 50.0 / (512.0 * 512.0))
    reconstruction_configs = [
        PostprocessConfig(
            foreground_threshold=foreground_threshold,
            boundary_threshold=boundary_threshold,
            min_area_fraction=min_area_fraction,
        )
        for foreground_threshold in (0.40, 0.50, 0.60)
        for boundary_threshold in BOUNDARY_CUTOFF_SWEEP
        for min_area_fraction in area_fractions
    ]
    reconstruction_results = evaluate_postprocessing_configs(
        records,
        reconstruction_configs,
    )
    cutoff_curves = boundary_cutoff_curves(reconstruction_results)
    # Use curve area to choose robust foreground/area families, then use each family's validation
    # peak as the deployable boundary cutoff. The iPhone still needs one concrete operating point.
    top_reconstructions = [
        PostprocessConfig(**curve["best_operating_point"]["config"])
        for curve in cutoff_curves[:4]
    ]
    filtered_configs = []
    for base in top_reconstructions:
        for mean_threshold, core_fraction, boundary_support in postprocessing_filter_grid():
            filtered_configs.append(
                replace(
                    base,
                    min_mean_foreground_probability=mean_threshold,
                    min_core_fraction=core_fraction,
                    min_boundary_support=boundary_support,
                )
            )
    filtered_configs = list(dict.fromkeys(filtered_configs))
    filter_results = evaluate_postprocessing_configs(records, filtered_configs)
    results_by_config = {
        tuple(sorted(row["config"].items())): row
        for row in reconstruction_results + filter_results
    }
    candidates = sorted(
        results_by_config.values(),
        key=lambda row: row["selection_score"],
        reverse=True,
    )
    baseline_key = tuple(sorted(asdict(DEFAULT_POSTPROCESS_CONFIG).items()))
    baseline = results_by_config[baseline_key]
    selected = candidates[0]
    metric_deltas = {
        key: float(selected["metrics"][key] - baseline["metrics"][key])
        for key in selected["metrics"]
    }
    return {
        "context": context,
        "selection_split": "validation/calibration only; no held-out test labels used",
        "selection_objective": (
            "0.25 foreground Dice + 0.15 reconstructed-boundary Dice + "
            "0.35 AP50 + 0.25 AP75"
        ),
        "boundary_cutoff_protocol": (
            "sweep 0.10 through 0.90 in 0.05 increments; normalized trapezoidal area "
            "selects robust foreground/area families, then the validation peak supplies the "
            "single deployable boundary cutoff"
        ),
        "boundary_cutoff_curves": cutoff_curves,
        "object_scores": {
            "mean_foreground_probability": "mean, not sum, to avoid favoring large cells",
            "core_fraction": "fraction of object pixels with foreground probability >= 0.70",
            "boundary_support": (
                "supported by the app but excluded from v3 automatic selection because the "
                "boundary head predicts internal contacts, not ordinary outer perimeters"
            ),
        },
        "validation_images": len(records),
        "validation_domains": dict(
            sorted(
                {
                    domain: sum(record[3] == domain for record in records)
                    for domain in {record[3] for record in records}
                }.items()
            )
        ),
        "domain_aggregation": "metrics averaged within domain, then macro-averaged across domains",
        "candidate_count": len(candidates),
        "baseline": baseline,
        "selected": selected,
        "selected_minus_baseline": metric_deltas,
        "candidates": candidates,
    }


@torch.inference_mode()
def tune_semantic_postprocessing(
    spec: ModelSpec,
    checkpoint_path: Path,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, object]:
    report_path = RUNS / spec.name / "postprocessing_search.json"
    checkpoint_digest = sha256(checkpoint_path)
    if report_path.is_file():
        existing_report = json.loads(report_path.read_text())
        if (
            existing_report.get("checkpoint_sha256") == checkpoint_digest
            and existing_report.get("postprocess_version") == POSTPROCESS_VERSION
            and existing_report.get("boundary_target_version")
            == BOUNDARY_TARGET_VERSION
        ):
            return existing_report
    model = build_model(spec).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    records: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
    for batch in tqdm(loader, desc=f"calibrate postprocessing {spec.name}"):
        images = batch["image"].to(device, non_blocking=True)
        probabilities = torch.sigmoid(model(images)).cpu().numpy().astype(np.float16)
        for index in range(images.shape[0]):
            records.append(
                (
                    probabilities[index, 0],
                    probabilities[index, 1],
                    batch["instances"][index].numpy().astype(np.int32),
                    str(batch["dataset"][index]),
                )
            )
    report = tune_postprocessing_records(records, spec.name)
    report["checkpoint_sha256"] = checkpoint_digest
    report["postprocess_version"] = POSTPROCESS_VERSION
    report["boundary_target_version"] = BOUNDARY_TARGET_VERSION
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


@torch.inference_mode()
def evaluate_model(
    spec: ModelSpec,
    checkpoint_path: Path,
    loader: DataLoader,
    device: torch.device,
    evaluation_name: str = "livecell_test",
    postprocess_config: PostprocessConfig = DEFAULT_POSTPROCESS_CONFIG,
) -> dict[str, object]:
    model = build_model(spec).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    raw_domain_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    postprocessed_domain_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    runtime_domain_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    previews = RUNS / spec.name / "previews" / evaluation_name
    previews.mkdir(parents=True, exist_ok=True)
    preview_count = 0
    for batch in tqdm(loader, desc=f"test {spec.name}"):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            logits = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        gpu_seconds_per_image = (time.perf_counter() - started) / images.shape[0]
        probabilities = torch.sigmoid(logits).cpu().numpy()
        for i in range(images.shape[0]):
            domain = str(batch["dataset"][i])
            sample_logits = logits[i : i + 1]
            sample_masks = masks[i : i + 1]
            iou, dice = pixel_metrics(sample_logits, sample_masks)
            valid = sample_masks[:, 2:3] if sample_masks.shape[1] >= 3 else None
            _, boundary_dice = channel_metrics(
                sample_logits[:, 1:2],
                sample_masks[:, 1:2],
                valid,
            )
            raw_domain_rows[domain].append(
                {
                    "foreground_iou": iou,
                    "foreground_dice": dice,
                    "boundary_dice": boundary_dice,
                }
            )
            postprocess_started = time.perf_counter()
            labels = instance_labels(
                probabilities[i, 0],
                probabilities[i, 1],
                postprocess_config,
            )
            postprocess_seconds = time.perf_counter() - postprocess_started
            runtime_domain_rows[domain].append(
                {
                    "gpu_seconds_per_image": gpu_seconds_per_image,
                    "postprocessing_seconds_per_image": postprocess_seconds,
                }
            )
            truth_labels = batch["instances"][i].numpy().astype(np.int32)
            postprocessed_domain_rows[domain].append(
                segmentation_metrics(labels, truth_labels)
            )
            if preview_count < 20:
                gray = (images[i, 0].cpu().numpy() * 255).astype(np.uint8)
                prediction = labels > 0
                overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                overlay[prediction] = (
                    0.45 * overlay[prediction] + 0.55 * np.array([40, 220, 40])
                ).astype(np.uint8)
                cv2.imwrite(
                    str(previews / f"{preview_count:02d}_{batch['name'][i]}.jpg"), overlay
                )
                preview_count += 1
    postprocessed, per_domain_postprocessed = macro_average_domain_metric_rows(
        postprocessed_domain_rows
    )
    raw_metrics, per_domain_raw = macro_average_domain_metric_rows(raw_domain_rows)
    runtime_metrics, per_domain_runtime = macro_average_domain_metric_rows(
        runtime_domain_rows
    )
    domains = set(per_domain_postprocessed)
    if domains != set(per_domain_raw) or domains != set(per_domain_runtime):
        raise RuntimeError("Evaluation metric domains became inconsistent")
    per_domain_metrics: dict[str, dict[str, object]] = {}
    for domain in sorted(domains):
        domain_postprocessed = per_domain_postprocessed[domain]
        per_domain_metrics[domain] = {
            "evaluation_images": len(postprocessed_domain_rows[domain]),
            **domain_postprocessed,
            "postprocessing_selection_score": postprocessing_selection_score(
                domain_postprocessed
            ),
            "raw_probability_threshold_metrics": per_domain_raw[domain],
            **per_domain_runtime[domain],
        }
    return {
        "test_iou": postprocessed["foreground_iou"],
        "test_dice": postprocessed["foreground_dice"],
        "test_boundary_dice": postprocessed["boundary_dice"],
        "foreground_precision": postprocessed["foreground_precision"],
        "foreground_recall": postprocessed["foreground_recall"],
        "boundary_precision": postprocessed["boundary_precision"],
        "boundary_recall": postprocessed["boundary_recall"],
        "false_positive_area_fraction": postprocessed["false_positive_area_fraction"],
        "false_negative_area_fraction": postprocessed["false_negative_area_fraction"],
        "count_mae": postprocessed["count_absolute_error"],
        "instance_average_precision": {
            "0.5": postprocessed["instance_ap50"],
            "0.75": postprocessed["instance_ap75"],
            "0.9": postprocessed["instance_ap90"],
        },
        "postprocessing": asdict(postprocess_config),
        "postprocessing_selection_score_components": postprocessed,
        "raw_probability_threshold_metrics": raw_metrics,
        "gpu_seconds_per_image": runtime_metrics["gpu_seconds_per_image"],
        "postprocessing_seconds_per_image": runtime_metrics[
            "postprocessing_seconds_per_image"
        ],
        "evaluation_images": sum(len(rows) for rows in postprocessed_domain_rows.values()),
        "domain_count": len(domains),
        "domain_image_counts": {
            domain: len(postprocessed_domain_rows[domain]) for domain in sorted(domains)
        },
        "domain_aggregation": (
            "per-image metrics averaged within dataset, then equally across datasets"
        ),
        "per_domain_metrics": per_domain_metrics,
        "checkpoint_parameters": sum(p.numel() for p in model.parameters()),
    }


def export_models(spec: ModelSpec, checkpoint_path: Path) -> tuple[Path, Path]:
    model = build_model(spec)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    onnx_destination = RUNS / spec.name / f"{spec.name}_{spec.image_size}.onnx"
    torchscript_destination = (
        RUNS / spec.name / f"{spec.name}_{spec.image_size}.torchscript.pt"
    )
    sample = torch.zeros(1, 3, spec.image_size, spec.image_size)
    traced = torch.jit.trace(model, sample, strict=False)
    torch.jit.save(torch.jit.freeze(traced), torchscript_destination)
    torch.onnx.export(
        model,
        sample,
        onnx_destination,
        input_names=["image"],
        output_names=["foreground_and_boundary_logits"],
        dynamic_axes={
            "image": {0: "batch"},
            "foreground_and_boundary_logits": {0: "batch"},
        },
        opset_version=17,
    )
    return onnx_destination, torchscript_destination


def combination_matrix(model_count: int, device: torch.device) -> tuple[list[tuple[int, ...]], torch.Tensor]:
    members = [
        member_tuple
        for size in range(1, model_count + 1)
        for member_tuple in combinations(range(model_count), size)
    ]
    matrix = torch.zeros((len(members), model_count), dtype=torch.float32, device=device)
    for row, member_tuple in enumerate(members):
        matrix[row, list(member_tuple)] = 1.0 / len(member_tuple)
    return members, matrix


def coco_instance_mask(coco: COCO, image_id: int, size: int) -> np.ndarray:
    metadata = coco.loadImgs([image_id])[0]
    labels = np.zeros((int(metadata["height"]), int(metadata["width"])), dtype=np.int32)
    annotations = coco.loadAnns(coco.getAnnIds(imgIds=[image_id], iscrowd=None))
    for label, annotation in enumerate(annotations, start=1):
        labels[coco.annToMask(annotation).astype(bool)] = label
    return cv2.resize(
        labels.astype(np.float32),
        (size, size),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int32)


ENSEMBLE_EVALUATOR_VERSION = "disk-stream-role-separated-deployment-grid-v3"


@torch.inference_mode()
def ensemble_member_probabilities(
    model: nn.Module,
    spec: ModelSpec,
    image: np.ndarray,
    common_size: int,
    device: torch.device,
) -> np.ndarray:
    """Run one member exactly once and align its maps to the app's fusion grid."""
    probabilities = predict_deployment(model, image, spec.image_size, device)
    if probabilities.shape != (2, spec.image_size, spec.image_size):
        raise RuntimeError(
            f"Unexpected {spec.name} output shape {probabilities.shape}; "
            f"expected (2, {spec.image_size}, {spec.image_size})."
        )
    if spec.image_size == common_size:
        return np.asarray(probabilities, dtype=np.float32)
    return np.stack(
        [
            cv2.resize(
                probabilities[channel],
                (common_size, common_size),
                interpolation=cv2.INTER_LINEAR,
            )
            for channel in range(2)
        ]
    ).astype(np.float32, copy=False)


def atomic_numpy_save(destination: Path, array: np.ndarray) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{os.getpid()}.tmp.npy"
    )
    np.save(temporary, np.ascontiguousarray(array), allow_pickle=False)
    temporary.replace(destination)


def atomic_json_save(destination: Path, payload: dict[str, object]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(destination)


def cached_array_is_valid(
    path: Path,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> bool:
    if not path.is_file():
        return False
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        return (
            array.shape == shape
            and array.dtype == dtype
            and bool(np.isfinite(array).all())
        )
    except (OSError, ValueError):
        return False


def evaluate_postprocessing_configs_streaming(
    record_factory: Callable[
        [], Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, str]]
    ],
    record_count: int,
    configs: list[PostprocessConfig],
    description: str,
) -> list[dict[str, object]]:
    """Evaluate a grid with one fused probability map and one reconstruction in RAM."""
    if record_count < 1:
        raise ValueError("Streaming post-processing requires at least one record.")
    grouped_configs: dict[
        tuple[float, float, float], list[PostprocessConfig]
    ] = defaultdict(list)
    for config in configs:
        grouped_configs[
            (
                config.foreground_threshold,
                config.boundary_threshold,
                config.min_area_fraction,
            )
        ].append(config)

    metric_sums: dict[
        PostprocessConfig, dict[str, dict[str, float]]
    ] = {config: {} for config in configs}
    metric_counts: dict[PostprocessConfig, dict[str, int]] = {
        config: {} for config in configs
    }
    observed_records = 0
    for foreground, boundary, truth, domain in tqdm(
        record_factory(),
        total=record_count,
        desc=description,
        leave=False,
    ):
        observed_records += 1
        for reconstruction_configs in grouped_configs.values():
            base_labels = reconstruct_instance_labels(
                foreground,
                boundary,
                reconstruction_configs[0],
            )
            for config in reconstruction_configs:
                labels = filter_instance_labels(
                    base_labels,
                    foreground,
                    boundary,
                    config,
                )
                metrics = segmentation_metrics(labels, truth)
                sums = metric_sums[config].setdefault(domain, {})
                for key, value in metrics.items():
                    sums[key] = sums.get(key, 0.0) + float(value)
                metric_counts[config][domain] = (
                    metric_counts[config].get(domain, 0) + 1
                )
    if observed_records != record_count:
        raise RuntimeError(
            f"Streaming record factory yielded {observed_records}, expected {record_count}."
        )

    results = []
    for config in configs:
        per_domain_metrics = {
            domain: {
                key: value / metric_counts[config][domain]
                for key, value in sums.items()
            }
            for domain, sums in sorted(metric_sums[config].items())
        }
        metrics = average_metric_rows(list(per_domain_metrics.values()))
        results.append(
            {
                "config": asdict(config),
                "metrics": metrics,
                "per_domain_metrics": per_domain_metrics,
                "selection_score": postprocessing_selection_score(metrics),
            }
        )
    return sorted(results, key=lambda row: row["selection_score"], reverse=True)


def tune_postprocessing_streaming(
    record_factory: Callable[
        [], Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, str]]
    ],
    record_count: int,
    validation_domains: dict[str, int],
    context: str,
    role: str,
) -> dict[str, object]:
    """Run the deployable two-stage search without retaining full-grid records."""
    area_fractions = (
        0.0,
        20.0 / (512.0 * 512.0),
        50.0 / (512.0 * 512.0),
    )
    reconstruction_configs = [
        PostprocessConfig(
            foreground_threshold=foreground_threshold,
            boundary_threshold=boundary_threshold,
            min_area_fraction=min_area_fraction,
        )
        for foreground_threshold in (0.40, 0.50, 0.60)
        for boundary_threshold in BOUNDARY_CUTOFF_SWEEP
        for min_area_fraction in area_fractions
    ]
    reconstruction_results = evaluate_postprocessing_configs_streaming(
        record_factory,
        record_count,
        reconstruction_configs,
        f"{context} reconstruction grid",
    )
    cutoff_curves = boundary_cutoff_curves(reconstruction_results)
    top_reconstructions = [
        PostprocessConfig(**curve["best_operating_point"]["config"])
        for curve in cutoff_curves[:4]
    ]
    filtered_configs = [
        replace(
            base,
            min_mean_foreground_probability=mean_threshold,
            min_core_fraction=core_fraction,
            min_boundary_support=boundary_support,
        )
        for base in top_reconstructions
        for mean_threshold, core_fraction, boundary_support in postprocessing_filter_grid()
    ]
    filtered_configs = list(dict.fromkeys(filtered_configs))
    filter_results = evaluate_postprocessing_configs_streaming(
        record_factory,
        record_count,
        filtered_configs,
        f"{context} object-filter grid",
    )
    results_by_config = {
        tuple(sorted(row["config"].items())): row
        for row in reconstruction_results + filter_results
    }
    candidates = sorted(
        results_by_config.values(),
        key=lambda row: row["selection_score"],
        reverse=True,
    )
    baseline = results_by_config[
        tuple(sorted(asdict(DEFAULT_POSTPROCESS_CONFIG).items()))
    ]
    selected = candidates[0]
    metric_deltas = {
        key: float(selected["metrics"][key] - baseline["metrics"][key])
        for key in selected["metrics"]
    }
    return {
        "context": context,
        "role": role,
        "selection_split": (
            f"disjoint {role} role only; held-out test labels stayed sealed and unparsed"
        ),
        "selection_objective": (
            "0.25 foreground Dice + 0.15 reconstructed-boundary Dice + "
            "0.35 AP50 + 0.25 AP75"
        ),
        "boundary_cutoff_protocol": (
            "sweep 0.10 through 0.90 in 0.05 increments; normalized trapezoidal "
            "area selects robust foreground/area families, then the role-specific "
            "peak supplies the deployable cutoff"
        ),
        "boundary_cutoff_curves": cutoff_curves,
        "object_scores": {
            "mean_foreground_probability": "mean, not sum, to avoid favoring large cells",
            "core_fraction": (
                "fraction of object pixels with foreground probability >= 0.70"
            ),
            "boundary_support": (
                "available as a research diagnostic but excluded from v3 auto-selection"
            ),
        },
        "validation_images": record_count,
        "validation_domains": dict(sorted(validation_domains.items())),
        "domain_aggregation": (
            "metrics averaged within domain, then macro-averaged across domains"
        ),
        "streaming_evaluation": True,
        "candidate_count": len(candidates),
        "baseline": baseline,
        "selected": selected,
        "selected_minus_baseline": metric_deltas,
        "candidates": candidates,
    }


def evaluate_fused_probabilities(
    foreground: np.ndarray,
    boundary: np.ndarray,
    truth_labels: np.ndarray,
    postprocess_config: PostprocessConfig = DEFAULT_POSTPROCESS_CONFIG,
) -> dict[str, float]:
    prediction_labels = instance_labels(
        foreground,
        boundary,
        postprocess_config,
    )
    return segmentation_metrics(prediction_labels, truth_labels)


def average_metric_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([row[key] for row in rows]))
        for key in rows[0]
    }


def ensemble_validation_score(metrics: dict[str, float]) -> float:
    return postprocessing_selection_score(metrics)


@torch.inference_mode()
def evaluate_all_ensembles(
    specs: list[ModelSpec],
    checkpoints: dict[str, Path],
    device: torch.device,
    experiment_fingerprint: str,
    external_samples: list[object] | None = None,
    proxy_size: int = 256,
    finalist_count: int = 16,
    role_limit_per_domain: int | None = None,
) -> dict[str, object]:
    """Select ensembles with disk-backed maps and at most one CUDA model."""
    if len(specs) < 2:
        raise ValueError("Ensemble evaluation requires at least two semantic models.")
    if len({spec.name for spec in specs}) != len(specs):
        raise ValueError("Ensemble model names must be unique.")
    missing_checkpoints = [spec.name for spec in specs if spec.name not in checkpoints]
    if missing_checkpoints:
        raise ValueError(
            "Missing ensemble checkpoints: " + ", ".join(missing_checkpoints)
        )
    if finalist_count < 1:
        raise ValueError("finalist_count must be positive.")
    if role_limit_per_domain is not None and role_limit_per_domain < 1:
        raise ValueError("role_limit_per_domain must be positive when supplied.")
    common_size = max(spec.image_size for spec in specs)
    resolved_proxy_size = min(proxy_size, common_size)
    if resolved_proxy_size < 32:
        raise ValueError("proxy_size must resolve to at least 32 pixels.")

    validation_roles = ("calibration", "ensemble_selection")
    livecell_cocos = {
        source_key: COCO(
            str(ANNOTATIONS / f"livecell_coco_{source_key}.json")
        )
        for source_key in ("train", "val")
    }
    records: list[dict[str, object]] = []
    for role in validation_roles:
        role_records: list[dict[str, object]] = []
        for coco_source, coco in livecell_cocos.items():
            for image_id in sorted(coco.getImgIds()):
                metadata = coco.loadImgs([image_id])[0]
                file_name = str(metadata["file_name"])
                if livecell_role(file_name) != role:
                    continue
                source_identifier = (
                    f"livecell:{role}:{coco_source}:{image_id}:{file_name}"
                )
                role_records.append(
                    {
                        "kind": "livecell",
                        "role": role,
                        "coco_source": coco_source,
                        "image_id": image_id,
                        "file_name": file_name,
                        "domain": "livecell",
                        "source": source_identifier,
                        "cache_key": hashlib.sha256(
                            source_identifier.encode()
                        ).hexdigest(),
                    }
                )
        role_records.sort(key=lambda record: str(record["cache_key"]))
        if role_limit_per_domain is not None:
            role_records = role_records[:role_limit_per_domain]
        records.extend(role_records)
    if external_samples:
        from accuracy_data import load_external_sample

        external_validation = sorted(
            (
                sample
                for sample in external_samples
                if sample.split == "val"
                and validation_role(sample.dataset, sample.image_path)
                in validation_roles
            ),
            key=lambda sample: (sample.dataset, str(sample.image_path)),
        )
        external_domain_counts: dict[tuple[str, str], int] = defaultdict(int)
        for sample in external_validation:
            role = validation_role(sample.dataset, sample.image_path)
            domain_key = (role, sample.dataset)
            if (
                role_limit_per_domain is not None
                and external_domain_counts[domain_key]
                >= role_limit_per_domain
            ):
                continue
            external_domain_counts[domain_key] += 1
            source_identifier = (
                f"{role}:{sample.dataset}:{sample.image_path.resolve()}:"
                f"{sample.instance_path.resolve()}"
            )
            records.append(
                {
                    "kind": "external",
                    "role": role,
                    "sample": sample,
                    "domain": sample.dataset,
                    "source": source_identifier,
                    "cache_key": hashlib.sha256(
                        source_identifier.encode()
                    ).hexdigest(),
                }
            )
    else:
        load_external_sample = None
    records_by_role = {
        role: [record for record in records if record["role"] == role]
        for role in validation_roles
    }
    missing_roles = [
        role for role, role_records in records_by_role.items() if not role_records
    ]
    if missing_roles:
        raise RuntimeError(
            "No validation records were discovered for roles: "
            + ", ".join(missing_roles)
        )
    record_keys = [str(record["cache_key"]) for record in records]
    if len(set(record_keys)) != len(record_keys):
        raise RuntimeError("Duplicate ensemble cache keys were discovered.")

    checkpoint_digests = {
        spec.name: sha256(checkpoints[spec.name]) for spec in specs
    }
    cache_core: dict[str, object] = {
        "schema_version": 3,
        "evaluator_version": ENSEMBLE_EVALUATOR_VERSION,
        "experiment_fingerprint": experiment_fingerprint,
        "roles": list(validation_roles),
        "common_size": common_size,
        "model_specs": [asdict(spec) for spec in specs],
        "checkpoint_sha256": checkpoint_digests,
        "records": [
            {
                "cache_key": record["cache_key"],
                "role": record["role"],
                "domain": record["domain"],
                "source": record["source"],
            }
            for record in records
        ],
    }
    cache_fingerprint = hashlib.sha256(
        json.dumps(cache_core, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    cache_directory = (
        RUNS
        / "ensemble_probability_cache"
        / experiment_fingerprint[:16]
        / cache_fingerprint[:16]
    )
    cache_directory.mkdir(parents=True, exist_ok=True)
    cache_manifest_path = cache_directory / "manifest.json"
    if cache_manifest_path.is_file():
        existing_manifest = json.loads(cache_manifest_path.read_text())
        if existing_manifest.get("cache_fingerprint") != cache_fingerprint:
            raise RuntimeError(
                f"Ensemble cache fingerprint collision in {cache_directory}."
            )
    atomic_json_save(
        cache_manifest_path,
        {
            **cache_core,
            "cache_fingerprint": cache_fingerprint,
            "status": "building",
        },
    )

    def truth_cache_path(record: dict[str, object]) -> Path:
        return cache_directory / "truth" / f"{record['cache_key']}.npy"

    def probability_cache_path(
        spec: ModelSpec,
        record: dict[str, object],
    ) -> Path:
        return (
            cache_directory
            / "models"
            / spec.name
            / f"{record['cache_key']}.npy"
        )

    def load_record_image(record: dict[str, object]) -> np.ndarray:
        if record["kind"] == "livecell":
            image = cv2.imread(
                str(locate_image(str(record["file_name"]))),
                cv2.IMREAD_GRAYSCALE,
            )
            if image is None:
                raise RuntimeError(f"Could not decode {record['file_name']}")
            return image
        if load_external_sample is None:
            raise RuntimeError("External ensemble loader was not initialized.")
        image, _ = load_external_sample(record["sample"])
        return image

    def load_record_truth(record: dict[str, object]) -> np.ndarray:
        if record["kind"] == "livecell":
            return coco_instance_mask(
                livecell_cocos[str(record["coco_source"])],
                int(record["image_id"]),
                common_size,
            )
        if load_external_sample is None:
            raise RuntimeError("External ensemble loader was not initialized.")
        _, truth = load_external_sample(record["sample"])
        truth = cv2.resize(
            truth.astype(np.float32),
            (common_size, common_size),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32, copy=False)
        return truth

    for record in tqdm(records, desc="cache ensemble truths"):
        path = truth_cache_path(record)
        if cached_array_is_valid(
            path,
            (common_size, common_size),
            np.dtype(np.int32),
        ):
            continue
        truth = load_record_truth(record)
        atomic_numpy_save(path, truth.astype(np.int32, copy=False))

    # A model is moved to CUDA only while its missing maps are generated. All maps are returned to
    # CPU and atomically cached before the model is released, bounding residency to one model.
    for spec in specs:
        missing_records = [
            record
            for record in records
            if not cached_array_is_valid(
                probability_cache_path(spec, record),
                (2, common_size, common_size),
                np.dtype(np.float16),
            )
        ]
        if not missing_records:
            continue
        model: nn.Module | None = None
        try:
            checkpoint = torch.load(
                checkpoints[spec.name],
                map_location="cpu",
                weights_only=False,
            )
            model = build_model(spec)
            model.load_state_dict(checkpoint["model"])
            del checkpoint
            model = model.to(device).eval()
            for record in tqdm(
                missing_records,
                desc=f"cache ensemble probabilities {spec.name}",
            ):
                image = load_record_image(record)
                probabilities = ensemble_member_probabilities(
                    model,
                    spec,
                    image,
                    common_size,
                    device,
                )
                if not np.isfinite(probabilities).all():
                    raise RuntimeError(
                        f"{spec.name} produced non-finite ensemble probabilities."
                    )
                atomic_numpy_save(
                    probability_cache_path(spec, record),
                    probabilities.astype(np.float16),
                )
        finally:
            if model is not None:
                del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    atomic_json_save(
        cache_manifest_path,
        {
            **cache_core,
            "cache_fingerprint": cache_fingerprint,
            "status": "complete",
            "probability_dtype": "float16",
            "probability_file_count": len(records) * len(specs),
            "truth_file_count": len(records),
        },
    )

    selection_fingerprint = hashlib.sha256(
        json.dumps(
            {
                "evaluator_version": ENSEMBLE_EVALUATOR_VERSION,
                "cache_fingerprint": cache_fingerprint,
                "proxy_size": resolved_proxy_size,
                "finalist_count": min(
                    finalist_count,
                    (1 << len(specs)) - 1 - len(specs),
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    destination = RUNS / "ensemble_evaluation.json"
    if destination.is_file():
        try:
            existing_result = json.loads(destination.read_text())
            if existing_result.get("selection_fingerprint") == selection_fingerprint:
                return existing_result
        except json.JSONDecodeError:
            pass

    member_tuples, weight_tensor = combination_matrix(
        len(specs),
        torch.device("cpu"),
    )
    weights = weight_tensor.numpy()
    combination_count = len(member_tuples)
    proxy_foreground_by_domain: dict[str, np.ndarray] = {}
    proxy_boundary_by_domain: dict[str, np.ndarray] = {}
    proxy_domain_counts: dict[str, int] = defaultdict(int)
    combination_chunk_size = 32
    selection_records = records_by_role["ensemble_selection"]
    calibration_records = records_by_role["calibration"]
    for record in tqdm(
        selection_records,
        desc="score all ensemble combinations on ensemble_selection",
    ):
        predictions = np.empty(
            (len(specs), 2, resolved_proxy_size, resolved_proxy_size),
            dtype=np.float32,
        )
        for model_index, spec in enumerate(specs):
            full_probability = np.load(
                probability_cache_path(spec, record),
                mmap_mode="r",
                allow_pickle=False,
            )
            if resolved_proxy_size == common_size:
                predictions[model_index] = full_probability
            else:
                for channel in range(2):
                    predictions[model_index, channel] = cv2.resize(
                        np.asarray(full_probability[channel], dtype=np.float32),
                        (resolved_proxy_size, resolved_proxy_size),
                        interpolation=cv2.INTER_AREA,
                    )
        truth = np.load(
            truth_cache_path(record),
            mmap_mode="r",
            allow_pickle=False,
        )
        if resolved_proxy_size != common_size:
            truth = cv2.resize(
                np.asarray(truth, dtype=np.float32),
                (resolved_proxy_size, resolved_proxy_size),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)
        truth_targets, _ = labels_to_targets(truth)
        truth_foreground = truth_targets[..., 0].astype(bool)
        truth_boundary = truth_targets[..., 1].astype(bool)
        domain = str(record["domain"])
        if domain not in proxy_foreground_by_domain:
            proxy_foreground_by_domain[domain] = np.zeros(
                combination_count,
                dtype=np.float64,
            )
            proxy_boundary_by_domain[domain] = np.zeros(
                combination_count,
                dtype=np.float64,
            )
        for start in range(0, combination_count, combination_chunk_size):
            stop = min(start + combination_chunk_size, combination_count)
            fused = np.tensordot(
                weights[start:stop],
                predictions,
                axes=(1, 0),
            )
            predicted_foreground = fused[:, 0] >= 0.5
            predicted_boundary = fused[:, 1] >= 0.45
            foreground_intersection = (
                predicted_foreground & truth_foreground
            ).sum(axis=(1, 2))
            foreground_denominator = (
                predicted_foreground.sum(axis=(1, 2)) + truth_foreground.sum()
            )
            boundary_intersection = (
                predicted_boundary & truth_boundary
            ).sum(axis=(1, 2))
            boundary_denominator = (
                predicted_boundary.sum(axis=(1, 2)) + truth_boundary.sum()
            )
            proxy_foreground_by_domain[domain][start:stop] += (
                2 * foreground_intersection + 1
            ) / (foreground_denominator + 1)
            proxy_boundary_by_domain[domain][start:stop] += (
                2 * boundary_intersection + 1
            ) / (boundary_denominator + 1)
        proxy_domain_counts[domain] += 1

    proxy_foreground = np.stack(
        [
            values / proxy_domain_counts[domain]
            for domain, values in proxy_foreground_by_domain.items()
        ]
    ).mean(axis=0)
    proxy_boundary = np.stack(
        [
            values / proxy_domain_counts[domain]
            for domain, values in proxy_boundary_by_domain.items()
        ]
    ).mean(axis=0)
    proxy_score = 0.75 * proxy_foreground + 0.25 * proxy_boundary
    proxy_order = np.argsort(-proxy_score, kind="stable").tolist()
    # Swift fuses at the largest grid among the selected members. The full-grid calibration below
    # uses ``common_size``, so a deployable finalist must contain at least one model with that
    # native grid. Otherwise a 512/640-only subset would be selected at 768 here but run at a
    # smaller grid in the app. All subsets still retain proxy scores in the research record.
    deployment_eligible_indices = {
        index
        for index, members in enumerate(member_tuples)
        if len(members) >= 2
        and max(specs[member].image_size for member in members) == common_size
    }
    finalist_indices = [
        index for index in proxy_order if index in deployment_eligible_indices
    ][:finalist_count]
    if not finalist_indices:
        raise RuntimeError(
            "No multi-model ensemble contains the maximum deployment fusion grid."
        )

    model_names = [spec.name for spec in specs]
    calibration_domain_counts: dict[str, int] = defaultdict(int)
    for record in calibration_records:
        calibration_domain_counts[str(record["domain"])] += 1

    def make_fused_record_factory(
        selected_members: tuple[int, ...],
        source_records: list[dict[str, object]],
    ) -> Callable[
        [], Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, str]]
    ]:
        def factory() -> Iterator[
            tuple[np.ndarray, np.ndarray, np.ndarray, str]
        ]:
            for record in source_records:
                fused = np.zeros(
                    (2, common_size, common_size),
                    dtype=np.float32,
                )
                for member_index in selected_members:
                    member_probability = np.load(
                        probability_cache_path(specs[member_index], record),
                        mmap_mode="r",
                        allow_pickle=False,
                    )
                    np.add(fused, member_probability, out=fused)
                fused /= len(selected_members)
                truth = np.load(
                    truth_cache_path(record),
                    mmap_mode="r",
                    allow_pickle=False,
                )
                yield fused[0], fused[1], truth, str(record["domain"])

        return factory

    finalist_reports: dict[int, dict[str, object]] = {}
    for combination_index in finalist_indices:
        member_tuple = member_tuples[combination_index]
        member_names = [model_names[member] for member in member_tuple]
        context = "ensemble:" + "+".join(member_names)
        calibration_report = tune_postprocessing_streaming(
            make_fused_record_factory(member_tuple, calibration_records),
            len(calibration_records),
            dict(calibration_domain_counts),
            context,
            role="calibration",
        )
        frozen_postprocessing = PostprocessConfig(
            **calibration_report["selected"]["config"]
        )
        ensemble_selection_evaluation = (
            evaluate_postprocessing_configs_streaming(
                make_fused_record_factory(member_tuple, selection_records),
                len(selection_records),
                [frozen_postprocessing],
                f"{context} frozen ensemble_selection evaluation",
            )[0]
        )
        finalist_reports[combination_index] = {
            "calibration_search": calibration_report,
            "frozen_postprocessing": asdict(frozen_postprocessing),
            "ensemble_selection_evaluation": ensemble_selection_evaluation,
        }
    winner_index = max(
        finalist_indices,
        key=lambda index: finalist_reports[index][
            "ensemble_selection_evaluation"
        ]["selection_score"],
    )
    winner_report = finalist_reports[winner_index]
    winner_postprocessing = PostprocessConfig(
        **winner_report["frozen_postprocessing"]
    )

    winner_members = member_tuples[winner_index]

    exhaustive = []
    for index, member_tuple in enumerate(member_tuples):
        exhaustive.append(
            {
                "members": [model_names[member] for member in member_tuple],
                "ensemble_selection_proxy_foreground_dice": float(
                    proxy_foreground[index]
                ),
                "ensemble_selection_proxy_boundary_dice": float(
                    proxy_boundary[index]
                ),
                "ensemble_selection_proxy_score": float(proxy_score[index]),
            }
        )
    finalists = []
    for index in finalist_indices:
        report = finalist_reports[index]
        calibration_report = report["calibration_search"]
        selection_evaluation = report["ensemble_selection_evaluation"]
        finalists.append(
            {
                "members": [model_names[member] for member in member_tuples[index]],
                "calibration_baseline_metrics": calibration_report["baseline"][
                    "metrics"
                ],
                "calibration_selected_metrics": calibration_report["selected"][
                    "metrics"
                ],
                "calibration_selection_score": calibration_report["selected"][
                    "selection_score"
                ],
                "calibration_selected_minus_baseline": calibration_report[
                    "selected_minus_baseline"
                ],
                "frozen_postprocessing": report["frozen_postprocessing"],
                "ensemble_selection_metrics": selection_evaluation["metrics"],
                "ensemble_selection_per_domain_metrics": selection_evaluation[
                    "per_domain_metrics"
                ],
                "ensemble_selection_score": selection_evaluation[
                    "selection_score"
                ],
                "calibration_search": calibration_report,
            }
        )
    result = {
        "schema_version": 3,
        "evaluator_version": ENSEMBLE_EVALUATOR_VERSION,
        "role": "ensemble_selection",
        "method": (
            "disk-backed float16 member probability maps; equal-weight soft foreground "
            "and boundary averaging; deployment-matched reconstruction and confidence filtering"
        ),
        "selection_policy": (
            "all 2^N-1 non-empty semantic-model combinations receive foreground/boundary "
            f"proxy scores on ensemble_selection at {resolved_proxy_size}x"
            f"{resolved_proxy_size}; deployable finalists must include a {common_size}px member "
            "so Python and Swift use the same largest-member fusion grid; the top "
            f"{len(finalist_indices)} eligible multi-model candidates "
            "have post-processing tuned and frozen on calibration, then are evaluated and "
            f"ranked on ensemble_selection with that frozen configuration at the app's "
            f"{common_size}x{common_size} "
            "maximum-model fusion grid; held-out test labels stay sealed and unparsed"
        ),
        "role_separation": {
            "proxy_screening": "ensemble_selection",
            "postprocessing_tuning": "calibration",
            "membership_ranking": "ensemble_selection with frozen postprocessing",
            "final_test": "labels sealed, unparsed, and unevaluated",
        },
        "experiment_fingerprint": experiment_fingerprint,
        "cache_fingerprint": cache_fingerprint,
        "selection_fingerprint": selection_fingerprint,
        "probability_cache": {
            "directory": str(cache_directory.relative_to(RUNS)),
            "dtype": "float16",
            "model_cuda_residency_limit": 1,
            "simultaneous_full_grid_fused_records_in_ram": 1,
        },
        "proxy_evaluation_size": resolved_proxy_size,
        "common_evaluation_size": common_size,
        "deployment_fusion_grid": common_size,
        "models": model_names,
        "combination_count": len(member_tuples),
        "deployment_grid_eligible_combination_count": len(
            deployment_eligible_indices
        ),
        "finalist_count": len(finalist_indices),
        "calibration_domains": dict(sorted(calibration_domain_counts.items())),
        "ensemble_selection_domains": dict(sorted(proxy_domain_counts.items())),
        "validation_domains": dict(sorted(proxy_domain_counts.items())),
        "domain_aggregation": "metrics averaged within domain, then macro-averaged across domains",
        "all_validation_combinations": exhaustive,
        "finalists": finalists,
        "selected_members": [model_names[member] for member in winner_members],
        "selected_postprocessing": asdict(winner_postprocessing),
        "selected_calibration_metrics": winner_report["calibration_search"][
            "selected"
        ]["metrics"],
        "selected_calibration_score": winner_report["calibration_search"][
            "selected"
        ]["selection_score"],
        "selected_calibration_minus_baseline": winner_report[
            "calibration_search"
        ]["selected_minus_baseline"],
        "selected_validation_metrics": winner_report[
            "ensemble_selection_evaluation"
        ]["metrics"],
        "selected_validation_per_domain_metrics": winner_report[
            "ensemble_selection_evaluation"
        ]["per_domain_metrics"],
        "selected_validation_score": winner_report[
            "ensemble_selection_evaluation"
        ]["selection_score"],
        "held_out_test_status": (
            "labels sealed and unparsed; run only after deployment_lock_v3.json is written"
        ),
    }
    atomic_json_save(destination, result)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def experiment_fingerprint(dataset_fingerprint: str, mode_family: str) -> str:
    digest = hashlib.sha256()
    digest.update(BUNDLE_VERSION.encode())
    digest.update(dataset_fingerprint.encode())
    digest.update(mode_family.encode())
    for path in sorted(ROOT.glob("*.py")) + [ROOT / "requirements.txt", ROOT / "run.sh"]:
        digest.update(path.name.encode())
        digest.update(sha256(path).encode())
    return digest.hexdigest()


def environment_report() -> dict[str, object]:
    try:
        nvidia_smi = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=False
        ).stdout
    except FileNotFoundError:
        nvidia_smi = "nvidia-smi not found"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvidia_smi": nvidia_smi,
    }


def select_cellpose_checkpoint(
    output_directory: Path,
    evaluation_limit: int | None = None,
) -> dict[str, object]:
    selected_model = output_directory / "model.pt"
    report_path = output_directory / "checkpoint_selection.json"
    completed = json.loads((output_directory / "completed.json").read_text())
    tile_size = int(completed["tile_size"])
    data_root = Path(str(completed["data_root"])).resolve()
    checkpoint_pairs_directory = data_root / "checkpoint"
    if not checkpoint_pairs_directory.is_dir():
        raise RuntimeError(
            "Cellpose checkpoint-selection pairs are missing: "
            f"{checkpoint_pairs_directory}"
        )
    if selected_model.is_file() and report_path.is_file():
        existing_report = json.loads(report_path.read_text())
        if existing_report.get("selection_pairs_directory") == str(
            checkpoint_pairs_directory
        ):
            return existing_report
    candidates = sorted((output_directory / "checkpoints").glob("*.pt"))
    final_model = output_directory / "final_model.pt"
    if final_model.is_file():
        candidates.append(final_model)
    if not candidates and selected_model.is_file():
        # Compatibility with the earlier 100-epoch bundle.
        return {"selected_model": selected_model.name, "legacy_single_checkpoint": True}
    if not candidates:
        raise RuntimeError(f"No Cellpose checkpoints found in {output_directory}")
    selection_directory = output_directory / "checkpoint_validation"
    selection_directory.mkdir(exist_ok=True)
    rows = []
    for candidate in candidates:
        evaluation_path = selection_directory / f"{candidate.stem}.json"
        if not cellpose_evaluation_matches(
            evaluation_path,
            sha256(candidate),
            0.0,
            0.4,
            pairs_directory=checkpoint_pairs_directory,
            tile_size=tile_size,
        ):
            command = [
                    sys.executable,
                    str(ROOT / "evaluate_cellpose_sam.py"),
                    "--model",
                    str(candidate),
                    "--pairs-dir",
                    str(checkpoint_pairs_directory),
                    "--output",
                    str(evaluation_path),
                    "--tile-size",
                    str(tile_size),
                    "--tile-overlap",
                    "0.25",
                    "--tile-batch-size",
                    "1",
                ]
            if evaluation_limit is not None:
                command.extend(["--limit", str(evaluation_limit)])
            subprocess.run(
                command,
                check=True,
            )
        metrics = json.loads(evaluation_path.read_text())
        ap = metrics["instance_average_precision"]
        score = 0.6 * float(ap["0.5"]) + 0.4 * float(ap["0.75"])
        rows.append(
            {
                "checkpoint": str(candidate.relative_to(output_directory)),
                "selection_instance_score": score,
                "metrics": metrics,
            }
        )
    winner = max(rows, key=lambda row: row["selection_instance_score"])
    shutil.copy2(output_directory / winner["checkpoint"], selected_model)
    report = {
        "policy": (
            "Select saved snapshot on the materialized checkpoint role using "
            "0.6*AP50 + 0.4*AP75; held-out test is never used for selection."
        ),
        "selection_pairs_directory": str(checkpoint_pairs_directory),
        "candidates": rows,
        "selected_checkpoint": winner["checkpoint"],
        "selected_model": selected_model.name,
        "selected_score": winner["selection_instance_score"],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def tune_cellpose_inference_settings(
    output_directory: Path,
    evaluation_limit: int | None = None,
) -> dict[str, object]:
    """Select Cellpose mask-quality settings on validation without touching test."""
    model_path = output_directory / "model.pt"
    model_digest = sha256(model_path)
    completed = json.loads((output_directory / "completed.json").read_text())
    tile_size = int(completed["tile_size"])
    data_root = Path(str(completed["data_root"])).resolve()
    calibration_pairs_directory = data_root / "calibration"
    if not calibration_pairs_directory.is_dir():
        raise RuntimeError(
            "Cellpose calibration pairs are missing: "
            f"{calibration_pairs_directory}"
        )
    report_path = output_directory / "inference_search.json"
    if report_path.is_file():
        existing = json.loads(report_path.read_text())
        if (
            existing.get("model_sha256") == model_digest
            and existing.get("selection_pairs_directory")
            == str(calibration_pairs_directory)
        ):
            return existing
    search_directory = output_directory / "inference_search"
    search_directory.mkdir(exist_ok=True)
    candidates = []
    for cellprob_threshold in (-2.0, -1.0, 0.0, 1.0, 2.0):
        for flow_threshold in (0.2, 0.4, 0.6, 0.8):
            evaluation_path = search_directory / (
                f"cellprob_{cellprob_threshold:+.1f}_flow_{flow_threshold:.1f}.json"
            )
            if not cellpose_evaluation_matches(
                evaluation_path,
                model_digest,
                cellprob_threshold,
                flow_threshold,
                pairs_directory=calibration_pairs_directory,
                tile_size=tile_size,
            ):
                command = [
                        sys.executable,
                        str(ROOT / "evaluate_cellpose_sam.py"),
                        "--model",
                        str(model_path),
                        "--pairs-dir",
                        str(calibration_pairs_directory),
                        "--output",
                        str(evaluation_path),
                        "--cellprob-threshold",
                        str(cellprob_threshold),
                        "--flow-threshold",
                        str(flow_threshold),
                        "--tile-size",
                        str(tile_size),
                        "--tile-overlap",
                        "0.25",
                        "--tile-batch-size",
                        "1",
                    ]
                if evaluation_limit is not None:
                    command.extend(["--limit", str(evaluation_limit)])
                subprocess.run(
                    command,
                    check=True,
                )
            metrics = json.loads(evaluation_path.read_text())
            candidates.append(
                {
                    "cellprob_threshold": cellprob_threshold,
                    "flow_threshold": flow_threshold,
                    "selection_score": metrics["selection_score"],
                    "metrics": metrics,
                    "evaluation": str(evaluation_path.relative_to(output_directory)),
                }
            )
    candidates.sort(key=lambda row: row["selection_score"], reverse=True)
    baseline = next(
        row
        for row in candidates
        if row["cellprob_threshold"] == 0.0 and row["flow_threshold"] == 0.4
    )
    selected = candidates[0]
    metric_deltas = {
        key: float(selected["metrics"][key] - baseline["metrics"][key])
        for key in (
            "pixel_dice",
            "pixel_iou",
            "foreground_precision",
            "foreground_recall",
            "boundary_dice",
            "boundary_precision",
            "boundary_recall",
            "count_mae",
            "selection_score",
        )
    }
    metric_deltas["instance_ap50"] = float(
        selected["metrics"]["instance_average_precision"]["0.5"]
        - baseline["metrics"]["instance_average_precision"]["0.5"]
    )
    metric_deltas["instance_ap75"] = float(
        selected["metrics"]["instance_average_precision"]["0.75"]
        - baseline["metrics"]["instance_average_precision"]["0.75"]
    )
    report = {
        "model_sha256": model_digest,
        "selection_split": (
            "disjoint materialized calibration role across eligible datasets; "
            "held-out test not used"
        ),
        "selection_pairs_directory": str(calibration_pairs_directory),
        "selection_objective": (
            "0.25 foreground Dice + 0.15 reconstructed-boundary Dice + "
            "0.35 AP50 + 0.25 AP75"
        ),
        "candidate_count": len(candidates),
        "baseline": baseline,
        "selected": selected,
        "selected_minus_baseline": metric_deltas,
        "candidates": candidates,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def cellpose_evaluation_matches(
    path: Path,
    model_digest: str,
    cellprob_threshold: float,
    flow_threshold: float,
    pairs_directory: Path,
    tile_size: int | None = None,
    tile_overlap: float = 0.25,
    tile_batch_size: int = 1,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text())
        settings = payload["inference_settings"]
        recorded_pairs_directory = payload.get("pairs_directory")
        return (
            payload.get("model_sha256") == model_digest
            and float(settings["cellprob_threshold"]) == cellprob_threshold
            and float(settings["flow_threshold"]) == flow_threshold
            and payload.get("source_mode") == "materialized_pairs"
            and recorded_pairs_directory is not None
            and Path(str(recorded_pairs_directory)).resolve()
            == pairs_directory.resolve()
            and (tile_size is None or int(settings["tile_size"]) == tile_size)
            and float(settings["tile_overlap"]) == tile_overlap
            and int(settings["tile_batch_size"]) == tile_batch_size
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def package_results(summary: dict[str, object], best_model: str) -> Path:
    forbidden_test_reports = sorted(RUNS.rglob("test_evaluation.json"))
    if forbidden_test_reports:
        locations = ", ".join(str(path) for path in forbidden_test_reports)
        raise RuntimeError(
            "Refusing to package held-out test evaluation artifacts: " + locations
        )
    result_root = OUTPUT / "return_to_cellect"
    if result_root.exists():
        shutil.rmtree(result_root)
    result_root.mkdir(parents=True)
    (result_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (result_root / "README.txt").write_text(
        "Copy this entire ZIP to cellect/WorkstationResults/inbox/ on the Mac.\n"
        f"Automatically selected model: {best_model}\n"
    )
    for spec in ALL_MODEL_SPECS:
        source = RUNS / spec.name
        if not source.exists():
            continue
        target = result_root / spec.name
        target.mkdir()
        for pattern in (
            "best.pt",
            "history.csv",
            "postprocessing_search.json",
            "*.onnx",
            "*.torchscript.pt",
        ):
            for path in source.glob(pattern):
                shutil.copy2(path, target / path.name)
        if (source / "previews").exists():
            shutil.copytree(source / "previews", target / "previews")
        if (source / "deployment_parity").exists():
            shutil.copytree(
                source / "deployment_parity",
                target / "deployment_parity",
            )
    for base_model in FOUNDATION_MODELS:
        foundation_run_name = f"{base_model}_eukaryotic_v3_finetuned"
        foundation_source = RUNS / foundation_run_name
        if foundation_source.exists():
            foundation_target = result_root / foundation_run_name
            foundation_target.mkdir()
            for name in (
                "model.pt",
                "history.csv",
                "training.log",
                "completed.json",
                "checkpoint_selection.json",
                "inference_search.json",
                "val_evaluation.json",
                "calibration_evaluation_tuned_v3.json",
            ):
                source_path = foundation_source / name
                if source_path.is_file():
                    shutil.copy2(source_path, foundation_target / name)
    ensemble_report = RUNS / "ensemble_evaluation.json"
    if ensemble_report.is_file():
        shutil.copy2(ensemble_report, result_root / ensemble_report.name)
    for name in ("deployment_lock_v3.json",):
        source_path = RUNS / name
        if source_path.is_file():
            shutil.copy2(source_path, result_root / name)
    for name in ("preflight_v3.json", "splits_v3.jsonl"):
        source_path = OUTPUT / name
        if source_path.is_file():
            shutil.copy2(source_path, result_root / name)
    checksums = []
    for path in sorted(result_root.rglob("*")):
        if path.is_file():
            checksums.append(f"{sha256(path)}  {path.relative_to(result_root)}")
    (result_root / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n")
    destination = OUTPUT / "cellect_workstation_results.zip"
    if destination.exists():
        destination.unlink()
    shutil.make_archive(str(destination.with_suffix("")), "zip", result_root)
    return destination


def main() -> None:
    global RUNS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["smoke", "full", "preflight", "best-smoke", "best"],
        default="full",
    )
    args = parser.parse_args()
    seed_everything()
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is unavailable. Update the NVIDIA driver, then rerun; CPU training is disabled."
        )
    smoke = args.mode == "smoke"
    best_smoke = args.mode == "best-smoke"
    accuracy_mode = args.mode in {"preflight", "best-smoke", "best"}
    torch.backends.cudnn.benchmark = not accuracy_mode
    torch.backends.cudnn.deterministic = accuracy_mode
    if accuracy_mode:
        torch.use_deterministic_algorithms(True, warn_only=True)
    prepare_data()
    if smoke:
        training_config = replace(
            STANDARD_CONFIG,
            epochs=2,
            early_stopping_patience=2,
        )
    elif best_smoke:
        training_config = replace(
            ACCURACY_CONFIG,
            epochs=1,
            early_stopping_patience=1,
            accumulation_steps=1,
            validation_instance_samples=2,
        )
    elif accuracy_mode:
        training_config = ACCURACY_CONFIG
    else:
        training_config = STANDARD_CONFIG
    limit = 64 if smoke else (3 if best_smoke else None)
    workers = (
        2
        if best_smoke
        else min(6, max(2, (os.cpu_count() or 4) - 2))
    )
    device = torch.device("cuda")
    external_samples = []
    external_manifest: dict[str, object] = {}
    optional_specialist_manifest: dict[str, object] = {}
    gated_dataset_manifest: dict[str, object] = {}
    if accuracy_mode:
        from accuracy_data import (
            DATASET_MANIFEST,
            GATED_DATASET_MANIFEST,
            OPTIONAL_SPECIALIST_DATASET_MANIFEST,
            prepare_external_datasets,
        )

        external_samples = prepare_external_datasets(DATA)
        from data_preflight import run_full_data_preflight

        preflight_report = run_full_data_preflight(
            ANNOTATIONS,
            IMAGES_ROOT,
            external_samples,
            OUTPUT / "preflight_v3.json",
        )
        dataset_fingerprint = str(preflight_report["dataset_fingerprint_sha256"])
        stage_fingerprint = experiment_fingerprint(
            dataset_fingerprint,
            "best-v3",
        )
        if args.mode == "preflight":
            print(json.dumps(preflight_report, indent=2))
            return
        experiment_kind = "best-smoke" if best_smoke else "best"
        RUNS = OUTPUT / "experiments" / experiment_kind / stage_fingerprint[:16] / "runs"
        if args.mode == "best":
            smoke_marker = OUTPUT / "best_smoke_pass_v3.json"
            if not smoke_marker.is_file():
                raise RuntimeError(
                    "Run ./run.sh best-smoke successfully before starting ./run.sh best."
                )
            marker = json.loads(smoke_marker.read_text())
            if (
                marker.get("bundle_version") != BUNDLE_VERSION
                or marker.get("dataset_fingerprint_sha256") != dataset_fingerprint
                or marker.get("stage_fingerprint") != stage_fingerprint
            ):
                raise RuntimeError(
                    "The best-smoke PASS marker does not match this v3 code/data fingerprint; "
                    "rerun ./run.sh best-smoke."
                )
        split_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for sample in external_samples:
            split_counts[sample.dataset][sample.split] += 1
        external_manifest = {
            name: {
                **details,
                "discovered_image_mask_pairs": dict(split_counts.get(name, {})),
            }
            for name, details in {**DATASET_MANIFEST, **GATED_DATASET_MANIFEST}.items()
        }
        optional_specialist_manifest = OPTIONAL_SPECIALIST_DATASET_MANIFEST
        gated_dataset_manifest = GATED_DATASET_MANIFEST
    else:
        stage_fingerprint = experiment_fingerprint("livecell-only", args.mode)
        RUNS = OUTPUT / "experiments" / args.mode / stage_fingerprint[:16] / "runs"
    summary: dict[str, object] = {
        "schema_version": 3,
        "bundle_version": BUNDLE_VERSION,
        "experiment_fingerprint": stage_fingerprint,
        "mode": args.mode,
        "environment": environment_report(),
        "configuration": {
            **asdict(training_config),
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau",
            "encoder_initialization": "ImageNet",
            "mixed_precision": "CUDA float16 autocast with GradScaler",
            "foreground_loss": "focal BCE + soft Dice + asymmetric Tversky",
            "boundary_loss": "positive-weighted focal BCE + soft Dice",
            "outputs": ["foreground_logit", "boundary_logit"],
            "boundary_target_version": BOUNDARY_TARGET_VERSION,
            "deployable_postprocessor": POSTPROCESS_VERSION,
            "tiled_inference_version": TILED_INFERENCE_VERSION,
            "selection_metric": (
                "0.35 foreground Dice + 0.15 boundary Dice + "
                "0.30 instance AP50 + 0.20 instance AP75"
            ),
            "checkpoint_domain_aggregation": (
                "per-image metrics averaged within dataset, then equally across datasets"
            ),
            "postprocessing_selection_metric": (
                "macro-domain 0.25 foreground Dice + 0.15 reconstructed-boundary Dice + "
                "0.35 instance AP50 + 0.25 instance AP75; validation only"
            ),
            "postprocessing_search": (
                "foreground threshold, boundary cutoff, scale-normalized minimum area, "
                "mean object foreground probability, and high-confidence core fraction"
            ),
            "dataset_sampling": (
                "per-image inverse square-root domain frequency; all yeast sources share "
                "one domain capped at 10% expected training probability"
                if accuracy_mode
                else "uniform image sampling"
            ),
            "input_scaling": (
                "grayscale float32 [0,1]; 35% of augmented LIVECell crops and all "
                "non-8-bit sources receive 0.5/99.5-percentile contrast scaling"
            ),
        },
        "datasets": {
            "livecell": {
                "license": "CC BY-NC 4.0",
                "role": (
                    "upstream train+val acquisitions are deterministically assigned to "
                    "train/checkpoint/calibration/ensemble_selection; upstream test labels "
                    "remain sealed, unparsed, and unevaluated"
                ),
            },
            **external_manifest,
        },
        "excluded_from_primary_training": optional_specialist_manifest,
        "permission_confirmed_but_local_archive_required": gated_dataset_manifest,
        "models": {},
    }
    if accuracy_mode:
        summary["data_preflight"] = {
            "report": str((OUTPUT / "preflight_v3.json").relative_to(ROOT)),
            "dataset_fingerprint_sha256": preflight_report[
                "dataset_fingerprint_sha256"
            ],
            "split_manifest": preflight_report["split_manifest"],
            "split_manifest_sha256": preflight_report["split_manifest_sha256"],
            "status": preflight_report["status"],
        }
    if accuracy_mode:
        from accuracy_data import materialize_cellpose_pairs

        # Version the materialized directory so an interrupted older run that included bacterial
        # or uncapped yeast images cannot silently contaminate this eukaryotic profile.
        cellpose_data_root = DATA / (
            f"cellpose_v3_{'smoke' if best_smoke else 'best'}_"
            f"{stage_fingerprint[:12]}"
        )
        livecell_cellpose_counts = materialize_livecell_for_cellpose(
            cellpose_data_root,
            limit_per_role=3 if best_smoke else None,
        )
        external_cellpose_counts = materialize_cellpose_pairs(
            external_samples,
            cellpose_data_root,
            limit_per_dataset_role=1 if best_smoke else None,
        )
        summary["foundation_models"] = {}
        for base_model in FOUNDATION_MODELS:
            foundation_output = RUNS / f"{base_model}_eukaryotic_v3_finetuned"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "train_cellpose_sam.py"),
                    "--data-root",
                    str(cellpose_data_root),
                    "--output-dir",
                    str(foundation_output),
                    "--base-model",
                    base_model,
                    "--epochs",
                    "1" if best_smoke else "300",
                    "--stage-fingerprint",
                    stage_fingerprint,
                ],
                check=True,
            )
            checkpoint_selection = select_cellpose_checkpoint(
                foundation_output,
                evaluation_limit=3 if best_smoke else None,
            )
            inference_search = tune_cellpose_inference_settings(
                foundation_output,
                evaluation_limit=3 if best_smoke else None,
            )
            selected_inference = inference_search["selected"]
            selected_cellprob = float(selected_inference["cellprob_threshold"])
            selected_flow = float(selected_inference["flow_threshold"])
            calibration_evaluation_path = (
                foundation_output / "calibration_evaluation_tuned_v3.json"
            )
            selected_validation_path = foundation_output / selected_inference[
                "evaluation"
            ]
            shutil.copy2(selected_validation_path, calibration_evaluation_path)
            summary["foundation_models"][base_model] = {
                "status": f"fine-tuned from {base_model}",
                "official_recipe": {
                    "epoch_ceiling": 1 if best_smoke else 300,
                    "learning_rate": 1e-5,
                    "weight_decay": 0.1,
                    "train_batch_size": 1,
                    "tile_size": 384 if base_model.startswith("cpdino") else 256,
                    "minimum_masks_per_image": 1,
                    "snapshot_interval": 25,
                },
                "checkpoint_selection": checkpoint_selection,
                "inference_search": inference_search,
                "selected_inference_settings": {
                    "cellprob_threshold": selected_cellprob,
                    "flow_threshold": selected_flow,
                },
                "livecell_pairs": livecell_cellpose_counts,
                "external_pairs": external_cellpose_counts,
                "calibration_metrics": json.loads(
                    calibration_evaluation_path.read_text()
                ),
                "held_out_test_status": "labels sealed and unparsed before deployment freeze",
                "deployment_status": (
                    "research-only until Core ML conversion and physical-iPhone parity pass"
                ),
                "output": str(foundation_output.relative_to(OUTPUT)),
                "history": str(
                    (foundation_output / "history.csv").relative_to(OUTPUT)
                ),
            }
        # Backwards-compatible key used by the Mac importer and older analysis notebooks.
        summary["cellpose_sam"] = summary["foundation_models"]["cpsam_v2"]
    if smoke:
        specs = MODEL_SPECS[:1]
    elif accuracy_mode:
        specs = ACCURACY_MODEL_SPECS
    else:
        specs = MODEL_SPECS
    semantic_checkpoints: dict[str, Path] = {}
    for spec in specs:
        requested_batch_size = 4 if smoke else spec.batch_size
        batch_size, model_training_config, memory_probe = probe_training_microbatch(
            spec,
            requested_batch_size,
            training_config,
            device,
        )
        livecell_annotations = (
            ANNOTATIONS / "livecell_coco_train.json",
            ANNOTATIONS / "livecell_coco_val.json",
        )
        livecell_train_set = LiveCellDataset(
            livecell_annotations,
            spec.image_size,
            True,
            limit,
            accuracy_augmentations=training_config.use_accuracy_augmentations,
            role_name="train",
        )
        livecell_val_set = LiveCellDataset(
            livecell_annotations,
            spec.image_size,
            False,
            limit,
            role_name="checkpoint",
        )
        livecell_calibration_set = LiveCellDataset(
            livecell_annotations,
            spec.image_size,
            False,
            limit,
            role_name="calibration",
        )
        livecell_selection_set = LiveCellDataset(
            livecell_annotations,
            spec.image_size,
            False,
            limit,
            role_name="ensemble_selection",
        )
        external_train_images = 0
        unique_external_train_images = 0
        training_sampling: dict[str, object] = {}
        train_set: Dataset = livecell_train_set
        val_set: Dataset = livecell_val_set
        calibration_set: Dataset = livecell_calibration_set
        selection_set: Dataset = livecell_selection_set
        train_sampler = None
        if accuracy_mode:
            from accuracy_data import ExternalCellDataset

            external_train_set = ExternalCellDataset(
                external_samples,
                split="train",
                image_size=spec.image_size,
                train=True,
                transform=training_transform(spec.image_size, accuracy=True),
                limit=limit,
            )
            unique_external_train_images = len(external_train_set)
            external_train_images = unique_external_train_images
            train_set = ConcatDataset([livecell_train_set, external_train_set])
            train_sampler, training_sampling = modality_balanced_sampler(
                len(livecell_train_set),
                external_train_set.samples,
            )
            external_val_set = ExternalCellDataset(
                external_samples,
                split="val",
                image_size=spec.image_size,
                train=False,
                transform=None,
                limit=limit,
                validation_role="checkpoint",
            )
            val_set = ConcatDataset([livecell_val_set, external_val_set])
            external_calibration_set = ExternalCellDataset(
                external_samples,
                split="val",
                image_size=spec.image_size,
                train=False,
                transform=None,
                limit=limit,
                validation_role="calibration",
            )
            calibration_set = ConcatDataset(
                [livecell_calibration_set, external_calibration_set]
            )
            external_selection_set = ExternalCellDataset(
                external_samples,
                split="val",
                image_size=spec.image_size,
                train=False,
                transform=None,
                limit=limit,
                validation_role="ensemble_selection",
            )
            selection_set = ConcatDataset(
                [livecell_selection_set, external_selection_set]
            )
        train_loader_args = {
            "batch_size": batch_size,
            "num_workers": workers,
            "pin_memory": True,
            "persistent_workers": workers > 0,
            "prefetch_factor": 1,
        }
        evaluation_loader_args = {
            "batch_size": min(2, batch_size),
            "num_workers": min(2, workers),
            "pin_memory": True,
            "persistent_workers": False,
            "prefetch_factor": 1,
        }
        train_loader = DataLoader(
            train_set,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            drop_last=True,
            **train_loader_args,
        )
        val_loader = DataLoader(val_set, shuffle=False, **evaluation_loader_args)
        calibration_loader = DataLoader(
            calibration_set,
            shuffle=False,
            **evaluation_loader_args,
        )
        selection_loader = DataLoader(
            selection_set,
            shuffle=False,
            **evaluation_loader_args,
        )
        checkpoint, history = train_model(
            spec,
            train_loader,
            val_loader,
            device,
            model_training_config,
            stage_fingerprint,
        )
        semantic_checkpoints[spec.name] = checkpoint
        postprocess_report: dict[str, object] | None = None
        postprocess_config = DEFAULT_POSTPROCESS_CONFIG
        if accuracy_mode:
            postprocess_report = tune_semantic_postprocessing(
                spec,
                checkpoint,
                calibration_loader,
                device,
            )
            postprocess_config = PostprocessConfig(
                **postprocess_report["selected"]["config"]
            )
        metrics = evaluate_model(
            spec,
            checkpoint,
            calibration_loader,
            device,
            evaluation_name="calibration",
            postprocess_config=postprocess_config,
        )
        selection_metrics = evaluate_model(
            spec,
            checkpoint,
            selection_loader,
            device,
            evaluation_name="ensemble_selection",
            postprocess_config=postprocess_config,
        )
        selection_score = postprocessing_selection_score(
            selection_metrics["postprocessing_selection_score_components"]
        )
        onnx_path, torchscript_path = export_models(spec, checkpoint)
        from deployment_parity import verify_deployment_artifacts

        parity_paths = livecell_calibration_set.image_paths(limit=3)
        parity_model = build_model(spec)
        parity_checkpoint = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        parity_model.load_state_dict(parity_checkpoint["model"])
        del parity_checkpoint
        parity_report = verify_deployment_artifacts(
            parity_model,
            parity_paths,
            spec.image_size,
            torchscript_path,
            onnx_path,
            RUNS / spec.name / "deployment_parity",
            device,
            postprocess_config=postprocess_config,
            postprocess_callback=instance_labels,
        )
        del parity_model
        torch.cuda.empty_cache()
        summary["models"][spec.name] = {
            "spec": asdict(spec),
            "train_images": len(train_set),
            "livecell_train_images": len(livecell_train_set),
            "external_train_images": external_train_images,
            "unique_external_train_images": unique_external_train_images,
            "training_sampling": training_sampling,
            "training_memory_probe": memory_probe,
            "effective_training_config": asdict(model_training_config),
            "val_images": len(val_set),
            "calibration_images": len(calibration_set),
            "ensemble_selection_images": len(selection_set),
            "held_out_test_status": "labels sealed and unparsed before deployment freeze",
            "best_validation_dice": max(row["val_dice"] for row in history),
            "best_validation_selection_score": max(
                row["val_selection_score"] for row in history
            ),
            "calibration_metrics": metrics,
            "ensemble_selection_metrics": selection_metrics,
            "ensemble_selection_score": selection_score,
            "postprocessing_search": postprocess_report,
            "selected_postprocessing": asdict(postprocess_config),
            "onnx": str(onnx_path.relative_to(OUTPUT)),
            "onnx_sha256": sha256(onnx_path),
            "torchscript": str(torchscript_path.relative_to(OUTPUT)),
            "torchscript_sha256": sha256(torchscript_path),
            "deployment_parity": parity_report,
        }
    if accuracy_mode:
        summary["ensemble_evaluation"] = evaluate_all_ensembles(
            specs,
            semantic_checkpoints,
            device,
            stage_fingerprint,
            external_samples=external_samples,
            proxy_size=128 if best_smoke else 256,
            finalist_count=2 if best_smoke else 16,
            role_limit_per_domain=1 if best_smoke else None,
        )
    best_model = max(
        summary["models"],
        key=lambda name: summary["models"][name]["ensemble_selection_score"],
    )
    summary["selected_model"] = best_model
    if accuracy_mode:
        selected_model_summary = summary["models"][best_model]
        deployment_lock = {
            "schema_version": 3,
            "bundle_version": BUNDLE_VERSION,
            "experiment_fingerprint": stage_fingerprint,
            "split_manifest_sha256": preflight_report["split_manifest_sha256"],
            "selection_roles": {
                "checkpoint": "checkpoint validation only",
                "postprocessing": "calibration only",
                "model_and_ensemble": "ensemble_selection only",
                "final_test": "labels sealed and unparsed during fitting, calibration, and selection",
            },
            "single_model": {
                "name": best_model,
                "checkpoint_sha256": sha256(semantic_checkpoints[best_model]),
                "torchscript_sha256": selected_model_summary["torchscript_sha256"],
                "onnx_sha256": selected_model_summary["onnx_sha256"],
                "postprocessing": selected_model_summary["selected_postprocessing"],
                "selection_score": selected_model_summary[
                    "ensemble_selection_score"
                ],
            },
            "semantic_ensemble": {
                "members": summary["ensemble_evaluation"]["selected_members"],
                "merge": "equal-weight probability mean",
                "grid": "largest selected model grid; bilinear align_corners=False",
                "postprocessing": summary["ensemble_evaluation"][
                    "selected_postprocessing"
                ],
                "selection_score": summary["ensemble_evaluation"][
                    "selected_validation_score"
                ],
            },
            "boundary_target_version": BOUNDARY_TARGET_VERSION,
            "postprocess_version": POSTPROCESS_VERSION,
            "tiled_inference_version": TILED_INFERENCE_VERSION,
            "coreml_and_physical_iphone_parity_required": True,
            "foundation_models": (
                "research-only until each model passes Core ML conversion and physical-device parity"
            ),
        }
        lock_path = RUNS / "deployment_lock_v3.json"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_temporary = lock_path.with_suffix(".json.tmp")
        lock_temporary.write_text(json.dumps(deployment_lock, indent=2) + "\n")
        lock_temporary.replace(lock_path)
        deployment_lock["lock_sha256"] = sha256(lock_path)
        summary["deployment_lock"] = deployment_lock
        summary["held_out_test_status"] = (
            "labels sealed and unparsed; evaluate frozen candidates only after Mac "
            "Core ML/Swift parity passes"
        )
    archive = package_results(summary, best_model)
    if best_smoke:
        smoke_marker = OUTPUT / "best_smoke_pass_v3.json"
        smoke_payload = {
            "status": "PASS",
            "bundle_version": BUNDLE_VERSION,
            "dataset_fingerprint_sha256": preflight_report[
                "dataset_fingerprint_sha256"
            ],
            "stage_fingerprint": stage_fingerprint,
            "archive": str(archive),
            "completed_unix_seconds": time.time(),
        }
        smoke_temporary = smoke_marker.with_suffix(".json.tmp")
        smoke_temporary.write_text(json.dumps(smoke_payload, indent=2) + "\n")
        smoke_temporary.replace(smoke_marker)
    print(json.dumps(summary, indent=2))
    print(f"\nReturn archive: {archive}")


if __name__ == "__main__":
    main()
