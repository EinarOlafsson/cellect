#!/usr/bin/env python3
"""Convert every workstation-trained candidate to reproducible Core ML packages.

Run this after ``import_results.py`` with a Python environment containing
numpy<2, torch 2.2.x, and coremltools 7.2. Conversion is intentionally separate
from training: it runs on the Mac so that the generated packages can be loaded
and numerically checked by the local Core ML runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path

import coremltools as ct
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
IMPORTED = ROOT / "WorkstationResults" / "imported"
DEFAULT_OUTPUT = ROOT / "Cellect" / "Resources" / "Models"
REPORT = ROOT / "WorkstationResults" / "coreml_evaluation.json"


@dataclass(frozen=True)
class Candidate:
    source_name: str
    resource_name: str
    tier: str
    use_float16: bool = True
    backend: str = "mlprogram"


# Keep the full nine-model workstation comparison available in the development
# app. The UI still identifies the recommended accuracy/size frontier, but no
# trained candidate is hidden from on-device testing.
CANDIDATES = (
    Candidate(
        "mobilenetv3_small_unet",
        "CellectMobileNetV3Small",
        "low",
        False,
    ),
    Candidate(
        "mobilenetv3_large_deeplab",
        "CellectMobileNetV3Large",
        "low",
        False,
    ),
    Candidate(
        "efficientnet_b0_unet",
        "CellectEfficientNetB0",
        "medium",
    ),
    Candidate(
        "resnet18_unet",
        "CellectResNet18",
        "medium",
    ),
    Candidate(
        "efficientnet_b3_unetpp",
        "CellectEfficientNetB3",
        "best",
    ),
    Candidate(
        "resnet50_deeplab",
        "CellectResNet50DeepLab",
        "high",
    ),
    Candidate(
        "resnet101_unetpp",
        "CellectResNet101",
        "best",
    ),
    Candidate(
        "segformer_b2",
        "CellectSegFormerB2",
        "high",
        False,
        "neuralnetwork",
    ),
    Candidate(
        "segformer_b5",
        "CellectSegFormerB5",
        "best",
        False,
        "neuralnetwork",
    ),
    Candidate(
        "efficientnet_b3_unetpp_accuracy",
        "CellectEfficientNetB3Accuracy",
        "best",
    ),
    Candidate(
        "resnet101_unetpp_accuracy",
        "CellectResNet101Accuracy",
        "best",
    ),
    Candidate(
        "segformer_b5_accuracy",
        "CellectSegFormerB5Accuracy",
        "best",
        False,
        "neuralnetwork",
    ),
    Candidate(
        "mobilenetv3_small_unet_accuracy_v2",
        "CellectMobileNetV3SmallAccuracyV2",
        "low",
        False,
    ),
    Candidate(
        "mobilenetv3_large_deeplab_accuracy_v2",
        "CellectMobileNetV3LargeAccuracyV2",
        "low",
        False,
    ),
    Candidate(
        "efficientnet_b0_unet_accuracy_v2",
        "CellectEfficientNetB0AccuracyV2",
        "medium",
    ),
    Candidate(
        "resnet18_unet_accuracy_v2",
        "CellectResNet18AccuracyV2",
        "medium",
    ),
    Candidate(
        "efficientnet_b3_unetpp_accuracy_v2",
        "CellectEfficientNetB3AccuracyV2",
        "best",
    ),
    Candidate(
        "resnet50_deeplab_accuracy_v2",
        "CellectResNet50DeepLabAccuracyV2",
        "high",
    ),
    Candidate(
        "resnet101_unetpp_accuracy_v2",
        "CellectResNet101AccuracyV2",
        "best",
    ),
    Candidate(
        "segformer_b2_accuracy_v2",
        "CellectSegFormerB2AccuracyV2",
        "high",
        False,
        "neuralnetwork",
    ),
    Candidate(
        "segformer_b5_accuracy_v2",
        "CellectSegFormerB5AccuracyV2",
        "best",
        False,
        "neuralnetwork",
    ),
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def representative_input(image_size: int) -> np.ndarray:
    """Return a deterministic grayscale-like tensor in the training range."""
    y, x = np.mgrid[0:image_size, 0:image_size]
    gray = (
        0.45
        + 0.22 * np.sin(x / 17.0)
        + 0.18 * np.cos(y / 29.0)
        + 0.08 * np.sin((x + y) / 7.0)
    )
    gray = np.clip(gray, 0.0, 1.0).astype(np.float32)
    return np.repeat(gray[None, None, :, :], 3, axis=1)


def convert_candidate(
    candidate: Candidate,
    summary: dict[str, object],
    output_dir: Path,
    benchmark_runs: int,
) -> dict[str, object]:
    details = summary["models"][candidate.source_name]
    image_size = int(details["spec"]["image_size"])
    source_dir = IMPORTED / candidate.source_name
    source = next(source_dir.glob("*.torchscript.pt"))
    expected = details["torchscript_sha256"]
    actual = sha256(source)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {source}")

    print(f"Loading {candidate.source_name}")
    torch_model = torch.jit.load(str(source), map_location="cpu").eval()
    sample = representative_input(image_size)

    print(f"Converting {candidate.source_name} -> {candidate.resource_name}.mlpackage")
    started = time.perf_counter()
    output_dtype = np.float16 if candidate.use_float16 else np.float32
    conversion_arguments = {
        "convert_to": candidate.backend,
        "inputs": [
            ct.TensorType(
                name="image",
                shape=(1, 3, image_size, image_size),
                dtype=np.float32,
            )
        ],
        "outputs": [
            ct.TensorType(
                name="segmentation_logits",
                dtype=output_dtype,
            )
        ],
        "minimum_deployment_target": (
            ct.target.iOS17
            if candidate.backend == "mlprogram"
            else ct.target.iOS14
        ),
    }
    if candidate.backend == "mlprogram":
        conversion_arguments["compute_precision"] = (
            ct.precision.FLOAT16
            if candidate.use_float16
            else ct.precision.FLOAT32
        )
    coreml_model = ct.convert(torch_model, **conversion_arguments)
    conversion_seconds = time.perf_counter() - started

    coreml_model.author = "Cellect workstation training pipeline"
    coreml_model.license = (
        "Derived from LIVECell (CC BY-NC 4.0); review licensing before distribution."
    )
    coreml_model.short_description = (
        "Two-channel LIVECell foreground and cell-boundary logits."
    )
    coreml_model.version = "1"
    coreml_model.user_defined_metadata["cellect_source_model"] = candidate.source_name
    coreml_model.user_defined_metadata["cellect_tier"] = candidate.tier
    coreml_model.user_defined_metadata["cellect_input"] = (
        f"1x3x{image_size}x{image_size} grayscale repeated as RGB, float32 [0,1]"
    )
    coreml_model.user_defined_metadata["cellect_output"] = (
        f"1x2x{image_size}x{image_size}: foreground logit, boundary logit"
    )
    selected_postprocessing = details.get("selected_postprocessing", {})
    coreml_model.user_defined_metadata["cellect_postprocessing"] = json.dumps(
        selected_postprocessing,
        sort_keys=True,
        separators=(",", ":"),
    )
    coreml_model.user_defined_metadata["source_torchscript_sha256"] = actual

    destination = output_dir / f"{candidate.resource_name}.mlpackage"
    if destination.exists():
        shutil.rmtree(destination)
    coreml_model.save(str(destination))

    print(f"Validating {candidate.resource_name} with Core ML")
    with torch.inference_mode():
        torch_output = torch_model(torch.from_numpy(sample)).cpu().numpy()
    first_started = time.perf_counter()
    prediction = coreml_model.predict({"image": sample})["segmentation_logits"]
    first_prediction_seconds = time.perf_counter() - first_started
    differences = np.abs(torch_output - prediction.astype(np.float32))
    parity_mean_absolute_error = float(differences.mean())
    parity_max_absolute_error = float(differences.max())
    if parity_mean_absolute_error > 0.1 or parity_max_absolute_error > 1.0:
        shutil.rmtree(destination)
        if candidate.use_float16:
            print(
                f"Float16 parity was unsafe for {candidate.source_name}; "
                "retrying at Float32."
            )
            return convert_candidate(
                replace(candidate, use_float16=False),
                summary,
                output_dir,
                benchmark_runs,
            )
        raise RuntimeError(
            f"Unsafe Core ML conversion for {candidate.source_name}: "
            f"mean error {parity_mean_absolute_error:.4f}, "
            f"max error {parity_max_absolute_error:.4f}"
        )

    timings = []
    for _ in range(benchmark_runs):
        run_started = time.perf_counter()
        coreml_model.predict({"image": sample})
        timings.append(time.perf_counter() - run_started)

    metrics = details["metrics"]
    return {
        "source_model": candidate.source_name,
        "resource_name": candidate.resource_name,
        "tier": candidate.tier,
        "image_size": image_size,
        "test_dice": metrics["test_dice"],
        "test_iou": metrics["test_iou"],
        "count_mae": metrics["count_mae"],
        "selected_postprocessing": selected_postprocessing,
        "parameters": metrics["checkpoint_parameters"],
        "compute_precision": "float16" if candidate.use_float16 else "float32",
        "coreml_backend": candidate.backend,
        "coreml_package_bytes": directory_size(destination),
        "conversion_seconds": conversion_seconds,
        "first_prediction_seconds": first_prediction_seconds,
        "median_prediction_seconds": statistics.median(timings),
        "parity_mean_absolute_error": parity_mean_absolute_error,
        "parity_max_absolute_error": parity_max_absolute_error,
        "torchscript_sha256": actual,
        "coreml_specification_version": coreml_model.get_spec().specificationVersion,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination for generated .mlpackage directories.",
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=3,
        help="Number of warm Core ML predictions per model.",
    )
    parser.add_argument(
        "--model",
        action="append",
        choices=[candidate.source_name for candidate in CANDIDATES],
        help="Convert only this model; repeat to select several.",
    )
    args = parser.parse_args()
    if args.benchmark_runs < 1:
        parser.error("--benchmark-runs must be at least 1")
    if not (IMPORTED / "summary.json").is_file():
        raise SystemExit("Run WorkstationResults/import_results.py first.")

    summary = json.loads((IMPORTED / "summary.json").read_text())
    if summary.get("mode") not in {"full", "best"}:
        raise SystemExit("Refusing to convert smoke-run results.")
    selected = set(args.model or ())
    candidates = [
        candidate
        for candidate in CANDIDATES
        if candidate.source_name in summary["models"]
        and (not selected or candidate.source_name in selected)
    ]
    missing = selected - set(summary["models"])
    if missing:
        raise SystemExit(
            "Requested models are absent from this result archive: "
            + ", ".join(sorted(missing))
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    results = [
        convert_candidate(candidate, summary, output_dir, args.benchmark_runs)
        for candidate in candidates
    ]
    report = {
        "workstation_selected_model": summary["selected_model"],
        "selection_policy": (
            "Convert every exportable candidate returned by the workstation run; "
            "retain only conversions that pass numerical parity checks."
        ),
        "conversion_environment": {
            "torch": torch.__version__,
            "coremltools": ct.__version__,
            "numpy": np.__version__,
        },
        "models": results,
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {REPORT}")
    for result in results:
        print(
            f"{result['resource_name']}: Dice={result['test_dice']:.4f}, "
            f"median={result['median_prediction_seconds']:.3f}s, "
            f"size={result['coreml_package_bytes'] / 1024 / 1024:.1f} MiB"
        )


if __name__ == "__main__":
    main()
