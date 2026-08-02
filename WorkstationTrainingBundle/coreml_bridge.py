#!/usr/bin/env python3
"""Convert on Linux and validate Core ML packages on macOS without TorchScript drift.

PyTorch 2.7 TorchScript is converted beside training.  The result archive carries
portable golden NumPy fixtures and hash-bound manifests, so an Intel Mac can run
Core ML parity with its native runtime without loading the newer TorchScript.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import statistics
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

# Keep this module importable for its pure manifest/hash contract tests on hosts
# that do not have Core ML Tools or PyTorch installed. Runtime dependencies are
# loaded explicitly by ``_require_conversion_runtime``.
ct = None
torch = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = ROOT / "WorkstationResults" / "imported"
DEFAULT_OUTPUT = ROOT / "Cellect" / "Resources" / "Models"
DEFAULT_REPORT = ROOT / "WorkstationResults" / "coreml_evaluation.json"
DEFAULT_ARTIFACT_DIRECTORY_NAME = "coreml_artifacts"
PACKAGES_DIRECTORY_NAME = "packages"
FIXTURES_DIRECTORY_NAME = "golden_fixtures"
MANIFESTS_DIRECTORY_NAME = "manifests"
CATALOG_FILENAME = "CellectModelCatalog.json"
CATALOG_SCHEMA_VERSION = 1
BRIDGE_SCHEMA_VERSION = 1
PARITY_MEAN_ABSOLUTE_ERROR_LIMIT = 0.05
PARITY_MAX_ABSOLUTE_ERROR_LIMIT = 0.5
SEGMENTATION_PROBABILITY_MEAN_ERROR_LIMIT = 0.005
SEGMENTATION_PROBABILITY_MAX_ERROR_LIMIT = 0.05
TRACKING_PROBABILITY_MEAN_ERROR_LIMIT = 0.002
TRACKING_PROBABILITY_MAX_ERROR_LIMIT = 0.02
DECISION_MISMATCH_FRACTION_LIMIT = 0.0001


def _require_torch_runtime() -> object:
    """Load PyTorch only for Linux conversion or an explicitly chosen legacy fallback."""

    global torch
    if torch is None:
        try:
            import torch as imported_torch
        except ImportError as error:  # pragma: no cover - dependency error path
            raise RuntimeError(
                "PyTorch is required for Core ML conversion or legacy fallback."
            ) from error
        torch = imported_torch
    return torch


def _require_coreml_runtime() -> object:
    """Load Core ML Tools without explicitly importing PyTorch on the Mac path."""

    global ct
    if ct is None:
        try:
            import coremltools as imported_coremltools
        except ImportError as error:  # pragma: no cover - dependency error path
            raise RuntimeError(
                "coremltools 9.0 is required for Core ML conversion/validation."
            ) from error
        ct = imported_coremltools
    return ct


def _require_conversion_runtime() -> tuple[object, object]:
    """Load both runtimes only where TorchScript conversion is actually requested."""

    return _require_coreml_runtime(), _require_torch_runtime()


@dataclass(frozen=True)
class Candidate:
    source_name: str
    resource_name: str
    tier: str
    use_float16: bool = True
    backend: str = "mlprogram"
    summary_section: str = "models"
    artifact_variant: str | None = None
    output_name: str = "segmentation_logits"
    output_channels: int = 2
    output_semantics: tuple[str, ...] = (
        "foreground_logit",
        "fused_boundary_logit",
    )


@dataclass(frozen=True)
class TrackingCandidate:
    source_name: str
    resource_name: str
    model_name: str
    manifest_path: Path
    use_float16: bool = True


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


V4_EXTENDED_OUTPUT_NAME = "extended_segmentation_logits"
V4_EXTENDED_OUTPUT_SEMANTICS = (
    "foreground_logit",
    "fused_boundary_logit",
    "context_boundary_logit",
    "flow_boundary_logit",
    "shape_boundary_logit",
)
TRACKING_OUTPUT_NAMES = (
    "association_logits",
    "association_mask",
    "division_probability",
    "birth_probability",
    "death_probability",
    "uncertainty",
)


def _v4_resource_name(source_name: str) -> str:
    """Derive a stable Swift/Core ML resource name for a manifest-defined v4 model."""

    special_tokens = {
        "v4": "V4",
        "mobilenetv3": "MobileNetV3",
        "segformer": "SegFormer",
        "unet": "UNet",
    }
    tokens = [token for token in re.split(r"[^A-Za-z0-9]+", source_name) if token]
    rendered = "".join(
        special_tokens.get(token.casefold(), token[:1].upper() + token[1:])
        for token in tokens
    )
    return f"CellectV4{rendered}"


def v4_candidates(summary: Mapping[str, object]) -> tuple[Candidate, ...]:
    """Create candidates from ``summary['cellect_v4']['models']`` when present."""

    section = summary.get("cellect_v4")
    if section is None:
        return ()
    if not isinstance(section, Mapping):
        raise ValueError("summary['cellect_v4'] must be an object")
    models = section.get("models")
    if models is None:
        return ()
    if not isinstance(models, Mapping):
        raise ValueError("summary['cellect_v4']['models'] must be an object")
    candidates: list[Candidate] = []
    for source_name, raw_details in sorted(models.items()):
        if not isinstance(source_name, str) or not isinstance(raw_details, Mapping):
            raise ValueError("Every cellect_v4 model requires a string name and object details")
        spec = raw_details.get("spec", {})
        if not isinstance(spec, Mapping):
            raise ValueError(f"cellect_v4 model {source_name!r} has an invalid spec")
        configured_resource = raw_details.get("coreml_resource_name")
        resource_name = (
            str(configured_resource)
            if isinstance(configured_resource, str) and configured_resource.strip()
            else _v4_resource_name(source_name)
        )
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", resource_name) is None:
            raise ValueError(
                f"cellect_v4 model {source_name!r} has an unsafe Core ML resource name"
            )
        candidates.append(
            Candidate(
                source_name=source_name,
                resource_name=resource_name,
                tier=str(spec.get("tier", "v4")),
                summary_section="cellect_v4",
                artifact_variant="extended",
                output_name=V4_EXTENDED_OUTPUT_NAME,
                output_channels=5,
                output_semantics=V4_EXTENDED_OUTPUT_SEMANTICS,
            )
        )
    return tuple(candidates)


# CellectTrack is intentionally a separate conversion path: its three fixed inputs and six outputs
# must never be mistaken for the image-segmentation contract above. Tracking manifests are
# self-identifying and pin the artifact digest, fixed shapes, feature order/normalization, output
# order, calibrated thresholds, and eager/TorchScript parity.


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def safe_relative_path(root: Path, relative: str, *, kind: str) -> Path:
    """Resolve an archive-declared path without permitting traversal or symlinks."""

    declared = Path(relative)
    if declared.is_absolute() or not relative or any(part == ".." for part in declared.parts):
        raise ValueError(f"Unsafe {kind} path: {relative!r}")
    root = root.resolve()
    resolved = (root / declared).resolve()
    if not _inside_root(resolved, root):
        raise ValueError(f"{kind.capitalize()} path escaped its artifact root: {relative!r}")
    current = root
    for part in declared.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{kind.capitalize()} path contains a symlink: {relative!r}")
    return resolved


def package_tree_sha256(path: Path) -> str:
    """Hash package bytes and canonical POSIX paths; reject symlink ambiguity."""

    root = path.resolve()
    if not root.is_dir() or path.is_symlink():
        raise ValueError(f"Core ML package is not a regular directory: {path}")
    files: list[Path] = []
    for entry in root.rglob("*"):
        if entry.is_symlink():
            raise ValueError(f"Core ML package contains a symlink: {entry}")
        if entry.is_file():
            files.append(entry)
        elif not entry.is_dir():
            raise ValueError(f"Core ML package contains a non-regular entry: {entry}")
    if not files:
        raise ValueError(f"Core ML package contains no files: {path}")
    digest = hashlib.sha256()
    for entry in sorted(files, key=lambda item: item.relative_to(root).as_posix()):
        relative = entry.relative_to(root).as_posix().encode("utf-8")
        content_digest = sha256(entry).encode("ascii")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(entry.stat().st_size.to_bytes(8, "big"))
        digest.update(content_digest)
    return digest.hexdigest()


def _relative_to(path: Path, root: Path, *, kind: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"{kind.capitalize()} must be inside {root}: {path}") from error


def _array_contract(value: np.ndarray, fixture_key: str, logical_name: str) -> dict[str, object]:
    return {
        "name": logical_name,
        "fixture_key": fixture_key,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as destination:
            np.savez(destination, **arrays)
            destination.flush()
            os.fsync(destination.fileno())
            temporary_path = Path(destination.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_golden_contract(
    *,
    artifact_root: Path,
    fixture_dir: Path,
    manifest_dir: Path,
    source_root: Path,
    source_path: Path,
    source_digest: str,
    package_path: Path,
    candidate_kind: str,
    source_model: str,
    resource_name: str,
    precision: str,
    backend: str,
    inputs: Mapping[str, np.ndarray],
    outputs: Mapping[str, np.ndarray],
    conversion_result: Mapping[str, object],
    extra_identity: Mapping[str, object] | None = None,
    contract_label: str | None = None,
) -> tuple[Path, Path, dict[str, object]]:
    """Write a non-pickle fixture and hash-bound manifest beside a package."""

    if candidate_kind not in {"segmentation", "tracking"}:
        raise ValueError(f"Unknown candidate kind: {candidate_kind!r}")
    artifact_root = artifact_root.resolve()
    fixture_dir = fixture_dir.resolve()
    manifest_dir = manifest_dir.resolve()
    _relative_to(fixture_dir, artifact_root, kind="fixture directory")
    _relative_to(manifest_dir, artifact_root, kind="manifest directory")
    package_relative = _relative_to(package_path, artifact_root, kind="package")
    if contract_label is not None and re.fullmatch(r"[a-z0-9_-]+", contract_label) is None:
        raise ValueError(f"Unsafe golden-contract label: {contract_label!r}")
    suffix = f".{contract_label}" if contract_label else ""
    fixture_path = fixture_dir / f"{resource_name}{suffix}.npz"
    manifest_path = manifest_dir / f"{resource_name}{suffix}.json"
    fixture_arrays: dict[str, np.ndarray] = {}
    input_contract: list[dict[str, object]] = []
    output_contract: list[dict[str, object]] = []
    for logical_name, value in inputs.items():
        array = np.asarray(value)
        if array.dtype.hasobject or not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"Golden input {logical_name} must be a numeric array")
        if not np.isfinite(array).all():
            raise ValueError(f"Golden input {logical_name} contains non-finite values")
        key = f"input__{logical_name}"
        fixture_arrays[key] = array
        input_contract.append(_array_contract(array, key, logical_name))
    for logical_name, value in outputs.items():
        array = np.asarray(value)
        if array.dtype.hasobject or not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"Golden output {logical_name} must be a numeric array")
        if not np.isfinite(array).all():
            raise ValueError(f"Golden output {logical_name} contains non-finite values")
        key = f"output__{logical_name}"
        fixture_arrays[key] = array
        output_contract.append(_array_contract(array, key, logical_name))
    _atomic_npz(fixture_path, fixture_arrays)
    environment = {
        "torch": str(torch.__version__),
        "coremltools": str(ct.__version__),
        "numpy": str(np.__version__),
    }
    manifest: dict[str, object] = {
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "candidate_kind": candidate_kind,
        "candidate_identity": {
            "source_model": source_model,
            "resource_name": resource_name,
            **dict(extra_identity or {}),
        },
        "source": {
            "path": _relative_to(source_path, source_root, kind="source TorchScript"),
            "sha256": source_digest,
        },
        "package": {
            "path": package_relative,
            "tree_sha256": package_tree_sha256(package_path),
            "bytes": directory_size(package_path),
        },
        "fixture": {
            "path": _relative_to(fixture_path, artifact_root, kind="golden fixture"),
            "sha256": sha256(fixture_path),
            "inputs": input_contract,
            "outputs": output_contract,
            "format": "npz-numeric-arrays-no-pickle",
        },
        "conversion": {
            "environment": environment,
            "precision": precision,
            "backend": backend,
            "runtime_prediction_performed": False,
        },
        "parity_tolerances": {
            "raw_mean_absolute_error": PARITY_MEAN_ABSOLUTE_ERROR_LIMIT,
            "raw_max_absolute_error": PARITY_MAX_ABSOLUTE_ERROR_LIMIT,
            "segmentation_probability_mean_absolute_error": (
                SEGMENTATION_PROBABILITY_MEAN_ERROR_LIMIT
            ),
            "segmentation_probability_max_absolute_error": (
                SEGMENTATION_PROBABILITY_MAX_ERROR_LIMIT
            ),
            "tracking_probability_mean_absolute_error": (
                TRACKING_PROBABILITY_MEAN_ERROR_LIMIT
            ),
            "tracking_probability_max_absolute_error": (
                TRACKING_PROBABILITY_MAX_ERROR_LIMIT
            ),
            "decision_mismatch_fraction": DECISION_MISMATCH_FRACTION_LIMIT,
        },
        "conversion_result": dict(conversion_result),
    }
    _atomic_json(manifest_path, manifest)
    return fixture_path, manifest_path, manifest


def candidate_details(
    candidate: Candidate,
    summary: Mapping[str, object],
) -> Mapping[str, object]:
    if candidate.summary_section == "models":
        models = summary.get("models")
    elif candidate.summary_section == "cellect_v4":
        section = summary.get("cellect_v4")
        if not isinstance(section, Mapping):
            raise ValueError("cellect_v4 summary section is missing or invalid")
        models = section.get("models")
    else:
        raise ValueError(f"Unknown candidate summary section: {candidate.summary_section}")
    if not isinstance(models, Mapping):
        raise ValueError(f"Summary section for {candidate.source_name} has no models object")
    details = models.get(candidate.source_name)
    if not isinstance(details, Mapping):
        raise ValueError(f"Summary has no details for {candidate.source_name}")
    return details


def _artifact_metadata(
    candidate: Candidate,
    details: Mapping[str, object],
) -> Mapping[str, object]:
    if candidate.artifact_variant is None:
        return details
    exports = details.get("exports")
    if not isinstance(exports, Mapping):
        raise ValueError(f"{candidate.source_name} has no exports manifest")
    artifact = exports.get(candidate.artifact_variant)
    if not isinstance(artifact, Mapping):
        raise ValueError(
            f"{candidate.source_name} has no {candidate.artifact_variant!r} export"
        )
    recorded_output_name = artifact.get("output_name")
    if recorded_output_name != candidate.output_name:
        raise ValueError(
            f"{candidate.source_name} export names {recorded_output_name!r}; "
            f"expected {candidate.output_name!r}"
        )
    output_shape = artifact.get("output_shape")
    if not (
        isinstance(output_shape, list)
        and len(output_shape) == 4
        and int(output_shape[1]) == candidate.output_channels
    ):
        raise ValueError(
            f"{candidate.source_name} has an invalid {candidate.output_channels}-channel "
            f"output shape: {output_shape!r}"
        )
    semantics = artifact.get("semantics")
    if semantics is not None and tuple(semantics) != candidate.output_semantics:
        raise ValueError(
            f"{candidate.source_name} extended-output semantics changed: {semantics!r}"
        )
    return artifact


def _inside_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def resolve_torchscript_artifact(
    candidate: Candidate,
    artifact: Mapping[str, object],
    source_root: Path,
) -> tuple[Path, str]:
    declared = artifact.get("torchscript")
    expected = artifact.get("torchscript_sha256")
    if not isinstance(declared, str) or not declared:
        raise ValueError(f"{candidate.source_name} has no TorchScript artifact path")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"{candidate.source_name} has no valid TorchScript SHA-256")

    declared_path = Path(declared)
    basename = declared_path.name
    possible: list[Path] = []
    if not declared_path.is_absolute():
        relative_candidate = source_root / declared_path
        if _inside_root(relative_candidate, source_root) and relative_candidate.is_file():
            possible.append(relative_candidate)
    for directory in (
        source_root / candidate.source_name,
        source_root / "cellect_v4" / candidate.source_name,
        source_root / "cellect_v4_shape" / candidate.source_name,
    ):
        if directory.is_dir():
            possible.extend(directory.rglob(basename))
    if not possible:
        possible.extend(source_root.rglob(basename))
    unique = sorted({path.resolve() for path in possible if path.is_file()})
    matching = [path for path in unique if sha256(path) == expected]
    if not matching:
        locations = ", ".join(str(path) for path in unique) or "none"
        raise RuntimeError(
            f"No imported TorchScript artifact for {candidate.source_name} matches "
            f"SHA-256 {expected}; basename candidates: {locations}"
        )
    if len(matching) > 1:
        raise RuntimeError(
            f"Ambiguous imported TorchScript artifact for {candidate.source_name}: "
            + ", ".join(str(path) for path in matching)
        )
    return matching[0], expected


def _tracking_manifest(path: Path, source_root: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not parse tracking manifest {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"Tracking manifest root is not an object: {path}")
    outputs = value.get("outputs")
    if not isinstance(outputs, list) or tuple(outputs) != TRACKING_OUTPUT_NAMES:
        raise ValueError(f"Tracking output contract changed in {path}: {outputs!r}")
    fixed = value.get("fixed_input_contract")
    if not isinstance(fixed, Mapping) or set(fixed) != {
        "time_yx",
        "cell_features",
        "cell_mask",
    }:
        raise ValueError(f"Tracking fixed-input contract is invalid in {path}")
    expected_ranks = {"time_yx": 3, "cell_features": 3, "cell_mask": 2}
    for name, rank in expected_ranks.items():
        item = fixed.get(name)
        if not isinstance(item, Mapping):
            raise ValueError(f"Tracking input {name} is invalid in {path}")
        shape = item.get("shape")
        if not (
            isinstance(shape, list)
            and len(shape) == rank
            and all(isinstance(value, int) and value > 0 for value in shape)
            and shape[0] == 1
            and item.get("dtype") == "float32"
        ):
            raise ValueError(f"Tracking input {name} shape/dtype is invalid in {path}")
    time_shape = fixed["time_yx"]["shape"]
    feature_shape = fixed["cell_features"]["shape"]
    mask_shape = fixed["cell_mask"]["shape"]
    if not (
        time_shape[1] == feature_shape[1] == mask_shape[1]
        and time_shape[2] == 3
    ):
        raise ValueError(f"Tracking token dimensions disagree in {path}")
    ordered_names = fixed["cell_features"].get("ordered_names")
    if not isinstance(ordered_names, list) or len(ordered_names) != feature_shape[2]:
        raise ValueError(f"Tracking feature-name contract is invalid in {path}")
    artifacts = value.get("artifacts")
    torchscript = artifacts.get("torchscript") if isinstance(artifacts, Mapping) else None
    if not isinstance(torchscript, Mapping):
        raise ValueError(f"Tracking manifest has no TorchScript artifact: {path}")
    relative = torchscript.get("path")
    expected_digest = torchscript.get("sha256")
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"Tracking TorchScript path is invalid in {path}")
    if not isinstance(expected_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_digest
    ):
        raise ValueError(f"Tracking TorchScript digest is invalid in {path}")
    artifact_path = (path.parent / relative).resolve()
    if not _inside_root(artifact_path, source_root) or not artifact_path.is_file():
        raise ValueError(f"Tracking TorchScript is missing or escaped imported results: {path}")
    if sha256(artifact_path) != expected_digest:
        raise ValueError(f"Tracking TorchScript SHA-256 mismatch: {artifact_path}")
    return value


def discover_tracking_candidates(source_root: Path) -> tuple[TrackingCandidate, ...]:
    """Discover only self-identifying CellectTrack export manifests in imported results."""

    candidates: list[TrackingCandidate] = []
    for manifest_path in sorted(source_root.rglob("manifest.json")):
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, Mapping):
            continue
        if not (
            isinstance(raw.get("fixed_input_contract"), Mapping)
            and isinstance(raw.get("artifacts"), Mapping)
            and raw.get("outputs") is not None
            and raw.get("model_name") is not None
        ):
            continue
        manifest = _tracking_manifest(manifest_path, source_root)
        model_name = str(manifest["model_name"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", model_name):
            raise ValueError(f"Unsafe tracking model name in {manifest_path}: {model_name!r}")
        rendered = "".join(part.capitalize() for part in re.split(r"[_-]+", model_name))
        candidates.append(
            TrackingCandidate(
                source_name=f"cellect_track_{model_name}_v4",
                resource_name=f"CellectTrack{rendered}V4",
                model_name=model_name,
                manifest_path=manifest_path.resolve(),
            )
        )
    names = [candidate.source_name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate CellectTrack model manifests were returned")
    return tuple(candidates)


def validate_tracking_export_inventory(
    source_root: Path,
    tracking_exports: Mapping[str, object],
    standalone_candidates: Sequence[TrackingCandidate],
) -> dict[str, object]:
    """Validate standalone tracker artifacts and non-standalone composite manifests.

    A probability ensemble is a runtime composition of already exported models, not a third
    TorchScript/Core ML model.  It must therefore remain in the returned scientific/deployment
    record without entering the standalone conversion candidate list.  This check binds every
    summary entry to exactly one on-disk manifest and binds each composite member reference to the
    hash-verified standalone manifest it names.
    """

    root = source_root.resolve()
    if not tracking_exports:
        raise ValueError("CellectTrack exports cannot be empty")
    exports: dict[str, Mapping[str, object]] = {}
    standalone_names: set[str] = set()
    composite_names: set[str] = set()
    for raw_name, raw_export in tracking_exports.items():
        name = str(raw_name)
        if not isinstance(raw_export, Mapping):
            raise ValueError(f"CellectTrack export {name!r} is not an object")
        if raw_export.get("model_name") != name:
            raise ValueError(f"CellectTrack export key/model_name disagree for {name!r}")
        standalone = raw_export.get("standalone_model_artifact", True)
        if not isinstance(standalone, bool):
            raise ValueError(
                f"CellectTrack export {name!r} has a non-boolean standalone flag"
            )
        digest = raw_export.get("manifest_sha256")
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError(f"CellectTrack export {name!r} has no valid manifest digest")
        exports[name] = raw_export
        (standalone_names if standalone else composite_names).add(name)

    candidate_by_name = {candidate.model_name: candidate for candidate in standalone_candidates}
    if len(candidate_by_name) != len(standalone_candidates):
        raise ValueError("Duplicate standalone CellectTrack candidates were discovered")
    if standalone_names != set(candidate_by_name):
        raise ValueError(
            "CellectTrack standalone exports/manifests disagree: "
            f"expected={sorted(standalone_names)}, discovered={sorted(candidate_by_name)}"
        )

    manifest_matches: dict[str, list[tuple[Path, Mapping[str, object]]]] = {
        name: [] for name in exports
    }
    for manifest_path in sorted(root.rglob("manifest.json")):
        try:
            raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw_manifest, Mapping):
            continue
        model_name = raw_manifest.get("model_name")
        if model_name not in manifest_matches:
            continue
        if manifest_path.is_symlink():
            raise ValueError(f"CellectTrack manifest cannot be a symlink: {manifest_path}")
        manifest_matches[str(model_name)].append((manifest_path.resolve(), raw_manifest))

    manifests: dict[str, tuple[Path, Mapping[str, object]]] = {}
    for name, matches in manifest_matches.items():
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one CellectTrack manifest for {name!r}, found {len(matches)}"
            )
        manifest_path, manifest = matches[0]
        if not _inside_root(manifest_path, root):
            raise ValueError(f"CellectTrack manifest escaped imported results: {manifest_path}")
        declared_digest = str(exports[name]["manifest_sha256"])
        if sha256(manifest_path) != declared_digest:
            raise ValueError(f"CellectTrack manifest SHA-256 mismatch for {name!r}")
        summary_manifest = dict(exports[name])
        summary_manifest.pop("manifest_sha256", None)
        if summary_manifest != dict(manifest):
            raise ValueError(
                f"CellectTrack summary and on-disk manifest disagree for {name!r}"
            )
        manifests[name] = (manifest_path, manifest)

    for name, candidate in candidate_by_name.items():
        if candidate.manifest_path.resolve() != manifests[name][0]:
            raise ValueError(f"Standalone CellectTrack candidate points at the wrong {name!r} manifest")

    composite_reports: list[dict[str, object]] = []
    for name in sorted(composite_names):
        manifest_path, manifest = manifests[name]
        if manifest.get("standalone_model_artifact") is not False:
            raise ValueError(f"Composite CellectTrack export {name!r} lost its non-standalone flag")
        if manifest.get("artifact_type") != "composite_probability_ensemble":
            raise ValueError(f"Unsupported CellectTrack composite type for {name!r}")
        if "artifacts" in manifest:
            raise ValueError(f"Composite CellectTrack export {name!r} must not claim model artifacts")
        if manifest.get("member_export_parity_required") is not True:
            raise ValueError(f"Composite CellectTrack export {name!r} does not require member parity")
        members = manifest.get("members")
        if not isinstance(members, Mapping) or not members:
            raise ValueError(f"Composite CellectTrack export {name!r} has no members")
        member_names = {str(member) for member in members}
        if not member_names.issubset(standalone_names):
            raise ValueError(
                f"Composite CellectTrack export {name!r} references non-standalone members"
            )
        composite_run = manifest.get("run_fingerprint")
        composite_normalization = manifest.get("normalization_fingerprint")
        for member_name in sorted(member_names):
            member_reference = members[member_name]
            if not isinstance(member_reference, Mapping):
                raise ValueError(
                    f"Composite CellectTrack member {member_name!r} is not an object"
                )
            expected_relative = f"../{member_name}/manifest.json"
            if member_reference.get("manifest_relative_path") != expected_relative:
                raise ValueError(
                    f"Composite CellectTrack member {member_name!r} has an unsafe manifest path"
                )
            member_path, member_manifest = manifests[member_name]
            referenced_path = (manifest_path.parent / expected_relative).resolve()
            if referenced_path != member_path:
                raise ValueError(
                    f"Composite CellectTrack member {member_name!r} resolves to the wrong manifest"
                )
            member_digest = str(exports[member_name]["manifest_sha256"])
            if (
                member_reference.get("manifest_sha256") != member_digest
                or sha256(member_path) != member_digest
            ):
                raise ValueError(
                    f"Composite CellectTrack member {member_name!r} manifest digest changed"
                )
            if member_reference.get("model_contract") != member_manifest.get("model_contract"):
                raise ValueError(
                    f"Composite CellectTrack member {member_name!r} model contract changed"
                )
            if member_manifest.get("run_fingerprint") != composite_run:
                raise ValueError(
                    f"Composite CellectTrack member {member_name!r} comes from another run"
                )
            parity = member_manifest.get("parity")
            if not isinstance(parity, Mapping) or parity.get("status") != "PASS":
                raise ValueError(
                    f"Composite CellectTrack member {member_name!r} lacks export parity"
                )
            fixed = member_manifest.get("fixed_input_contract")
            features = fixed.get("cell_features") if isinstance(fixed, Mapping) else None
            if (
                not isinstance(features, Mapping)
                or features.get("normalization_fingerprint") != composite_normalization
            ):
                raise ValueError(
                    f"Composite CellectTrack member {member_name!r} normalization changed"
                )
        if not isinstance(manifest.get("runtime_contract"), Mapping):
            raise ValueError(f"Composite CellectTrack export {name!r} has no runtime contract")
        if not isinstance(manifest.get("frozen_thresholds"), Mapping):
            raise ValueError(f"Composite CellectTrack export {name!r} has no frozen thresholds")
        composite_reports.append(
            {
                "model_name": name,
                "artifact_type": str(manifest["artifact_type"]),
                "manifest_path": manifest_path.relative_to(root).as_posix(),
                "manifest_sha256": str(exports[name]["manifest_sha256"]),
                "members": sorted(member_names),
                "standalone_model_artifact": False,
            }
        )

    return {
        "standalone_models": sorted(standalone_names),
        "composite_exports": composite_reports,
    }


def _candidate_postprocessing(
    candidate: Candidate,
    details: Mapping[str, object],
    summary: Mapping[str, object],
) -> object:
    direct = details.get("selected_postprocessing")
    if direct is not None:
        return direct
    if candidate.summary_section == "cellect_v4":
        # Boundary operating points are selected independently for each frozen
        # student.  Keep a top-level fallback for development archives made
        # before that report was moved beside the model it evaluates.
        search = details.get("boundary_search")
        if not isinstance(search, Mapping):
            section = summary.get("cellect_v4")
            search = section.get("boundary_search") if isinstance(section, Mapping) else None
        if isinstance(search, Mapping):
            selected = search.get("selected_candidate")
            if isinstance(selected, Mapping) and selected.get("config") is not None:
                return selected["config"]
        calibration = details.get("calibration")
        if isinstance(calibration, Mapping):
            return calibration.get("selected_thresholds", {})
    return {}


def _candidate_validation_metrics(
    candidate: Candidate,
    details: Mapping[str, object],
) -> tuple[Mapping[str, object], float | None, str]:
    if candidate.summary_section == "models":
        selection_metrics = details.get("ensemble_selection_metrics")
        if isinstance(selection_metrics, Mapping):
            raw_score = details.get("ensemble_selection_score")
            score = float(raw_score) if raw_score is not None else None
            return selection_metrics, score, "ensemble_selection"
        metrics = details.get("metrics", {})
        if not isinstance(metrics, Mapping):
            raise ValueError(f"{candidate.source_name} metrics must be an object")
        return metrics, None, "held_out_test"
    selection = details.get("ensemble_selection", {})
    if not isinstance(selection, Mapping):
        raise ValueError(f"{candidate.source_name} ensemble_selection must be an object")
    metrics = selection.get("macro_domain_metrics", {})
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{candidate.source_name} macro-domain metrics must be an object")
    score = selection.get("selection_score")
    return metrics, float(score) if score is not None else None, "ensemble_selection"


def _candidate_export_version(candidate: Candidate) -> str:
    if candidate.summary_section == "cellect_v4":
        return "4"
    if candidate.source_name.endswith(("_accuracy", "_accuracy_v2")):
        return "2"
    return "1"


def _catalog_tier(value: str) -> str:
    """Map training labels onto the five tiers understood by the iOS app."""

    normalized = value.strip().casefold().replace("-", "_")
    aliases = {
        "classical": "classical",
        "light": "low",
        "mobile": "low",
        "low": "low",
        "balanced": "medium",
        "medium": "medium",
        "accurate": "high",
        "high": "high",
        "best": "best",
        "high_accuracy": "best",
        "largest": "best",
    }
    try:
        return aliases[normalized]
    except KeyError as error:
        raise ValueError(f"Unsupported app model tier {value!r}") from error


def _architecture_display(spec: Mapping[str, object], *, is_v4: bool) -> str:
    raw = str(spec.get("architecture", "Unknown"))
    architecture = {
        "unet": "U-Net",
        "unetplusplus": "U-Net++",
        "deeplabv3plus": "DeepLabV3+",
        "segformer": "SegFormer",
    }.get(raw.casefold(), raw)
    if not is_v4:
        return architecture
    context_layers = int(spec.get("context_layers", 0) or 0)
    context = " + transformer context" if context_layers > 0 else ""
    return f"{architecture} + Cellect multi-head shape refinement{context}"


def _encoder_display(spec: Mapping[str, object]) -> str:
    raw = str(spec.get("encoder", "Model"))
    known = {
        "timm-mobilenetv3_small_100": "MobileNetV3-Small",
        "timm-mobilenetv3_large_100": "MobileNetV3-Large",
        "efficientnet-b0": "EfficientNet-B0",
        "efficientnet-b3": "EfficientNet-B3",
        "resnet18": "ResNet18",
        "resnet50": "ResNet50",
        "resnet101": "ResNet101",
        "mit_b2": "SegFormer-B2",
        "mit_b5": "SegFormer-B5",
    }
    return known.get(raw.casefold(), raw.replace("_", "-"))


def _catalog_display_name(
    candidate: Candidate,
    spec: Mapping[str, object],
) -> str:
    encoder = _encoder_display(spec)
    architecture = _architecture_display(spec, is_v4=False)
    if candidate.summary_section == "cellect_v4":
        return f"Cellect v4 · {encoder} shape model"
    suffix = ""
    if candidate.source_name.endswith("_accuracy_v2"):
        suffix = " · accuracy v2"
    elif candidate.source_name.endswith("_accuracy"):
        suffix = " · accuracy"
    return f"{encoder} {architecture}{suffix}"


def _finite_numeric_metrics(metrics: Mapping[str, object]) -> dict[str, float]:
    available: dict[str, float] = {}
    for key, value in sorted(metrics.items()):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numeric = float(value)
        if np.isfinite(numeric):
            available[str(key)] = numeric
    return available


def _catalog_postprocessing_defaults(
    candidate: Candidate,
    selected: object,
) -> dict[str, object]:
    """Translate frozen Python operating points to the app's explicit option names."""

    if not isinstance(selected, Mapping):
        return {}
    defaults: dict[str, object] = {}

    def probability(source_key: str, destination_key: str) -> None:
        value = selected.get(source_key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if np.isfinite(numeric) and 0 <= numeric <= 1:
                defaults[destination_key] = numeric

    probability("foreground_threshold", "foregroundProbabilityThreshold")
    probability("boundary_threshold", "boundaryProbabilityThreshold")
    probability("boundary_cutoff", "boundaryProbabilityThreshold")
    probability(
        "min_mean_foreground_probability",
        "minimumMeanCellProbability",
    )
    probability("core_probability_threshold", "coreProbabilityThreshold")
    probability("min_core_fraction", "minimumCoreFraction")
    probability("min_boundary_support", "minimumBoundarySupport")
    minimum_area = selected.get("min_area_fraction")
    if isinstance(minimum_area, (int, float)) and not isinstance(minimum_area, bool):
        numeric_area = float(minimum_area)
        if np.isfinite(numeric_area) and 0 <= numeric_area < 1:
            defaults["minimumAreaFraction"] = numeric_area

    fusion = selected.get("fusion")
    if isinstance(fusion, Mapping):
        mode = str(fusion.get("mode", ""))
        head = fusion.get("head")
        mode_mapping = {
            "learned": "learnedFusion",
            "any": "anyBranch",
            "majority": "majorityBranches",
            "all": "allBranches",
            "weighted": "weightedBranches",
        }
        if mode == "single":
            mode_mapping_value = {
                "context": "contextOnly",
                "flow": "flowOnly",
                "shape": "shapeOnly",
            }.get(str(head))
        else:
            mode_mapping_value = mode_mapping.get(mode)
        if mode_mapping_value is None:
            raise ValueError(
                f"{candidate.source_name} selected an unsupported boundary fusion: {fusion!r}"
            )
        defaults["boundaryEvidenceMode"] = mode_mapping_value
        for source_key, destination_key in (
            ("context_threshold", "contextBoundaryProbabilityThreshold"),
            ("flow_threshold", "flowBoundaryProbabilityThreshold"),
            ("shape_threshold", "shapeBoundaryProbabilityThreshold"),
        ):
            value = fusion.get(source_key)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(
                    f"{candidate.source_name} has no numeric {source_key} in its frozen fusion"
                )
            numeric = float(value)
            if not np.isfinite(numeric) or not 0 < numeric < 1:
                raise ValueError(
                    f"{candidate.source_name} has an invalid {source_key} in its frozen fusion"
                )
            defaults[destination_key] = numeric
        weights = fusion.get("weights")
        if not (
            isinstance(weights, (list, tuple))
            and len(weights) == 3
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and np.isfinite(float(value))
                and float(value) >= 0
                for value in weights
            )
            and sum(float(value) for value in weights) > 0
        ):
            raise ValueError(f"{candidate.source_name} has invalid frozen boundary weights")
        for destination_key, value in zip(
            (
                "contextBoundaryWeight",
                "flowBoundaryWeight",
                "shapeBoundaryWeight",
            ),
            weights,
        ):
            defaults[destination_key] = float(value)
    elif candidate.summary_section == "cellect_v4":
        raise ValueError(f"{candidate.source_name} has no frozen v4 boundary fusion")
    else:
        defaults["boundaryEvidenceMode"] = "learnedFusion"

    # These are app behavior defaults, not fitted validation results. Record them explicitly so
    # switching models never silently resurrects unrelated state from the previous model.
    defaults["automaticBoundaryThreshold"] = False
    defaults["separateTouchingCells"] = True
    return defaults


def _catalog_record(
    result: Mapping[str, object],
    candidate: Candidate,
    summary: Mapping[str, object],
) -> dict[str, object]:
    """Build one honest, app-facing record from a parity-validated conversion."""

    details = candidate_details(candidate, summary)
    raw_spec = details.get("spec")
    if not isinstance(raw_spec, Mapping):
        raise ValueError(f"{candidate.source_name} has no model spec for the app catalog")
    raw_metrics = result.get("validation_metrics")
    if not isinstance(raw_metrics, Mapping):
        raise ValueError(f"{candidate.source_name} has no validation metrics")
    available_metrics = _finite_numeric_metrics(raw_metrics)

    validation_role = str(result["validation_role"])
    if validation_role == "held_out_test":
        dice = float(result["test_dice"])
        iou = float(result["test_iou"])
        count_mae: float | None = float(result["count_mae"])
        evaluation_role = "Held-out test"
    else:
        if candidate.summary_section == "cellect_v4":
            dice_key = "foreground_logit_dice"
            iou_key = "foreground_logit_iou"
        else:
            # The current two-channel accuracy-v2 pipeline retains historical metric key names,
            # but those values are from ensemble_selection, not from the sealed test labels.
            dice_key = "test_dice"
            iou_key = "test_iou"
        try:
            dice = float(raw_metrics[dice_key])
            iou = float(raw_metrics[iou_key])
        except KeyError as error:
            raise ValueError(
                f"{candidate.source_name} has no foreground Dice/IoU on ensemble_selection"
            ) from error
        # Count MAE is optional in the app contract. Current two-channel selection reports
        # measure it directly; the v4 shape report does not, so never synthesize a replacement.
        measured_count_mae = raw_metrics.get("count_mae")
        count_mae = (
            float(measured_count_mae)
            if candidate.summary_section == "models"
            and isinstance(measured_count_mae, (int, float))
            and not isinstance(measured_count_mae, bool)
            else None
        )
        evaluation_role = "Ensemble-selection development set"

    export_version = str(result["export_version"])
    selection_score = result.get("selection_score")
    if isinstance(selection_score, (int, float)) and not isinstance(
        selection_score, bool
    ):
        numeric_score = float(selection_score)
        if np.isfinite(numeric_score):
            available_metrics["selection_score"] = numeric_score

    if not (
        np.isfinite(dice)
        and 0 <= dice <= 1
        and np.isfinite(iou)
        and 0 <= iou <= 1
        and (count_mae is None or (np.isfinite(count_mae) and count_mae >= 0))
    ):
        raise ValueError(f"{candidate.source_name} has invalid app catalog metrics")

    precision = str(result["compute_precision"]).casefold()
    if precision not in {"float16", "float32"}:
        raise ValueError(
            f"{candidate.source_name} has unsupported Core ML precision {precision!r}"
        )
    record: dict[str, object] = {
        "id": candidate.source_name,
        "resourceName": str(result["resource_name"]),
        "displayName": _catalog_display_name(candidate, raw_spec),
        "tier": _catalog_tier(str(result["tier"])),
        "architecture": _architecture_display(
            raw_spec,
            is_v4=candidate.summary_section == "cellect_v4",
        ),
        "inputSize": int(result["image_size"]),
        "parameterCount": int(result["parameters"]),
        "dice": dice,
        "iou": iou,
        "precision": "Float16" if precision == "float16" else "Float32",
        "sourceTorchScriptSHA256": str(result["torchscript_sha256"]),
        # Normalized globally after current and retained records are merged. Keeping this false
        # here prevents two independently selected training families from both becoming defaults.
        "isRecommended": False,
        "evaluationRole": evaluation_role,
        "exportVersion": export_version,
        "availableMetrics": available_metrics,
    }
    selected_postprocessing = result.get("selected_postprocessing", {})
    if isinstance(selected_postprocessing, Mapping):
        record["selectedPostprocessing"] = dict(selected_postprocessing)
    defaults = _catalog_postprocessing_defaults(candidate, selected_postprocessing)
    if defaults:
        record["defaultPostprocessing"] = defaults
    if count_mae is not None:
        record["countMAE"] = count_mae
    return record


def _valid_existing_catalog_record(record: object, output_dir: Path) -> bool:
    """Keep old records only when their package and minimum typed contract still exist."""

    if not isinstance(record, Mapping):
        return False
    required_types: dict[str, type | tuple[type, ...]] = {
        "id": str,
        "resourceName": str,
        "displayName": str,
        "tier": str,
        "architecture": str,
        "inputSize": int,
        "parameterCount": int,
        "dice": (int, float),
        "iou": (int, float),
        "precision": str,
        "sourceTorchScriptSHA256": str,
        "isRecommended": bool,
        "evaluationRole": str,
        "exportVersion": str,
    }
    if any(
        key not in record or not isinstance(record[key], expected)
        for key, expected in required_types.items()
    ):
        return False
    resource_name = str(record["resourceName"])
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", resource_name) is None:
        return False
    try:
        _catalog_tier(str(record["tier"]))
    except ValueError:
        return False
    dice = float(record["dice"])
    iou = float(record["iou"])
    count_mae = record.get("countMAE")
    defaults = record.get("defaultPostprocessing")
    if defaults is not None:
        if not isinstance(defaults, Mapping):
            return False
        probability_keys = {
            "foregroundProbabilityThreshold",
            "boundaryProbabilityThreshold",
            "contextBoundaryProbabilityThreshold",
            "flowBoundaryProbabilityThreshold",
            "shapeBoundaryProbabilityThreshold",
            "minimumMeanCellProbability",
            "coreProbabilityThreshold",
            "minimumCoreFraction",
            "minimumBoundarySupport",
        }
        for key in probability_keys & set(defaults):
            value = defaults[key]
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(float(value))
                or not 0 <= float(value) <= 1
            ):
                return False
        area = defaults.get("minimumAreaFraction")
        if area is not None and (
            not isinstance(area, (int, float))
            or isinstance(area, bool)
            or not np.isfinite(float(area))
            or not 0 <= float(area) < 1
        ):
            return False
        mode = defaults.get("boundaryEvidenceMode")
        if mode is not None and mode not in {
            "learnedFusion",
            "contextOnly",
            "flowOnly",
            "shapeOnly",
            "anyBranch",
            "majorityBranches",
            "allBranches",
            "weightedBranches",
        }:
            return False
        weight_keys = (
            "contextBoundaryWeight",
            "flowBoundaryWeight",
            "shapeBoundaryWeight",
        )
        weights = [defaults.get(key) for key in weight_keys]
        if any(value is not None for value in weights) and not (
            all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and np.isfinite(float(value))
                and float(value) >= 0
                for value in weights
            )
            and sum(float(value) for value in weights) > 0
        ):
            return False
        for key in ("automaticBoundaryThreshold", "separateTouchingCells"):
            if key in defaults and not isinstance(defaults[key], bool):
                return False
    if not (
        str(record["id"])
        and str(record["displayName"])
        and str(record["architecture"])
        and int(record["inputSize"]) > 0
        and int(record["parameterCount"]) > 0
        and np.isfinite(dice)
        and 0 <= dice <= 1
        and np.isfinite(iou)
        and 0 <= iou <= 1
        and (
            count_mae is None
            or (
                isinstance(count_mae, (int, float))
                and not isinstance(count_mae, bool)
                and np.isfinite(float(count_mae))
                and float(count_mae) >= 0
            )
        )
        and str(record["precision"]) in {"Float16", "Float32"}
        and re.fullmatch(
            r"[0-9a-f]{64}", str(record["sourceTorchScriptSHA256"])
        )
        is not None
        and str(record["evaluationRole"]).strip()
        and str(record["exportVersion"]).strip()
    ):
        return False
    return (output_dir / f"{resource_name}.mlpackage").is_dir()


def normalize_catalog_recommendation(
    records: list[dict[str, object]],
    summary: Mapping[str, object],
) -> list[dict[str, object]]:
    """Choose exactly one app default: selected deployable v4, then legacy, then first."""

    if not records:
        return []
    identifiers = {str(record["id"]) for record in records}
    v4 = summary.get("cellect_v4")
    v4_selected = v4.get("selected_model") if isinstance(v4, Mapping) else None
    legacy_selected = summary.get("selected_model")
    preferred: str
    if isinstance(v4_selected, str) and v4_selected in identifiers:
        preferred = v4_selected
    elif isinstance(legacy_selected, str) and legacy_selected in identifiers:
        preferred = legacy_selected
    else:
        preferred = str(records[0]["id"])
    normalized: list[dict[str, object]] = []
    for record in records:
        updated = dict(record)
        updated["isRecommended"] = str(record["id"]) == preferred
        normalized.append(updated)
    if sum(bool(record["isRecommended"]) for record in normalized) != 1:
        raise AssertionError("Generated model catalog must have exactly one recommendation")
    return normalized


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as destination:
            json.dump(value, destination, indent=2, sort_keys=False)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
            temporary_path = Path(destination.name)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_model_catalog(
    results: list[dict[str, object]],
    candidates: list[Candidate],
    summary: Mapping[str, object],
    output_dir: Path,
    catalog_path: Path | None = None,
) -> Path:
    """Atomically expose every known converted segmentation model to the iOS app."""

    by_source_name = {candidate.source_name: candidate for candidate in candidates}
    current: list[dict[str, object]] = []
    for result in results:
        source_name = str(result["source_model"])
        try:
            candidate = by_source_name[source_name]
        except KeyError as error:
            raise ValueError(f"No catalog candidate for {source_name!r}") from error
        current.append(_catalog_record(result, candidate, summary))

    catalog_path = catalog_path or (output_dir / CATALOG_FILENAME)
    retained: list[dict[str, object]] = []
    if catalog_path.is_file():
        try:
            existing = json.loads(catalog_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = None
        raw_records = existing.get("models") if isinstance(existing, Mapping) else None
        if isinstance(raw_records, list):
            retained = [
                dict(record)
                for record in raw_records
                if _valid_existing_catalog_record(record, output_dir)
            ]

    # A selective --model conversion updates that model without hiding previously converted
    # packages. The just-validated record wins, and the first retained ID wins thereafter.
    merged: list[dict[str, object]] = []
    seen: set[str] = set()
    for record in current + retained:
        identifier = str(record["id"])
        if identifier in seen:
            continue
        seen.add(identifier)
        merged.append(record)
    merged = normalize_catalog_recommendation(merged, summary)
    _atomic_json(
        catalog_path,
        {
            "schemaVersion": CATALOG_SCHEMA_VERSION,
            "models": merged,
        },
    )
    return catalog_path


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


def _load_torchscript(source: Path, source_model: str) -> object:
    try:
        return torch.jit.load(str(source), map_location="cpu").eval()
    except Exception as error:
        raise RuntimeError(
            f"Could not load TorchScript for {source_model} with installed PyTorch "
            f"{torch.__version__}. Newer TorchScript is not forward-compatible with older "
            "PyTorch. Use a result archive containing preconverted coreml_artifacts, or rerun "
            "./run.sh best-smoke and ./run.sh best on the workstation."
        ) from error


def convert_candidate(
    candidate: Candidate,
    summary: dict[str, object],
    source_root: Path,
    output_dir: Path,
    benchmark_runs: int,
    *,
    conversion_only: bool = False,
    artifact_root: Path | None = None,
    fixture_dir: Path | None = None,
    manifest_dir: Path | None = None,
    contract_label: str | None = None,
) -> dict[str, object]:
    _require_conversion_runtime()
    details = candidate_details(candidate, summary)
    spec = details.get("spec")
    if not isinstance(spec, Mapping) or spec.get("image_size") is None:
        raise ValueError(f"{candidate.source_name} has no image_size in its spec")
    image_size = int(spec["image_size"])
    metrics, selection_score, validation_role = _candidate_validation_metrics(
        candidate,
        details,
    )
    export_version = _candidate_export_version(candidate)
    artifact = _artifact_metadata(candidate, details)
    source, expected = resolve_torchscript_artifact(candidate, artifact, source_root)
    actual = sha256(source)
    if actual != expected:
        raise RuntimeError(f"SHA-256 mismatch for {source}")

    print(f"Loading {candidate.source_name}")
    torch_model = _load_torchscript(source, candidate.source_name)
    sample = representative_input(image_size)
    with torch.inference_mode():
        torch_output_tensor = torch_model(torch.from_numpy(sample))
    if not isinstance(torch_output_tensor, torch.Tensor):
        raise RuntimeError(f"{candidate.source_name} did not return one tensor")
    expected_shape = (1, candidate.output_channels, image_size, image_size)
    if tuple(torch_output_tensor.shape) != expected_shape:
        raise RuntimeError(
            f"{candidate.source_name} returned {tuple(torch_output_tensor.shape)}; "
            f"expected {expected_shape}"
        )
    if not torch.isfinite(torch_output_tensor).all():
        raise RuntimeError(f"{candidate.source_name} returned non-finite logits")
    torch_output = torch_output_tensor.cpu().numpy()

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
                name=candidate.output_name,
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
    if candidate.summary_section == "cellect_v4":
        coreml_model.license = (
            "Mixed microscopy training sources; review the returned v4 provenance and licenses "
            "before distribution."
        )
        coreml_model.short_description = (
            "Five-channel Cellect v4 foreground and learned/context/flow/shape boundary logits."
        )
    elif validation_role == "ensemble_selection":
        coreml_model.license = (
            "Mixed microscopy training sources; review the returned provenance and licenses "
            "before distribution."
        )
        coreml_model.short_description = (
            "Two-channel Cellect accuracy-v2 foreground and contact-boundary logits."
        )
    else:
        coreml_model.license = (
            "Derived from LIVECell (CC BY-NC 4.0); review licensing before distribution."
        )
        coreml_model.short_description = (
            "Two-channel LIVECell foreground and cell-boundary logits."
        )
    coreml_model.version = export_version
    coreml_model.user_defined_metadata["cellect_source_model"] = candidate.source_name
    coreml_model.user_defined_metadata["cellect_tier"] = candidate.tier
    coreml_model.user_defined_metadata["cellect_summary_section"] = (
        candidate.summary_section
    )
    coreml_model.user_defined_metadata["cellect_evaluation_role"] = validation_role
    coreml_model.user_defined_metadata["cellect_export_version"] = export_version
    coreml_model.user_defined_metadata["cellect_input"] = (
        f"1x3x{image_size}x{image_size} grayscale repeated as RGB, float32 [0,1]"
    )
    coreml_model.user_defined_metadata["cellect_output"] = (
        f"1x{candidate.output_channels}x{image_size}x{image_size}: "
        + ", ".join(candidate.output_semantics)
    )
    coreml_model.user_defined_metadata["cellect_output_name"] = candidate.output_name
    coreml_model.user_defined_metadata["cellect_output_semantics"] = json.dumps(
        candidate.output_semantics,
        separators=(",", ":"),
    )
    selected_postprocessing = _candidate_postprocessing(candidate, details, summary)
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

    parameter_count = sum(parameter.numel() for parameter in torch_model.parameters())
    result: dict[str, object] = {
        "source_model": candidate.source_name,
        "resource_name": candidate.resource_name,
        "tier": candidate.tier,
        "summary_section": candidate.summary_section,
        "image_size": image_size,
        "output_name": candidate.output_name,
        "output_channels": candidate.output_channels,
        "output_semantics": list(candidate.output_semantics),
        "validation_metrics": dict(metrics),
        "validation_role": validation_role,
        "export_version": export_version,
        "selected_postprocessing": selected_postprocessing,
        "parameters": int(metrics.get("checkpoint_parameters", parameter_count)),
        "compute_precision": "float16" if candidate.use_float16 else "float32",
        "coreml_backend": candidate.backend,
        "coreml_package_bytes": directory_size(destination),
        "coreml_package_tree_sha256": package_tree_sha256(destination),
        "conversion_seconds": conversion_seconds,
        "first_prediction_seconds": None,
        "median_prediction_seconds": None,
        "parity_mean_absolute_error": None,
        "parity_max_absolute_error": None,
        "torchscript_sha256": actual,
        "coreml_specification_version": coreml_model.get_spec().specificationVersion,
        "runtime_validation_status": (
            "pending_macos_coreml_runtime" if conversion_only else "in_progress"
        ),
    }
    if validation_role == "held_out_test":
        # Preserve the historical report schema for existing app/report consumers.
        result.update(
            {
                "test_dice": metrics["test_dice"],
                "test_iou": metrics["test_iou"],
                "count_mae": metrics["count_mae"],
            }
        )
    else:
        result.update(
            {
                "selection_score": selection_score,
                "held_out_test_status": "not evaluated by Core ML conversion",
            }
        )

    if conversion_only:
        if artifact_root is None or fixture_dir is None or manifest_dir is None:
            raise ValueError(
                "conversion-only mode requires artifact_root, fixture_dir, and manifest_dir"
            )
        fixture_path, manifest_path, manifest = write_golden_contract(
            artifact_root=artifact_root,
            fixture_dir=fixture_dir,
            manifest_dir=manifest_dir,
            source_root=source_root,
            source_path=source,
            source_digest=actual,
            package_path=destination,
            candidate_kind="segmentation",
            source_model=candidate.source_name,
            resource_name=candidate.resource_name,
            precision=str(result["compute_precision"]),
            backend=candidate.backend,
            inputs={"image": sample},
            outputs={candidate.output_name: torch_output},
            conversion_result=result,
            extra_identity={
                "summary_section": candidate.summary_section,
                "output_name": candidate.output_name,
                "output_channels": candidate.output_channels,
            },
            contract_label=contract_label,
        )
        result.update(
            {
                "golden_fixture": _relative_to(
                    fixture_path, artifact_root, kind="golden fixture"
                ),
                "golden_fixture_sha256": sha256(fixture_path),
                "conversion_manifest": _relative_to(
                    manifest_path, artifact_root, kind="conversion manifest"
                ),
                "conversion_manifest_sha256": sha256(manifest_path),
                "coreml_package_tree_sha256": manifest["package"]["tree_sha256"],
            }
        )
        return result

    print(f"Validating {candidate.resource_name} with Core ML")
    first_started = time.perf_counter()
    prediction_values = coreml_model.predict({"image": sample})
    if candidate.output_name not in prediction_values:
        raise RuntimeError(
            f"Core ML output for {candidate.source_name} has keys "
            f"{sorted(prediction_values)}; expected {candidate.output_name!r}"
        )
    prediction = prediction_values[candidate.output_name]
    first_prediction_seconds = time.perf_counter() - first_started
    differences = np.abs(torch_output - prediction.astype(np.float32))
    parity_mean_absolute_error = float(differences.mean())
    parity_max_absolute_error = float(differences.max())
    semantic_parity = segmentation_semantic_parity(
        torch_output,
        prediction.astype(np.float32),
        result,
    )
    if (
        parity_mean_absolute_error > PARITY_MEAN_ABSOLUTE_ERROR_LIMIT
        or parity_max_absolute_error > PARITY_MAX_ABSOLUTE_ERROR_LIMIT
        or float(semantic_parity["probability_mean_absolute_error"])
        > SEGMENTATION_PROBABILITY_MEAN_ERROR_LIMIT
        or float(semantic_parity["probability_max_absolute_error"])
        > SEGMENTATION_PROBABILITY_MAX_ERROR_LIMIT
        or float(semantic_parity["maximum_decision_mismatch_fraction"])
        > DECISION_MISMATCH_FRACTION_LIMIT
    ):
        shutil.rmtree(destination)
        if candidate.use_float16:
            print(
                f"Float16 parity was unsafe for {candidate.source_name}; "
                "retrying at Float32."
            )
            return convert_candidate(
                replace(candidate, use_float16=False),
                summary,
                source_root,
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

    result.update(
        {
            "first_prediction_seconds": first_prediction_seconds,
            "median_prediction_seconds": statistics.median(timings),
            "parity_mean_absolute_error": parity_mean_absolute_error,
            "parity_max_absolute_error": parity_max_absolute_error,
            "runtime_validation_status": "passed_macos_coreml_runtime",
            "semantic_probability_and_decision_parity": semantic_parity,
        }
    )
    return result


def tracking_representative_inputs(
    manifest: Mapping[str, object],
) -> dict[str, np.ndarray]:
    fixed = manifest["fixed_input_contract"]
    time_shape = tuple(int(value) for value in fixed["time_yx"]["shape"])
    feature_shape = tuple(int(value) for value in fixed["cell_features"]["shape"])
    mask_shape = tuple(int(value) for value in fixed["cell_mask"]["shape"])
    token_count = time_shape[1]
    active = min(token_count, 12)
    time_yx = np.zeros(time_shape, dtype=np.float32)
    if active:
        indices = np.arange(active, dtype=np.float32)
        time_yx[0, :active, 0] = np.floor(indices / 4.0)
        time_yx[0, :active, 1] = np.linspace(-0.75, 0.75, active, dtype=np.float32)
        time_yx[0, :active, 2] = np.sin(indices * 0.7).astype(np.float32) * 0.6
    feature_index = np.arange(np.prod(feature_shape), dtype=np.float32).reshape(
        feature_shape
    )
    cell_features = (
        0.45 * np.sin(feature_index * 0.071)
        + 0.20 * np.cos(feature_index * 0.037)
    ).astype(np.float32)
    cell_mask = np.zeros(mask_shape, dtype=np.float32)
    cell_mask[0, :active] = 1.0
    return {
        "time_yx": time_yx,
        "cell_features": cell_features,
        "cell_mask": cell_mask,
    }


def convert_tracking_candidate(
    candidate: TrackingCandidate,
    source_root: Path,
    output_dir: Path,
    benchmark_runs: int,
    *,
    conversion_only: bool = False,
    artifact_root: Path | None = None,
    fixture_dir: Path | None = None,
    manifest_dir: Path | None = None,
    contract_label: str | None = None,
) -> dict[str, object]:
    """Convert one fixed-token CellectTrack manifest via its separate six-output path."""

    _require_conversion_runtime()
    manifest = _tracking_manifest(candidate.manifest_path, source_root)
    artifacts = manifest["artifacts"]
    artifact = artifacts["torchscript"]
    source = (candidate.manifest_path.parent / str(artifact["path"])).resolve()
    expected_digest = str(artifact["sha256"])
    actual_digest = sha256(source)
    if actual_digest != expected_digest:
        raise RuntimeError(f"SHA-256 mismatch for {source}")
    print(f"Loading tracking model {candidate.model_name}")
    torch_model = _load_torchscript(source, candidate.source_name)
    sample = tracking_representative_inputs(manifest)
    torch_inputs = tuple(
        torch.from_numpy(sample[name])
        for name in ("time_yx", "cell_features", "cell_mask")
    )
    with torch.inference_mode():
        raw_outputs = torch_model(*torch_inputs)
    if not isinstance(raw_outputs, (tuple, list)) or len(raw_outputs) != len(
        TRACKING_OUTPUT_NAMES
    ):
        raise RuntimeError(
            f"{candidate.source_name} must return {len(TRACKING_OUTPUT_NAMES)} tensors"
        )
    torch_outputs: dict[str, np.ndarray] = {}
    token_count = sample["cell_mask"].shape[1]
    expected_shapes = {
        "association_logits": (1, token_count, token_count),
        "association_mask": (1, token_count, token_count),
        "division_probability": (1, token_count),
        "birth_probability": (1, token_count),
        "death_probability": (1, token_count),
        "uncertainty": (1, token_count),
    }
    for name, value in zip(TRACKING_OUTPUT_NAMES, raw_outputs):
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected_shapes[name]:
            shape = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value)
            raise RuntimeError(
                f"Tracking output {name} has {shape}; expected {expected_shapes[name]}"
            )
        if not torch.isfinite(value).all():
            raise RuntimeError(f"Tracking output {name} contains non-finite values")
        torch_outputs[name] = value.detach().cpu().numpy()

    fixed = manifest["fixed_input_contract"]
    output_dtype = np.float16 if candidate.use_float16 else np.float32
    conversion_arguments = {
        "convert_to": "mlprogram",
        "inputs": [
            ct.TensorType(
                name=name,
                shape=tuple(int(value) for value in fixed[name]["shape"]),
                dtype=np.float32,
            )
            for name in ("time_yx", "cell_features", "cell_mask")
        ],
        "outputs": [
            ct.TensorType(name=name, dtype=output_dtype)
            for name in TRACKING_OUTPUT_NAMES
        ],
        "minimum_deployment_target": ct.target.iOS17,
        "compute_precision": (
            ct.precision.FLOAT16 if candidate.use_float16 else ct.precision.FLOAT32
        ),
    }
    print(
        f"Converting tracking {candidate.model_name} -> "
        f"{candidate.resource_name}.mlpackage"
    )
    started = time.perf_counter()
    coreml_model = ct.convert(torch_model, **conversion_arguments)
    conversion_seconds = time.perf_counter() - started
    coreml_model.author = "Cellect workstation training pipeline"
    coreml_model.license = (
        "Mixed microscopy tracking sources; inspect the returned tracking provenance before "
        "distribution."
    )
    coreml_model.short_description = (
        "Fixed-token CellectTrack association, division, birth, death, and uncertainty model."
    )
    coreml_model.version = "4"
    coreml_model.user_defined_metadata["cellect_source_model"] = candidate.source_name
    coreml_model.user_defined_metadata["cellect_model_contract"] = str(
        manifest.get("model_contract", "")
    )
    coreml_model.user_defined_metadata["cellect_tracking_inputs"] = json.dumps(
        fixed,
        sort_keys=True,
        separators=(",", ":"),
    )
    coreml_model.user_defined_metadata["cellect_tracking_outputs"] = json.dumps(
        TRACKING_OUTPUT_NAMES,
        separators=(",", ":"),
    )
    coreml_model.user_defined_metadata["cellect_tracking_thresholds"] = json.dumps(
        manifest.get("frozen_thresholds", {}),
        sort_keys=True,
        separators=(",", ":"),
    )
    coreml_model.user_defined_metadata["source_torchscript_sha256"] = actual_digest
    destination = output_dir / f"{candidate.resource_name}.mlpackage"
    if destination.exists():
        shutil.rmtree(destination)
    coreml_model.save(str(destination))

    result: dict[str, object] = {
        "source_model": candidate.source_name,
        "resource_name": candidate.resource_name,
        "model_name": candidate.model_name,
        "model_contract": manifest.get("model_contract"),
        "fixed_input_contract": fixed,
        "output_names": list(TRACKING_OUTPUT_NAMES),
        "frozen_thresholds": manifest.get("frozen_thresholds"),
        "compute_precision": "float16" if candidate.use_float16 else "float32",
        "coreml_backend": "mlprogram",
        "coreml_package_bytes": directory_size(destination),
        "coreml_package_tree_sha256": package_tree_sha256(destination),
        "conversion_seconds": conversion_seconds,
        "first_prediction_seconds": None,
        "median_prediction_seconds": None,
        "parity_mean_absolute_error": None,
        "parity_max_absolute_error": None,
        "per_output_parity": None,
        "torchscript_sha256": actual_digest,
        "tracking_manifest_sha256": sha256(candidate.manifest_path),
        "coreml_specification_version": coreml_model.get_spec().specificationVersion,
        "physical_iphone_parity_required": True,
        "runtime_validation_status": (
            "pending_macos_coreml_runtime" if conversion_only else "in_progress"
        ),
    }
    if conversion_only:
        if artifact_root is None or fixture_dir is None or manifest_dir is None:
            raise ValueError(
                "conversion-only mode requires artifact_root, fixture_dir, and manifest_dir"
            )
        fixture_path, manifest_path, golden_manifest = write_golden_contract(
            artifact_root=artifact_root,
            fixture_dir=fixture_dir,
            manifest_dir=manifest_dir,
            source_root=source_root,
            source_path=source,
            source_digest=actual_digest,
            package_path=destination,
            candidate_kind="tracking",
            source_model=candidate.source_name,
            resource_name=candidate.resource_name,
            precision=str(result["compute_precision"]),
            backend="mlprogram",
            inputs=sample,
            outputs=torch_outputs,
            conversion_result=result,
            extra_identity={
                "model_name": candidate.model_name,
                "tracking_manifest_sha256": sha256(candidate.manifest_path),
            },
            contract_label=contract_label,
        )
        result.update(
            {
                "golden_fixture": _relative_to(
                    fixture_path, artifact_root, kind="golden fixture"
                ),
                "golden_fixture_sha256": sha256(fixture_path),
                "conversion_manifest": _relative_to(
                    manifest_path, artifact_root, kind="conversion manifest"
                ),
                "conversion_manifest_sha256": sha256(manifest_path),
                "coreml_package_tree_sha256": golden_manifest["package"][
                    "tree_sha256"
                ],
            }
        )
        return result

    first_started = time.perf_counter()
    prediction = coreml_model.predict(sample)
    first_prediction_seconds = time.perf_counter() - first_started
    missing = set(TRACKING_OUTPUT_NAMES) - set(prediction)
    if missing:
        shutil.rmtree(destination)
        raise RuntimeError(
            f"Core ML tracking outputs are missing: {', '.join(sorted(missing))}"
        )
    per_output_error: dict[str, dict[str, float]] = {}
    maximum_error = 0.0
    mean_errors: list[float] = []
    for name in TRACKING_OUTPUT_NAMES:
        difference = np.abs(
            torch_outputs[name] - np.asarray(prediction[name], dtype=np.float32)
        )
        mean_error = float(difference.mean())
        max_error = float(difference.max())
        per_output_error[name] = {"mean_absolute_error": mean_error, "max_absolute_error": max_error}
        mean_errors.append(mean_error)
        maximum_error = max(maximum_error, max_error)
    mean_error = float(np.mean(mean_errors))
    prediction_outputs = {
        name: np.asarray(prediction[name], dtype=np.float32)
        for name in TRACKING_OUTPUT_NAMES
    }
    semantic_parity = tracking_semantic_parity(
        torch_outputs,
        prediction_outputs,
        result,
    )
    if (
        mean_error > PARITY_MEAN_ABSOLUTE_ERROR_LIMIT
        or maximum_error > PARITY_MAX_ABSOLUTE_ERROR_LIMIT
        or float(semantic_parity["probability_mean_absolute_error"])
        > TRACKING_PROBABILITY_MEAN_ERROR_LIMIT
        or float(semantic_parity["probability_max_absolute_error"])
        > TRACKING_PROBABILITY_MAX_ERROR_LIMIT
        or float(semantic_parity["maximum_decision_mismatch_fraction"])
        > DECISION_MISMATCH_FRACTION_LIMIT
    ):
        shutil.rmtree(destination)
        if candidate.use_float16:
            print(
                f"Float16 parity was unsafe for tracking {candidate.model_name}; "
                "retrying at Float32."
            )
            return convert_tracking_candidate(
                replace(candidate, use_float16=False),
                source_root,
                output_dir,
                benchmark_runs,
            )
        raise RuntimeError(
            f"Unsafe Core ML tracking conversion for {candidate.model_name}: "
            f"mean error {mean_error:.4f}, max error {maximum_error:.4f}"
        )
    timings: list[float] = []
    for _ in range(benchmark_runs):
        run_started = time.perf_counter()
        coreml_model.predict(sample)
        timings.append(time.perf_counter() - run_started)
    result.update(
        {
            "first_prediction_seconds": first_prediction_seconds,
            "median_prediction_seconds": statistics.median(timings),
            "parity_mean_absolute_error": mean_error,
            "parity_max_absolute_error": maximum_error,
            "per_output_parity": per_output_error,
            "runtime_validation_status": "passed_macos_coreml_runtime",
            "semantic_probability_and_decision_parity": semantic_parity,
        }
    )
    return result


def _manifest_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Golden manifest field {field!r} must be an object")
    return value


def _manifest_contract_entries(
    value: object,
    *,
    field: str,
) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"Golden manifest fixture.{field} must be a non-empty list")
    entries: list[Mapping[str, object]] = []
    names: set[str] = set()
    keys: set[str] = set()
    for raw in value:
        if not isinstance(raw, Mapping):
            raise ValueError(f"Golden manifest fixture.{field} entry must be an object")
        name = raw.get("name")
        key = raw.get("fixture_key")
        shape = raw.get("shape")
        dtype = raw.get("dtype")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError(f"Golden manifest fixture.{field} has an invalid/duplicate name")
        if not isinstance(key, str) or not key or key in keys:
            raise ValueError(f"Golden manifest fixture.{field} has an invalid/duplicate key")
        if not (
            isinstance(shape, list)
            and shape
            and all(isinstance(dimension, int) and dimension > 0 for dimension in shape)
        ):
            raise ValueError(f"Golden manifest fixture.{field}.{name} has an invalid shape")
        try:
            parsed_dtype = np.dtype(dtype)
        except TypeError as error:
            raise ValueError(
                f"Golden manifest fixture.{field}.{name} has an invalid dtype"
            ) from error
        if parsed_dtype.hasobject or not np.issubdtype(parsed_dtype, np.number):
            raise ValueError(
                f"Golden manifest fixture.{field}.{name} must be a numeric dtype"
            )
        names.add(name)
        keys.add(key)
        entries.append(raw)
    return entries


def validate_golden_manifest(
    manifest_path: Path,
    *,
    artifact_root: Path,
    source_root: Path,
    expected_kind: str,
    expected_source_model: str,
    expected_resource_name: str,
    expected_source_path: Path,
    expected_source_sha256: str,
    expected_input_names: tuple[str, ...],
    expected_output_names: tuple[str, ...],
    expected_tracking_manifest_sha256: str | None = None,
) -> dict[str, object]:
    """Validate hashes, paths, identities, and portable fixture contents without Core ML."""

    try:
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not parse golden manifest {manifest_path}: {error}") from error
    manifest = _manifest_mapping(raw_manifest, field="root")
    if manifest.get("schema_version") != BRIDGE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported golden manifest schema in {manifest_path}")
    if manifest.get("candidate_kind") != expected_kind:
        raise ValueError(f"Golden manifest candidate kind mismatch: {manifest_path}")
    identity = _manifest_mapping(manifest.get("candidate_identity"), field="candidate_identity")
    if identity.get("source_model") != expected_source_model:
        raise ValueError(f"Golden manifest source identity mismatch: {manifest_path}")
    if identity.get("resource_name") != expected_resource_name:
        raise ValueError(f"Golden manifest resource identity mismatch: {manifest_path}")
    if (
        expected_tracking_manifest_sha256 is not None
        and identity.get("tracking_manifest_sha256")
        != expected_tracking_manifest_sha256
    ):
        raise ValueError(f"Golden tracking-manifest identity mismatch: {manifest_path}")

    source = _manifest_mapping(manifest.get("source"), field="source")
    declared_source = source.get("path")
    declared_source_digest = source.get("sha256")
    if not isinstance(declared_source, str):
        raise ValueError(f"Golden manifest source path is invalid: {manifest_path}")
    source_path = safe_relative_path(source_root, declared_source, kind="source")
    if source_path != expected_source_path.resolve():
        raise ValueError(f"Golden manifest points at the wrong source: {manifest_path}")
    if declared_source_digest != expected_source_sha256 or sha256(source_path) != expected_source_sha256:
        raise ValueError(f"Golden manifest source SHA-256 mismatch: {manifest_path}")

    package = _manifest_mapping(manifest.get("package"), field="package")
    declared_package = package.get("path")
    expected_tree_digest = package.get("tree_sha256")
    if not isinstance(declared_package, str):
        raise ValueError(f"Golden manifest package path is invalid: {manifest_path}")
    package_path = safe_relative_path(artifact_root, declared_package, kind="package")
    if package_path.name != f"{expected_resource_name}.mlpackage":
        raise ValueError(f"Golden manifest package resource mismatch: {manifest_path}")
    if not isinstance(expected_tree_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_tree_digest
    ):
        raise ValueError(f"Golden manifest package tree digest is invalid: {manifest_path}")
    if package_tree_sha256(package_path) != expected_tree_digest:
        raise ValueError(f"Golden manifest package tree SHA-256 mismatch: {manifest_path}")
    if package.get("bytes") != directory_size(package_path):
        raise ValueError(f"Golden manifest package byte count mismatch: {manifest_path}")

    fixture = _manifest_mapping(manifest.get("fixture"), field="fixture")
    if fixture.get("format") != "npz-numeric-arrays-no-pickle":
        raise ValueError(f"Golden fixture format changed: {manifest_path}")
    declared_fixture = fixture.get("path")
    expected_fixture_digest = fixture.get("sha256")
    if not isinstance(declared_fixture, str):
        raise ValueError(f"Golden manifest fixture path is invalid: {manifest_path}")
    fixture_path = safe_relative_path(artifact_root, declared_fixture, kind="fixture")
    if (
        not isinstance(expected_fixture_digest, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_fixture_digest) is None
        or not fixture_path.is_file()
        or sha256(fixture_path) != expected_fixture_digest
    ):
        raise ValueError(f"Golden fixture SHA-256 mismatch: {manifest_path}")
    input_entries = _manifest_contract_entries(fixture.get("inputs"), field="inputs")
    output_entries = _manifest_contract_entries(fixture.get("outputs"), field="outputs")
    if tuple(entry["name"] for entry in input_entries) != expected_input_names:
        raise ValueError(f"Golden fixture input names changed: {manifest_path}")
    if tuple(entry["name"] for entry in output_entries) != expected_output_names:
        raise ValueError(f"Golden fixture output names changed: {manifest_path}")
    all_entries = input_entries + output_entries
    expected_keys = {str(entry["fixture_key"]) for entry in all_entries}
    arrays: dict[str, np.ndarray] = {}
    try:
        with np.load(fixture_path, allow_pickle=False) as loaded:
            if set(loaded.files) != expected_keys:
                raise ValueError(f"Golden fixture keys changed: {manifest_path}")
            for entry in all_entries:
                key = str(entry["fixture_key"])
                array = np.asarray(loaded[key])
                if list(array.shape) != entry["shape"] or str(array.dtype) != entry["dtype"]:
                    raise ValueError(f"Golden fixture array contract changed for {key}")
                if array.dtype.hasobject or not np.isfinite(array).all():
                    raise ValueError(f"Golden fixture array {key} is unsafe or non-finite")
                arrays[key] = array.copy()
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith("Golden fixture"):
            raise
        raise ValueError(f"Could not load safe golden fixture {fixture_path}: {error}") from error

    conversion = _manifest_mapping(manifest.get("conversion"), field="conversion")
    environment = _manifest_mapping(conversion.get("environment"), field="conversion.environment")
    if str(environment.get("coremltools", "")) != "9.0":
        raise ValueError(f"Golden package was not converted with coremltools 9: {manifest_path}")
    if not str(environment.get("torch", "")).startswith("2.7"):
        raise ValueError(f"Golden package was not converted with PyTorch 2.7: {manifest_path}")
    if str(environment.get("numpy", "")) != "1.26.4":
        raise ValueError(f"Golden package was not converted with NumPy 1.26.4: {manifest_path}")
    if conversion.get("precision") not in {"float16", "float32"}:
        raise ValueError(f"Golden package precision is invalid: {manifest_path}")
    if conversion.get("backend") not in {"mlprogram", "neuralnetwork"}:
        raise ValueError(f"Golden package backend is invalid: {manifest_path}")
    if conversion.get("runtime_prediction_performed") is not False:
        raise ValueError(f"Linux conversion manifest falsely claims runtime prediction: {manifest_path}")
    tolerances = _manifest_mapping(manifest.get("parity_tolerances"), field="parity_tolerances")
    expected_tolerances = {
        "raw_mean_absolute_error": PARITY_MEAN_ABSOLUTE_ERROR_LIMIT,
        "raw_max_absolute_error": PARITY_MAX_ABSOLUTE_ERROR_LIMIT,
        "segmentation_probability_mean_absolute_error": (
            SEGMENTATION_PROBABILITY_MEAN_ERROR_LIMIT
        ),
        "segmentation_probability_max_absolute_error": (
            SEGMENTATION_PROBABILITY_MAX_ERROR_LIMIT
        ),
        "tracking_probability_mean_absolute_error": TRACKING_PROBABILITY_MEAN_ERROR_LIMIT,
        "tracking_probability_max_absolute_error": TRACKING_PROBABILITY_MAX_ERROR_LIMIT,
        "decision_mismatch_fraction": DECISION_MISMATCH_FRACTION_LIMIT,
    }
    if set(tolerances) != set(expected_tolerances) or any(
        float(tolerances.get(name, -1)) != value
        for name, value in expected_tolerances.items()
    ):
        raise ValueError(f"Golden manifest parity tolerances changed: {manifest_path}")
    conversion_result = _manifest_mapping(manifest.get("conversion_result"), field="conversion_result")
    if conversion_result.get("source_model") != expected_source_model:
        raise ValueError(f"Golden conversion result source mismatch: {manifest_path}")
    if conversion_result.get("resource_name") != expected_resource_name:
        raise ValueError(f"Golden conversion result resource mismatch: {manifest_path}")
    if conversion_result.get("torchscript_sha256") != expected_source_sha256:
        raise ValueError(f"Golden conversion result source digest mismatch: {manifest_path}")
    if conversion_result.get("coreml_package_tree_sha256") != expected_tree_digest:
        raise ValueError(f"Golden conversion result package digest mismatch: {manifest_path}")
    if conversion_result.get("compute_precision") != conversion.get("precision"):
        raise ValueError(f"Golden conversion result precision mismatch: {manifest_path}")
    if conversion_result.get("coreml_backend") != conversion.get("backend"):
        raise ValueError(f"Golden conversion result backend mismatch: {manifest_path}")
    return {
        "manifest": dict(manifest),
        "manifest_path": manifest_path.resolve(),
        "package_path": package_path,
        "fixture_path": fixture_path,
        "inputs": {
            str(entry["name"]): arrays[str(entry["fixture_key"])]
            for entry in input_entries
        },
        "outputs": {
            str(entry["name"]): arrays[str(entry["fixture_key"])]
            for entry in output_entries
        },
        "conversion_result": dict(conversion_result),
    }


def _validate_preconverted_segmentation(
    candidate: Candidate,
    summary: Mapping[str, object],
    source_root: Path,
    artifact_root: Path,
    manifest_path: Path,
) -> dict[str, object]:
    details = candidate_details(candidate, summary)
    artifact = _artifact_metadata(candidate, details)
    source, expected_digest = resolve_torchscript_artifact(candidate, artifact, source_root)
    validated = validate_golden_manifest(
        manifest_path,
        artifact_root=artifact_root,
        source_root=source_root,
        expected_kind="segmentation",
        expected_source_model=candidate.source_name,
        expected_resource_name=candidate.resource_name,
        expected_source_path=source,
        expected_source_sha256=expected_digest,
        expected_input_names=("image",),
        expected_output_names=(candidate.output_name,),
    )
    spec = details.get("spec")
    if not isinstance(spec, Mapping) or spec.get("image_size") is None:
        raise ValueError(f"{candidate.source_name} has no image size for golden validation")
    image_size = int(spec["image_size"])
    if validated["inputs"]["image"].shape != (1, 3, image_size, image_size):
        raise ValueError(f"{candidate.source_name} golden image shape is invalid")
    if validated["outputs"][candidate.output_name].shape != (
        1,
        candidate.output_channels,
        image_size,
        image_size,
    ):
        raise ValueError(f"{candidate.source_name} golden output shape is invalid")
    manifest = validated["manifest"]
    conversion_environment = manifest["conversion"]["environment"]
    summary_environment = summary.get("environment")
    if (
        isinstance(summary_environment, Mapping)
        and summary_environment.get("torch") is not None
        and conversion_environment.get("torch") != str(summary_environment["torch"])
    ):
        raise ValueError(
            f"{candidate.source_name} was converted under a different PyTorch environment"
        )
    return validated


def _validate_preconverted_tracking(
    candidate: TrackingCandidate,
    summary: Mapping[str, object],
    source_root: Path,
    artifact_root: Path,
    manifest_path: Path,
) -> dict[str, object]:
    tracking_manifest = _tracking_manifest(candidate.manifest_path, source_root)
    artifact = tracking_manifest["artifacts"]["torchscript"]
    source = (candidate.manifest_path.parent / str(artifact["path"])).resolve()
    source_digest = str(artifact["sha256"])
    validated = validate_golden_manifest(
        manifest_path,
        artifact_root=artifact_root,
        source_root=source_root,
        expected_kind="tracking",
        expected_source_model=candidate.source_name,
        expected_resource_name=candidate.resource_name,
        expected_source_path=source,
        expected_source_sha256=source_digest,
        expected_input_names=("time_yx", "cell_features", "cell_mask"),
        expected_output_names=TRACKING_OUTPUT_NAMES,
        expected_tracking_manifest_sha256=sha256(candidate.manifest_path),
    )
    fixed = tracking_manifest["fixed_input_contract"]
    for name in ("time_yx", "cell_features", "cell_mask"):
        if tuple(validated["inputs"][name].shape) != tuple(fixed[name]["shape"]):
            raise ValueError(f"{candidate.source_name} golden input {name} shape is invalid")
    token_count = int(fixed["cell_mask"]["shape"][1])
    expected_output_shapes = {
        "association_logits": (1, token_count, token_count),
        "association_mask": (1, token_count, token_count),
        "division_probability": (1, token_count),
        "birth_probability": (1, token_count),
        "death_probability": (1, token_count),
        "uncertainty": (1, token_count),
    }
    for name, shape in expected_output_shapes.items():
        if tuple(validated["outputs"][name].shape) != shape:
            raise ValueError(f"{candidate.source_name} golden output {name} shape is invalid")
    conversion_environment = validated["manifest"]["conversion"]["environment"]
    summary_environment = summary.get("environment")
    if (
        isinstance(summary_environment, Mapping)
        and summary_environment.get("torch") is not None
        and conversion_environment.get("torch") != str(summary_environment["torch"])
    ):
        raise ValueError(
            f"{candidate.source_name} was converted under a different PyTorch environment"
        )
    return validated


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float32), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _recenter_probability(value: np.ndarray, threshold: float) -> np.ndarray:
    probability = np.asarray(value, dtype=np.float32)
    return np.where(
        probability < threshold,
        0.5 * probability / threshold,
        0.5 + 0.5 * (probability - threshold) / (1.0 - threshold),
    )


def _segmentation_boundary_probability(
    probabilities: np.ndarray,
    postprocessing: Mapping[str, object],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    branch_decisions: dict[str, np.ndarray] = {}
    fusion = postprocessing.get("fusion")
    if not isinstance(fusion, Mapping):
        return probabilities[:, 1], branch_decisions
    thresholds = {
        "context": float(fusion["context_threshold"]),
        "flow": float(fusion["flow_threshold"]),
        "shape": float(fusion["shape_threshold"]),
    }
    raw_branches = {
        "context": probabilities[:, 2],
        "flow": probabilities[:, 3],
        "shape": probabilities[:, 4],
    }
    for name in ("context", "flow", "shape"):
        branch_decisions[f"{name}_raw_threshold"] = (
            raw_branches[name] >= thresholds[name]
        )
    mode = str(fusion.get("mode"))
    if mode == "learned":
        return probabilities[:, 1], branch_decisions
    branches = np.stack(
        [
            _recenter_probability(raw_branches[name], thresholds[name])
            for name in ("context", "flow", "shape")
        ],
        axis=0,
    )
    if mode == "single":
        boundary = branches[("context", "flow", "shape").index(str(fusion.get("head")))]
    elif mode == "any":
        boundary = branches.max(axis=0)
    elif mode == "all":
        boundary = branches.min(axis=0)
    elif mode == "majority":
        boundary = np.median(branches, axis=0)
    elif mode == "weighted":
        weights = np.asarray(fusion.get("weights"), dtype=np.float32)
        weights = weights / weights.sum()
        boundary = np.sum(branches * weights[:, None, None, None], axis=0)
    else:
        raise ValueError(f"Unsupported frozen boundary fusion mode: {mode!r}")
    return boundary, branch_decisions


def _decision_mismatch(
    expected: Mapping[str, np.ndarray],
    actual: Mapping[str, np.ndarray],
) -> tuple[dict[str, float], float]:
    if set(expected) != set(actual):
        raise ValueError("Decision-parity maps have different names")
    fractions: dict[str, float] = {}
    for name in sorted(expected):
        expected_value = np.asarray(expected[name], dtype=bool)
        actual_value = np.asarray(actual[name], dtype=bool)
        if expected_value.shape != actual_value.shape:
            raise ValueError(f"Decision-parity shape mismatch for {name}")
        fractions[name] = float(np.not_equal(expected_value, actual_value).mean())
    return fractions, max(fractions.values(), default=0.0)


def segmentation_semantic_parity(
    expected_logits: np.ndarray,
    actual_logits: np.ndarray,
    conversion_result: Mapping[str, object],
) -> dict[str, object]:
    expected_probability = _sigmoid(expected_logits)
    actual_probability = _sigmoid(actual_logits)
    probability_difference = np.abs(expected_probability - actual_probability)
    postprocessing = conversion_result.get("selected_postprocessing")
    if not isinstance(postprocessing, Mapping):
        postprocessing = {}
    foreground_threshold = float(postprocessing.get("foreground_threshold", 0.5))
    boundary_threshold = float(
        postprocessing.get(
            "boundary_cutoff",
            postprocessing.get("boundary_threshold", 0.45),
        )
    )
    expected_boundary, expected_branches = _segmentation_boundary_probability(
        expected_probability, postprocessing
    )
    actual_boundary, actual_branches = _segmentation_boundary_probability(
        actual_probability, postprocessing
    )
    expected_decisions = {
        "foreground": expected_probability[:, 0] >= foreground_threshold,
        "post_fusion_boundary": expected_boundary >= boundary_threshold,
        **expected_branches,
    }
    actual_decisions = {
        "foreground": actual_probability[:, 0] >= foreground_threshold,
        "post_fusion_boundary": actual_boundary >= boundary_threshold,
        **actual_branches,
    }
    mismatch_by_decision, maximum_mismatch = _decision_mismatch(
        expected_decisions, actual_decisions
    )
    return {
        "probability_mean_absolute_error": float(probability_difference.mean()),
        "probability_max_absolute_error": float(probability_difference.max()),
        "decision_mismatch_fraction_by_output": mismatch_by_decision,
        "maximum_decision_mismatch_fraction": maximum_mismatch,
    }


def tracking_semantic_parity(
    expected_outputs: Mapping[str, np.ndarray],
    actual_outputs: Mapping[str, np.ndarray],
    conversion_result: Mapping[str, object],
) -> dict[str, object]:
    expected_mask = np.asarray(expected_outputs["association_mask"]) > 0.5
    actual_mask = np.asarray(actual_outputs["association_mask"]) > 0.5
    probability_differences: list[np.ndarray] = []
    expected_probability: dict[str, np.ndarray] = {}
    actual_probability: dict[str, np.ndarray] = {}
    expected_probability["association"] = _sigmoid(
        expected_outputs["association_logits"]
    )
    actual_probability["association"] = _sigmoid(actual_outputs["association_logits"])
    if expected_mask.any():
        probability_differences.append(
            np.abs(
                expected_probability["association"][expected_mask]
                - actual_probability["association"][expected_mask]
            )
        )
    for short_name, output_name in (
        ("division", "division_probability"),
        ("birth", "birth_probability"),
        ("death", "death_probability"),
        ("uncertainty", "uncertainty"),
    ):
        expected_probability[short_name] = np.asarray(
            expected_outputs[output_name], dtype=np.float32
        )
        actual_probability[short_name] = np.asarray(
            actual_outputs[output_name], dtype=np.float32
        )
        probability_differences.append(
            np.abs(expected_probability[short_name] - actual_probability[short_name])
        )
    concatenated = np.concatenate([value.ravel() for value in probability_differences])
    thresholds = conversion_result.get("frozen_thresholds")
    if not isinstance(thresholds, Mapping):
        raise ValueError("Tracking conversion result has no frozen thresholds")
    expected_decisions: dict[str, np.ndarray] = {"association_mask": expected_mask}
    actual_decisions: dict[str, np.ndarray] = {"association_mask": actual_mask}
    expected_decisions["association"] = expected_mask & (
        expected_probability["association"] >= float(thresholds["association"])
    )
    actual_decisions["association"] = actual_mask & (
        actual_probability["association"] >= float(thresholds["association"])
    )
    for name in ("division", "birth", "death"):
        expected_decisions[name] = expected_probability[name] >= float(thresholds[name])
        actual_decisions[name] = actual_probability[name] >= float(thresholds[name])
    mismatch_by_decision, maximum_mismatch = _decision_mismatch(
        expected_decisions, actual_decisions
    )
    return {
        "probability_mean_absolute_error": float(concatenated.mean()),
        "probability_max_absolute_error": float(concatenated.max()),
        "decision_mismatch_fraction_by_output": mismatch_by_decision,
        "maximum_decision_mismatch_fraction": maximum_mismatch,
    }


def validate_coreml_runtime(
    validated: Mapping[str, object],
    *,
    benchmark_runs: int,
) -> dict[str, object]:
    """Run native Core ML against the portable golden fixture; never load TorchScript."""

    _require_coreml_runtime()
    package_path = Path(str(validated["package_path"]))
    inputs = validated["inputs"]
    expected_outputs = validated["outputs"]
    if not isinstance(inputs, Mapping) or not isinstance(expected_outputs, Mapping):
        raise ValueError("Validated golden fixture lost its input/output mapping")
    print(f"Validating preconverted {package_path.name} with the macOS Core ML runtime")
    model = ct.models.MLModel(str(package_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    first_started = time.perf_counter()
    prediction = model.predict(dict(inputs))
    first_prediction_seconds = time.perf_counter() - first_started
    missing = set(expected_outputs) - set(prediction)
    if missing:
        raise RuntimeError(
            f"Core ML outputs are missing from {package_path.name}: "
            + ", ".join(sorted(missing))
        )
    per_output: dict[str, dict[str, float]] = {}
    actual_outputs: dict[str, np.ndarray] = {}
    mean_errors: list[float] = []
    maximum_error = 0.0
    for name, expected in expected_outputs.items():
        actual = np.asarray(prediction[name], dtype=np.float32)
        expected_array = np.asarray(expected, dtype=np.float32)
        if actual.shape != expected_array.shape or not np.isfinite(actual).all():
            raise RuntimeError(
                f"Core ML output {name} has unsafe shape/data: {actual.shape}; "
                f"expected {expected_array.shape}"
            )
        difference = np.abs(expected_array - actual)
        actual_outputs[str(name)] = actual
        mean_error = float(difference.mean())
        max_error = float(difference.max())
        per_output[str(name)] = {
            "mean_absolute_error": mean_error,
            "max_absolute_error": max_error,
        }
        mean_errors.append(mean_error)
        maximum_error = max(maximum_error, max_error)
    aggregate_mean_error = float(np.mean(mean_errors))
    if (
        aggregate_mean_error > PARITY_MEAN_ABSOLUTE_ERROR_LIMIT
        or maximum_error > PARITY_MAX_ABSOLUTE_ERROR_LIMIT
    ):
        raise RuntimeError(
            f"Unsafe Core ML package {package_path.name}: mean error "
            f"{aggregate_mean_error:.4f}, max error {maximum_error:.4f}"
        )
    manifest = validated.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("Validated package lost its golden manifest")
    conversion_result = validated.get("conversion_result")
    if not isinstance(conversion_result, Mapping):
        raise ValueError("Validated package lost its conversion result")
    candidate_kind = manifest.get("candidate_kind")
    if candidate_kind == "segmentation":
        output_name = str(conversion_result["output_name"])
        semantic_parity = segmentation_semantic_parity(
            np.asarray(expected_outputs[output_name]),
            actual_outputs[output_name],
            conversion_result,
        )
        probability_mean_limit = SEGMENTATION_PROBABILITY_MEAN_ERROR_LIMIT
        probability_max_limit = SEGMENTATION_PROBABILITY_MAX_ERROR_LIMIT
    elif candidate_kind == "tracking":
        semantic_parity = tracking_semantic_parity(
            expected_outputs,
            actual_outputs,
            conversion_result,
        )
        probability_mean_limit = TRACKING_PROBABILITY_MEAN_ERROR_LIMIT
        probability_max_limit = TRACKING_PROBABILITY_MAX_ERROR_LIMIT
    else:
        raise ValueError(f"Unknown golden candidate kind: {candidate_kind!r}")
    if (
        float(semantic_parity["probability_mean_absolute_error"])
        > probability_mean_limit
        or float(semantic_parity["probability_max_absolute_error"])
        > probability_max_limit
        or float(semantic_parity["maximum_decision_mismatch_fraction"])
        > DECISION_MISMATCH_FRACTION_LIMIT
    ):
        raise RuntimeError(
            f"Core ML semantic parity failed for {package_path.name}: "
            f"probability mean={semantic_parity['probability_mean_absolute_error']:.6f}, "
            f"probability max={semantic_parity['probability_max_absolute_error']:.6f}, "
            f"decision mismatch={semantic_parity['maximum_decision_mismatch_fraction']:.6f}"
        )
    timings: list[float] = []
    for _ in range(benchmark_runs):
        started = time.perf_counter()
        model.predict(dict(inputs))
        timings.append(time.perf_counter() - started)
    result = dict(conversion_result)
    result.update(
        {
            "runtime_validation_status": "passed_macos_coreml_runtime",
            "first_prediction_seconds": first_prediction_seconds,
            "median_prediction_seconds": statistics.median(timings),
            "parity_mean_absolute_error": aggregate_mean_error,
            "parity_max_absolute_error": maximum_error,
            "per_output_parity": per_output,
            "semantic_probability_and_decision_parity": semantic_parity,
            "validated_package_tree_sha256": package_tree_sha256(package_path),
            "golden_fixture_sha256": sha256(Path(str(validated["fixture_path"]))),
            "conversion_manifest_sha256": sha256(Path(str(validated["manifest_path"]))),
        }
    )
    return result


def install_verified_packages(
    validated_candidates: list[Mapping[str, object]],
    output_dir: Path,
) -> None:
    """Copy a fully verified candidate set into app resources as one transaction."""

    output_dir.mkdir(parents=True, exist_ok=True)
    sources: dict[str, Path] = {}
    for validated in validated_candidates:
        source = Path(str(validated["package_path"])).resolve()
        if source.name in sources:
            raise ValueError(f"Duplicate Core ML package resource: {source.name}")
        sources[source.name] = source
    temporary = Path(tempfile.mkdtemp(prefix=".cellect-coreml-install.", dir=output_dir))
    staged_root = temporary / "staged"
    backup_root = temporary / "previous"
    staged_root.mkdir()
    backup_root.mkdir()
    installed: list[str] = []
    backed_up: list[str] = []
    try:
        # Complete and verify every potentially fallible copy before changing app resources.
        for name, source in sources.items():
            staged = staged_root / name
            shutil.copytree(source, staged)
            if package_tree_sha256(staged) != package_tree_sha256(source):
                raise RuntimeError(f"Package copy verification failed: {name}")

        for name in sources:
            destination = output_dir / name
            if destination.exists():
                os.replace(destination, backup_root / name)
                backed_up.append(name)
        for name in sources:
            os.replace(staged_root / name, output_dir / name)
            installed.append(name)
    except Exception:
        for name in reversed(installed):
            destination = output_dir / name
            if destination.exists():
                shutil.rmtree(destination)
        for name in reversed(backed_up):
            backup = backup_root / name
            if backup.exists():
                os.replace(backup, output_dir / name)
        raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Root containing summary.json and returned TorchScript/manifests.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Package destination. Defaults to coreml_artifacts/packages for "
            "--conversion-only, otherwise the app model resources directory."
        ),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Root containing portable packages, manifests, fixtures, report, and catalog.",
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        help="Golden .npz destination (conversion-only only).",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        help="Per-candidate golden manifest destination/discovery directory.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        help="Conversion/runtime validation report path.",
    )
    parser.add_argument(
        "--catalog-path",
        type=Path,
        help="Segmentation-only generated app catalog path.",
    )
    parser.add_argument(
        "--conversion-only",
        action="store_true",
        help=(
            "Convert under the workstation PyTorch environment and emit golden fixtures; "
            "never call MLModel.predict (safe on Linux)."
        ),
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
        help=(
            "Convert only this source model; repeat to select several. Names are validated "
            "against both legacy models and dynamic summary['cellect_v4']['models'] entries."
        ),
    )
    args = parser.parse_args()
    if args.benchmark_runs < 1:
        parser.error("--benchmark-runs must be at least 1")
    source_root = args.source_root.resolve()
    summary_path = source_root / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(
            f"Missing {summary_path}. Import a workstation archive or pass --source-root."
        )
    artifact_root = (
        args.artifact_root.resolve()
        if args.artifact_root is not None
        else (source_root / DEFAULT_ARTIFACT_DIRECTORY_NAME).resolve()
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (
            artifact_root / PACKAGES_DIRECTORY_NAME
            if args.conversion_only
            else DEFAULT_OUTPUT
        ).resolve()
    )
    fixture_dir = (
        args.fixture_dir.resolve()
        if args.fixture_dir is not None
        else (artifact_root / FIXTURES_DIRECTORY_NAME).resolve()
    )
    manifest_dir = (
        args.manifest_dir.resolve()
        if args.manifest_dir is not None
        else (artifact_root / MANIFESTS_DIRECTORY_NAME).resolve()
    )
    report_path = (
        args.report_path.resolve()
        if args.report_path is not None
        else (
            artifact_root / "coreml_conversion_report.json"
            if args.conversion_only
            else DEFAULT_REPORT
        ).resolve()
    )
    catalog_path = (
        args.catalog_path.resolve()
        if args.catalog_path is not None
        else (
            artifact_root / CATALOG_FILENAME
            if args.conversion_only
            else output_dir / CATALOG_FILENAME
        ).resolve()
    )
    if args.conversion_only:
        for path, kind in (
            (output_dir, "package directory"),
            (fixture_dir, "fixture directory"),
            (manifest_dir, "manifest directory"),
            (report_path, "report"),
            (catalog_path, "catalog"),
        ):
            _relative_to(path, artifact_root, kind=kind)

    summary = json.loads(summary_path.read_text())
    if not isinstance(summary, dict):
        raise SystemExit("Imported summary root must be a JSON object.")
    allowed_modes = {"full", "best", "best-smoke"} if args.conversion_only else {"full", "best"}
    if summary.get("mode") not in allowed_modes:
        if args.conversion_only:
            raise SystemExit("Core ML conversion is supported only for full/best-smoke/best runs.")
        raise SystemExit(
            "Refusing to install smoke-run results. Validate the final ./run.sh best archive."
        )
    legacy_models = summary.get("models", {})
    if not isinstance(legacy_models, Mapping):
        raise SystemExit("Imported summary has no valid legacy models object.")
    known_legacy_names = {candidate.source_name for candidate in CANDIDATES}
    unsupported_legacy = set(legacy_models) - known_legacy_names
    if unsupported_legacy:
        raise SystemExit(
            "Refusing to silently skip unsupported segmentation candidates: "
            + ", ".join(sorted(str(name) for name in unsupported_legacy))
        )
    available_candidates = [
        candidate for candidate in CANDIDATES if candidate.source_name in legacy_models
    ]
    try:
        available_candidates.extend(v4_candidates(summary))
    except ValueError as error:
        raise SystemExit(f"Invalid cellect_v4 summary: {error}") from error
    try:
        available_tracking_candidates = list(discover_tracking_candidates(source_root))
    except ValueError as error:
        raise SystemExit(f"Invalid CellectTrack export manifest: {error}") from error
    tracking_export_inventory: dict[str, object] | None = None
    tracking_section_for_exports = summary.get("cellect_track")
    if isinstance(tracking_section_for_exports, Mapping):
        tracking_exports = tracking_section_for_exports.get("exports")
        if not isinstance(tracking_exports, Mapping):
            raise SystemExit("cellect_track summary has no valid exports object.")
        try:
            tracking_export_inventory = validate_tracking_export_inventory(
                source_root,
                tracking_exports,
                available_tracking_candidates,
            )
        except ValueError as error:
            raise SystemExit(f"Invalid CellectTrack export inventory: {error}") from error
    names = [candidate.source_name for candidate in available_candidates]
    tracking_names = [
        candidate.source_name for candidate in available_tracking_candidates
    ]
    all_names = names + tracking_names
    if len(set(all_names)) != len(all_names):
        raise SystemExit("Legacy and cellect_v4 candidate names overlap.")
    resources = [candidate.resource_name for candidate in available_candidates] + [
        candidate.resource_name for candidate in available_tracking_candidates
    ]
    if len(set(resources)) != len(resources):
        raise SystemExit("Core ML resource names collide across returned candidates.")
    selected = set(args.model or ())
    candidates: list[Candidate] = [
        candidate
        for candidate in available_candidates
        if not selected or candidate.source_name in selected
    ]
    tracking_candidates = [
        candidate
        for candidate in available_tracking_candidates
        if not selected or candidate.source_name in selected
    ]
    missing = selected - set(all_names)
    if missing:
        raise SystemExit(
            "Requested models are absent from this result archive: "
            + ", ".join(sorted(missing))
        )
    if not candidates and not tracking_candidates:
        raise SystemExit("The imported result archive has no supported conversion candidates.")
    output_dir.mkdir(parents=True, exist_ok=True)

    preconverted_available = manifest_dir.is_dir() and any(manifest_dir.glob("*.json"))
    validated_packages: list[dict[str, object]] = []
    fallback_results: list[dict[str, object]] = []
    if args.conversion_only:
        _require_conversion_runtime()
        artifact_root.mkdir(parents=True, exist_ok=True)
        fixture_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        results = [
            convert_candidate(
                candidate,
                summary,
                source_root,
                output_dir,
                args.benchmark_runs,
                conversion_only=True,
                artifact_root=artifact_root,
                fixture_dir=fixture_dir,
                manifest_dir=manifest_dir,
            )
            for candidate in candidates
        ]
        tracking_results = [
            convert_tracking_candidate(
                candidate,
                source_root,
                output_dir,
                args.benchmark_runs,
                conversion_only=True,
                artifact_root=artifact_root,
                fixture_dir=fixture_dir,
                manifest_dir=manifest_dir,
            )
            for candidate in tracking_candidates
        ]
        fallback_output_dir = artifact_root / "float32_fallback" / PACKAGES_DIRECTORY_NAME
        fallback_fixture_dir = artifact_root / "float32_fallback" / FIXTURES_DIRECTORY_NAME
        fallback_output_dir.mkdir(parents=True, exist_ok=True)
        fallback_fixture_dir.mkdir(parents=True, exist_ok=True)
        fallback_results.extend(
            convert_candidate(
                replace(candidate, use_float16=False),
                summary,
                source_root,
                fallback_output_dir,
                args.benchmark_runs,
                conversion_only=True,
                artifact_root=artifact_root,
                fixture_dir=fallback_fixture_dir,
                manifest_dir=manifest_dir,
                contract_label="float32",
            )
            for candidate in candidates
            if candidate.use_float16
        )
        fallback_results.extend(
            convert_tracking_candidate(
                replace(candidate, use_float16=False),
                source_root,
                fallback_output_dir,
                args.benchmark_runs,
                conversion_only=True,
                artifact_root=artifact_root,
                fixture_dir=fallback_fixture_dir,
                manifest_dir=manifest_dir,
                contract_label="float32",
            )
            for candidate in tracking_candidates
            if candidate.use_float16
        )
        execution_mode = "linux_conversion_only"
    elif preconverted_available:
        # Validate every hash and every portable fixture before invoking Core ML. Package
        # installation happens only after all candidates pass native runtime parity.
        for candidate in candidates:
            manifest_path = manifest_dir / f"{candidate.resource_name}.json"
            if not manifest_path.is_file():
                raise SystemExit(
                    f"Preconverted artifact set is incomplete: missing {manifest_path.name}. "
                    "Do not fall back to an older local PyTorch for a partial v4 archive."
                )
            validated_packages.append(
                _validate_preconverted_segmentation(
                    candidate, summary, source_root, artifact_root, manifest_path
                )
            )
        for candidate in tracking_candidates:
            manifest_path = manifest_dir / f"{candidate.resource_name}.json"
            if not manifest_path.is_file():
                raise SystemExit(
                    f"Preconverted artifact set is incomplete: missing {manifest_path.name}. "
                    "Do not fall back to an older local PyTorch for a partial v4 archive."
                )
            validated_packages.append(
                _validate_preconverted_tracking(
                    candidate, summary, source_root, artifact_root, manifest_path
                )
            )
        fallbacks_by_source: dict[str, dict[str, object]] = {}
        for candidate in candidates:
            primary = next(
                item
                for item in validated_packages
                if item["conversion_result"]["source_model"] == candidate.source_name
            )
            if primary["conversion_result"].get("compute_precision") == "float16":
                fallback_manifest = manifest_dir / f"{candidate.resource_name}.float32.json"
                if not fallback_manifest.is_file():
                    raise SystemExit(
                        f"Missing required Float32 safety fallback: {fallback_manifest.name}"
                    )
                fallbacks_by_source[candidate.source_name] = (
                    _validate_preconverted_segmentation(
                        candidate,
                        summary,
                        source_root,
                        artifact_root,
                        fallback_manifest,
                    )
                )
                if (
                    fallbacks_by_source[candidate.source_name]["conversion_result"].get(
                        "compute_precision"
                    )
                    != "float32"
                ):
                    raise SystemExit(f"{fallback_manifest.name} is not a Float32 fallback")
        for candidate in tracking_candidates:
            primary = next(
                item
                for item in validated_packages
                if item["conversion_result"]["source_model"] == candidate.source_name
            )
            if primary["conversion_result"].get("compute_precision") == "float16":
                fallback_manifest = manifest_dir / f"{candidate.resource_name}.float32.json"
                if not fallback_manifest.is_file():
                    raise SystemExit(
                        f"Missing required Float32 safety fallback: {fallback_manifest.name}"
                    )
                fallbacks_by_source[candidate.source_name] = _validate_preconverted_tracking(
                    candidate,
                    summary,
                    source_root,
                    artifact_root,
                    fallback_manifest,
                )
                if (
                    fallbacks_by_source[candidate.source_name]["conversion_result"].get(
                        "compute_precision"
                    )
                    != "float32"
                ):
                    raise SystemExit(f"{fallback_manifest.name} is not a Float32 fallback")
        runtime_results: list[dict[str, object]] = []
        selected_packages: list[dict[str, object]] = []
        for primary in validated_packages:
            source_name = str(primary["conversion_result"]["source_model"])
            try:
                runtime_result = validate_coreml_runtime(
                    primary, benchmark_runs=args.benchmark_runs
                )
                selected_package = primary
            except RuntimeError as primary_error:
                fallback = fallbacks_by_source.get(source_name)
                if fallback is None:
                    raise
                print(
                    f"Primary Float16 package failed strict parity for {source_name}; "
                    "validating its preconverted Float32 fallback."
                )
                runtime_result = validate_coreml_runtime(
                    fallback, benchmark_runs=args.benchmark_runs
                )
                runtime_result["float32_fallback_used"] = True
                runtime_result["float16_rejection_reason"] = str(primary_error)
                selected_package = fallback
            runtime_results.append(runtime_result)
            selected_packages.append(selected_package)
        by_source = {str(result["source_model"]): result for result in runtime_results}
        results = [by_source[candidate.source_name] for candidate in candidates]
        tracking_results = [
            by_source[candidate.source_name] for candidate in tracking_candidates
        ]
        install_verified_packages(selected_packages, output_dir)
        execution_mode = "macos_preconverted_runtime_validation"
    else:
        # Backwards compatibility for old archives. A clear compatibility error from
        # _load_torchscript replaces opaque deserialization failures for new artifacts.
        producer_environment = summary.get("environment")
        producer_torch = (
            str(producer_environment.get("torch", ""))
            if isinstance(producer_environment, Mapping)
            else ""
        )
        producer_version = re.match(r"^(\d+)\.(\d+)", producer_torch)
        if producer_version is not None and tuple(
            int(value) for value in producer_version.groups()
        ) > (2, 2):
            raise SystemExit(
                f"This archive was exported by PyTorch {producer_torch} but contains no "
                "preconverted coreml_artifacts. Intel macOS cannot safely load that newer "
                "TorchScript. Rerun ./run.sh best-smoke and ./run.sh best with the v4 "
                "workstation bundle and return the replacement ZIP."
            )
        _require_conversion_runtime()
        results = [
            convert_candidate(
                candidate,
                summary,
                source_root,
                output_dir,
                args.benchmark_runs,
            )
            for candidate in candidates
        ]
        tracking_results = [
            convert_tracking_candidate(
                candidate,
                source_root,
                output_dir,
                args.benchmark_runs,
            )
            for candidate in tracking_candidates
        ]
        execution_mode = "legacy_local_conversion_and_runtime_validation"
    tracking_section = summary.get("cellect_track")
    tracking_selection = (
        tracking_section.get("selection")
        if isinstance(tracking_section, Mapping)
        else None
    )
    catalog_path = (
        write_model_catalog(
            results,
            candidates,
            summary,
            output_dir,
            catalog_path=catalog_path,
        )
        if results
        else None
    )
    _require_coreml_runtime()
    report = {
        "workstation_selected_model": summary.get("selected_model"),
        "workstation_v4_selected_model": (
            summary.get("cellect_v4", {}).get("selected_model")
            if isinstance(summary.get("cellect_v4"), Mapping)
            else None
        ),
        "selection_policy": (
            "Convert every exportable candidate beside training; on macOS retain only packages "
            "that pass raw, probability, and frozen-threshold decision parity, using the "
            "preconverted Float32 fallback when a Float16 package is unsafe."
        ),
        "bridge_schema_version": BRIDGE_SCHEMA_VERSION,
        "execution_mode": execution_mode,
        "source_root": str(source_root),
        "artifact_root": str(artifact_root),
        "conversion_environment": {
            "torch": str(torch.__version__) if torch is not None else None,
            "coremltools": ct.__version__,
            "numpy": np.__version__,
        },
        "models": results,
        "model_catalog": str(catalog_path) if catalog_path is not None else None,
        "model_catalog_schema_version": (
            CATALOG_SCHEMA_VERSION if catalog_path is not None else None
        ),
        "tracking_selection": tracking_selection,
        "tracking_export_inventory": tracking_export_inventory,
        "tracking_models": tracking_results,
        "float32_safety_fallbacks": fallback_results,
    }
    _atomic_json(report_path, report)
    print(f"Wrote {report_path}")
    if catalog_path is not None:
        print(f"Wrote {catalog_path}")
    for result in results:
        if result["validation_role"] == "ensemble_selection":
            score = result.get("selection_score")
            score_text = f"selection={float(score):.4f}, " if score is not None else ""
        else:
            score_text = f"Dice={float(result['test_dice']):.4f}, "
        print(
            f"{result['resource_name']}: {score_text}"
            f"channels={result['output_channels']}, "
            f"runtime={result['runtime_validation_status']}, "
            f"size={result['coreml_package_bytes'] / 1024 / 1024:.1f} MiB"
        )
    for result in tracking_results:
        print(
            f"{result['resource_name']}: tracking outputs=6, "
            f"runtime={result['runtime_validation_status']}, "
            f"size={result['coreml_package_bytes'] / 1024 / 1024:.1f} MiB"
        )


if __name__ == "__main__":
    main()
