#!/usr/bin/env python3
"""Pure artifact-contract tests; intentionally requires neither torch nor coremltools."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import coreml_bridge as bridge


class CoreMLBridgeContractTests(unittest.TestCase):
    def test_package_tree_hash_is_path_sensitive_and_rejects_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = root / "Model.mlpackage"
            (package / "Data").mkdir(parents=True)
            (package / "Manifest.json").write_text("{}\n", encoding="utf-8")
            (package / "Data" / "weights.bin").write_bytes(b"weights")
            first = bridge.package_tree_sha256(package)
            self.assertEqual(first, bridge.package_tree_sha256(package))
            (package / "Data" / "weights.bin").write_bytes(b"changed")
            self.assertNotEqual(first, bridge.package_tree_sha256(package))
            link = package / "Data" / "unsafe-link"
            try:
                link.symlink_to(package / "Manifest.json")
            except OSError:
                return
            with self.assertRaisesRegex(ValueError, "symlink"):
                bridge.package_tree_sha256(package)

    def test_safe_relative_path_rejects_traversal_absolute_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "inside").mkdir()
            self.assertEqual(
                bridge.safe_relative_path(root, "inside/model.json", kind="fixture"),
                (root / "inside" / "model.json").resolve(),
            )
            for unsafe in ("../escape", "/absolute", "inside/../../escape"):
                with self.assertRaises(ValueError):
                    bridge.safe_relative_path(root, unsafe, kind="fixture")
            link = root / "linked"
            try:
                link.symlink_to(root / "inside", target_is_directory=True)
            except OSError:
                return
            with self.assertRaisesRegex(ValueError, "symlink"):
                bridge.safe_relative_path(root, "linked/model.json", kind="fixture")

    def _fixture_tree(self, root: Path) -> tuple[Path, Path, Path, dict[str, object]]:
        source_root = root / "source"
        artifact_root = source_root / "coreml_artifacts"
        source_path = source_root / "models" / "source.pt"
        package = artifact_root / "packages" / "FixtureModel.mlpackage"
        fixture = artifact_root / "golden_fixtures" / "FixtureModel.npz"
        manifest_path = artifact_root / "manifests" / "FixtureModel.json"
        source_path.parent.mkdir(parents=True)
        package.mkdir(parents=True)
        fixture.parent.mkdir(parents=True)
        manifest_path.parent.mkdir(parents=True)
        source_path.write_bytes(b"torchscript-fixture")
        (package / "Manifest.json").write_text('{"model":"fixture"}\n', encoding="utf-8")
        image = np.arange(12, dtype=np.float32).reshape(1, 3, 2, 2)
        logits = np.arange(8, dtype=np.float32).reshape(1, 2, 2, 2)
        np.savez(fixture, input__image=image, output__logits=logits)
        source_digest = bridge.sha256(source_path)
        tree_digest = bridge.package_tree_sha256(package)
        conversion_result = {
            "source_model": "fixture_model",
            "resource_name": "FixtureModel",
            "torchscript_sha256": source_digest,
            "coreml_package_tree_sha256": tree_digest,
            "compute_precision": "float32",
            "coreml_backend": "mlprogram",
        }
        manifest: dict[str, object] = {
            "schema_version": bridge.BRIDGE_SCHEMA_VERSION,
            "candidate_kind": "segmentation",
            "candidate_identity": {
                "source_model": "fixture_model",
                "resource_name": "FixtureModel",
            },
            "source": {
                "path": "models/source.pt",
                "sha256": source_digest,
            },
            "package": {
                "path": "packages/FixtureModel.mlpackage",
                "tree_sha256": tree_digest,
                "bytes": bridge.directory_size(package),
            },
            "fixture": {
                "path": "golden_fixtures/FixtureModel.npz",
                "sha256": bridge.sha256(fixture),
                "format": "npz-numeric-arrays-no-pickle",
                "inputs": [
                    {
                        "name": "image",
                        "fixture_key": "input__image",
                        "shape": [1, 3, 2, 2],
                        "dtype": "float32",
                    }
                ],
                "outputs": [
                    {
                        "name": "logits",
                        "fixture_key": "output__logits",
                        "shape": [1, 2, 2, 2],
                        "dtype": "float32",
                    }
                ],
            },
            "conversion": {
                "environment": {
                    "torch": "2.7.1+cu128",
                    "coremltools": "9.0",
                    "numpy": "1.26.4",
                },
                "precision": "float32",
                "backend": "mlprogram",
                "runtime_prediction_performed": False,
            },
            "parity_tolerances": {
                "raw_mean_absolute_error": bridge.PARITY_MEAN_ABSOLUTE_ERROR_LIMIT,
                "raw_max_absolute_error": bridge.PARITY_MAX_ABSOLUTE_ERROR_LIMIT,
                "segmentation_probability_mean_absolute_error": (
                    bridge.SEGMENTATION_PROBABILITY_MEAN_ERROR_LIMIT
                ),
                "segmentation_probability_max_absolute_error": (
                    bridge.SEGMENTATION_PROBABILITY_MAX_ERROR_LIMIT
                ),
                "tracking_probability_mean_absolute_error": (
                    bridge.TRACKING_PROBABILITY_MEAN_ERROR_LIMIT
                ),
                "tracking_probability_max_absolute_error": (
                    bridge.TRACKING_PROBABILITY_MAX_ERROR_LIMIT
                ),
                "decision_mismatch_fraction": bridge.DECISION_MISMATCH_FRACTION_LIMIT,
            },
            "conversion_result": conversion_result,
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return source_root, artifact_root, manifest_path, manifest

    def test_fixture_manifest_and_source_identity_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_root, artifact_root, manifest_path, manifest = self._fixture_tree(
                Path(temporary)
            )
            source_path = source_root / "models" / "source.pt"
            validated = bridge.validate_golden_manifest(
                manifest_path,
                artifact_root=artifact_root,
                source_root=source_root,
                expected_kind="segmentation",
                expected_source_model="fixture_model",
                expected_resource_name="FixtureModel",
                expected_source_path=source_path,
                expected_source_sha256=bridge.sha256(source_path),
                expected_input_names=("image",),
                expected_output_names=("logits",),
            )
            self.assertEqual(set(validated["inputs"]), {"image"})
            self.assertEqual(set(validated["outputs"]), {"logits"})

            bad_identity = dict(manifest)
            bad_identity["candidate_identity"] = {
                "source_model": "another_model",
                "resource_name": "FixtureModel",
            }
            manifest_path.write_text(json.dumps(bad_identity), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source identity"):
                bridge.validate_golden_manifest(
                    manifest_path,
                    artifact_root=artifact_root,
                    source_root=source_root,
                    expected_kind="segmentation",
                    expected_source_model="fixture_model",
                    expected_resource_name="FixtureModel",
                    expected_source_path=source_path,
                    expected_source_sha256=bridge.sha256(source_path),
                    expected_input_names=("image",),
                    expected_output_names=("logits",),
                )

    def test_fixture_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source_root, artifact_root, manifest_path, _ = self._fixture_tree(Path(temporary))
            fixture = artifact_root / "golden_fixtures" / "FixtureModel.npz"
            fixture.write_bytes(fixture.read_bytes() + b"tamper")
            source_path = source_root / "models" / "source.pt"
            with self.assertRaisesRegex(ValueError, "fixture SHA-256"):
                bridge.validate_golden_manifest(
                    manifest_path,
                    artifact_root=artifact_root,
                    source_root=source_root,
                    expected_kind="segmentation",
                    expected_source_model="fixture_model",
                    expected_resource_name="FixtureModel",
                    expected_source_path=source_path,
                    expected_source_sha256=bridge.sha256(source_path),
                    expected_input_names=("image",),
                    expected_output_names=("logits",),
                )

    def test_cli_help_is_available_without_optional_runtimes(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(Path(bridge.__file__).resolve()), "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--conversion-only", completed.stdout)
        self.assertIn("--source-root", completed.stdout)

    def test_frozen_v4_postprocessing_maps_to_app_defaults(self) -> None:
        candidate = bridge.Candidate(
            source_name="shape_mobile_v4",
            resource_name="CellectShapeMobileV4",
            tier="low",
            summary_section="cellect_v4",
        )
        defaults = bridge._catalog_postprocessing_defaults(
            candidate,
            {
                "foreground_threshold": 0.55,
                "boundary_cutoff": 0.35,
                "min_area_fraction": 0.00008,
                "fusion": {
                    "mode": "weighted",
                    "head": None,
                    "context_threshold": 0.4,
                    "flow_threshold": 0.5,
                    "shape_threshold": 0.6,
                    "weights": [0.5, 0.3, 0.2],
                },
            },
        )
        self.assertEqual(defaults["foregroundProbabilityThreshold"], 0.55)
        self.assertEqual(defaults["boundaryProbabilityThreshold"], 0.35)
        self.assertEqual(defaults["boundaryEvidenceMode"], "weightedBranches")
        self.assertEqual(defaults["contextBoundaryProbabilityThreshold"], 0.4)
        self.assertEqual(defaults["flowBoundaryProbabilityThreshold"], 0.5)
        self.assertEqual(defaults["shapeBoundaryProbabilityThreshold"], 0.6)
        self.assertEqual(defaults["contextBoundaryWeight"], 0.5)
        self.assertEqual(defaults["flowBoundaryWeight"], 0.3)
        self.assertEqual(defaults["shapeBoundaryWeight"], 0.2)
        self.assertEqual(defaults["minimumAreaFraction"], 0.00008)

    def test_semantic_parity_catches_threshold_flips_below_raw_limit(self) -> None:
        expected = np.zeros((1, 2, 8, 8), dtype=np.float32)
        expected[:, 0] = -0.01
        actual = expected.copy()
        actual[:, 0] = 0.01
        parity = bridge.segmentation_semantic_parity(
            expected,
            actual,
            {
                "selected_postprocessing": {
                    "foreground_threshold": 0.5,
                    "boundary_threshold": 0.45,
                }
            },
        )
        self.assertLess(float(np.abs(actual - expected).max()), 0.05)
        self.assertEqual(parity["maximum_decision_mismatch_fraction"], 1.0)

        mask = np.ones((1, 2, 2), dtype=np.float32)
        tracking_expected = {
            "association_logits": np.full((1, 2, 2), -0.01, dtype=np.float32),
            "association_mask": mask,
            "division_probability": np.full((1, 2), 0.49, dtype=np.float32),
            "birth_probability": np.full((1, 2), 0.49, dtype=np.float32),
            "death_probability": np.full((1, 2), 0.49, dtype=np.float32),
            "uncertainty": np.full((1, 2), 0.5, dtype=np.float32),
        }
        tracking_actual = {name: value.copy() for name, value in tracking_expected.items()}
        tracking_actual["association_logits"][:] = 0.01
        tracking_parity = bridge.tracking_semantic_parity(
            tracking_expected,
            tracking_actual,
            {
                "frozen_thresholds": {
                    "association": 0.5,
                    "division": 0.5,
                    "birth": 0.5,
                    "death": 0.5,
                }
            },
        )
        self.assertEqual(
            tracking_parity["decision_mismatch_fraction_by_output"]["association"],
            1.0,
        )

    def test_global_recommendation_prefers_v4_and_clears_stale_flags(self) -> None:
        records = [
            {"id": "legacy_selected", "isRecommended": True},
            {"id": "retained_stale", "isRecommended": True},
            {"id": "v4_selected", "isRecommended": False},
        ]
        summary = {
            "selected_model": "legacy_selected",
            "cellect_v4": {"selected_model": "v4_selected"},
        }
        normalized = bridge.normalize_catalog_recommendation(records, summary)
        self.assertEqual(
            [record["id"] for record in normalized if record["isRecommended"]],
            ["v4_selected"],
        )

        # A selective legacy-only conversion still has one safe fallback; when the retained v4
        # record is present on the next merge, the same normalizer promotes it globally.
        selective = bridge.normalize_catalog_recommendation(records[:1], summary)
        self.assertTrue(selective[0]["isRecommended"])

    def test_tracking_inventory_preserves_hash_bound_mean_composite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            exports_root = root / "cellect_track_v4" / "exports"
            run_fingerprint = "1" * 64
            normalization_fingerprint = "2" * 64

            def write_manifest(
                model_name: str,
                manifest: dict[str, object],
            ) -> tuple[Path, dict[str, object]]:
                path = exports_root / model_name / "manifest.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
                return path, {**manifest, "manifest_sha256": bridge.sha256(path)}

            def standalone_manifest(model_name: str) -> dict[str, object]:
                directory = exports_root / model_name
                directory.mkdir(parents=True, exist_ok=True)
                torchscript = directory / f"cellect_track_{model_name}.torchscript.pt"
                torchscript.write_bytes(f"fixture-{model_name}".encode())
                return {
                    "schema_version": 4,
                    "model_name": model_name,
                    "model_contract": "cellect-track-v4",
                    "run_fingerprint": run_fingerprint,
                    "fixed_input_contract": {
                        "time_yx": {"shape": [1, 4, 3], "dtype": "float32"},
                        "cell_features": {
                            "shape": [1, 4, 2],
                            "dtype": "float32",
                            "ordered_names": ["area", "intensity"],
                            "normalization_fingerprint": normalization_fingerprint,
                        },
                        "cell_mask": {"shape": [1, 4], "dtype": "float32"},
                    },
                    "outputs": list(bridge.TRACKING_OUTPUT_NAMES),
                    "artifacts": {
                        "torchscript": {
                            "path": torchscript.name,
                            "sha256": bridge.sha256(torchscript),
                        }
                    },
                    "parity": {"status": "PASS"},
                }

            _, heavy_export = write_manifest(
                "heavy", standalone_manifest("heavy")
            )
            _, mobile_export = write_manifest(
                "mobile", standalone_manifest("mobile")
            )
            mean_manifest: dict[str, object] = {
                "schema_version": 4,
                "model_name": "mean_heavy_mobile",
                "artifact_type": "composite_probability_ensemble",
                "run_fingerprint": run_fingerprint,
                "normalization_fingerprint": normalization_fingerprint,
                "members": {
                    name: {
                        "manifest_relative_path": f"../{name}/manifest.json",
                        "manifest_sha256": export["manifest_sha256"],
                        "model_contract": "cellect-track-v4",
                    }
                    for name, export in (
                        ("heavy", heavy_export),
                        ("mobile", mobile_export),
                    )
                },
                "runtime_contract": {"decoder": "decode fused probabilities once"},
                "frozen_thresholds": {
                    "association": 0.5,
                    "division": 0.5,
                    "birth": 1.0,
                    "death": 1.0,
                },
                "standalone_model_artifact": False,
                "member_export_parity_required": True,
            }
            mean_path, mean_export = write_manifest(
                "mean_heavy_mobile", mean_manifest
            )
            exports = {
                "heavy": heavy_export,
                "mobile": mobile_export,
                "mean_heavy_mobile": mean_export,
            }

            candidates = bridge.discover_tracking_candidates(root)
            self.assertEqual(
                {candidate.model_name for candidate in candidates},
                {"heavy", "mobile"},
            )
            inventory = bridge.validate_tracking_export_inventory(
                root, exports, candidates
            )
            self.assertEqual(inventory["standalone_models"], ["heavy", "mobile"])
            self.assertEqual(
                inventory["composite_exports"][0]["members"],
                ["heavy", "mobile"],
            )
            self.assertEqual(
                inventory["composite_exports"][0]["manifest_sha256"],
                bridge.sha256(mean_path),
            )
            tampered_mean = json.loads(json.dumps(mean_manifest))
            tampered_mean["members"]["heavy"]["manifest_sha256"] = "0" * 64
            _, tampered_export = write_manifest("mean_heavy_mobile", tampered_mean)
            tampered_exports = {**exports, "mean_heavy_mobile": tampered_export}
            with self.assertRaisesRegex(ValueError, "member 'heavy' manifest digest changed"):
                bridge.validate_tracking_export_inventory(
                    root, tampered_exports, candidates
                )

    def test_catalog_merge_keeps_retained_model_but_recommends_v4_once(self) -> None:
        def record(identifier: str, resource_name: str) -> dict[str, object]:
            return {
                "id": identifier,
                "resourceName": resource_name,
                "displayName": identifier,
                "tier": "Accurate",
                "architecture": "fixture",
                "inputSize": 384,
                "parameterCount": 100,
                "dice": 0.8,
                "iou": 0.7,
                "precision": "Float32",
                "sourceTorchScriptSHA256": "a" * 64,
                "isRecommended": True,
                "evaluationRole": "Ensemble-selection development set",
                "exportVersion": "fixture-v1",
            }

        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            legacy = record("legacy_selected", "LegacyModel")
            v4 = record("v4_selected", "V4Model")
            for resource_name in ("LegacyModel", "V4Model"):
                (output_dir / f"{resource_name}.mlpackage").mkdir()
            catalog_path = output_dir / bridge.CATALOG_FILENAME
            catalog_path.write_text(
                json.dumps(
                    {
                        "schemaVersion": bridge.CATALOG_SCHEMA_VERSION,
                        "models": [legacy],
                    }
                ),
                encoding="utf-8",
            )
            candidate = bridge.Candidate(
                source_name="v4_selected",
                resource_name="V4Model",
                tier="high",
                summary_section="cellect_v4",
            )
            summary = {
                "selected_model": "legacy_selected",
                "cellect_v4": {"selected_model": "v4_selected"},
            }
            with mock.patch.object(bridge, "_catalog_record", return_value=v4):
                bridge.write_model_catalog(
                    [{"source_model": "v4_selected"}],
                    [candidate],
                    summary,
                    output_dir,
                )

            written = json.loads(catalog_path.read_text(encoding="utf-8"))["models"]
            self.assertEqual(
                {entry["id"] for entry in written},
                {"legacy_selected", "v4_selected"},
            )
            self.assertEqual(
                [entry["id"] for entry in written if entry["isRecommended"]],
                ["v4_selected"],
            )


if __name__ == "__main__":
    unittest.main()
