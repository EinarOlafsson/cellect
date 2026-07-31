#!/usr/bin/env python3
"""Fine-tune one current Cellpose foundation model on all manual instance masks.

This follows the official Cellpose recommendation: always start from the built-in model, put all
training images into the same training round, use 1e-5 learning rate, 0.1 weight decay, batch size
1, and validate on a separate folder. A completion marker prevents accidentally fine-tuning an
already fine-tuned model again, which the Cellpose documentation warns can reweight data badly.

The Cellpose CLI only prints its sampled loss history, so this wrapper also parses those records
into history.csv for the research record.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


LOSS_PATTERN = re.compile(
    r"(?P<epoch>\d+),\s+train_loss=(?P<train>[0-9.eE+-]+),\s+"
    r"test_loss=(?P<test>[0-9.eE+-]+),\s+LR=(?P<lr>[0-9.eE+-]+)"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_history(log_path: Path, destination: Path) -> int:
    rows: dict[int, dict[str, object]] = {}
    for line in log_path.read_text(errors="replace").splitlines():
        match = LOSS_PATTERN.search(line)
        if match is None:
            continue
        epoch = int(match.group("epoch"))
        rows[epoch] = {
            "epoch": epoch,
            "train_loss": float(match.group("train")),
            "validation_loss": float(match.group("test")),
            "learning_rate": float(match.group("lr")),
        }
    with destination.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=[
                "epoch",
                "train_loss",
                "validation_loss",
                "learning_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows[epoch] for epoch in sorted(rows))
    if not rows:
        # Cellpose emits its sampled loss line only at logging intervals.  A one-epoch
        # ``best-smoke`` run can therefore be successful without printing a parseable row.
        # Keep the raw log and a valid header-only CSV; full 300-epoch runs still contain the
        # sampled loss history needed for the paper record.
        print(
            f"Warning: Cellpose emitted no sampled loss row in {log_path}; "
            "preserving the raw log and an empty history CSV."
        )
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--base-model",
        choices=["cpsam_v2", "cpdino", "cpdino-vitb"],
        default="cpsam_v2",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--stage-fingerprint", required=True)
    args = parser.parse_args()
    tile_size = 384 if args.base_model.startswith("cpdino") else 256

    train_directory = (args.data_root / "train").resolve()
    validation_directory = (args.data_root / "checkpoint").resolve()
    output_directory = args.output_dir.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    completion_marker = output_directory / "completed.json"
    if completion_marker.is_file():
        try:
            completed = json.loads(completion_marker.read_text())
            completed_model = output_directory / str(completed["final_model"])
            reusable = (
                completed.get("stage_fingerprint") == args.stage_fingerprint
                and completed.get("base_model") == args.base_model
                and int(completed.get("epochs", -1)) == args.epochs
                and completed_model.is_file()
                and completed.get("final_model_sha256") == file_sha256(completed_model)
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            reusable = False
        if reusable:
            print(f"Reusing fingerprint-matched Cellpose fine-tune: {output_directory}")
            return
        raise RuntimeError(
            f"Refusing stale Cellpose completion marker in {output_directory}; "
            "use the fingerprint-specific v3 output directory."
        )
    if not any(train_directory.glob("*_masks.tif")):
        raise SystemExit(f"No Cellpose training masks found in {train_directory}")
    if not any(validation_directory.glob("*_masks.tif")):
        raise SystemExit(f"No Cellpose validation masks found in {validation_directory}")

    command = [
        sys.executable,
        "-m",
        "cellpose",
        "--train",
        "--use_gpu",
        "--dir",
        str(train_directory),
        "--test_dir",
        str(validation_directory),
        "--pretrained_model",
        args.base_model,
        "--img_filter",
        "_img",
        "--mask_filter",
        "_masks",
        "--learning_rate",
        "0.00001",
        "--weight_decay",
        "0.1",
        "--n_epochs",
        str(args.epochs),
        "--train_batch_size",
        "1",
        "--bsize",
        str(tile_size),
        "--min_train_masks",
        "1",
        "--model_name_out",
        f"cellect_{args.base_model}_{args.stage_fingerprint[:12]}",
        "--save_every",
        "25",
        "--save_each",
        "--verbose",
    ]
    print("Running:", " ".join(command))
    log_path = output_directory / "training.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)

    model_stem = f"cellect_{args.base_model}_{args.stage_fingerprint[:12]}"
    expected_model = train_directory / "models" / model_stem
    if expected_model.is_file():
        trained_model = expected_model
    else:
        produced_models = sorted(
            (train_directory / "models").glob(f"{model_stem}*"),
            key=lambda path: path.stat().st_mtime,
        )
        if not produced_models:
            raise RuntimeError(
                f"Cellpose completed without producing the expected {args.base_model} model."
            )
        trained_model = produced_models[-1]
    destination = output_directory / "final_model.pt"
    shutil.copy2(trained_model, destination)
    checkpoint_directory = output_directory / "checkpoints"
    checkpoint_directory.mkdir(exist_ok=True)
    snapshots = []
    for snapshot in sorted(
        (train_directory / "models").glob(f"{model_stem}_epoch_*"),
    ):
        snapshot_destination = checkpoint_directory / f"{snapshot.name}.pt"
        shutil.copy2(snapshot, snapshot_destination)
        snapshots.append(str(snapshot_destination.relative_to(output_directory)))
    history_rows = write_history(log_path, output_directory / "history.csv")
    payload = json.dumps(
        {
            "base_model": args.base_model,
            "stage_fingerprint": args.stage_fingerprint,
            "data_root": str(args.data_root.resolve()),
            "epochs": args.epochs,
            "learning_rate": 1e-5,
            "weight_decay": 0.1,
            "train_batch_size": 1,
            "tile_size": tile_size,
            "minimum_masks_per_image": 1,
            "final_model": destination.name,
            "final_model_sha256": file_sha256(destination),
            "checkpoint_interval_epochs": 25,
            "checkpoint_candidates": snapshots + [destination.name],
            "training_log": log_path.name,
            "history": "history.csv",
            "logged_history_rows": history_rows,
            "command": command,
        },
        indent=2,
    ) + "\n"
    temporary_marker = completion_marker.with_suffix(".json.tmp")
    temporary_marker.write_text(payload)
    temporary_marker.replace(completion_marker)


if __name__ == "__main__":
    main()
