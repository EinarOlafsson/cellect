#!/usr/bin/env python3
"""Validate and unpack the single archive returned by the training workstation."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INBOX = ROOT / "inbox"
IMPORTED = ROOT / "imported"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    archive = INBOX / "cellect_workstation_results.zip"
    if not archive.exists():
        raise SystemExit(f"Missing {archive}")
    with tempfile.TemporaryDirectory(prefix="cellect-results-") as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(archive) as source:
            source.extractall(staging)
        checksums = staging / "SHA256SUMS.txt"
        if not checksums.exists():
            raise SystemExit("The result archive has no SHA256SUMS.txt")
        for line in checksums.read_text().splitlines():
            expected, relative = line.split("  ", 1)
            candidate = staging / relative
            if not candidate.is_file() or digest(candidate) != expected:
                raise SystemExit(f"Checksum failed: {relative}")
        summary = json.loads((staging / "summary.json").read_text())
        if IMPORTED.exists():
            shutil.rmtree(IMPORTED)
        shutil.copytree(staging, IMPORTED)
    selected = summary["selected_model"]
    metrics = summary["models"][selected]["metrics"]
    print(f"Imported and verified: {selected}")
    print(json.dumps(metrics, indent=2))
    print(f"Files: {IMPORTED}")


if __name__ == "__main__":
    try:
        main()
    except (KeyError, ValueError, zipfile.BadZipFile) as error:
        print(f"Invalid result archive: {error}", file=sys.stderr)
        raise SystemExit(1)
