#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-full}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="$SCRIPT_DIR/.venv"

case "$MODE" in
  smoke|full|preflight|best-smoke|best) ;;
  *)
    echo "Usage: ./run.sh [smoke|full|preflight|best-smoke|best]" >&2
    exit 2
    ;;
esac

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3 was not found. Install Python 3.11 or 3.12 and rerun." >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "Git was not found. Install it (for Ubuntu: sudo apt install git) and rerun." >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if (3, 11) <= sys.version_info[:2] <= (3, 12) else 1)'; then
  echo "The accuracy bundle requires Python 3.11 or 3.12 (DINOv3 requires 3.11+)." >&2
  exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r requirements.txt
"$VENV_DIR/bin/python" bundle_self_check.py
"$VENV_DIR/bin/python" pipeline.py --mode "$MODE"

echo
if [[ "$MODE" == "preflight" ]]; then
  echo "Preflight finished. Review:"
  echo "$SCRIPT_DIR/output/preflight_v4.json"
else
  echo "Finished. Copy this file back to the Mac:"
  echo "$SCRIPT_DIR/output/cellect_workstation_results.zip"
fi
