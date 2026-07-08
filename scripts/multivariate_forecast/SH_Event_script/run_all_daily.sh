#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

if ! ls dataset/forecasting/event_0_daily.csv >/dev/null 2>&1; then
  echo "Daily CSVs not found. Run: python ./scripts/aggregate_sh_event_daily.py"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")/daily" && pwd)"
for script in "$SCRIPT_DIR"/*.sh; do
  echo "Running $(basename "$script")..."
  bash "$script"
done
