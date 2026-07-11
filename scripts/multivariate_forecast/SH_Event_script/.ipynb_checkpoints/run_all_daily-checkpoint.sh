#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

if ! ls dataset/forecasting/event_0_daily.csv >/dev/null 2>&1; then
  echo "Daily CSVs not found. Run: python ./scripts/aggregate_sh_event_daily.py"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=_config.sh
source "$SCRIPT_DIR/_config.sh"
eval "DAILY_DATA_NAMES=(event_{${SH_EVENT_RANGE}}_daily.csv)"
echo "Using ${#DAILY_DATA_NAMES[@]} daily series: ${DAILY_DATA_NAMES[*]}"

DAILY_SCRIPT_DIR="$(cd "$(dirname "$0")/daily" && pwd)"
for script in "$DAILY_SCRIPT_DIR"/*.sh; do
  echo "Running $(basename "$script")..."
  bash "$script"
done
