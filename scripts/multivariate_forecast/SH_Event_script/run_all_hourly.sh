#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

SCRIPT_DIR="$(cd "$(dirname "$0")/hourly" && pwd)"
for script in "$SCRIPT_DIR"/*.sh; do
  echo "Running $(basename "$script")..."
  bash "$script"
done
