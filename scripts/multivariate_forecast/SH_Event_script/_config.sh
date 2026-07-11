#!/bin/bash
TFB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Space-separated SH event ids, one benchmark job per series (no parallel load).
# Override: SH_EVENT_IDS="0 1" bash run_all_hourly.sh
SH_EVENT_IDS="${SH_EVENT_IDS:-0 1 2 3 4 5 6 7}"

# Strictly sequential evaluation (no Ray, single worker).
TFB_RUN_FLAGS=(
  --eval-backend ray
  --num-workers 10
  --num-cpus 15
  --gpus 1
  --timeout 60000
)

run_benchmark_one_series_at_a_time() {
  local config_path="$1"
  local strategy_args="$2"
  local model_name="$3"
  local model_hp="${4:-}"
  local save_path="$5"
  local hp_args=()
  if [[ -n "$model_hp" ]]; then
    hp_args=(--model-hyper-params "$model_hp")
  fi
  for event_id in ${SH_EVENT_IDS}; do
    echo "=== event_${event_id}.csv | ${model_name} | ${save_path} ==="
    (cd "$TFB_ROOT" && python ./scripts/run_benchmark.py \
      --config-path "$config_path" \
      --data-name-list "event_${event_id}.csv" \
      --strategy-args "$strategy_args" \
      --model-name "$model_name" \
      "${hp_args[@]}" \
      "${TFB_RUN_FLAGS[@]}" \
      --save-path "$save_path")
  done
}
