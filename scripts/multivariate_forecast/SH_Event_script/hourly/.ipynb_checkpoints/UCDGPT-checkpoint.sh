#!/bin/bash
set -euo pipefail
source "$(dirname "$0")/../_config.sh"

CFG="fixed_forecast_config_sh_event_hourly.json"
COMMON_HP='{"seq_len": 576, "hour_patch_size": 1, "patch_size": 4, "t_patch_size": 8, "model_size": "medium", "mask_strategy": "combined", "t_mask_ratio": 0.15, "s_mask_ratio": 0.15, "contrastive_weight": 1.5, "meta_weight": 0.5, "lr": 0.0005, "min_lr": 0.0001, "num_epochs": 200, "num_workers": 1, "patience": 3, "curriculum_mask": 1, "curriculum_mask_ratio": 0.05, "curriculum_mask_rate": 2, "cycle_gamma": 0.5, "psych_top_k": 4, "batch_size": 64, "norm": true}'

run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 288}' "ucdgpt.UCDGPT" \
  "${COMMON_HP%\}} , \"pred_len\": 288, \"horizon\": 288}" "SH_Event/UCDGPT/hourly/d12"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 576}' "ucdgpt.UCDGPT" \
  "${COMMON_HP%\}} , \"pred_len\": 576, \"horizon\": 576}" "SH_Event/UCDGPT/hourly/d24"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 864}' "ucdgpt.UCDGPT" \
  "${COMMON_HP%\}} , \"pred_len\": 864, \"horizon\": 864}" "SH_Event/UCDGPT/hourly/d36"
run_benchmark_one_series_at_a_time "$CFG" '{"horizon": 1152}' "ucdgpt.UCDGPT" \
  "${COMMON_HP%\}} , \"pred_len\": 1152, \"horizon\": 1152}" "SH_Event/UCDGPT/hourly/d48"
