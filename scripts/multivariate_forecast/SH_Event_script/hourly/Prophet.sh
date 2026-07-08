python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_sh_event_hourly.json" --data-name-list event_{0..7}.csv --strategy-args '{"horizon": 288}' --model-name "prophet.Prophet" --gpus 0 --num-workers 1 --timeout 60000 --save-path "SH_Event/Prophet/hourly/d12"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_sh_event_hourly.json" --data-name-list event_{0..7}.csv --strategy-args '{"horizon": 576}' --model-name "prophet.Prophet" --gpus 0 --num-workers 1 --timeout 60000 --save-path "SH_Event/Prophet/hourly/d24"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_sh_event_hourly.json" --data-name-list event_{0..7}.csv --strategy-args '{"horizon": 864}' --model-name "prophet.Prophet" --gpus 0 --num-workers 1 --timeout 60000 --save-path "SH_Event/Prophet/hourly/d36"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_sh_event_hourly.json" --data-name-list event_{0..7}.csv --strategy-args '{"horizon": 1152}' --model-name "prophet.Prophet" --gpus 0 --num-workers 1 --timeout 60000 --save-path "SH_Event/Prophet/hourly/d48"
