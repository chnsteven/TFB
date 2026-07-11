python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_sh_event_daily.json" --data-name-list event_{0..1}_daily.csv --strategy-args '{"horizon": 12}' --model-name "prophet.Prophet" --gpus 0 --num-workers 1 --timeout 60000 --save-path "SH_Event/Prophet/daily/d12"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_sh_event_daily.json" --data-name-list event_{0..1}_daily.csv --strategy-args '{"horizon": 24}' --model-name "prophet.Prophet" --gpus 0 --num-workers 1 --timeout 60000 --save-path "SH_Event/Prophet/daily/d24"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_sh_event_daily.json" --data-name-list event_{0..1}_daily.csv --strategy-args '{"horizon": 36}' --model-name "prophet.Prophet" --gpus 0 --num-workers 1 --timeout 60000 --save-path "SH_Event/Prophet/daily/d36"

python ./scripts/run_benchmark.py --config-path "fixed_forecast_config_sh_event_daily.json" --data-name-list event_{0..1}_daily.csv --strategy-args '{"horizon": 48}' --model-name "prophet.Prophet" --gpus 0 --num-workers 1 --timeout 60000 --save-path "SH_Event/Prophet/daily/d48"
