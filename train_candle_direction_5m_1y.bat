@echo off
set PYTHONPATH=src
"C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m btc_agent.cli train-candle-direction --csv data_btc_5m_1y.csv --lookback 240 --train-fraction 0.70 --cost 0 --out-model btc_candle_direction_model_1y.json --sample-step 5 --epochs 450
