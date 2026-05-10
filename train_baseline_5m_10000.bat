@echo off
set PYTHONPATH=src
"C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m btc_agent.cli train-baseline --csv data_btc_5m_10000.csv --lookback 240 --horizon 3 --train-fraction 0.70 --cost 0.0008 --out-model btc_baseline_model_10000.json
