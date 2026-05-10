@echo off
set PYTHONPATH=src
"C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m btc_agent.cli walk-forward-regime --csv data_btc_5m_10000.csv --lookback 240 --horizon 12 --cost 0.0008 --train-size 3000 --test-size 750 --mode long_only --threshold 0.52 --min-train-trades 40
