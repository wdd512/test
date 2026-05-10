@echo off
set PYTHONPATH=src
"C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m btc_agent.cli walk-forward-candle-direction --csv data_btc_5m_10000.csv --lookback 240 --cost 0 --train-size 3000 --test-size 750 --mode long_short --threshold 0.52
