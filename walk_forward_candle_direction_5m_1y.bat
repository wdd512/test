@echo off
set PYTHONPATH=src
"C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m btc_agent.cli walk-forward-candle-direction --csv data_btc_5m_1y.csv --lookback 240 --cost 0 --train-size 10000 --test-size 2500 --mode long_short --threshold 0.52
