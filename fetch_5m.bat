@echo off
set PYTHONPATH=src
"C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m btc_agent.cli fetch --symbol BTCUSDT --interval 5m --limit 1000 --out data_btc_5m.csv
