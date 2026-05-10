@echo off
set PYTHONPATH=src
"C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m btc_agent.cli label-intrabar --input intrabar_signals.csv --output intrabar_labeled.csv --symbol BTCUSDT --interval 5m
