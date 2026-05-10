@echo off
set PYTHONPATH=src
"C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m btc_agent.cli intrabar-watch --symbol BTCUSDT --interval 5m --model btc_candle_direction_model.json --seconds 300 --poll-seconds 20 --out intrabar_signals.csv --threshold 0.52 --flip-threshold 0.535
