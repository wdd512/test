@echo off
set PYTHONPATH=src
"C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m btc_agent.cli candle-direction-live --symbol BTCUSDT --interval 5m --model btc_candle_direction_model.json --threshold 0.52
