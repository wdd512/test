@echo off
set PYTHONPATH=src
set MARKET_DATA_PROVIDER=coinbase
"C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m btc_agent.cli paper-trade --symbol BTC-USD --interval 5m --base-model btc_candle_direction_model.json --intrabar-model btc_intrabar_model.json --seconds 3600 --poll-seconds 20 --balance 100 --stake-fraction 0.20 --threshold 0.52 --flip-threshold 0.535 --out paper_trades.csv
