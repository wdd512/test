@echo off
set PYTHONPATH=src
"C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m btc_agent.cli download-binance-history --symbol BTCUSDT --interval 5m --start-month 2025-05 --end-month 2026-04 --out data_btc_5m_1y.csv
