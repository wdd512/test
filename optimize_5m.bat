@echo off
set PYTHONPATH=src
"C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m btc_agent.cli optimize --csv data_btc_5m.csv --interval 5m --horizon 3 --limit 240 --min-trades 20 --top 10
