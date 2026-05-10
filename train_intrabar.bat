@echo off
set PYTHONPATH=src
"C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m btc_agent.cli train-intrabar --csv intrabar_labeled.csv --out-model btc_intrabar_model.json --train-fraction 0.70
