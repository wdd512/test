@echo off
set PYTHONPATH=src
"C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m btc_agent.cli experiment-matrix --csv data_btc_5m_10000.csv --lookback 240 --horizons 1,3,6,12 --costs 0,0.0008 --train-fraction 0.70
