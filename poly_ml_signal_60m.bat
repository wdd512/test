@echo off
set "BTC_AGENT_ROOT=%~dp0"
set "BTC_AGENT_PYTHON=C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
set "BTC_AGENT_SYMBOL=BTC-USD"
set "BTC_AGENT_INTERVAL=5m"
set "BTC_AGENT_MODEL=btc_candle_direction_model.json"
set "BTC_AGENT_THRESHOLD=0.52"
set "BTC_AGENT_FLIP_THRESHOLD=0.535"
set "MARKET_DATA_PROVIDER=coinbase"

set "WALLET_BALANCE=100"
set "POLY_MODEL_SHARES=5"
set "POLY_MODEL_POLL_MS=10000"
set "POLY_MODEL_MIN_CONFIDENCE=0.535"
set "POLY_MODEL_FLIP_CONFIDENCE=0.58"
set "POLY_MODEL_MAX_ASK=0.68"
set "POLY_MODEL_MIN_LIQUIDITY=1"
set "POLY_MODEL_MIN_REMAINING_SEC=20"

cd /d "%~dp0Poly"
bun run index.ts --strategy ml-signal --rounds 12 --always-log
