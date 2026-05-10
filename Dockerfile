FROM python:3.12-slim

WORKDIR /app

COPY . .

ENV PYTHONPATH=/app/src
ENV COLLECT_SECONDS=18000
ENV POLL_SECONDS=20
ENV INTRABAR_OUT=intrabar_signals.csv
ENV THRESHOLD=0.52
ENV FLIP_THRESHOLD=0.535
ENV MARKET_DATA_PROVIDER=coinbase
ENV SYMBOL=BTC-USD
ENV INTERVAL=5m

CMD ["sh", "-c", "python -m btc_agent.cli intrabar-watch --symbol ${SYMBOL} --interval ${INTERVAL} --model btc_candle_direction_model.json --seconds ${COLLECT_SECONDS} --poll-seconds ${POLL_SECONDS} --out ${INTRABAR_OUT} --threshold ${THRESHOLD} --flip-threshold ${FLIP_THRESHOLD}"]
