# Railway intrabar collector

Мета: запустити збір intrabar snapshots у хмарі, потім забрати `intrabar_signals.csv`
і продовжити навчання локально.

## Що запускати

Railway використовує `railway.json` + `Dockerfile`. Команда за замовчуванням:

```bash
python -m btc_agent.cli intrabar-watch \
  --symbol BTCUSDT \
  --interval 5m \
  --model btc_candle_direction_model.json \
  --seconds 18000 \
  --poll-seconds 20 \
  --out intrabar_signals.csv \
  --threshold 0.52 \
  --flip-threshold 0.535
```

`18000` секунд = 5 годин = приблизно 60 п'ятихвилинних свічок.
При `poll-seconds=20` це приблизно 900 snapshots.

## Важливо про файл

Railway filesystem може не бути зручним як довготривале сховище. Для тесту це ок,
але перед redeploy/restart потрібно забрати `intrabar_signals.csv` з логів/файлів або
підключити persistent storage.

Надійніші варіанти:

- Railway Volume і писати CSV туди.
- S3/R2 bucket.
- Postgres table.

Поточний MVP пише CSV у локальний файл контейнера.

## Змінні середовища

Можна задати в Railway:

```text
COLLECT_SECONDS=18000
POLL_SECONDS=20
INTRABAR_OUT=intrabar_signals.csv
THRESHOLD=0.52
FLIP_THRESHOLD=0.535
MARKET_DATA_PROVIDER=coinbase
SYMBOL=BTC-USD
INTERVAL=5m
```

Railway US West може отримувати `HTTP 451` від `https://api.binance.com`.
Тому Docker deployment за замовчуванням використовує Coinbase `BTC-USD`.
Якщо зміните Railway region на європейський і хочете повернути Binance, задайте:

```text
MARKET_DATA_PROVIDER=binance
SYMBOL=BTCUSDT
BINANCE_BASE_URL=https://api.binance.com
```

## Після збору

Коли забрали `intrabar_signals.csv` у локальну папку проєкту:

```bat
label_intrabar.bat
train_intrabar.bat
```

Якщо snapshots менше 100, `train_intrabar.bat` попросить зібрати більше.
