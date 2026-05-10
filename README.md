# BTC Agent

MVP ШІ-агента для аналізу BTC на 5-15-хвилинних графіках на основі дослідження
`Розробка власного ШІ-агента для аналізу BTC на 5-15-хвилинних графіках.docx`.

Це не торговий бот і не фінансова порада. Поточна версія є research/MVP-шаром:
вона збирає market data, будує ознаки, формує directional signal, накладає risk overlay
і повертає пояснюваний JSON-рішення.

## Що реалізовано

- Binance REST collector для `klines` і order book snapshot.
- Ознаки з дослідження:
  - log returns на коротких вікнах;
  - realized volatility, ATR, range, close-location;
  - RSI, MACD histogram, Bollinger position, VWAP deviation;
  - volume/trade intensity z-score;
  - taker buy/sell imbalance;
  - spread, relative spread, order book imbalance, microprice deviation.
- Alpha layer:
  - окремий аналіз для `5m` і `15m`;
  - score + confidence + reasons;
  - combined decision `LONG`, `SHORT` або `HOLD`.
- Risk layer:
  - position fraction;
  - ATR-based stop/take-profit;
  - kill-switch flags для wide spread, volatility expansion, disagreement між таймфреймами.
- CSV export і простий walk-forward backtest.

## Структура

```text
src/btc_agent/
  agent.py      orchestration
  alpha.py      explainable scoring baseline
  backtest.py   simple walk-forward backtest
  cli.py        command line interface
  config.py     agent settings
  data.py       Binance/CSV data access
  features.py   feature engineering
  models.py     dataclasses
  risk.py       deterministic risk overlay
```

## Запуск

У цій папці можна запускати модулі напряму через `PYTHONPATH`.

```powershell
$env:PYTHONPATH="src"
python -m btc_agent.cli analyze --symbol BTCUSDT --limit 240
```

Якщо звичайний `python` не доступний у PATH, використайте Python runtime Codex:

```powershell
$env:PYTHONPATH="src"
& "C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m btc_agent.cli analyze --symbol BTCUSDT --limit 240
```

Зберегти 5m-свічки:

```powershell
$env:PYTHONPATH="src"
& "C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m btc_agent.cli fetch --symbol BTCUSDT --interval 5m --limit 1000 --out data_btc_5m.csv
```

Запустити простий backtest:

```powershell
$env:PYTHONPATH="src"
& "C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m btc_agent.cli backtest --csv data_btc_5m.csv --interval 5m --horizon 3 --limit 240
```

Перебрати пороги confidence/score і перевірити normal/inverse режим:

```powershell
$env:PYTHONPATH="src"
& "C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m btc_agent.cli optimize --csv data_btc_5m.csv --interval 5m --horizon 3 --limit 240 --min-trades 20 --top 10
```

У `cmd.exe` можна користуватись готовими файлами:

```bat
fetch_5m.bat
backtest_5m.bat
backtest_5m_inverse.bat
optimize_5m.bat
train_baseline_5m.bat
fetch_5m_10000.bat
train_baseline_5m_10000.bat
experiment_matrix_5m_10000.bat
walk_forward_h12_5m_10000.bat
walk_forward_regime_h12_5m_10000.bat
train_candle_direction_5m_10000.bat
walk_forward_candle_direction_5m_10000.bat
candle_direction_live.bat
intrabar_live.bat
intrabar_watch_5m.bat
label_intrabar.bat
train_intrabar.bat
```

## Як читати результат

`action`:

- `LONG` - перевага вгору після risk overlay.
- `SHORT` - перевага вниз після risk overlay.
- `HOLD` - сигнал слабкий або risk layer заблокував угоду.

`confidence` - не ймовірність прибутку, а внутрішня впевненість score-моделі.

`position_fraction` - рекомендована частка equity для позиції в межах risk constraints.

`risk_flags` - причини, чому агент міг зменшити позицію або перейти в `HOLD`.

У backtest-звіті:

- `profit_factor` нижче `1.0` означає, що gross loss більший за gross profit.
- `long` і `short` показують, яка сторона сигналу працює гірше.
- `avg_net_return` рахується після умовних trading costs.

## Supervised baseline

`train_baseline_5m.bat` навчає просту logistic regression модель без зовнішніх бібліотек:

- labels: `1`, якщо forward return через 3 свічки більший за умовну вартість угоди;
- split: перші 70% рядків train, останні 30% test;
- scaling: robust median/MAD тільки на train;
- звіт: classification metrics, trading threshold sweep, top feature weights;
- модель зберігається в `btc_baseline_model.json`.

Для ML-звіту краще використовувати довшу історію:

```bat
fetch_5m_10000.bat
train_baseline_5m_10000.bat
experiment_matrix_5m_10000.bat
walk_forward_h12_5m_10000.bat
walk_forward_regime_h12_5m_10000.bat
```

`walk_forward_h12_5m_10000.bat` перевіряє фіксовану політику `horizon=12`,
`cost=0.0008`, `long_only`, `threshold=0.52` на кількох послідовних folds.
`walk_forward_regime_h12_5m_10000.bat` на кожному fold підбирає простий regime
filter тільки на train-частині й застосовує його до наступного test-відрізка.

## Current 5m candle direction

Цей режим відповідає на питання: коли відкрилась нова 5m-свічка BTC, чи буде її close
вище або нижче її open.

```bat
train_candle_direction_5m_10000.bat
walk_forward_candle_direction_5m_10000.bat
candle_direction_live.bat
intrabar_live.bat
intrabar_watch_5m.bat
```

Модель використовує тільки закриту історію до поточної свічки, без даних з її майбутнього close.

`intrabar_live.bat` додає динамічний intrabar overlay: базовий open-прогноз не змінюється,
але по ходу поточної 5m-свічки overlay враховує partial close/open move, close-location
і taker imbalance. `intrabar_watch_5m.bat` опитує сигнал протягом 5 хвилин і пише
історію в `intrabar_signals.csv`.

Щоб навчити окрему intrabar-модель:

```bat
intrabar_watch_5m.bat
label_intrabar.bat
train_intrabar.bat
```

Для `train_intrabar.bat` потрібно хоча б 100 labeled snapshots, тому збір треба повторити
на багатьох свічках.

Для запуску collector у Railway дивіться [RAILWAY.md](RAILWAY.md).

## Наступні production-кроки

1. Додати WebSocket ingestion і локальний raw/silver store.
2. Зберігати історію L2 snapshots/depth updates для replay.
3. Додати supervised baseline: logistic/CatBoost/XGBoost на engineered features.
4. Додати purged/walk-forward validation з embargo.
5. Додати execution-aware simulator: fees, spread, slippage, latency, partial fills.
6. Додати paper trading і monitoring: feature health, drift, latency, PnL attribution.
