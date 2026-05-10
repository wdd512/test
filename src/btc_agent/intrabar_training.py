from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from .data import BinanceMarketData
from .ml_baseline import ClassificationMetrics, LogisticRegression, RobustScaler


INTRABAR_FEATURE_COLUMNS = [
    "elapsed_fraction",
    "current_return",
    "candle_range_pct",
    "close_location",
    "taker_imbalance",
    "base_probability_up",
    "dynamic_probability_up",
]


@dataclass(frozen=True)
class IntrabarLabelReport:
    input_csv: str
    output_csv: str
    labeled_rows: int


@dataclass(frozen=True)
class IntrabarTrainingReport:
    input_csv: str
    output_model: str
    samples: int
    train_samples: int
    test_samples: int
    positive_rate: float
    train_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics
    top_weights: list[dict[str, float]]


def label_intrabar_csv(input_csv: str, output_csv: str, symbol: str = "BTCUSDT", interval: str = "5m") -> IntrabarLabelReport:
    rows = _read_rows(input_csv)
    if not rows:
        raise ValueError("Input intrabar CSV has no rows.")

    market_data = BinanceMarketData()
    candles = market_data.historical_klines(symbol, interval, limit=1000)
    label_by_open = {candle.open_time.isoformat(): int(candle.close > candle.open) for candle in candles}
    close_by_open = {candle.open_time.isoformat(): candle.close for candle in candles}

    labeled: list[dict[str, object]] = []
    for row in rows:
        open_time = row["candle_open_time"]
        if open_time not in label_by_open:
            continue
        enriched: dict[str, object] = dict(row)
        enriched["label_up"] = label_by_open[open_time]
        enriched["final_close"] = close_by_open[open_time]
        labeled.append(enriched)

    if not labeled:
        raise ValueError("No rows could be labeled. Wait until watched candles are closed, then try again.")

    fieldnames = list(labeled[0].keys())
    with Path(output_csv).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(labeled)

    return IntrabarLabelReport(input_csv=input_csv, output_csv=output_csv, labeled_rows=len(labeled))


def train_intrabar_model(
    input_csv: str,
    output_model: str = "btc_intrabar_model.json",
    train_fraction: float = 0.70,
) -> IntrabarTrainingReport:
    rows = [row for row in _read_rows(input_csv) if row.get("label_up") not in {"", None}]
    dataset = [(_feature_row(row), int(float(row["label_up"]))) for row in rows if _has_feature_columns(row)]
    if len(dataset) < 100:
        raise ValueError("Need at least 100 labeled intrabar rows. Collect more with intrabar_watch_5m.bat.")

    split = max(1, min(len(dataset) - 1, int(len(dataset) * train_fraction)))
    train = dataset[:split]
    test = dataset[split:]

    scaler = RobustScaler()
    scaler.fit([features for features, _ in train])
    x_train = [scaler.transform(features) for features, _ in train]
    y_train = [target for _, target in train]
    x_test = [scaler.transform(features) for features, _ in test]
    y_test = [target for _, target in test]

    model = LogisticRegression()
    model.fit(x_train, y_train)
    train_probs = model.predict_proba(x_train)
    test_probs = model.predict_proba(x_test)

    payload = {
        "label_type": "intrabar_candle_direction",
        "feature_columns": INTRABAR_FEATURE_COLUMNS,
        "weights": model.weights,
        "bias": model.bias,
        "centers": scaler.centers,
        "scales": scaler.scales,
    }
    Path(output_model).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return IntrabarTrainingReport(
        input_csv=input_csv,
        output_model=output_model,
        samples=len(dataset),
        train_samples=len(train),
        test_samples=len(test),
        positive_rate=sum(target for _, target in dataset) / len(dataset),
        train_metrics=_classification_metrics(y_train, train_probs),
        test_metrics=_classification_metrics(y_test, test_probs),
        top_weights=_top_weights(model.weights),
    )


def _classification_metrics(targets: list[int], probabilities: list[float]) -> ClassificationMetrics:
    predictions = [1 if probability >= 0.5 else 0 for probability in probabilities]
    tp = sum(1 for y, p in zip(targets, predictions) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(targets, predictions) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(targets, predictions) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(targets, predictions) if y == 1 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    brier = mean([(probability - target) ** 2 for target, probability in zip(targets, probabilities)])
    return ClassificationMetrics(
        samples=len(targets),
        positive_rate=sum(targets) / len(targets),
        accuracy=(tp + tn) / len(targets),
        precision=precision,
        recall=recall,
        f1=f1,
        brier=brier,
    )


def _read_rows(path: str) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _has_feature_columns(row: dict[str, str]) -> bool:
    return all(row.get(name) not in {"", None} for name in INTRABAR_FEATURE_COLUMNS)


def _feature_row(row: dict[str, str]) -> list[float]:
    return [float(row[name]) for name in INTRABAR_FEATURE_COLUMNS]


def _top_weights(weights: list[float], limit: int = 10) -> list[dict[str, float]]:
    pairs = sorted(zip(INTRABAR_FEATURE_COLUMNS, weights), key=lambda item: abs(item[1]), reverse=True)
    return [{"feature": name, "weight": weight} for name, weight in pairs[:limit]]
