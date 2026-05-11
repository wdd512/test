from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

from .features import build_features
from .models import Candle


FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_6",
    "return_12",
    "realized_vol_20",
    "realized_vol_60",
    "volatility_ratio",
    "atr_pct",
    "range_pct",
    "close_location",
    "vwap_deviation",
    "volume_z",
    "trade_intensity_z",
    "taker_imbalance",
    "rsi_14",
    "macd_hist",
    "bb_position",
    "hour_utc",
    "weekend",
]


@dataclass(frozen=True)
class DatasetRow:
    features: list[float]
    feature_map: dict[str, float]
    target: int
    forward_return: float


@dataclass(frozen=True)
class ClassificationMetrics:
    samples: int
    positive_rate: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    brier: float


@dataclass(frozen=True)
class TradeMetrics:
    mode: str
    threshold: float
    trades: int
    wins: int
    losses: int
    hit_rate: float
    cumulative_return: float
    avg_net_return: float
    max_drawdown: float
    profit_factor: float


@dataclass(frozen=True)
class TrainingReport:
    feature_columns: list[str]
    train_samples: int
    test_samples: int
    horizon: int
    train_positive_rate: float
    test_positive_rate: float
    train_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics
    train_probability_summary: dict[str, float]
    test_probability_summary: dict[str, float]
    best_trade: TradeMetrics
    top_trades: list[TradeMetrics]
    top_weights: list[dict[str, float]]


@dataclass(frozen=True)
class CandleDirectionSignal:
    symbol: str
    interval: str
    current_open_time: str
    current_open: float
    probability_up: float
    prediction: str
    confidence: float
    threshold: float
    model_path: str


@dataclass(frozen=True)
class CandleDirectionWalkForwardReport:
    cost: float
    mode: str
    threshold: float
    train_size: int
    test_size: int
    folds: list[WalkForwardFold]
    total_trades: int
    profitable_folds: int
    average_fold_return: float
    compounded_return: float
    average_profit_factor: float
    average_hit_rate: float


@dataclass(frozen=True)
class ExperimentSummary:
    horizon: int
    cost: float
    train_samples: int
    test_samples: int
    test_positive_rate: float
    test_precision: float
    test_recall: float
    test_brier: float
    best_mode: str
    best_threshold: float
    best_trades: int
    best_cumulative_return: float
    best_profit_factor: float
    best_hit_rate: float
    best_avg_net_return: float


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    train_samples: int
    test_samples: int
    test_positive_rate: float
    test_precision: float
    test_recall: float
    test_brier: float
    trade: TradeMetrics


@dataclass(frozen=True)
class WalkForwardReport:
    horizon: int
    cost: float
    mode: str
    threshold: float
    train_size: int
    test_size: int
    folds: list[WalkForwardFold]
    total_trades: int
    profitable_folds: int
    average_fold_return: float
    compounded_return: float
    average_profit_factor: float
    average_hit_rate: float


@dataclass(frozen=True)
class RegimeFilter:
    max_atr_pct: float | None = None
    max_abs_vwap_deviation: float | None = None
    max_volatility_ratio: float | None = None
    min_volume_z: float | None = None
    max_volume_z: float | None = None
    weekdays_only: bool = False
    weekends_only: bool = False


@dataclass(frozen=True)
class RegimeWalkForwardFold:
    fold: int
    train_filter_return: float
    selected_filter: RegimeFilter
    train_trade: TradeMetrics
    test_trade: TradeMetrics
    test_precision: float
    test_recall: float
    test_brier: float


@dataclass(frozen=True)
class RegimeWalkForwardReport:
    horizon: int
    cost: float
    mode: str
    threshold: float
    train_size: int
    test_size: int
    min_train_trades: int
    folds: list[RegimeWalkForwardFold]
    total_trades: int
    profitable_folds: int
    compounded_return: float
    average_fold_return: float
    average_profit_factor: float
    average_hit_rate: float


class RobustScaler:
    def __init__(self) -> None:
        self.centers: list[float] = []
        self.scales: list[float] = []

    def fit(self, rows: list[list[float]]) -> None:
        columns = list(zip(*rows))
        self.centers = [median(col) for col in columns]
        self.scales = []
        for col, center in zip(columns, self.centers):
            deviations = [abs(value - center) for value in col]
            scale = median(deviations) * 1.4826
            self.scales.append(scale if scale > 1e-9 else 1.0)

    def transform(self, row: list[float]) -> list[float]:
        return [(value - center) / scale for value, center, scale in zip(row, self.centers, self.scales)]


class LogisticRegression:
    def __init__(self, learning_rate: float = 0.035, epochs: int = 900, l2: float = 0.003) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.weights: list[float] = []
        self.bias = 0.0

    def fit(self, rows: list[list[float]], targets: list[int]) -> None:
        feature_count = len(rows[0])
        self.weights = [0.0] * feature_count
        self.bias = _logit(sum(targets) / len(targets))
        n = len(rows)
        for _ in range(self.epochs):
            grad_w = [0.0] * feature_count
            grad_b = 0.0
            for row, target in zip(rows, targets):
                probability = self.predict_proba_one(row)
                error = probability - target
                grad_b += error
                for index, value in enumerate(row):
                    grad_w[index] += error * value
            for index in range(feature_count):
                grad = grad_w[index] / n + self.l2 * self.weights[index]
                self.weights[index] -= self.learning_rate * grad
            self.bias -= self.learning_rate * grad_b / n

    def predict_proba_one(self, row: list[float]) -> float:
        z = self.bias + sum(weight * value for weight, value in zip(self.weights, row))
        return _sigmoid(z)

    def predict_proba(self, rows: list[list[float]]) -> list[float]:
        return [self.predict_proba_one(row) for row in rows]


def build_supervised_dataset(candles: list[Candle], lookback: int, horizon: int, cost: float) -> list[DatasetRow]:
    rows: list[DatasetRow] = []
    for end in range(lookback, len(candles) - horizon):
        window = candles[end - lookback : end]
        current_close = candles[end - 1].close
        future_close = candles[end + horizon - 1].close
        forward_return = (future_close / current_close) - 1
        target = 1 if forward_return > cost else 0
        feature_map = build_features(window)
        rows.append(DatasetRow([feature_map[name] for name in FEATURE_COLUMNS], feature_map, target, forward_return))
    return rows


def build_candle_direction_dataset(candles: list[Candle], lookback: int, cost: float = 0.0) -> list[DatasetRow]:
    rows: list[DatasetRow] = []
    for target_index in range(lookback, len(candles)):
        window = candles[target_index - lookback : target_index]
        target_candle = candles[target_index]
        candle_return = (target_candle.close / target_candle.open) - 1
        target = 1 if candle_return > cost else 0
        feature_map = build_features(window)
        rows.append(DatasetRow([feature_map[name] for name in FEATURE_COLUMNS], feature_map, target, candle_return))
    return rows


def train_and_evaluate(
    candles: list[Candle],
    lookback: int = 240,
    horizon: int = 3,
    train_fraction: float = 0.70,
    cost: float = 0.0008,
    output_model: str | None = None,
) -> TrainingReport:
    dataset = build_supervised_dataset(candles, lookback=lookback, horizon=horizon, cost=cost)
    if len(dataset) < 100:
        raise ValueError("Need at least 100 supervised rows. Fetch more candles first.")

    split = max(1, min(len(dataset) - 1, int(len(dataset) * train_fraction)))
    train_rows = dataset[:split]
    test_rows = dataset[split:]

    scaler = RobustScaler()
    scaler.fit([row.features for row in train_rows])
    x_train = [scaler.transform(row.features) for row in train_rows]
    y_train = [row.target for row in train_rows]
    x_test = [scaler.transform(row.features) for row in test_rows]
    y_test = [row.target for row in test_rows]

    model = LogisticRegression()
    model.fit(x_train, y_train)

    train_probs = model.predict_proba(x_train)
    test_probs = model.predict_proba(x_test)
    trade_results = _trade_sweep(test_rows, test_probs, cost=cost)
    top_weights = _top_weights(model.weights)

    if output_model:
        _save_model(output_model, model, scaler, lookback, horizon, cost, label_type="forward_return")

    return TrainingReport(
        feature_columns=FEATURE_COLUMNS,
        train_samples=len(train_rows),
        test_samples=len(test_rows),
        horizon=horizon,
        train_positive_rate=sum(y_train) / len(y_train),
        test_positive_rate=sum(y_test) / len(y_test),
        train_metrics=_classification_metrics(y_train, train_probs),
        test_metrics=_classification_metrics(y_test, test_probs),
        train_probability_summary=_probability_summary(train_probs),
        test_probability_summary=_probability_summary(test_probs),
        best_trade=trade_results[0],
        top_trades=trade_results[:10],
        top_weights=top_weights,
    )


def train_candle_direction(
    candles: list[Candle],
    lookback: int = 240,
    train_fraction: float = 0.70,
    cost: float = 0.0,
    output_model: str | None = None,
    sample_step: int = 1,
    epochs: int = 900,
) -> TrainingReport:
    dataset = build_candle_direction_dataset(candles, lookback=lookback, cost=cost)
    if sample_step > 1:
        dataset = dataset[::sample_step]
    return _train_dataset_report(
        dataset,
        lookback=lookback,
        horizon=1,
        train_fraction=train_fraction,
        cost=cost,
        output_model=output_model,
        label_type="candle_direction",
        epochs=epochs,
    )


def _train_dataset_report(
    dataset: list[DatasetRow],
    lookback: int,
    horizon: int,
    train_fraction: float,
    cost: float,
    output_model: str | None,
    label_type: str,
    epochs: int = 900,
) -> TrainingReport:
    if len(dataset) < 100:
        raise ValueError("Need at least 100 supervised rows. Fetch more candles first.")

    split = max(1, min(len(dataset) - 1, int(len(dataset) * train_fraction)))
    train_rows = dataset[:split]
    test_rows = dataset[split:]

    scaler = RobustScaler()
    scaler.fit([row.features for row in train_rows])
    x_train = [scaler.transform(row.features) for row in train_rows]
    y_train = [row.target for row in train_rows]
    x_test = [scaler.transform(row.features) for row in test_rows]
    y_test = [row.target for row in test_rows]

    model = LogisticRegression(epochs=epochs)
    model.fit(x_train, y_train)

    train_probs = model.predict_proba(x_train)
    test_probs = model.predict_proba(x_test)
    trade_results = _trade_sweep(test_rows, test_probs, cost=cost)
    top_weights = _top_weights(model.weights)

    if output_model:
        _save_model(output_model, model, scaler, lookback, horizon, cost, label_type=label_type)

    return TrainingReport(
        feature_columns=FEATURE_COLUMNS,
        train_samples=len(train_rows),
        test_samples=len(test_rows),
        horizon=horizon,
        train_positive_rate=sum(y_train) / len(y_train),
        test_positive_rate=sum(y_test) / len(y_test),
        train_metrics=_classification_metrics(y_train, train_probs),
        test_metrics=_classification_metrics(y_test, test_probs),
        train_probability_summary=_probability_summary(train_probs),
        test_probability_summary=_probability_summary(test_probs),
        best_trade=trade_results[0],
        top_trades=trade_results[:10],
        top_weights=top_weights,
    )


def run_experiment_matrix(
    candles: list[Candle],
    lookback: int = 240,
    horizons: tuple[int, ...] = (1, 3, 6, 12),
    costs: tuple[float, ...] = (0.0, 0.0008),
    train_fraction: float = 0.70,
) -> list[ExperimentSummary]:
    summaries: list[ExperimentSummary] = []
    for horizon in horizons:
        for cost in costs:
            report = train_and_evaluate(
                candles,
                lookback=lookback,
                horizon=horizon,
                train_fraction=train_fraction,
                cost=cost,
                output_model=None,
            )
            best = report.best_trade
            summaries.append(
                ExperimentSummary(
                    horizon=horizon,
                    cost=cost,
                    train_samples=report.train_samples,
                    test_samples=report.test_samples,
                    test_positive_rate=report.test_positive_rate,
                    test_precision=report.test_metrics.precision,
                    test_recall=report.test_metrics.recall,
                    test_brier=report.test_metrics.brier,
                    best_mode=best.mode,
                    best_threshold=best.threshold,
                    best_trades=best.trades,
                    best_cumulative_return=best.cumulative_return,
                    best_profit_factor=best.profit_factor,
                    best_hit_rate=best.hit_rate,
                    best_avg_net_return=best.avg_net_return,
                )
            )
    return sorted(
        summaries,
        key=lambda item: (item.best_cumulative_return, item.best_profit_factor, item.best_trades),
        reverse=True,
    )


def walk_forward_evaluate(
    candles: list[Candle],
    lookback: int = 240,
    horizon: int = 12,
    cost: float = 0.0008,
    train_size: int = 3000,
    test_size: int = 750,
    mode: str = "long_only",
    threshold: float = 0.52,
) -> WalkForwardReport:
    dataset = build_supervised_dataset(candles, lookback=lookback, horizon=horizon, cost=cost)
    if len(dataset) < train_size + test_size:
        raise ValueError("Not enough rows for requested walk-forward train/test sizes.")

    folds: list[WalkForwardFold] = []
    start = 0
    fold_number = 1
    while start + train_size + test_size <= len(dataset):
        train_rows = dataset[start : start + train_size]
        test_rows = dataset[start + train_size : start + train_size + test_size]
        scaler = RobustScaler()
        scaler.fit([row.features for row in train_rows])
        x_train = [scaler.transform(row.features) for row in train_rows]
        y_train = [row.target for row in train_rows]
        x_test = [scaler.transform(row.features) for row in test_rows]
        y_test = [row.target for row in test_rows]

        model = LogisticRegression()
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)
        trade = evaluate_fixed_policy(test_rows, probabilities, mode=mode, threshold=threshold, cost=cost)
        metrics = _classification_metrics(y_test, probabilities)
        folds.append(
            WalkForwardFold(
                fold=fold_number,
                train_start=start,
                train_end=start + train_size - 1,
                test_start=start + train_size,
                test_end=start + train_size + test_size - 1,
                train_samples=len(train_rows),
                test_samples=len(test_rows),
                test_positive_rate=sum(y_test) / len(y_test),
                test_precision=metrics.precision,
                test_recall=metrics.recall,
                test_brier=metrics.brier,
                trade=trade,
            )
        )
        start += test_size
        fold_number += 1

    compounded = 1.0
    for fold in folds:
        compounded *= 1 + fold.trade.cumulative_return

    return WalkForwardReport(
        horizon=horizon,
        cost=cost,
        mode=mode,
        threshold=threshold,
        train_size=train_size,
        test_size=test_size,
        folds=folds,
        total_trades=sum(fold.trade.trades for fold in folds),
        profitable_folds=sum(1 for fold in folds if fold.trade.cumulative_return > 0),
        average_fold_return=mean([fold.trade.cumulative_return for fold in folds]) if folds else 0.0,
        compounded_return=compounded - 1,
        average_profit_factor=mean([fold.trade.profit_factor for fold in folds]) if folds else 0.0,
        average_hit_rate=mean([fold.trade.hit_rate for fold in folds]) if folds else 0.0,
    )


def walk_forward_regime_evaluate(
    candles: list[Candle],
    lookback: int = 240,
    horizon: int = 12,
    cost: float = 0.0008,
    train_size: int = 3000,
    test_size: int = 750,
    mode: str = "long_only",
    threshold: float = 0.52,
    min_train_trades: int = 20,
) -> RegimeWalkForwardReport:
    dataset = build_supervised_dataset(candles, lookback=lookback, horizon=horizon, cost=cost)
    if len(dataset) < train_size + test_size:
        raise ValueError("Not enough rows for requested walk-forward train/test sizes.")

    folds: list[RegimeWalkForwardFold] = []
    start = 0
    fold_number = 1
    while start + train_size + test_size <= len(dataset):
        train_rows = dataset[start : start + train_size]
        test_rows = dataset[start + train_size : start + train_size + test_size]
        scaler = RobustScaler()
        scaler.fit([row.features for row in train_rows])
        x_train = [scaler.transform(row.features) for row in train_rows]
        y_train = [row.target for row in train_rows]
        x_test = [scaler.transform(row.features) for row in test_rows]
        y_test = [row.target for row in test_rows]

        model = LogisticRegression()
        model.fit(x_train, y_train)
        train_probs = model.predict_proba(x_train)
        test_probs = model.predict_proba(x_test)
        selected_filter, train_trade = select_regime_filter(
            train_rows,
            train_probs,
            mode=mode,
            threshold=threshold,
            cost=cost,
            min_train_trades=min_train_trades,
        )
        test_trade = evaluate_fixed_policy(
            test_rows,
            test_probs,
            mode=mode,
            threshold=threshold,
            cost=cost,
            regime_filter=selected_filter,
        )
        metrics = _classification_metrics(y_test, test_probs)
        folds.append(
            RegimeWalkForwardFold(
                fold=fold_number,
                train_filter_return=train_trade.cumulative_return,
                selected_filter=selected_filter,
                train_trade=train_trade,
                test_trade=test_trade,
                test_precision=metrics.precision,
                test_recall=metrics.recall,
                test_brier=metrics.brier,
            )
        )
        start += test_size
        fold_number += 1

    compounded = 1.0
    for fold in folds:
        compounded *= 1 + fold.test_trade.cumulative_return

    return RegimeWalkForwardReport(
        horizon=horizon,
        cost=cost,
        mode=mode,
        threshold=threshold,
        train_size=train_size,
        test_size=test_size,
        min_train_trades=min_train_trades,
        folds=folds,
        total_trades=sum(fold.test_trade.trades for fold in folds),
        profitable_folds=sum(1 for fold in folds if fold.test_trade.cumulative_return > 0),
        compounded_return=compounded - 1,
        average_fold_return=mean([fold.test_trade.cumulative_return for fold in folds]) if folds else 0.0,
        average_profit_factor=mean([fold.test_trade.profit_factor for fold in folds]) if folds else 0.0,
        average_hit_rate=mean([fold.test_trade.hit_rate for fold in folds]) if folds else 0.0,
    )


def walk_forward_candle_direction(
    candles: list[Candle],
    lookback: int = 240,
    cost: float = 0.0,
    train_size: int = 3000,
    test_size: int = 750,
    mode: str = "long_short",
    threshold: float = 0.52,
) -> CandleDirectionWalkForwardReport:
    dataset = build_candle_direction_dataset(candles, lookback=lookback, cost=cost)
    if len(dataset) < train_size + test_size:
        raise ValueError("Not enough rows for requested candle-direction walk-forward sizes.")

    folds: list[WalkForwardFold] = []
    start = 0
    fold_number = 1
    while start + train_size + test_size <= len(dataset):
        train_rows = dataset[start : start + train_size]
        test_rows = dataset[start + train_size : start + train_size + test_size]
        scaler = RobustScaler()
        scaler.fit([row.features for row in train_rows])
        x_train = [scaler.transform(row.features) for row in train_rows]
        y_train = [row.target for row in train_rows]
        x_test = [scaler.transform(row.features) for row in test_rows]
        y_test = [row.target for row in test_rows]

        model = LogisticRegression()
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)
        trade = evaluate_fixed_policy(test_rows, probabilities, mode=mode, threshold=threshold, cost=cost)
        metrics = _classification_metrics(y_test, probabilities)
        folds.append(
            WalkForwardFold(
                fold=fold_number,
                train_start=start,
                train_end=start + train_size - 1,
                test_start=start + train_size,
                test_end=start + train_size + test_size - 1,
                train_samples=len(train_rows),
                test_samples=len(test_rows),
                test_positive_rate=sum(y_test) / len(y_test),
                test_precision=metrics.precision,
                test_recall=metrics.recall,
                test_brier=metrics.brier,
                trade=trade,
            )
        )
        start += test_size
        fold_number += 1

    compounded = 1.0
    for fold in folds:
        compounded *= 1 + fold.trade.cumulative_return

    return CandleDirectionWalkForwardReport(
        cost=cost,
        mode=mode,
        threshold=threshold,
        train_size=train_size,
        test_size=test_size,
        folds=folds,
        total_trades=sum(fold.trade.trades for fold in folds),
        profitable_folds=sum(1 for fold in folds if fold.trade.cumulative_return > 0),
        average_fold_return=mean([fold.trade.cumulative_return for fold in folds]) if folds else 0.0,
        compounded_return=compounded - 1,
        average_profit_factor=mean([fold.trade.profit_factor for fold in folds]) if folds else 0.0,
        average_hit_rate=mean([fold.trade.hit_rate for fold in folds]) if folds else 0.0,
    )


def predict_candle_direction_live(
    candles: list[Candle],
    model_path: str,
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    threshold: float = 0.52,
) -> CandleDirectionSignal:
    payload = json.loads(Path(model_path).read_text(encoding="utf-8"))
    if payload.get("label_type") != "candle_direction":
        raise ValueError("Model is not a candle_direction model. Train it with train-candle-direction first.")
    if payload["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("Model feature columns do not match current FEATURE_COLUMNS.")

    lookback = int(payload["lookback"])
    if len(candles) < lookback + 1:
        raise ValueError(f"Need at least {lookback + 1} candles for live candle direction signal.")

    now = datetime.now(timezone.utc)
    current = candles[-1]
    history = candles[-lookback - 1 : -1]
    if current.close_time <= now and len(candles) >= lookback + 2:
        current = candles[-1]
        history = candles[-lookback - 1 : -1]
    if len(history) < lookback:
        raise ValueError(f"Need {lookback} closed candles before the current candle.")

    feature_map = build_features(history)
    row = [feature_map[name] for name in FEATURE_COLUMNS]
    scaler = RobustScaler()
    scaler.centers = payload["centers"]
    scaler.scales = payload["scales"]
    model = LogisticRegression()
    model.weights = payload["weights"]
    model.bias = payload["bias"]
    probability_up = model.predict_proba_one(scaler.transform(row))

    if probability_up >= threshold:
        prediction = "UP"
        confidence = probability_up
    elif probability_up <= 1 - threshold:
        prediction = "DOWN"
        confidence = 1 - probability_up
    else:
        prediction = "NO_EDGE"
        confidence = max(probability_up, 1 - probability_up)

    return CandleDirectionSignal(
        symbol=symbol,
        interval=interval,
        current_open_time=current.open_time.isoformat(),
        current_open=current.open,
        probability_up=probability_up,
        prediction=prediction,
        confidence=confidence,
        threshold=threshold,
        model_path=model_path,
    )


def select_regime_filter(
    rows: list[DatasetRow],
    probabilities: list[float],
    mode: str,
    threshold: float,
    cost: float,
    min_train_trades: int,
) -> tuple[RegimeFilter, TradeMetrics]:
    candidates = _regime_filter_candidates(rows)
    scored: list[tuple[RegimeFilter, TradeMetrics]] = []
    for candidate in candidates:
        trade = evaluate_fixed_policy(
            rows,
            probabilities,
            mode=mode,
            threshold=threshold,
            cost=cost,
            regime_filter=candidate,
        )
        if trade.trades >= min_train_trades:
            scored.append((candidate, trade))
    if not scored:
        empty = RegimeFilter()
        return empty, evaluate_fixed_policy(rows, probabilities, mode, threshold, cost, empty)
    return sorted(
        scored,
        key=lambda item: (item[1].cumulative_return, item[1].profit_factor, -item[1].max_drawdown),
        reverse=True,
    )[0]


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


def _trade_sweep(rows: list[DatasetRow], probabilities: list[float], cost: float) -> list[TradeMetrics]:
    results: list[TradeMetrics] = []
    for mode in ("long_only", "short_only", "long_short"):
        for threshold in (0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70):
            returns: list[float] = []
            for row, probability in zip(rows, probabilities):
                if mode in {"long_only", "long_short"} and probability >= threshold:
                    returns.append(row.forward_return - cost)
                elif mode in {"short_only", "long_short"} and probability <= 1 - threshold:
                    returns.append(-row.forward_return - cost)
            if len(returns) >= 5:
                results.append(_trade_metrics(mode, threshold, returns))
    return sorted(
        results,
        key=lambda item: (item.cumulative_return, item.profit_factor, -item.max_drawdown),
        reverse=True,
    )


def evaluate_fixed_policy(
    rows: list[DatasetRow],
    probabilities: list[float],
    mode: str,
    threshold: float,
    cost: float,
    regime_filter: RegimeFilter | None = None,
) -> TradeMetrics:
    returns: list[float] = []
    for row, probability in zip(rows, probabilities):
        if regime_filter and not _passes_regime_filter(row, regime_filter):
            continue
        if mode in {"long_only", "long_short"} and probability >= threshold:
            returns.append(row.forward_return - cost)
        elif mode in {"short_only", "long_short"} and probability <= 1 - threshold:
            returns.append(-row.forward_return - cost)
    return _trade_metrics(mode, threshold, returns)


def _regime_filter_candidates(rows: list[DatasetRow]) -> list[RegimeFilter]:
    atr_q = _feature_quantiles(rows, "atr_pct", (0.60, 0.70, 0.80, 0.90))
    vwap_q = _abs_feature_quantiles(rows, "vwap_deviation", (0.60, 0.70, 0.80, 0.90))
    filters = [RegimeFilter()]
    for atr in atr_q:
        filters.append(RegimeFilter(max_atr_pct=atr))
    for vwap in vwap_q:
        filters.append(RegimeFilter(max_abs_vwap_deviation=vwap))
    for vol_ratio in (1.0, 1.25, 1.5, 2.0):
        filters.append(RegimeFilter(max_volatility_ratio=vol_ratio))
    for volume_max in (1.0, 1.5, 2.0):
        filters.append(RegimeFilter(max_volume_z=volume_max))
    filters.extend([RegimeFilter(weekdays_only=True), RegimeFilter(weekends_only=True)])
    for atr in atr_q:
        for vwap in vwap_q:
            filters.append(RegimeFilter(max_atr_pct=atr, max_abs_vwap_deviation=vwap))
    for atr in atr_q:
        for vol_ratio in (1.25, 1.5, 2.0):
            filters.append(RegimeFilter(max_atr_pct=atr, max_volatility_ratio=vol_ratio))
    return filters


def _passes_regime_filter(row: DatasetRow, regime_filter: RegimeFilter) -> bool:
    features = row.feature_map
    if regime_filter.max_atr_pct is not None and features["atr_pct"] > regime_filter.max_atr_pct:
        return False
    if (
        regime_filter.max_abs_vwap_deviation is not None
        and abs(features["vwap_deviation"]) > regime_filter.max_abs_vwap_deviation
    ):
        return False
    if regime_filter.max_volatility_ratio is not None and features["volatility_ratio"] > regime_filter.max_volatility_ratio:
        return False
    if regime_filter.min_volume_z is not None and features["volume_z"] < regime_filter.min_volume_z:
        return False
    if regime_filter.max_volume_z is not None and features["volume_z"] > regime_filter.max_volume_z:
        return False
    if regime_filter.weekdays_only and features["weekend"] >= 0.5:
        return False
    if regime_filter.weekends_only and features["weekend"] < 0.5:
        return False
    return True


def _feature_quantiles(rows: list[DatasetRow], feature: str, quantiles: tuple[float, ...]) -> list[float]:
    values = sorted(row.feature_map[feature] for row in rows)
    return [_quantile(values, q) for q in quantiles]


def _abs_feature_quantiles(rows: list[DatasetRow], feature: str, quantiles: tuple[float, ...]) -> list[float]:
    values = sorted(abs(row.feature_map[feature]) for row in rows)
    return [_quantile(values, q) for q in quantiles]


def _trade_metrics(mode: str, threshold: float, returns: list[float]) -> TradeMetrics:
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    wins = 0
    for value in returns:
        equity *= 1 + value
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, (peak - equity) / peak)
        if value > 0:
            wins += 1
            gross_profit += value
        else:
            gross_loss += abs(value)
    trades = len(returns)
    return TradeMetrics(
        mode=mode,
        threshold=threshold,
        trades=trades,
        wins=wins,
        losses=trades - wins,
        hit_rate=wins / trades if trades else 0.0,
        cumulative_return=equity - 1,
        avg_net_return=sum(returns) / trades if trades else 0.0,
        max_drawdown=max_drawdown,
        profit_factor=gross_profit / gross_loss if gross_loss else 0.0,
    )


def _top_weights(weights: list[float], limit: int = 10) -> list[dict[str, float]]:
    pairs = sorted(zip(FEATURE_COLUMNS, weights), key=lambda item: abs(item[1]), reverse=True)
    return [{"feature": name, "weight": weight} for name, weight in pairs[:limit]]


def _probability_summary(probabilities: list[float]) -> dict[str, float]:
    ordered = sorted(probabilities)
    return {
        "min": ordered[0],
        "p10": _quantile(ordered, 0.10),
        "p50": _quantile(ordered, 0.50),
        "p90": _quantile(ordered, 0.90),
        "max": ordered[-1],
        "mean": mean(ordered),
    }


def _quantile(ordered_values: list[float], q: float) -> float:
    if not ordered_values:
        return 0.0
    index = (len(ordered_values) - 1) * q
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered_values[lower]
    weight = index - lower
    return ordered_values[lower] * (1 - weight) + ordered_values[upper] * weight


def _save_model(
    path: str,
    model: LogisticRegression,
    scaler: RobustScaler,
    lookback: int,
    horizon: int,
    cost: float,
    label_type: str,
) -> None:
    payload = {
        "label_type": label_type,
        "feature_columns": FEATURE_COLUMNS,
        "weights": model.weights,
        "bias": model.bias,
        "centers": scaler.centers,
        "scales": scaler.scales,
        "lookback": lookback,
        "horizon": horizon,
        "cost": cost,
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def _logit(value: float) -> float:
    clipped = min(max(value, 1e-6), 1 - 1e-6)
    return math.log(clipped / (1 - clipped))
