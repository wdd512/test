from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .features import build_features
from .ml_baseline import FEATURE_COLUMNS, LogisticRegression, RobustScaler
from .models import Candle


@dataclass(frozen=True)
class IntrabarSignal:
    symbol: str
    interval: str
    candle_open_time: str
    candle_open: float
    last_price: float
    elapsed_fraction: float
    current_return: float
    candle_range_pct: float
    close_location: float
    taker_imbalance: float
    base_probability_up: float
    dynamic_probability_up: float
    base_prediction: str
    dynamic_prediction: str
    flipped: bool
    confidence: float
    threshold: float
    flip_threshold: float
    reasons: list[str]


def dynamic_candle_signal(
    candles: list[Candle],
    model_path: str,
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    threshold: float = 0.52,
    flip_threshold: float = 0.535,
) -> IntrabarSignal:
    payload = json.loads(Path(model_path).read_text(encoding="utf-8"))
    if payload.get("label_type") != "candle_direction":
        raise ValueError("Model is not a candle_direction model.")
    if payload["feature_columns"] != FEATURE_COLUMNS:
        raise ValueError("Model feature columns do not match current FEATURE_COLUMNS.")

    lookback = int(payload["lookback"])
    if len(candles) < lookback + 1:
        raise ValueError(f"Need at least {lookback + 1} candles.")

    current = candles[-1]
    history = candles[-lookback - 1 : -1]
    feature_map = build_features(history)
    row = [feature_map[name] for name in FEATURE_COLUMNS]

    scaler = RobustScaler()
    scaler.centers = payload["centers"]
    scaler.scales = payload["scales"]
    model = LogisticRegression()
    model.weights = payload["weights"]
    model.bias = payload["bias"]

    base_probability = model.predict_proba_one(scaler.transform(row))
    intrabar_features = _intrabar_features(current)
    dynamic_probability, reasons = _apply_intrabar_overlay(base_probability, current, feature_map, intrabar_features)
    base_prediction = _classify(base_probability, threshold)
    dynamic_prediction = _classify(dynamic_probability, flip_threshold)
    confidence = max(dynamic_probability, 1 - dynamic_probability)

    return IntrabarSignal(
        symbol=symbol,
        interval=interval,
        candle_open_time=current.open_time.isoformat(),
        candle_open=current.open,
        last_price=current.close,
        elapsed_fraction=intrabar_features["elapsed_fraction"],
        current_return=intrabar_features["current_return"],
        candle_range_pct=intrabar_features["candle_range_pct"],
        close_location=intrabar_features["close_location"],
        taker_imbalance=intrabar_features["taker_imbalance"],
        base_probability_up=base_probability,
        dynamic_probability_up=dynamic_probability,
        base_prediction=base_prediction,
        dynamic_prediction=dynamic_prediction,
        flipped=base_prediction in {"UP", "DOWN"} and dynamic_prediction in {"UP", "DOWN"} and base_prediction != dynamic_prediction,
        confidence=confidence,
        threshold=threshold,
        flip_threshold=flip_threshold,
        reasons=reasons,
    )


def watch_intrabar(
    market_data,
    symbol: str,
    interval: str,
    model_path: str,
    seconds: int,
    poll_seconds: int,
    out_csv: str,
    threshold: float = 0.52,
    flip_threshold: float = 0.535,
) -> list[IntrabarSignal]:
    signals: list[IntrabarSignal] = []
    deadline = time.time() + seconds
    output_path = Path(out_csv)
    write_header = not output_path.exists()
    with output_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=_intrabar_csv_fields())
        if write_header:
            writer.writeheader()
        while time.time() <= deadline:
            try:
                candles = market_data.klines(symbol, interval, 260)
                signal = dynamic_candle_signal(
                    candles,
                    model_path=model_path,
                    symbol=symbol,
                    interval=interval,
                    threshold=threshold,
                    flip_threshold=flip_threshold,
                )
                signals.append(signal)
                writer.writerow(_flatten_signal(signal))
                file.flush()
                print(json.dumps(asdict(signal), ensure_ascii=False), flush=True)
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "level": "error",
                            "message": "intrabar poll failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            time.sleep(poll_seconds)
    return signals


def _apply_intrabar_overlay(
    base_probability: float,
    current: Candle,
    previous_features: dict[str, float],
    intrabar_features: dict[str, float],
) -> tuple[float, list[str]]:
    reasons: list[str] = []
    elapsed = intrabar_features["elapsed_fraction"]
    current_return = intrabar_features["current_return"]
    atr_pct = max(previous_features.get("atr_pct", 0.0), 0.0003)
    range_pct = intrabar_features["candle_range_pct"]
    close_location = intrabar_features["close_location"]
    taker_imbalance = intrabar_features["taker_imbalance"]

    pressure = math.tanh(current_return / max(atr_pct * 0.75, 0.00025))
    location_pressure = (close_location - 0.5) * min(range_pct / max(atr_pct, 1e-12), 2.0)
    flow_pressure = 0.45 * taker_imbalance
    early_dampener = min(max(elapsed, 0.15), 1.0)
    adjustment = early_dampener * (0.80 * pressure + 0.35 * location_pressure + flow_pressure)

    if current_return > atr_pct * 0.35:
        reasons.append("current candle is moving above open")
    elif current_return < -atr_pct * 0.35:
        reasons.append("current candle is moving below open")
    if close_location > 0.72:
        reasons.append("price is holding near candle high")
    elif close_location < 0.28:
        reasons.append("price is holding near candle low")
    if abs(taker_imbalance) > 0.18:
        reasons.append("partial candle taker flow is imbalanced")
    if not reasons:
        reasons.append("intrabar evidence is weak")

    dynamic_logit = _logit(base_probability) + adjustment
    return _sigmoid(dynamic_logit), reasons


def _intrabar_features(current: Candle) -> dict[str, float]:
    candle_range = current.high - current.low
    return {
        "elapsed_fraction": _elapsed_fraction(current),
        "current_return": (current.close / current.open) - 1 if current.open > 0 else 0.0,
        "candle_range_pct": candle_range / current.open if current.open > 0 else 0.0,
        "close_location": (current.close - current.low) / max(candle_range, 1e-12),
        "taker_imbalance": (
            current.taker_buy_base_volume - (current.volume - current.taker_buy_base_volume)
        )
        / max(current.volume, 1e-12),
    }


def _classify(probability_up: float, threshold: float) -> str:
    if probability_up >= threshold:
        return "UP"
    if probability_up <= 1 - threshold:
        return "DOWN"
    return "NO_EDGE"


def _elapsed_fraction(candle: Candle) -> float:
    now = datetime.now(timezone.utc)
    total = max((candle.close_time - candle.open_time).total_seconds(), 1.0)
    elapsed = (now - candle.open_time).total_seconds()
    return max(0.0, min(1.0, elapsed / total))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def _logit(value: float) -> float:
    clipped = min(max(value, 1e-6), 1 - 1e-6)
    return math.log(clipped / (1 - clipped))


def _intrabar_csv_fields() -> list[str]:
    return [
        "logged_at",
        "symbol",
        "interval",
        "candle_open_time",
        "candle_open",
        "last_price",
        "elapsed_fraction",
        "current_return",
        "candle_range_pct",
        "close_location",
        "taker_imbalance",
        "base_probability_up",
        "dynamic_probability_up",
        "base_prediction",
        "dynamic_prediction",
        "flipped",
        "confidence",
        "reasons",
    ]


def _flatten_signal(signal: IntrabarSignal) -> dict[str, object]:
    return {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "symbol": signal.symbol,
        "interval": signal.interval,
        "candle_open_time": signal.candle_open_time,
        "candle_open": signal.candle_open,
        "last_price": signal.last_price,
        "elapsed_fraction": signal.elapsed_fraction,
        "current_return": signal.current_return,
        "candle_range_pct": signal.candle_range_pct,
        "close_location": signal.close_location,
        "taker_imbalance": signal.taker_imbalance,
        "base_probability_up": signal.base_probability_up,
        "dynamic_probability_up": signal.dynamic_probability_up,
        "base_prediction": signal.base_prediction,
        "dynamic_prediction": signal.dynamic_prediction,
        "flipped": signal.flipped,
        "confidence": signal.confidence,
        "reasons": " | ".join(signal.reasons),
    }
