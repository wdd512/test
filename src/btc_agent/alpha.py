from __future__ import annotations

import math

from .models import IntervalAnalysis


def analyze_interval(interval: str, features: dict[str, float], min_confidence: float) -> IntervalAnalysis:
    score = _score(features)
    confidence = _confidence(score, features)
    signal = _signal(score, confidence, min_confidence)
    reasons = _reasons(features, score)
    return IntervalAnalysis(
        interval=interval,
        features=features,
        signal=signal,
        confidence=confidence,
        score=score,
        reasons=reasons,
    )


def combine_analyses(analyses: list[IntervalAnalysis], min_confidence: float) -> tuple[str, float]:
    if not analyses:
        return "HOLD", 0.0

    weights = {"5m": 0.58, "15m": 0.42}
    weighted_score = 0.0
    total_weight = 0.0
    for item in analyses:
        weight = weights.get(item.interval, 1.0 / len(analyses))
        weighted_score += item.score * item.confidence * weight
        total_weight += weight

    combined_score = weighted_score / max(total_weight, 1e-12)
    confidence = _sigmoid(abs(combined_score) * 1.65)
    if confidence < min_confidence or abs(combined_score) < 0.18:
        return "HOLD", confidence
    return ("LONG" if combined_score > 0 else "SHORT"), confidence


def _score(f: dict[str, float]) -> float:
    trend = 1.15 * f["return_3"] + 0.85 * f["return_6"] + 0.55 * f["return_12"]
    trend_scaled = trend / max(f["realized_vol_20"], 0.0005)

    flow = (
        0.80 * f["taker_imbalance"]
        + 0.65 * f["book_imbalance"]
        + 0.45 * math.tanh(f["trade_intensity_z"] / 2)
        + 0.35 * math.tanh(f["volume_z"] / 2)
    )

    technical = (
        0.45 * _centered_rsi(f["rsi_14"])
        + 0.35 * math.tanh(f["macd_hist"] / max(f["close"] * 0.0008, 1e-12))
        + 0.25 * _bb_bias(f["bb_position"])
        - 0.20 * math.tanh(f["vwap_deviation"] / 0.003)
    )

    micro = 0.55 * math.tanh(f["microprice_deviation"] / 0.0004)
    spread_penalty = min(max(f["spread_pct"] / 0.0008, 0.0), 2.0) * 0.22
    volatility_penalty = max(f["volatility_ratio"] - 1.7, 0.0) * 0.25

    raw_score = 0.42 * trend_scaled + 0.32 * flow + 0.22 * technical + micro
    raw_score -= math.copysign(spread_penalty + volatility_penalty, raw_score)
    return max(min(raw_score, 3.0), -3.0)


def _confidence(score: float, f: dict[str, float]) -> float:
    base = _sigmoid(abs(score) * 1.55)
    liquidity_drag = min(f["spread_pct"] / 0.0012, 0.35)
    unstable_vol_drag = min(max(f["volatility_ratio"] - 2.0, 0.0) * 0.10, 0.20)
    return max(0.0, min(1.0, base - liquidity_drag - unstable_vol_drag))


def _signal(score: float, confidence: float, min_confidence: float) -> str:
    if confidence < min_confidence or abs(score) < 0.18:
        return "HOLD"
    return "LONG" if score > 0 else "SHORT"


def _reasons(f: dict[str, float], score: float) -> list[str]:
    reasons: list[str] = []
    if abs(f["return_6"]) > f["realized_vol_20"] * 0.55:
        direction = "upward" if f["return_6"] > 0 else "downward"
        reasons.append(f"{direction} short-horizon momentum")
    if abs(f["taker_imbalance"]) > 0.18:
        side = "buy" if f["taker_imbalance"] > 0 else "sell"
        reasons.append(f"strong taker {side} imbalance")
    if abs(f["book_imbalance"]) > 0.18:
        side = "bid" if f["book_imbalance"] > 0 else "ask"
        reasons.append(f"order book tilted to {side} side")
    if f["spread_pct"] > 0.0008:
        reasons.append("spread is elevated, execution risk is higher")
    if f["volatility_ratio"] > 1.7:
        reasons.append("fast volatility is expanding")
    if not reasons:
        reasons.append("mixed or weak signal; no dominant feature block")
    if abs(score) >= 1.2:
        reasons.append("multi-feature score is strong")
    return reasons


def _centered_rsi(rsi: float) -> float:
    if rsi > 70:
        return -0.7
    if rsi < 30:
        return 0.7
    return (rsi - 50) / 50


def _bb_bias(position: float) -> float:
    if position > 0.9:
        return -0.45
    if position < 0.1:
        return 0.45
    return position - 0.5


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))
