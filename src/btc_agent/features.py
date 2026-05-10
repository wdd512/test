from __future__ import annotations

import math
from statistics import mean, pstdev

from .models import Candle, OrderBookSnapshot


def build_features(candles: list[Candle], book: OrderBookSnapshot | None = None) -> dict[str, float]:
    if len(candles) < 60:
        raise ValueError("Need at least 60 candles to build robust short-horizon features.")

    closes = [c.close for c in candles]
    highs = [c.high for c in candles]
    lows = [c.low for c in candles]
    volumes = [c.volume for c in candles]
    last = candles[-1]

    returns = _log_returns(closes)
    realized_vol_20 = pstdev(returns[-20:]) * math.sqrt(20) if len(returns) >= 20 else 0.0
    realized_vol_60 = pstdev(returns[-60:]) * math.sqrt(60) if len(returns) >= 60 else 0.0
    atr_14 = _atr(candles, 14)
    atr_pct = atr_14 / max(last.close, 1e-12)
    vwap_30 = _vwap(candles[-30:])
    volume_z = _zscore(volumes[-1], volumes[-60:-1])
    trade_intensity_z = _zscore(float(last.trade_count), [float(c.trade_count) for c in candles[-60:-1]])
    taker_imbalance = _safe_divide(last.taker_buy_base_volume - (last.volume - last.taker_buy_base_volume), last.volume)

    features = {
        "close": last.close,
        "return_1": returns[-1],
        "return_3": _window_return(closes, 3),
        "return_6": _window_return(closes, 6),
        "return_12": _window_return(closes, 12),
        "realized_vol_20": realized_vol_20,
        "realized_vol_60": realized_vol_60,
        "volatility_ratio": _safe_divide(realized_vol_20, realized_vol_60),
        "atr_pct": atr_pct,
        "range_pct": _safe_divide(last.high - last.low, last.close),
        "close_location": _safe_divide(last.close - last.low, last.high - last.low),
        "vwap_deviation": _safe_divide(last.close - vwap_30, vwap_30),
        "volume_z": volume_z,
        "trade_intensity_z": trade_intensity_z,
        "taker_imbalance": taker_imbalance,
        "rsi_14": _rsi(closes, 14),
        "macd_hist": _macd_hist(closes),
        "bb_position": _bollinger_position(closes, 20),
        "hour_utc": float(last.close_time.hour),
        "weekend": 1.0 if last.close_time.weekday() >= 5 else 0.0,
    }

    if book:
        features.update(
            {
                "spread_pct": book.relative_spread,
                "book_imbalance": book.imbalance,
                "microprice_deviation": _safe_divide(book.microprice - last.close, last.close),
            }
        )
    else:
        features.update({"spread_pct": 0.0, "book_imbalance": 0.0, "microprice_deviation": 0.0})

    return features


def _log_returns(values: list[float]) -> list[float]:
    return [math.log(values[i] / values[i - 1]) for i in range(1, len(values)) if values[i - 1] > 0]


def _window_return(values: list[float], window: int) -> float:
    if len(values) <= window or values[-window - 1] <= 0:
        return 0.0
    return math.log(values[-1] / values[-window - 1])


def _safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) < 1e-12:
        return 0.0
    return numerator / denominator


def _zscore(value: float, history: list[float]) -> float:
    if not history:
        return 0.0
    sigma = pstdev(history)
    if sigma < 1e-12:
        return 0.0
    return (value - mean(history)) / sigma


def _ema(values: list[float], span: int) -> list[float]:
    alpha = 2 / (span + 1)
    result = [values[0]]
    for value in values[1:]:
        result.append(alpha * value + (1 - alpha) * result[-1])
    return result


def _rsi(closes: list[float], period: int) -> float:
    if len(closes) <= period:
        return 50.0
    gains: list[float] = []
    losses: list[float] = []
    for i in range(-period, 0):
        delta = closes[i] - closes[i - 1]
        gains.append(max(delta, 0.0))
        losses.append(abs(min(delta, 0.0)))
    avg_gain = mean(gains)
    avg_loss = mean(losses)
    if avg_loss < 1e-12:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _macd_hist(closes: list[float]) -> float:
    if len(closes) < 35:
        return 0.0
    ema_12 = _ema(closes, 12)
    ema_26 = _ema(closes, 26)
    macd_line = [a - b for a, b in zip(ema_12[-len(ema_26):], ema_26)]
    signal = _ema(macd_line, 9)
    return macd_line[-1] - signal[-1]


def _atr(candles: list[Candle], period: int) -> float:
    if len(candles) <= period:
        return 0.0
    true_ranges: list[float] = []
    for index in range(-period, 0):
        current = candles[index]
        previous = candles[index - 1]
        true_ranges.append(
            max(
                current.high - current.low,
                abs(current.high - previous.close),
                abs(current.low - previous.close),
            )
        )
    return mean(true_ranges)


def _vwap(candles: list[Candle]) -> float:
    quote = sum(c.quote_volume for c in candles)
    volume = sum(c.volume for c in candles)
    if volume > 0 and quote > 0:
        return quote / volume
    weighted = sum(((c.high + c.low + c.close) / 3) * c.volume for c in candles)
    return weighted / max(volume, 1e-12)


def _bollinger_position(closes: list[float], period: int) -> float:
    window = closes[-period:]
    basis = mean(window)
    sigma = pstdev(window)
    if sigma < 1e-12:
        return 0.5
    upper = basis + 2 * sigma
    lower = basis - 2 * sigma
    return _safe_divide(closes[-1] - lower, upper - lower)
