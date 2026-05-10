from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class Candle:
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time: datetime
    quote_volume: float
    trade_count: int
    taker_buy_base_volume: float
    taker_buy_quote_volume: float


@dataclass(frozen=True)
class OrderBookSnapshot:
    bid: float
    ask: float
    bid_qty: float
    ask_qty: float
    spread: float
    relative_spread: float
    imbalance: float
    microprice: float
    captured_at: datetime


@dataclass(frozen=True)
class IntervalAnalysis:
    interval: str
    features: dict[str, float]
    signal: str
    confidence: float
    score: float
    reasons: list[str]


@dataclass(frozen=True)
class AgentDecision:
    symbol: str
    timestamp: datetime
    action: str
    confidence: float
    position_fraction: float
    risk_usdt: float
    stop_loss_pct: float
    take_profit_pct: float
    analyses: list[IntervalAnalysis]
    risk_flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.astimezone(timezone.utc).isoformat(),
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "position_fraction": round(self.position_fraction, 4),
            "risk_usdt": round(self.risk_usdt, 2),
            "stop_loss_pct": round(self.stop_loss_pct, 4),
            "take_profit_pct": round(self.take_profit_pct, 4),
            "risk_flags": self.risk_flags,
            "analyses": [
                {
                    "interval": item.interval,
                    "signal": item.signal,
                    "confidence": round(item.confidence, 4),
                    "score": round(item.score, 4),
                    "reasons": item.reasons,
                    "features": {k: round(v, 6) for k, v in item.features.items()},
                }
                for item in self.analyses
            ],
        }
