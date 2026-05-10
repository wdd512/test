from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    symbol: str = "BTCUSDT"
    intervals: tuple[str, ...] = ("5m", "15m")
    limit: int = 240
    account_equity_usdt: float = 10_000.0
    max_risk_per_trade: float = 0.005
    max_position_fraction: float = 0.20
    min_confidence: float = 0.58
    stale_seconds: int = 180
    base_url: str = "https://api.binance.com"


INTERVAL_SECONDS: dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
}
