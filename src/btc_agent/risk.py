from __future__ import annotations

from datetime import datetime, timezone

from .config import AgentConfig
from .models import AgentDecision, IntervalAnalysis


def apply_risk(
    config: AgentConfig,
    symbol: str,
    action: str,
    confidence: float,
    analyses: list[IntervalAnalysis],
) -> AgentDecision:
    flags: list[str] = []
    now = datetime.now(timezone.utc)
    primary = analyses[0] if analyses else None
    features = primary.features if primary else {}

    spread_pct = features.get("spread_pct", 0.0)
    atr_pct = features.get("atr_pct", 0.0)
    volatility_ratio = features.get("volatility_ratio", 1.0)

    if spread_pct > 0.0012:
        flags.append("wide_spread")
    if volatility_ratio > 2.2:
        flags.append("volatility_expansion")
    if atr_pct <= 0:
        flags.append("atr_unavailable")
    if analyses and _disagree(analyses):
        flags.append("interval_disagreement")

    stop_loss_pct = max(atr_pct * 1.2, 0.0025)
    take_profit_pct = max(stop_loss_pct * 1.55, 0.004)
    position_fraction = _position_fraction(config, confidence, stop_loss_pct, flags)

    if flags or action == "HOLD":
        if "wide_spread" in flags or "interval_disagreement" in flags:
            action = "HOLD"
            position_fraction = 0.0

    return AgentDecision(
        symbol=symbol,
        timestamp=now,
        action=action,
        confidence=confidence,
        position_fraction=position_fraction,
        risk_usdt=config.account_equity_usdt * config.max_risk_per_trade * position_fraction / max(config.max_position_fraction, 1e-12),
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        analyses=analyses,
        risk_flags=flags,
    )


def _position_fraction(config: AgentConfig, confidence: float, stop_loss_pct: float, flags: list[str]) -> float:
    if confidence < config.min_confidence or flags:
        return 0.0
    risk_budget_fraction = config.max_risk_per_trade / max(stop_loss_pct, 1e-12)
    confidence_scale = max(0.0, min(1.0, (confidence - config.min_confidence) / (1 - config.min_confidence)))
    return min(config.max_position_fraction, risk_budget_fraction * confidence_scale)


def _disagree(analyses: list[IntervalAnalysis]) -> bool:
    directional = {item.signal for item in analyses if item.signal in {"LONG", "SHORT"}}
    return len(directional) > 1
