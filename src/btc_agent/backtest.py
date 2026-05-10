from __future__ import annotations

from dataclasses import dataclass

from .agent import BTCAgent
from .models import Candle


@dataclass(frozen=True)
class SideStats:
    trades: int = 0
    wins: int = 0
    losses: int = 0
    gross_return: float = 0.0
    avg_net_return: float = 0.0
    hit_rate: float = 0.0


@dataclass(frozen=True)
class BacktestResult:
    trades: int
    wins: int
    losses: int
    flat: int
    hit_rate: float
    cumulative_return: float
    max_drawdown: float
    avg_net_return: float
    profit_factor: float
    long: SideStats
    short: SideStats


@dataclass(frozen=True)
class ThresholdResult:
    mode: str
    min_confidence: float
    min_score: float
    result: BacktestResult


def walk_forward_backtest(
    agent: BTCAgent,
    candles: list[Candle],
    interval: str = "5m",
    horizon: int = 3,
    min_score: float = 0.0,
    cost: float = 0.0008,
    mode: str = "normal",
) -> BacktestResult:
    if len(candles) < agent.config.limit + horizon + 1:
        raise ValueError("Not enough candles for backtest window.")

    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    trades = wins = losses = flat = 0
    gross_profit = 0.0
    gross_loss = 0.0
    net_returns: list[float] = []
    side_returns: dict[str, list[float]] = {"LONG": [], "SHORT": []}

    for end in range(agent.config.limit, len(candles) - horizon):
        window = candles[end - agent.config.limit : end]
        decision = agent.analyze_candles({interval: window})
        score = abs(decision.analyses[0].score) if decision.analyses else 0.0
        if decision.action == "HOLD" or score < min_score:
            flat += 1
            continue
        entry = candles[end - 1].close
        exit_price = candles[end + horizon - 1].close
        executed_action = _executed_action(decision.action, mode)
        direction = 1 if executed_action == "LONG" else -1
        raw_return = direction * ((exit_price / entry) - 1)
        net_return = raw_return - cost
        equity *= 1 + net_return * decision.position_fraction
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, (peak - equity) / peak)
        net_returns.append(net_return)
        side_returns[executed_action].append(net_return)
        trades += 1
        if net_return > 0:
            wins += 1
            gross_profit += net_return
        else:
            losses += 1
            gross_loss += abs(net_return)

    hit_rate = wins / trades if trades else 0.0
    return BacktestResult(
        trades=trades,
        wins=wins,
        losses=losses,
        flat=flat,
        hit_rate=hit_rate,
        cumulative_return=equity - 1,
        max_drawdown=max_drawdown,
        avg_net_return=sum(net_returns) / len(net_returns) if net_returns else 0.0,
        profit_factor=gross_profit / gross_loss if gross_loss > 0 else 0.0,
        long=_side_stats(side_returns["LONG"]),
        short=_side_stats(side_returns["SHORT"]),
    )


def optimize_thresholds(
    agent_factory,
    candles: list[Candle],
    interval: str = "5m",
    horizon: int = 3,
    confidence_values: tuple[float, ...] = (0.58, 0.62, 0.66, 0.70, 0.74, 0.78),
    score_values: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0),
    modes: tuple[str, ...] = ("normal", "inverse"),
    min_trades: int = 20,
) -> list[ThresholdResult]:
    results: list[ThresholdResult] = []
    for mode in modes:
        for confidence in confidence_values:
            agent = agent_factory(confidence)
            for score in score_values:
                result = walk_forward_backtest(
                    agent,
                    candles,
                    interval=interval,
                    horizon=horizon,
                    min_score=score,
                    mode=mode,
                )
                if result.trades >= min_trades:
                    results.append(ThresholdResult(mode, confidence, score, result))
    return sorted(
        results,
        key=lambda item: (
            item.result.cumulative_return,
            item.result.profit_factor,
            -item.result.max_drawdown,
            item.result.trades,
        ),
        reverse=True,
    )


def _side_stats(values: list[float]) -> SideStats:
    trades = len(values)
    wins = sum(1 for value in values if value > 0)
    losses = trades - wins
    gross_return = sum(values)
    return SideStats(
        trades=trades,
        wins=wins,
        losses=losses,
        gross_return=gross_return,
        avg_net_return=gross_return / trades if trades else 0.0,
        hit_rate=wins / trades if trades else 0.0,
    )


def _executed_action(action: str, mode: str) -> str:
    if mode == "normal":
        return action
    if mode == "inverse":
        if action == "LONG":
            return "SHORT"
        if action == "SHORT":
            return "LONG"
        return action
    raise ValueError(f"Unknown backtest mode: {mode}")
