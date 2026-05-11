from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .intrabar import IntrabarSignal, dynamic_candle_signal
from .intrabar_training import INTRABAR_FEATURE_COLUMNS
from .ml_baseline import LogisticRegression, RobustScaler


@dataclass
class PaperPosition:
    direction: str
    entry_price: float
    entry_time: str
    candle_open_time: str
    notional: float


@dataclass(frozen=True)
class PaperEvent:
    logged_at: str
    action: str
    balance: float
    equity: float
    position: str
    price: float
    pnl: float
    reason: str
    candle_open_time: str
    prediction: str
    probability_up: float
    confidence: float


def paper_trade_intrabar(
    market_data,
    symbol: str,
    interval: str,
    base_model_path: str,
    intrabar_model_path: str | None,
    seconds: int = 3600,
    poll_seconds: int = 20,
    starting_balance: float = 100.0,
    stake_fraction: float = 0.20,
    threshold: float = 0.52,
    flip_threshold: float = 0.535,
    out_csv: str = "paper_trades.csv",
) -> list[PaperEvent]:
    balance = starting_balance
    position: PaperPosition | None = None
    events: list[PaperEvent] = []
    deadline = time.time() + seconds
    output_path = Path(out_csv)
    write_header = not output_path.exists()

    with output_path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(PaperEvent.__dataclass_fields__.keys()))
        if write_header:
            writer.writeheader()

        while time.time() <= deadline:
            try:
                candles = market_data.klines(symbol, interval, 260)
                signal = dynamic_candle_signal(
                    candles,
                    model_path=base_model_path,
                    symbol=symbol,
                    interval=interval,
                    threshold=threshold,
                    flip_threshold=flip_threshold,
                )
                prediction, probability_up, confidence = _final_prediction(
                    signal,
                    intrabar_model_path=intrabar_model_path,
                    threshold=threshold,
                )

                if position and position.candle_open_time != signal.candle_open_time:
                    balance, event = _close_position(balance, position, signal, prediction, probability_up, confidence, "candle_closed")
                    position = None
                    _write_event(writer, event)
                    events.append(event)

                if prediction in {"UP", "DOWN"}:
                    if position is None:
                        notional = max(balance * stake_fraction, 0.0)
                        if notional > 0:
                            position = PaperPosition(
                                direction=prediction,
                                entry_price=signal.last_price,
                                entry_time=datetime.now(timezone.utc).isoformat(),
                                candle_open_time=signal.candle_open_time,
                                notional=notional,
                            )
                            event = _event("OPEN", balance, balance, position, signal, 0.0, "new_signal", prediction, probability_up, confidence)
                            _write_event(writer, event)
                            events.append(event)
                    elif position.direction != prediction and confidence >= flip_threshold:
                        balance, close_event = _close_position(
                            balance,
                            position,
                            signal,
                            prediction,
                            probability_up,
                            confidence,
                            "flip_signal",
                        )
                        _write_event(writer, close_event)
                        events.append(close_event)
                        notional = max(balance * stake_fraction, 0.0)
                        position = PaperPosition(
                            direction=prediction,
                            entry_price=signal.last_price,
                            entry_time=datetime.now(timezone.utc).isoformat(),
                            candle_open_time=signal.candle_open_time,
                            notional=notional,
                        )
                        open_event = _event("FLIP_OPEN", balance, balance, position, signal, 0.0, "flip_signal", prediction, probability_up, confidence)
                        _write_event(writer, open_event)
                        events.append(open_event)
                    else:
                        equity = balance + _unrealized_pnl(position, signal.last_price)
                        event = _event("KEEP", balance, equity, position, signal, 0.0, "same_signal", prediction, probability_up, confidence)
                        _write_event(writer, event)
                        events.append(event)
                else:
                    equity = balance + (_unrealized_pnl(position, signal.last_price) if position else 0.0)
                    event = _event("HOLD", balance, equity, position, signal, 0.0, "no_edge", prediction, probability_up, confidence)
                    _write_event(writer, event)
                    events.append(event)

                file.flush()
                print(json.dumps(asdict(events[-1]), ensure_ascii=False), flush=True)
            except Exception as exc:
                error_event = PaperEvent(
                    logged_at=datetime.now(timezone.utc).isoformat(),
                    action="ERROR",
                    balance=balance,
                    equity=balance,
                    position=position.direction if position else "FLAT",
                    price=0.0,
                    pnl=0.0,
                    reason=f"{type(exc).__name__}: {exc}",
                    candle_open_time="",
                    prediction="ERROR",
                    probability_up=0.0,
                    confidence=0.0,
                )
                _write_event(writer, error_event)
                events.append(error_event)
                file.flush()
                print(json.dumps(asdict(error_event), ensure_ascii=False), flush=True)
            time.sleep(poll_seconds)

    return events


def _final_prediction(signal: IntrabarSignal, intrabar_model_path: str | None, threshold: float) -> tuple[str, float, float]:
    if intrabar_model_path and Path(intrabar_model_path).exists():
        probability_up = _intrabar_model_probability(signal, intrabar_model_path)
        if probability_up >= threshold:
            return "UP", probability_up, probability_up
        if probability_up <= 1 - threshold:
            return "DOWN", probability_up, 1 - probability_up
        return "NO_EDGE", probability_up, max(probability_up, 1 - probability_up)
    return signal.dynamic_prediction, signal.dynamic_probability_up, signal.confidence


def _intrabar_model_probability(signal: IntrabarSignal, model_path: str) -> float:
    payload = json.loads(Path(model_path).read_text(encoding="utf-8"))
    if payload.get("label_type") != "intrabar_candle_direction":
        raise ValueError("Intrabar model has wrong label_type.")
    row_map = {
        "elapsed_fraction": signal.elapsed_fraction,
        "current_return": signal.current_return,
        "candle_range_pct": signal.candle_range_pct,
        "close_location": signal.close_location,
        "taker_imbalance": signal.taker_imbalance,
        "base_probability_up": signal.base_probability_up,
        "dynamic_probability_up": signal.dynamic_probability_up,
    }
    row = [row_map[name] for name in INTRABAR_FEATURE_COLUMNS]
    scaler = RobustScaler()
    scaler.centers = payload["centers"]
    scaler.scales = payload["scales"]
    model = LogisticRegression()
    model.weights = payload["weights"]
    model.bias = payload["bias"]
    return model.predict_proba_one(scaler.transform(row))


def _close_position(
    balance: float,
    position: PaperPosition,
    signal: IntrabarSignal,
    prediction: str,
    probability_up: float,
    confidence: float,
    reason: str,
) -> tuple[float, PaperEvent]:
    pnl = _unrealized_pnl(position, signal.last_price)
    new_balance = balance + pnl
    event = _event("CLOSE", new_balance, new_balance, position, signal, pnl, reason, prediction, probability_up, confidence)
    return new_balance, event


def _unrealized_pnl(position: PaperPosition | None, price: float) -> float:
    if not position:
        return 0.0
    direction = 1 if position.direction == "UP" else -1
    return position.notional * direction * ((price / position.entry_price) - 1)


def _event(
    action: str,
    balance: float,
    equity: float,
    position: PaperPosition | None,
    signal: IntrabarSignal,
    pnl: float,
    reason: str,
    prediction: str,
    probability_up: float,
    confidence: float,
) -> PaperEvent:
    return PaperEvent(
        logged_at=datetime.now(timezone.utc).isoformat(),
        action=action,
        balance=balance,
        equity=equity,
        position=position.direction if position else "FLAT",
        price=signal.last_price,
        pnl=pnl,
        reason=reason,
        candle_open_time=signal.candle_open_time,
        prediction=prediction,
        probability_up=probability_up,
        confidence=confidence,
    )


def _write_event(writer, event: PaperEvent) -> None:
    writer.writerow(asdict(event))
