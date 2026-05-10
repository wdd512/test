from __future__ import annotations

import csv
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .models import Candle, OrderBookSnapshot
from .config import INTERVAL_SECONDS


def _utc_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)


def _http_json(url: str, timeout: float = 10.0) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "btc-agent/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


class BinanceMarketData:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("BINANCE_BASE_URL") or "https://api.binance.com").rstrip("/")

    def klines(self, symbol: str, interval: str, limit: int = 240) -> list[Candle]:
        params = urllib.parse.urlencode({"symbol": symbol, "interval": interval, "limit": limit})
        rows = _http_json(f"{self.base_url}/api/v3/klines?{params}")
        return [self._parse_kline(row) for row in rows]  # type: ignore[arg-type]

    def historical_klines(self, symbol: str, interval: str, limit: int = 10_000) -> list[Candle]:
        rows: list[list] = []
        remaining = limit
        end_time: int | None = None
        while remaining > 0:
            batch_limit = min(1000, remaining)
            params_dict = {"symbol": symbol, "interval": interval, "limit": batch_limit}
            if end_time is not None:
                params_dict["endTime"] = end_time
            params = urllib.parse.urlencode(params_dict)
            batch = _http_json(f"{self.base_url}/api/v3/klines?{params}")
            if not batch:
                break
            typed_batch = list(batch)  # type: ignore[arg-type]
            rows = typed_batch + rows
            remaining -= len(typed_batch)
            first_open_time = int(typed_batch[0][0])
            next_end_time = first_open_time - 1
            if end_time == next_end_time or len(typed_batch) < batch_limit:
                break
            end_time = next_end_time
        parsed = [self._parse_kline(row) for row in rows]
        return parsed[-limit:]

    def order_book(self, symbol: str, limit: int = 20) -> OrderBookSnapshot:
        params = urllib.parse.urlencode({"symbol": symbol, "limit": limit})
        payload = _http_json(f"{self.base_url}/api/v3/depth?{params}")
        bids = [(float(p), float(q)) for p, q in payload["bids"]]  # type: ignore[index]
        asks = [(float(p), float(q)) for p, q in payload["asks"]]  # type: ignore[index]
        best_bid, best_bid_qty = bids[0]
        best_ask, best_ask_qty = asks[0]
        bid_depth = sum(q for _, q in bids)
        ask_depth = sum(q for _, q in asks)
        mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        imbalance = (bid_depth - ask_depth) / max(bid_depth + ask_depth, 1e-12)
        microprice = (best_ask * best_bid_qty + best_bid * best_ask_qty) / max(best_bid_qty + best_ask_qty, 1e-12)
        return OrderBookSnapshot(
            bid=best_bid,
            ask=best_ask,
            bid_qty=best_bid_qty,
            ask_qty=best_ask_qty,
            spread=spread,
            relative_spread=spread / max(mid, 1e-12),
            imbalance=imbalance,
            microprice=microprice,
            captured_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _parse_kline(row: list) -> Candle:
        return Candle(
            open_time=_utc_ms(int(row[0])),
            open=float(row[1]),
            high=float(row[2]),
            low=float(row[3]),
            close=float(row[4]),
            volume=float(row[5]),
            close_time=_utc_ms(int(row[6])),
            quote_volume=float(row[7]),
            trade_count=int(row[8]),
            taker_buy_base_volume=float(row[9]),
            taker_buy_quote_volume=float(row[10]),
        )


class CoinbaseMarketData:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or os.getenv("COINBASE_BASE_URL") or "https://api.exchange.coinbase.com").rstrip("/")

    def klines(self, symbol: str, interval: str, limit: int = 240) -> list[Candle]:
        granularity = INTERVAL_SECONDS[interval]
        encoded_symbol = urllib.parse.quote(symbol, safe="")
        params = urllib.parse.urlencode({"granularity": granularity})
        rows = _http_json(f"{self.base_url}/products/{encoded_symbol}/candles?{params}")
        parsed = [self._parse_candle(row, granularity) for row in rows]  # type: ignore[arg-type]
        parsed.sort(key=lambda candle: candle.open_time)
        return parsed[-limit:]

    def historical_klines(self, symbol: str, interval: str, limit: int = 300) -> list[Candle]:
        return self.klines(symbol, interval, min(limit, 300))

    @staticmethod
    def _parse_candle(row: list, granularity: int) -> Candle:
        # Coinbase format: [time, low, high, open, close, volume]
        open_time = datetime.fromtimestamp(int(row[0]), tz=timezone.utc)
        close_time = datetime.fromtimestamp(int(row[0]) + granularity, tz=timezone.utc)
        volume = float(row[5])
        close = float(row[4])
        return Candle(
            open_time=open_time,
            open=float(row[3]),
            high=float(row[2]),
            low=float(row[1]),
            close=close,
            volume=volume,
            close_time=close_time,
            quote_volume=volume * close,
            trade_count=0,
            taker_buy_base_volume=volume / 2,
            taker_buy_quote_volume=volume * close / 2,
        )


def create_market_data():
    provider = os.getenv("MARKET_DATA_PROVIDER", "binance").lower()
    if provider == "coinbase":
        return CoinbaseMarketData()
    if provider == "binance":
        return BinanceMarketData()
    raise ValueError(f"Unsupported MARKET_DATA_PROVIDER: {provider}")


def load_candles_csv(path: str | Path) -> list[Candle]:
    candles: list[Candle] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            open_time = _parse_time(row.get("open_time") or row.get("timestamp") or row.get("time"))
            close_time = _parse_time(row.get("close_time")) if row.get("close_time") else open_time
            candles.append(
                Candle(
                    open_time=open_time,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0.0)),
                    close_time=close_time,
                    quote_volume=float(row.get("quote_volume", 0.0)),
                    trade_count=int(float(row.get("trade_count", 0))),
                    taker_buy_base_volume=float(row.get("taker_buy_base_volume", 0.0)),
                    taker_buy_quote_volume=float(row.get("taker_buy_quote_volume", 0.0)),
                )
            )
    return candles


def save_candles_csv(path: str | Path, candles: Iterable[Candle]) -> None:
    fieldnames = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for candle in candles:
            writer.writerow(
                {
                    "open_time": candle.open_time.isoformat(),
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                    "close_time": candle.close_time.isoformat(),
                    "quote_volume": candle.quote_volume,
                    "trade_count": candle.trade_count,
                    "taker_buy_base_volume": candle.taker_buy_base_volume,
                    "taker_buy_quote_volume": candle.taker_buy_quote_volume,
                }
            )


def _parse_time(value: str | None) -> datetime:
    if not value:
        raise ValueError("CSV row is missing time/open_time/timestamp")
    cleaned = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(cleaned)
    except ValueError:
        parsed = datetime.fromtimestamp(int(float(value)) / 1000, tz=timezone.utc)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
