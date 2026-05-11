from __future__ import annotations

import csv
import io
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .data import save_candles_csv
from .models import Candle


@dataclass(frozen=True)
class DownloadReport:
    symbol: str
    interval: str
    start_month: str
    end_month: str
    output_csv: str
    downloaded_files: int
    skipped_files: int
    candles: int


def download_binance_public_monthly_klines(
    symbol: str,
    interval: str,
    start_month: str,
    end_month: str,
    output_csv: str,
    base_url: str = "https://data.binance.vision/data/spot/monthly/klines",
) -> DownloadReport:
    candles: list[Candle] = []
    downloaded = 0
    skipped = 0
    for month in _month_range(start_month, end_month):
        url = f"{base_url.rstrip('/')}/{symbol}/{interval}/{symbol}-{interval}-{month}.zip"
        try:
            payload = _download(url)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                skipped += 1
                continue
            raise
        candles.extend(_candles_from_zip(payload))
        downloaded += 1

    candles.sort(key=lambda candle: candle.open_time)
    save_candles_csv(output_csv, candles)
    return DownloadReport(
        symbol=symbol,
        interval=interval,
        start_month=start_month,
        end_month=end_month,
        output_csv=output_csv,
        downloaded_files=downloaded,
        skipped_files=skipped,
        candles=len(candles),
    )


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "btc-agent/0.1"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def _candles_from_zip(payload: bytes) -> list[Candle]:
    candles: list[Candle] = []
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            return candles
        with archive.open(csv_names[0]) as file:
            text = io.TextIOWrapper(file, encoding="utf-8")
            reader = csv.reader(text)
            first = True
            for row in reader:
                if not row:
                    continue
                if first and row[0].lower() in {"open_time", "open time"}:
                    first = False
                    continue
                first = False
                candles.append(_parse_binance_kline(row))
    return candles


def _parse_binance_kline(row: list[str]) -> Candle:
    return Candle(
        open_time=_parse_binance_time(int(row[0])),
        open=float(row[1]),
        high=float(row[2]),
        low=float(row[3]),
        close=float(row[4]),
        volume=float(row[5]),
        close_time=_parse_binance_time(int(row[6])),
        quote_volume=float(row[7]),
        trade_count=int(float(row[8])),
        taker_buy_base_volume=float(row[9]),
        taker_buy_quote_volume=float(row[10]),
    )


def _parse_binance_time(value: int) -> datetime:
    # Binance spot archive switched to microsecond timestamps in 2025.
    if value > 10_000_000_000_000:
        value = value // 1000
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc)


def _month_range(start_month: str, end_month: str) -> list[str]:
    start_year, start_m = [int(part) for part in start_month.split("-")]
    end_year, end_m = [int(part) for part in end_month.split("-")]
    months: list[str] = []
    year = start_year
    month = start_m
    while (year, month) <= (end_year, end_m):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            month = 1
            year += 1
    return months
