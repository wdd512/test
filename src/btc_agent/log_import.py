from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


INTRABAR_FIELDS = [
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


@dataclass(frozen=True)
class ImportReport:
    input_path: str
    output_path: str
    imported_rows: int


def import_railway_logs(input_path: str, output_path: str = "intrabar_signals.csv") -> ImportReport:
    text = Path(input_path).read_text(encoding="utf-8-sig", errors="replace")
    rows = _extract_rows(text)
    if not rows:
        raise ValueError("No intrabar signal rows found in the log file.")

    with Path(output_path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=INTRABAR_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in INTRABAR_FIELDS})

    return ImportReport(input_path=input_path, output_path=output_path, imported_rows=len(rows))


def _extract_rows(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    stripped = text.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
            rows.extend(_rows_from_json_payload(payload))
        except json.JSONDecodeError:
            pass

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.extend(_rows_from_line(line))

    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row.get("logged_at", "")),
            str(row.get("candle_open_time", "")),
            str(row.get("elapsed_fraction", "")),
        )
        unique[key] = row
    return list(unique.values())


def _rows_from_json_payload(payload: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(payload, list):
        for item in payload:
            rows.extend(_rows_from_json_payload(item))
    elif isinstance(payload, dict):
        normalized = _normalize_row(payload)
        if normalized:
            rows.append(normalized)
        for value in payload.values():
            if isinstance(value, str):
                rows.extend(_rows_from_line(value))
            elif isinstance(value, (dict, list)):
                rows.extend(_rows_from_json_payload(value))
    elif isinstance(payload, str):
        rows.extend(_rows_from_line(payload))
    return rows


def _rows_from_line(line: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        payload = json.loads(line)
        rows.extend(_rows_from_json_payload(payload))
        return rows
    except json.JSONDecodeError:
        pass

    for match in re.finditer(r"\{.*?\}", line):
        try:
            payload = json.loads(match.group(0))
            rows.extend(_rows_from_json_payload(payload))
        except json.JSONDecodeError:
            continue

    key_value_row = _parse_key_value_line(line)
    if key_value_row:
        rows.append(key_value_row)
    return rows


def _parse_key_value_line(line: str) -> dict[str, Any] | None:
    if "symbol:" not in line or "candle_open_time:" not in line:
        return None

    matches = re.findall(r"([A-Za-z_]+(?:\[\d+\])?):\s*(.*?)(?=\s+[A-Za-z_]+(?:\[\d+\])?:|$)", line)
    if not matches:
        return None

    row: dict[str, Any] = {}
    reasons: list[str] = []
    for key, value in matches:
        value = value.strip()
        if key.startswith("reasons["):
            reasons.append(value)
        elif key in INTRABAR_FIELDS:
            row[key] = value
    if reasons:
        row["reasons"] = " | ".join(reasons)
    return _normalize_row(row)


def _normalize_row(payload: dict[str, Any]) -> dict[str, Any] | None:
    if "symbol" not in payload or "candle_open_time" not in payload:
        return None

    row = dict(payload)
    if "logged_at" not in row:
        row["logged_at"] = row.get("timestamp") or row.get("time") or ""
    if isinstance(row.get("reasons"), list):
        row["reasons"] = " | ".join(str(item) for item in row["reasons"])
    return row
