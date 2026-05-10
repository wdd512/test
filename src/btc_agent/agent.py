from __future__ import annotations

from .alpha import analyze_interval, combine_analyses
from .config import AgentConfig
from .data import BinanceMarketData
from .features import build_features
from .models import AgentDecision, Candle
from .risk import apply_risk


class BTCAgent:
    def __init__(self, config: AgentConfig | None = None, market_data: BinanceMarketData | None = None) -> None:
        self.config = config or AgentConfig()
        self.market_data = market_data or BinanceMarketData(self.config.base_url)

    def analyze_live(self) -> AgentDecision:
        book = self.market_data.order_book(self.config.symbol)
        candles_by_interval = {
            interval: self.market_data.klines(self.config.symbol, interval, self.config.limit)
            for interval in self.config.intervals
        }
        return self.analyze_candles(candles_by_interval, use_book=True, book=book)

    def analyze_candles(
        self,
        candles_by_interval: dict[str, list[Candle]],
        use_book: bool = False,
        book=None,
    ) -> AgentDecision:
        analyses = []
        for interval, candles in candles_by_interval.items():
            features = build_features(candles, book if use_book else None)
            analyses.append(analyze_interval(interval, features, self.config.min_confidence))

        action, confidence = combine_analyses(analyses, self.config.min_confidence)
        return apply_risk(self.config, self.config.symbol, action, confidence, analyses)
