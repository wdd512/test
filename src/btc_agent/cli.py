from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .agent import BTCAgent
from .backtest import optimize_thresholds, walk_forward_backtest
from .config import AgentConfig
from .data import BinanceMarketData, create_market_data, load_candles_csv, save_candles_csv
from .history_download import download_binance_public_monthly_klines
from .intrabar import dynamic_candle_signal, watch_intrabar
from .intrabar_training import label_intrabar_csv, train_intrabar_model
from .log_import import import_railway_logs
from .ml_baseline import (
    predict_candle_direction_live,
    run_experiment_matrix,
    train_and_evaluate,
    train_candle_direction,
    walk_forward_candle_direction,
    walk_forward_evaluate,
    walk_forward_regime_evaluate,
)
from .paper_trading import paper_trade_intrabar


def main() -> None:
    parser = argparse.ArgumentParser(description="BTC 5m/15m short-horizon analysis agent")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze live BTC market data from Binance")
    analyze.add_argument("--symbol", default="BTCUSDT")
    analyze.add_argument("--limit", type=int, default=240)

    fetch = subparsers.add_parser("fetch", help="Download candles to CSV")
    fetch.add_argument("--symbol", default="BTCUSDT")
    fetch.add_argument("--interval", default="5m")
    fetch.add_argument("--limit", type=int, default=1000)
    fetch.add_argument("--out", required=True)
    fetch.add_argument("--history", action="store_true", help="Page backwards to fetch more than 1000 candles")

    backtest = subparsers.add_parser("backtest", help="Run a simple walk-forward CSV backtest")
    backtest.add_argument("--csv", required=True)
    backtest.add_argument("--interval", default="5m")
    backtest.add_argument("--horizon", type=int, default=3)
    backtest.add_argument("--limit", type=int, default=240)
    backtest.add_argument("--min-confidence", type=float, default=0.58)
    backtest.add_argument("--min-score", type=float, default=0.0)
    backtest.add_argument("--mode", choices=("normal", "inverse"), default="normal")

    optimize = subparsers.add_parser("optimize", help="Sweep confidence/score thresholds on CSV data")
    optimize.add_argument("--csv", required=True)
    optimize.add_argument("--interval", default="5m")
    optimize.add_argument("--horizon", type=int, default=3)
    optimize.add_argument("--limit", type=int, default=240)
    optimize.add_argument("--min-trades", type=int, default=20)
    optimize.add_argument("--top", type=int, default=10)

    train = subparsers.add_parser("train-baseline", help="Train a stdlib logistic regression baseline on CSV data")
    train.add_argument("--csv", required=True)
    train.add_argument("--lookback", type=int, default=240)
    train.add_argument("--horizon", type=int, default=3)
    train.add_argument("--train-fraction", type=float, default=0.70)
    train.add_argument("--cost", type=float, default=0.0008)
    train.add_argument("--out-model", default="btc_baseline_model.json")

    candle_train = subparsers.add_parser(
        "train-candle-direction",
        help="Train model to predict whether the next/current candle closes above its open",
    )
    candle_train.add_argument("--csv", required=True)
    candle_train.add_argument("--lookback", type=int, default=240)
    candle_train.add_argument("--train-fraction", type=float, default=0.70)
    candle_train.add_argument("--cost", type=float, default=0.0)
    candle_train.add_argument("--out-model", default="btc_candle_direction_model.json")
    candle_train.add_argument("--sample-step", type=int, default=1)
    candle_train.add_argument("--epochs", type=int, default=900)

    matrix = subparsers.add_parser("experiment-matrix", help="Run horizon/cost experiments for the ML baseline")
    matrix.add_argument("--csv", required=True)
    matrix.add_argument("--lookback", type=int, default=240)
    matrix.add_argument("--horizons", default="1,3,6,12")
    matrix.add_argument("--costs", default="0,0.0008")
    matrix.add_argument("--train-fraction", type=float, default=0.70)

    wf = subparsers.add_parser("walk-forward-ml", help="Walk-forward evaluation for a fixed ML trading policy")
    wf.add_argument("--csv", required=True)
    wf.add_argument("--lookback", type=int, default=240)
    wf.add_argument("--horizon", type=int, default=12)
    wf.add_argument("--cost", type=float, default=0.0008)
    wf.add_argument("--train-size", type=int, default=3000)
    wf.add_argument("--test-size", type=int, default=750)
    wf.add_argument("--mode", choices=("long_only", "short_only", "long_short"), default="long_only")
    wf.add_argument("--threshold", type=float, default=0.52)

    wf_regime = subparsers.add_parser(
        "walk-forward-regime",
        help="Walk-forward ML evaluation with train-only regime filter selection",
    )
    wf_regime.add_argument("--csv", required=True)
    wf_regime.add_argument("--lookback", type=int, default=240)
    wf_regime.add_argument("--horizon", type=int, default=12)
    wf_regime.add_argument("--cost", type=float, default=0.0008)
    wf_regime.add_argument("--train-size", type=int, default=3000)
    wf_regime.add_argument("--test-size", type=int, default=750)
    wf_regime.add_argument("--mode", choices=("long_only", "short_only", "long_short"), default="long_only")
    wf_regime.add_argument("--threshold", type=float, default=0.52)
    wf_regime.add_argument("--min-train-trades", type=int, default=40)

    candle_wf = subparsers.add_parser(
        "walk-forward-candle-direction",
        help="Walk-forward test for current-candle close-above-open prediction",
    )
    candle_wf.add_argument("--csv", required=True)
    candle_wf.add_argument("--lookback", type=int, default=240)
    candle_wf.add_argument("--cost", type=float, default=0.0)
    candle_wf.add_argument("--train-size", type=int, default=3000)
    candle_wf.add_argument("--test-size", type=int, default=750)
    candle_wf.add_argument("--mode", choices=("long_only", "short_only", "long_short"), default="long_short")
    candle_wf.add_argument("--threshold", type=float, default=0.52)

    candle_live = subparsers.add_parser(
        "candle-direction-live",
        help="Predict whether the current live candle will close above or below its open",
    )
    candle_live.add_argument("--symbol", default="BTCUSDT")
    candle_live.add_argument("--interval", default="5m")
    candle_live.add_argument("--model", default="btc_candle_direction_model.json")
    candle_live.add_argument("--threshold", type=float, default=0.52)

    intrabar = subparsers.add_parser(
        "intrabar-live",
        help="Dynamic current-candle signal that can flip during the 5m candle",
    )
    intrabar.add_argument("--symbol", default="BTCUSDT")
    intrabar.add_argument("--interval", default="5m")
    intrabar.add_argument("--model", default="btc_candle_direction_model.json")
    intrabar.add_argument("--threshold", type=float, default=0.52)
    intrabar.add_argument("--flip-threshold", type=float, default=0.535)

    intrabar_watch = subparsers.add_parser(
        "intrabar-watch",
        help="Poll dynamic current-candle signal and append snapshots to CSV",
    )
    intrabar_watch.add_argument("--symbol", default="BTCUSDT")
    intrabar_watch.add_argument("--interval", default="5m")
    intrabar_watch.add_argument("--model", default="btc_candle_direction_model.json")
    intrabar_watch.add_argument("--seconds", type=int, default=300)
    intrabar_watch.add_argument("--poll-seconds", type=int, default=20)
    intrabar_watch.add_argument("--out", default="intrabar_signals.csv")
    intrabar_watch.add_argument("--threshold", type=float, default=0.52)
    intrabar_watch.add_argument("--flip-threshold", type=float, default=0.535)

    intrabar_label = subparsers.add_parser("label-intrabar", help="Label intrabar snapshots after candles close")
    intrabar_label.add_argument("--input", default="intrabar_signals.csv")
    intrabar_label.add_argument("--output", default="intrabar_labeled.csv")
    intrabar_label.add_argument("--symbol", default="BTCUSDT")
    intrabar_label.add_argument("--interval", default="5m")

    intrabar_train = subparsers.add_parser("train-intrabar", help="Train an intrabar flip/update model")
    intrabar_train.add_argument("--csv", default="intrabar_labeled.csv")
    intrabar_train.add_argument("--out-model", default="btc_intrabar_model.json")
    intrabar_train.add_argument("--train-fraction", type=float, default=0.70)

    paper = subparsers.add_parser("paper-trade", help="Run virtual-money intrabar paper trading")
    paper.add_argument("--symbol", default="BTC-USD")
    paper.add_argument("--interval", default="5m")
    paper.add_argument("--base-model", default="btc_candle_direction_model.json")
    paper.add_argument("--intrabar-model", default="btc_intrabar_model.json")
    paper.add_argument("--seconds", type=int, default=3600)
    paper.add_argument("--poll-seconds", type=int, default=20)
    paper.add_argument("--balance", type=float, default=100.0)
    paper.add_argument("--stake-fraction", type=float, default=0.20)
    paper.add_argument("--threshold", type=float, default=0.52)
    paper.add_argument("--flip-threshold", type=float, default=0.535)
    paper.add_argument("--out", default="paper_trades.csv")

    import_logs = subparsers.add_parser("import-railway-logs", help="Convert downloaded Railway logs to intrabar CSV")
    import_logs.add_argument("--input", required=True)
    import_logs.add_argument("--output", default="intrabar_signals.csv")

    hist = subparsers.add_parser("download-binance-history", help="Download Binance public monthly kline ZIPs")
    hist.add_argument("--symbol", default="BTCUSDT")
    hist.add_argument("--interval", default="5m")
    hist.add_argument("--start-month", required=True)
    hist.add_argument("--end-month", required=True)
    hist.add_argument("--out", required=True)
    hist.add_argument("--base-url", default="https://data.binance.vision/data/spot/monthly/klines")

    args = parser.parse_args()

    if args.command == "analyze":
        config = AgentConfig(symbol=args.symbol, limit=args.limit)
        decision = BTCAgent(config).analyze_live()
        print(json.dumps(decision.as_dict(), ensure_ascii=False, indent=2))
    elif args.command == "fetch":
        data = create_market_data()
        if args.history:
            candles = data.historical_klines(args.symbol, args.interval, args.limit)
        else:
            candles = data.klines(args.symbol, args.interval, args.limit)
        save_candles_csv(args.out, candles)
        print(f"Saved {len(candles)} candles to {args.out}")
    elif args.command == "backtest":
        config = AgentConfig(limit=args.limit, intervals=(args.interval,), min_confidence=args.min_confidence)
        candles = load_candles_csv(args.csv)
        result = walk_forward_backtest(
            BTCAgent(config),
            candles,
            args.interval,
            args.horizon,
            min_score=args.min_score,
            mode=args.mode,
        )
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    elif args.command == "optimize":
        candles = load_candles_csv(args.csv)

        def make_agent(min_confidence: float) -> BTCAgent:
            config = AgentConfig(limit=args.limit, intervals=(args.interval,), min_confidence=min_confidence)
            return BTCAgent(config)

        results = optimize_thresholds(
            make_agent,
            candles,
            interval=args.interval,
            horizon=args.horizon,
            min_trades=args.min_trades,
        )
        payload = [
            {
                "rank": index + 1,
                "mode": item.mode,
                "min_confidence": item.min_confidence,
                "min_score": item.min_score,
                **asdict(item.result),
            }
            for index, item in enumerate(results[: args.top])
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    elif args.command == "train-baseline":
        candles = load_candles_csv(args.csv)
        report = train_and_evaluate(
            candles,
            lookback=args.lookback,
            horizon=args.horizon,
            train_fraction=args.train_fraction,
            cost=args.cost,
            output_model=args.out_model,
        )
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    elif args.command == "train-candle-direction":
        candles = load_candles_csv(args.csv)
        report = train_candle_direction(
            candles,
            lookback=args.lookback,
            train_fraction=args.train_fraction,
            cost=args.cost,
            output_model=args.out_model,
            sample_step=args.sample_step,
            epochs=args.epochs,
        )
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    elif args.command == "experiment-matrix":
        candles = load_candles_csv(args.csv)
        horizons = tuple(int(value.strip()) for value in args.horizons.split(",") if value.strip())
        costs = tuple(float(value.strip()) for value in args.costs.split(",") if value.strip())
        summaries = run_experiment_matrix(
            candles,
            lookback=args.lookback,
            horizons=horizons,
            costs=costs,
            train_fraction=args.train_fraction,
        )
        print(json.dumps([asdict(item) for item in summaries], ensure_ascii=False, indent=2))
    elif args.command == "walk-forward-ml":
        candles = load_candles_csv(args.csv)
        report = walk_forward_evaluate(
            candles,
            lookback=args.lookback,
            horizon=args.horizon,
            cost=args.cost,
            train_size=args.train_size,
            test_size=args.test_size,
            mode=args.mode,
            threshold=args.threshold,
        )
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    elif args.command == "walk-forward-regime":
        candles = load_candles_csv(args.csv)
        report = walk_forward_regime_evaluate(
            candles,
            lookback=args.lookback,
            horizon=args.horizon,
            cost=args.cost,
            train_size=args.train_size,
            test_size=args.test_size,
            mode=args.mode,
            threshold=args.threshold,
            min_train_trades=args.min_train_trades,
        )
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    elif args.command == "walk-forward-candle-direction":
        candles = load_candles_csv(args.csv)
        report = walk_forward_candle_direction(
            candles,
            lookback=args.lookback,
            cost=args.cost,
            train_size=args.train_size,
            test_size=args.test_size,
            mode=args.mode,
            threshold=args.threshold,
        )
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    elif args.command == "candle-direction-live":
        data = create_market_data()
        candles = data.klines(args.symbol, args.interval, 260)
        signal = predict_candle_direction_live(
            candles,
            model_path=args.model,
            symbol=args.symbol,
            interval=args.interval,
            threshold=args.threshold,
        )
        print(json.dumps(asdict(signal), ensure_ascii=False, indent=2))
    elif args.command == "intrabar-live":
        data = create_market_data()
        candles = data.klines(args.symbol, args.interval, 260)
        signal = dynamic_candle_signal(
            candles,
            model_path=args.model,
            symbol=args.symbol,
            interval=args.interval,
            threshold=args.threshold,
            flip_threshold=args.flip_threshold,
        )
        print(json.dumps(asdict(signal), ensure_ascii=False, indent=2))
    elif args.command == "intrabar-watch":
        data = create_market_data()
        watch_intrabar(
            data,
            symbol=args.symbol,
            interval=args.interval,
            model_path=args.model,
            seconds=args.seconds,
            poll_seconds=args.poll_seconds,
            out_csv=args.out,
            threshold=args.threshold,
            flip_threshold=args.flip_threshold,
        )
    elif args.command == "label-intrabar":
        report = label_intrabar_csv(args.input, args.output, symbol=args.symbol, interval=args.interval)
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    elif args.command == "train-intrabar":
        report = train_intrabar_model(args.csv, output_model=args.out_model, train_fraction=args.train_fraction)
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    elif args.command == "import-railway-logs":
        report = import_railway_logs(args.input, args.output)
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    elif args.command == "download-binance-history":
        report = download_binance_public_monthly_klines(
            symbol=args.symbol,
            interval=args.interval,
            start_month=args.start_month,
            end_month=args.end_month,
            output_csv=args.out,
            base_url=args.base_url,
        )
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    elif args.command == "paper-trade":
        data = create_market_data()
        paper_trade_intrabar(
            data,
            symbol=args.symbol,
            interval=args.interval,
            base_model_path=args.base_model,
            intrabar_model_path=args.intrabar_model,
            seconds=args.seconds,
            poll_seconds=args.poll_seconds,
            starting_balance=args.balance,
            stake_fraction=args.stake_fraction,
            threshold=args.threshold,
            flip_threshold=args.flip_threshold,
            out_csv=args.out,
        )


if __name__ == "__main__":
    main()
