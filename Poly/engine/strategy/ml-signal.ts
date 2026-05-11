import { spawn } from "node:child_process";
import { resolve } from "node:path";
import type { Strategy, StrategyContext } from "./types.ts";
import { Env } from "../../utils/config.ts";

type ModelSide = "UP" | "DOWN";

type ModelSignal = {
  side: ModelSide | "NO_EDGE";
  confidence: number;
  probabilityUp: number | null;
};

type Position = {
  side: ModelSide;
  tokenId: string;
  entryPrice: number;
  shares: number;
};

type State = {
  position: Position | null;
  pendingEntry: boolean;
  pendingExit: boolean;
};

const DEFAULT_POLL_MS = 10_000;
const DEFAULT_SHARES = 5;
const DEFAULT_MIN_CONFIDENCE = 0.535;
const DEFAULT_FLIP_CONFIDENCE = 0.58;
const DEFAULT_MAX_ASK = 0.68;
const DEFAULT_MIN_LIQUIDITY = 1;
const DEFAULT_MIN_REMAINING_SEC = 20;

function numberEnv(name: string, fallback: number): number {
  const raw = process.env[name];
  if (!raw) return fallback;
  const value = Number(raw);
  return Number.isFinite(value) ? value : fallback;
}

function stringEnv(name: string, fallback: string): string {
  return process.env[name] || fallback;
}

function decodeModelPrediction(json: Record<string, unknown>): ModelSignal {
  const prediction =
    typeof json.dynamic_prediction === "string"
      ? json.dynamic_prediction
      : typeof json.prediction === "string"
        ? json.prediction
        : "NO_EDGE";

  const side: ModelSignal["side"] =
    prediction === "UP" || prediction === "DOWN" ? prediction : "NO_EDGE";

  const confidence =
    typeof json.confidence === "number" && Number.isFinite(json.confidence)
      ? json.confidence
      : 0;

  const probabilityUp =
    typeof json.dynamic_probability_up === "number"
      ? json.dynamic_probability_up
      : typeof json.probability_up === "number"
        ? json.probability_up
        : null;

  return { side, confidence, probabilityUp };
}

async function runPythonSignal(): Promise<ModelSignal> {
  const root = resolve(stringEnv("BTC_AGENT_ROOT", ".."));
  const python = stringEnv("BTC_AGENT_PYTHON", "python");
  const symbol = stringEnv("BTC_AGENT_SYMBOL", "BTC-USD");
  const interval = stringEnv("BTC_AGENT_INTERVAL", "5m");
  const modelPath = resolve(
    root,
    stringEnv("BTC_AGENT_MODEL", "btc_candle_direction_model.json"),
  );
  const threshold = stringEnv("BTC_AGENT_THRESHOLD", "0.52");
  const flipThreshold = stringEnv("BTC_AGENT_FLIP_THRESHOLD", "0.535");

  const args = [
    "-m",
    "btc_agent.cli",
    "intrabar-live",
    "--symbol",
    symbol,
    "--interval",
    interval,
    "--model",
    modelPath,
    "--threshold",
    threshold,
    "--flip-threshold",
    flipThreshold,
  ];

  const env = {
    ...process.env,
    PYTHONPATH: resolve(root, "src"),
    MARKET_DATA_PROVIDER: stringEnv("MARKET_DATA_PROVIDER", "coinbase"),
  };

  const output = await new Promise<string>((resolveOutput, reject) => {
    const child = spawn(python, args, { cwd: root, env });
    let stdout = "";
    let stderr = "";

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
    });
    child.on("error", reject);
    child.on("close", (code) => {
      if (code === 0) {
        resolveOutput(stdout);
        return;
      }
      reject(new Error(stderr.trim() || `Python exited with code ${code}`));
    });
  });

  const json = JSON.parse(output) as Record<string, unknown>;
  return decodeModelPrediction(json);
}

function tokenForSide(ctx: StrategyContext, side: ModelSide): string {
  return side === "UP" ? ctx.clobTokenIds[0] : ctx.clobTokenIds[1];
}

function placeEntry(
  ctx: StrategyContext,
  state: State,
  side: ModelSide,
  confidence: number,
): void {
  const ask = ctx.orderBook.bestAskInfo(side);
  if (!ask) return;

  const maxAsk = numberEnv("POLY_MODEL_MAX_ASK", DEFAULT_MAX_ASK);
  const minLiquidity = numberEnv(
    "POLY_MODEL_MIN_LIQUIDITY",
    DEFAULT_MIN_LIQUIDITY,
  );
  const shares = numberEnv("POLY_MODEL_SHARES", DEFAULT_SHARES);

  if (ask.price > maxAsk || ask.liquidity < minLiquidity) {
    ctx.log(
      `[${ctx.slug}] ml-signal: skip ${side}, ask=${ask.price.toFixed(2)}, liq=${ask.liquidity.toFixed(2)}`,
      "dim",
    );
    return;
  }

  state.pendingEntry = true;
  const tokenId = tokenForSide(ctx, side);
  ctx.log(
    `[${ctx.slug}] ml-signal: BUY ${side} @ ${ask.price.toFixed(2)} confidence=${confidence.toFixed(3)}`,
    "cyan",
  );

  ctx.postOrders([
    {
      req: {
        tokenId,
        action: "buy",
        price: ask.price,
        shares,
        orderType: "FOK",
      },
      expireAtMs: Date.now() + 3_000,
      onFilled(filledShares) {
        state.pendingEntry = false;
        state.position = {
          side,
          tokenId,
          entryPrice: ask.price,
          shares: filledShares,
        };
        ctx.log(
          `[${ctx.slug}] ml-signal: BUY ${side} filled @ ${ask.price.toFixed(2)} (${filledShares} shares)`,
          "green",
        );
      },
      onExpired() {
        state.pendingEntry = false;
      },
      onFailed(reason) {
        state.pendingEntry = false;
        ctx.log(`[${ctx.slug}] ml-signal: BUY failed (${reason})`, "yellow");
      },
    },
  ]);
}

function flipPosition(
  ctx: StrategyContext,
  state: State,
  nextSide: ModelSide,
  confidence: number,
): void {
  const position = state.position;
  if (!position) return;

  const bid = ctx.orderBook.bestBidPrice(position.side);
  if (bid === null) return;

  state.pendingExit = true;
  ctx.log(
    `[${ctx.slug}] ml-signal: FLIP ${position.side} -> ${nextSide}, SELL @ ${bid.toFixed(2)} confidence=${confidence.toFixed(3)}`,
    "yellow",
  );

  ctx.postOrders([
    {
      req: {
        tokenId: position.tokenId,
        action: "sell",
        price: bid,
        shares: position.shares,
        orderType: "FOK",
      },
      expireAtMs: Date.now() + 3_000,
      onFilled() {
        state.pendingExit = false;
        state.position = null;
        placeEntry(ctx, state, nextSide, confidence);
      },
      onExpired() {
        state.pendingExit = false;
      },
      onFailed(reason) {
        state.pendingExit = false;
        ctx.log(`[${ctx.slug}] ml-signal: SELL failed (${reason})`, "yellow");
      },
    },
  ]);
}

export const mlSignal: Strategy = async (ctx) => {
  if (Env.get("PROD")) {
    ctx.log("[ml-signal] Strategy is simulation-only for now.", "red");
    process.exit(1);
  }

  const releaseLock = ctx.hold();
  const state: State = {
    position: null,
    pendingEntry: false,
    pendingExit: false,
  };

  let busy = false;
  let released = false;
  let lastWaitingLogAt = 0;

  const pollMs = numberEnv("POLY_MODEL_POLL_MS", DEFAULT_POLL_MS);
  const minConfidence = numberEnv(
    "POLY_MODEL_MIN_CONFIDENCE",
    DEFAULT_MIN_CONFIDENCE,
  );
  const flipConfidence = numberEnv(
    "POLY_MODEL_FLIP_CONFIDENCE",
    DEFAULT_FLIP_CONFIDENCE,
  );
  const minRemainingSec = numberEnv(
    "POLY_MODEL_MIN_REMAINING_SEC",
    DEFAULT_MIN_REMAINING_SEC,
  );

  const release = () => {
    if (released) return;
    released = true;
    releaseLock();
  };

  const tick = async () => {
    if (busy || released) return;

    const remaining = Math.floor((ctx.slotEndMs - Date.now()) / 1000);
    if (remaining <= 0) {
      release();
      return;
    }

    if (remaining < minRemainingSec && !state.position) {
      release();
      return;
    }

    if (!ctx.getMarketResult()?.openPrice) {
      const now = Date.now();
      if (now - lastWaitingLogAt >= 30_000) {
        lastWaitingLogAt = now;
        ctx.log(
          `[${ctx.slug}] ml-signal: waiting for market open price`,
          "dim",
        );
      }
      return;
    }

    busy = true;
    try {
      const signal = await runPythonSignal();
      const probability =
        signal.probabilityUp === null
          ? "n/a"
          : signal.probabilityUp.toFixed(3);
      ctx.log(
        `[${ctx.slug}] ml-signal: model=${signal.side} confidence=${signal.confidence.toFixed(3)} p_up=${probability}`,
        "dim",
      );

      if (signal.side === "NO_EDGE" || signal.confidence < minConfidence) {
        return;
      }

      if (!state.position && !state.pendingEntry) {
        placeEntry(ctx, state, signal.side, signal.confidence);
        return;
      }

      if (
        state.position &&
        !state.pendingExit &&
        state.position.side !== signal.side &&
        signal.confidence >= flipConfidence
      ) {
        flipPosition(ctx, state, signal.side, signal.confidence);
      }
    } catch (error) {
      ctx.log(`[${ctx.slug}] ml-signal: model error: ${error}`, "red");
    } finally {
      busy = false;
    }
  };

  ctx.log(`[${ctx.slug}] ml-signal: strategy active`, "dim");
  void tick();
  const interval = setInterval(() => void tick(), pollMs);

  return () => {
    clearInterval(interval);
    release();
  };
};
