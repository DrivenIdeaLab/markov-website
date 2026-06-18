"""Markov Regime Dashboard — FastAPI backend.

Framework: Roan (@RohOnChain) — https://x.com/RohOnChain
Website:  Lewis Jackson — https://github.com/lucusmartin1-bit/markov-website
Repo:     https://github.com/jackson-video-resources/markov-hedge-fund-method

Enhanced to match the full framework:
  - Full 3x3 transition matrix + persistence diagonal in API response
  - Stationary distribution (left eigenvector)
  - Walk-forward backtest with Sharpe + max drawdown
  - Crypto assets (BTC, ETH, SOL, etc.)
  - Proper lifespan (no deprecated @app.on_event)
  - All TFs seeded on startup
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ── Asset universe ──────────────────────────────────────────────────────────
ASSETS: list[dict] = [
    # Crypto
    {"label": "Bitcoin",       "ticker": "BTC-USD",    "group": "Crypto"},
    {"label": "Ethereum",      "ticker": "ETH-USD",    "group": "Crypto"},
    {"label": "Solana",        "ticker": "SOL-USD",    "group": "Crypto"},
    {"label": "XRP",           "ticker": "XRP-USD",    "group": "Crypto"},
    {"label": "Cardano",       "ticker": "ADA-USD",    "group": "Crypto"},
    {"label": "Avalanche",     "ticker": "AVAX-USD",   "group": "Crypto"},
    {"label": "Chainlink",     "ticker": "LINK-USD",   "group": "Crypto"},
    {"label": "Dogecoin",      "ticker": "DOGE-USD",   "group": "Crypto"},
    # Forex
    {"label": "EUR/USD", "ticker": "EURUSD=X", "group": "Forex"},
    {"label": "GBP/USD", "ticker": "GBPUSD=X", "group": "Forex"},
    {"label": "USD/JPY", "ticker": "USDJPY=X", "group": "Forex"},
    {"label": "GBP/JPY", "ticker": "GBPJPY=X", "group": "Forex"},
    {"label": "EUR/GBP", "ticker": "EURGBP=X", "group": "Forex"},
    {"label": "AUD/USD", "ticker": "AUDUSD=X", "group": "Forex"},
    {"label": "USD/CAD", "ticker": "USDCAD=X", "group": "Forex"},
    {"label": "EUR/JPY", "ticker": "EURJPY=X", "group": "Forex"},
    # US Indices
    {"label": "S&P 500",      "ticker": "SPY",      "group": "US Indices"},
    {"label": "NASDAQ 100",   "ticker": "QQQ",      "group": "US Indices"},
    {"label": "Dow Jones",    "ticker": "DIA",      "group": "US Indices"},
    {"label": "Russell 2000", "ticker": "IWM",      "group": "US Indices"},
    {"label": "VIX",          "ticker": "^VIX",     "group": "US Indices"},
    # US Sectors
    {"label": "Technology",    "ticker": "XLK",  "group": "US Sectors"},
    {"label": "Financials",    "ticker": "XLF",  "group": "US Sectors"},
    {"label": "Energy",        "ticker": "XLE",  "group": "US Sectors"},
    {"label": "Healthcare",    "ticker": "XLV",  "group": "US Sectors"},
    {"label": "Industrials",   "ticker": "XLI",  "group": "US Sectors"},
    {"label": "Consumer Disc", "ticker": "XLY",  "group": "US Sectors"},
    {"label": "Utilities",     "ticker": "XLU",  "group": "US Sectors"},
    {"label": "Real Estate",   "ticker": "XLRE", "group": "US Sectors"},
    # Global Indices
    {"label": "FTSE 100",      "ticker": "^FTSE",     "group": "Global Indices"},
    {"label": "DAX",           "ticker": "^GDAXI",    "group": "Global Indices"},
    {"label": "Nikkei 225",    "ticker": "^N225",     "group": "Global Indices"},
    {"label": "Hang Seng",     "ticker": "^HSI",      "group": "Global Indices"},
    {"label": "Euro Stoxx 50", "ticker": "^STOXX50E", "group": "Global Indices"},
    {"label": "ASX 200",       "ticker": "^AXJO",     "group": "Global Indices"},
    {"label": "CAC 40",        "ticker": "^FCHI",     "group": "Global Indices"},
    # Commodities
    {"label": "Gold",   "ticker": "GLD",  "group": "Commodities"},
    {"label": "Oil",    "ticker": "USO",  "group": "Commodities"},
    {"label": "Silver", "ticker": "SLV",  "group": "Commodities"},
    {"label": "Copper", "ticker": "CPER", "group": "Commodities"},
]

# ── Timeframe configs ───────────────────────────────────────────────────────
# lookback = rolling window for regime label
# threshold = ±threshold for bull/bear classification
TF_CONFIG: dict[str, dict] = {
    "15m": {"interval": "15m", "period": "60d",  "resample": None, "lookback": 48,  "threshold": 0.015},
    "1h":  {"interval": "1h",  "period": "730d", "resample": None, "lookback": 48,  "threshold": 0.020},
    "4h":  {"interval": "1h",  "period": "730d", "resample": "4h", "lookback": 42,  "threshold": 0.025},
    "1d":  {"interval": "1d",  "period": "5y",   "resample": None, "lookback": 20,  "threshold": 0.020},
}

# State indices: 0=Bear, 1=Sideways, 2=Bull
STATES   = ["Bear", "Sideways", "Bull"]
ATR_LEN  = 14
CACHE_TTL = timedelta(minutes=60)

_cache:      dict[str, dict[str, Any]] = {tf: {} for tf in TF_CONFIG}
_cache_time: dict[str, datetime] = {}


# ── Core framework functions (from markov_regime.py) ─────────────────────────

def _label(close: pd.Series, lookback: int, threshold: float) -> pd.Series:
    """Label each bar from the rolling log-return. 0=Bear, 1=Sideways, 2=Bull."""
    lr  = np.log(close / close.shift(lookback))
    lbl = pd.Series(1, index=close.index, dtype=int)  # default Sideways
    lbl[lr >  threshold] = 2  # Bull
    lbl[lr < -threshold] = 0  # Bear
    return lbl.dropna()


def _transition_matrix(labels: pd.Series) -> np.ndarray:
    """MLE 3x3 transition matrix."""
    counts = np.zeros((3, 3))
    arr    = labels.to_numpy()
    for i in range(len(arr) - 1):
        counts[arr[i], arr[i + 1]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return counts / row_sums


def _stationary(P: np.ndarray) -> np.ndarray:
    """Stationary distribution — left eigenvector for eigenvalue 1."""
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    vec = np.abs(np.real(eigvecs[:, idx]))
    return vec / vec.sum()


def _persistence(P: np.ndarray) -> dict:
    """Diagonal of the transition matrix = how sticky each regime is."""
    return {
        "bear":     round(float(P[0, 0]) * 100, 1),
        "sideways": round(float(P[1, 1]) * 100, 1),
        "bull":     round(float(P[2, 2]) * 100, 1),
    }


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    tr   = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(ATR_LEN).mean()


def _backtest(df: pd.DataFrame, lookback: int, threshold: float) -> dict:
    """Simple ATR-based backtest: enter on regime flip, TP/SL at 3x ATR."""
    ATR_MULT = 3.0
    close  = df["Close"].values.astype(float)
    high   = df["High"].values.astype(float)
    low    = df["Low"].values.astype(float)
    atr_a  = _compute_atr(df["High"], df["Low"], df["Close"]).values.astype(float)
    labels = _label(df["Close"], lookback, threshold).values.astype(int)

    wins, losses, skip_until = 0, 0, 0
    for k in range(lookback + 1, len(labels)):
        pos = k
        if pos < skip_until:
            continue
        prev_r, curr_r = labels[k - 1], labels[k]
        if prev_r == curr_r or curr_r == 1:
            continue
        atr_val = atr_a[pos]
        if np.isnan(atr_val) or atr_val == 0:
            continue
        entry = close[pos]
        long  = curr_r == 2
        tp = entry + ATR_MULT * atr_val if long else entry - ATR_MULT * atr_val
        sl = entry - ATR_MULT * atr_val if long else entry + ATR_MULT * atr_val
        for j in range(pos + 1, len(close)):
            h, l = high[j], low[j]
            if long:
                if h >= tp: wins   += 1; skip_until = j + 1; break
                if l <= sl: losses += 1; skip_until = j + 1; break
            else:
                if l <= tp: wins   += 1; skip_until = j + 1; break
                if h >= sl: losses += 1; skip_until = j + 1; break

    total = wins + losses
    return {
        "trades":   total,
        "wins":     wins,
        "losses":   losses,
        "win_rate": round(wins / max(total, 1) * 100, 1),
    }


def _walk_forward(close: pd.Series, labels: pd.Series, min_train: int = 252) -> dict:
    """No-lookahead walk-forward backtest with Sharpe and max drawdown."""
    daily_returns = close.pct_change().dropna()
    common = labels.index.intersection(daily_returns.index)
    labels = labels.loc[common]
    daily_returns = daily_returns.loc[common]

    if len(labels) < min_train + 30:
        return {"sharpe": None, "max_drawdown": None, "n_trades": 0}

    lab  = np.asarray(labels, dtype=int)
    rets = daily_returns.to_numpy(dtype=float)

    counts = np.zeros((3, 3), dtype=float)
    for i in range(min_train - 1):
        counts[lab[i], lab[i + 1]] += 1.0

    strategy = np.empty(len(lab) - 1 - min_train, dtype=float)
    for k, t in enumerate(range(min_train, len(lab) - 1)):
        rs = counts.sum(axis=1, keepdims=True)
        safe = np.where(rs == 0, 1.0, rs)
        P_t = counts / safe
        sig = float(P_t[lab[t], 2] - P_t[lab[t], 0])
        strategy[k] = np.sign(sig) * rets[t + 1]
        counts[lab[t - 1], lab[t]] += 1.0

    std = strategy.std(ddof=1) if len(strategy) > 1 else 0.0
    if std == 0 or not np.isfinite(std):
        sharpe = None
    else:
        sharpe = round(float(strategy.mean() / std * np.sqrt(252)), 3)

    equity = (1.0 + strategy).cumprod()
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max
    max_dd = round(float(drawdown.min() * 100), 2) if len(drawdown) else None

    return {"sharpe": sharpe, "max_drawdown": max_dd, "n_trades": int(len(strategy))}


def _analyse(ticker: str, tf: str) -> dict:
    cfg = TF_CONFIG[tf]
    df  = yf.download(ticker, period=cfg["period"], interval=cfg["interval"],
                      progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()

    if cfg["resample"]:
        df = df.resample(cfg["resample"]).agg(
            {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
        ).dropna()

    if len(df) < cfg["lookback"] + 20:
        raise ValueError("Not enough data")

    close  = df["Close"]
    labels = _label(close, cfg["lookback"], cfg["threshold"])
    P      = _transition_matrix(labels)
    pi     = _stationary(P)
    pers   = _persistence(P)
    atr_s  = _compute_atr(df["High"], df["Low"], close)

    cur   = int(labels.iloc[-1])
    price = float(close.iloc[-1])
    atr   = float(atr_s.iloc[-1])

    bull_p = round(float(P[cur, 2]) * 100, 1)
    bear_p = round(float(P[cur, 0]) * 100, 1)
    side_p = round(float(P[cur, 1]) * 100, 1)

    edge = bull_p - bear_p
    if edge >= 10:
        signal = "LONG"
    elif edge <= -10:
        signal = "SHORT"
    else:
        signal = "FLAT"

    bt = _backtest(df, cfg["lookback"], cfg["threshold"])
    wf = _walk_forward(close, labels)

    return {
        "regime":   STATES[cur],
        "signal":   signal,
        "price":    price,
        "atr":      round(atr, 5),
        "tp":       round(price + 3 * atr, 5),
        "sl":       round(price - 3 * atr, 5),
        "bull_pct": bull_p,
        "bear_pct": bear_p,
        "side_pct": side_p,
        "edge":     round(edge, 1),
        # Full framework fields (from source markov_regime.py)
        "transition_matrix": [[round(float(P[i, j]) * 100, 1) for j in range(3)] for i in range(3)],
        "persistence":       pers,
        "stationary": {
            "bull":     round(float(pi[2]) * 100, 1),
            "sideways": round(float(pi[1]) * 100, 1),
            "bear":     round(float(pi[0]) * 100, 1),
        },
        "walk_forward": wf,
        "backtest": bt,
    }


# ── Cache management ────────────────────────────────────────────────────────

_EXECUTOR = ThreadPoolExecutor(max_workers=4)


def _refresh_tf_sync(tf: str):
    """Synchronous refresh — run in thread pool to avoid blocking the event loop."""
    for a in ASSETS:
        try:
            _cache[tf][a["ticker"]] = _analyse(a["ticker"], tf)
            log.info(f"  [{tf}] {a['label']} → {_cache[tf][a['ticker']]['signal']}")
        except Exception as exc:
            log.warning(f"  [{tf}] {a['label']} failed: {exc}")
            _cache[tf][a["ticker"]] = {"error": str(exc)}
    _cache_time[tf] = datetime.now(timezone.utc)


async def _refresh_tf(tf: str):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_EXECUTOR, _refresh_tf_sync, tf)


async def _bg_loop():
    while True:
        await asyncio.sleep(CACHE_TTL.seconds)
        for tf in TF_CONFIG:
            await _refresh_tf(tf)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: seed all timeframes in background — don't block the server
    for tf in TF_CONFIG:
        asyncio.create_task(_refresh_tf(tf))
    asyncio.create_task(_bg_loop())
    yield


# ── FastAPI app ───────────────────────────────────────────────────────────

app = FastAPI(title="Markov Dashboard", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/api/regimes")
async def get_regimes(tf: str = Query(default="1h")):
    if tf not in TF_CONFIG:
        tf = "1h"
    if tf not in _cache_time:
        # Seed on first request for this TF (non-blocking)
        asyncio.create_task(_refresh_tf(tf))
    results = []
    for a in ASSETS:
        data = _cache[tf].get(a["ticker"], {"error": "loading..."})
        results.append({"label": a["label"], "ticker": a["ticker"],
                         "group": a["group"], **data})
    updated = _cache_time.get(tf)
    return {
        "assets":  results,
        "updated": updated.strftime("%Y-%m-%d %H:%M UTC") if updated else "loading...",
        "tf":      tf,
    }


@app.get("/")
def index():
    return FileResponse("static/index.html")
