"""Markov Regime Dashboard — FastAPI backend."""

from __future__ import annotations
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(title="Markov Dashboard")
app.mount("/static", StaticFiles(directory="static"), name="static")

ASSETS: list[dict] = [
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

TF_CONFIG: dict[str, dict] = {
    "15m": {"interval": "15m", "period": "60d",  "resample": None, "lookback": 48,  "threshold": 0.015},
    "1h":  {"interval": "1h",  "period": "730d", "resample": None, "lookback": 48,  "threshold": 0.020},
    "4h":  {"interval": "1h",  "period": "730d", "resample": "4h", "lookback": 42,  "threshold": 0.025},
    "1d":  {"interval": "1d",  "period": "5y",   "resample": None, "lookback": 20,  "threshold": 0.020},
}

STATES   = ["Bear", "Sideways", "Bull"]
ATR_LEN  = 14
CACHE_TTL = timedelta(minutes=60)

_cache:      dict[str, dict[str, Any]] = {tf: {} for tf in TF_CONFIG}
_cache_time: dict[str, datetime]       = {}


def _label(close: pd.Series, lookback: int, threshold: float) -> pd.Series:
    lr  = np.log(close / close.shift(lookback))
    lbl = pd.Series(1, index=close.index, dtype=int)
    lbl[lr >  threshold] = 2
    lbl[lr < -threshold] = 0
    return lbl.dropna()


def _transition_matrix(labels: pd.Series) -> np.ndarray:
    counts = np.zeros((3, 3))
    arr    = labels.to_numpy()
    for i in range(len(arr) - 1):
        counts[arr[i], arr[i + 1]] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return counts / row_sums


def _stationary(P: np.ndarray) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eig(P.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    vec = np.abs(np.real(eigvecs[:, idx]))
    return vec / vec.sum()


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    tr   = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(ATR_LEN).mean()


def _backtest(df: pd.DataFrame, lookback: int, threshold: float) -> dict:
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
    atr_s  = _compute_atr(df["High"], df["Low"], close)

    cur   = int(labels.iloc[-1])
    price = float(close.iloc[-1])
    atr   = float(atr_s.iloc[-1])

    bull_p = round(float(P[cur, 2]) * 100, 1)
    bear_p = round(float(P[cur, 0]) * 100, 1)
    side_p = round(float(P[cur, 1]) * 100, 1)

    # Signal: follow the dominant next-bar probability
    edge = bull_p - bear_p
    if edge >= 10:
        signal = "LONG"
    elif edge <= -10:
        signal = "SHORT"
    else:
        signal = "FLAT"

    bt = _backtest(df, cfg["lookback"], cfg["threshold"])

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
        "stationary": {
            "bull": round(float(pi[2]) * 100, 1),
            "side": round(float(pi[1]) * 100, 1),
            "bear": round(float(pi[0]) * 100, 1),
        },
        "backtest": bt,
    }


async def _refresh_tf(tf: str):
    log.info(f"Refreshing [{tf}]...")
    for a in ASSETS:
        try:
            _cache[tf][a["ticker"]] = _analyse(a["ticker"], tf)
            log.info(f"  [{tf}] {a['label']} → {_cache[tf][a['ticker']]['signal']}")
        except Exception as exc:
            log.warning(f"  [{tf}] {a['label']} failed: {exc}")
            _cache[tf][a["ticker"]] = {"error": str(exc)}
    _cache_time[tf] = datetime.now(timezone.utc)


@app.on_event("startup")
async def startup():
    asyncio.create_task(_refresh_tf("1h"))
    asyncio.create_task(_bg_loop())


async def _bg_loop():
    while True:
        await asyncio.sleep(CACHE_TTL.seconds)
        for tf, last in list(_cache_time.items()):
            if datetime.now(timezone.utc) - last >= CACHE_TTL:
                await _refresh_tf(tf)


@app.get("/api/regimes")
async def get_regimes(tf: str = Query(default="1h")):
    if tf not in TF_CONFIG:
        tf = "1h"
    if tf not in _cache_time:
        await _refresh_tf(tf)
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
