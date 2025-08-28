#!/usr/bin/env python3
"""
Discord Watchlist Webhook Bot — Stocks + Crypto (1m or faster)

What it does
- Polls your watchlist on a schedule (per-minute by default; you can lower to seconds for crypto)
- Builds a chart image for each symbol and posts to a Discord channel via Webhook
- Calculates dynamic support/resistance, RSI, EMAs, ATR trailing stop
- Suggests: optimal BUY-in (nearest support), short-term signal, long-term trend, take-profit, and a trailing stop
- Auto-scales with the market: indicators and levels recalculated on every run
- "changed" mode only posts when signal/levels change (reduces spam)

Data sources
- Crypto: Binance public REST (near-real-time)
- Stocks/indices/commodities/FX: Yahoo Finance via yfinance (free feeds can be delayed)
  For true second-level equities, plug in a paid feed (Polygon/Finnhub/Alpaca). Provider layer is modular.

Environment (Railway → Variables)
- DISCORD_WEBHOOK_URL (secret) — REQUIRED
- WATCHLIST_STOCKS — e.g. AAPL,MSFT,^NDX,^GSPC,GC=F,CL=F,JPY=X
- WATCHLIST_CRYPTO — e.g. ETHUSDT,TRXUSDT,BTCUSDT,ADAUSDT
- POLL_SECONDS (default 60), POST_MODE=changed|always, INTERVAL=1m
- RSI_PERIOD, RSI_OVERBOUGHT, RSI_OVERSOLD, EMA_FAST, EMA_SLOW
- ATR_PERIOD, ATR_MULTIPLIER, LEVEL_TOL_PCT, TIMEZONE
"""
import os
import json
import time
import logging
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# -------------------- Config --------------------
load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
WATCHLIST_STOCKS = [s.strip() for s in os.getenv("WATCHLIST_STOCKS", "AAPL,MSFT,TSLA").split(",") if s.strip()]
WATCHLIST_CRYPTO = [s.strip().upper() for s in os.getenv("WATCHLIST_CRYPTO", "ETHUSDT,TRXUSDT,BTCUSDT").split(",") if s.strip()]
POLL_SECONDS = int(os.getenv("POLL_SECONDS", "60"))
POST_MODE = os.getenv("POST_MODE", "changed").lower()  # "changed" | "always"
INTERVAL = os.getenv("INTERVAL", "1m").lower()
STATE_FILE = os.getenv("STATE_FILE", ".bot_state.json")

RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", "70"))
RSI_OVERSOLD = float(os.getenv("RSI_OVERSOLD", "30"))
EMA_FAST = int(os.getenv("EMA_FAST", "50"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "200"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))
ATR_MULTIPLIER = float(os.getenv("ATR_MULTIPLIER", "2.0"))
LEVEL_TOL_PCT = float(os.getenv("LEVEL_TOL_PCT", "0.004"))  # 0.4%

CHART_DIR = os.path.join(os.getcwd(), "charts")
os.makedirs(CHART_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)

# -------------------- Helpers --------------------
def as_float(x):
    """Return a plain Python float from a numpy/pandas scalar."""
    try:
        return float(x.item())  # numpy scalar path
    except AttributeError:
        return float(x)

# -------------------- Providers --------------------
class BaseProvider:
    name = "base"
    def get_recent_candles(self, symbol: str, interval: str = "1m", limit: int = 500) -> pd.DataFrame:
        raise NotImplementedError
    def get_last_price(self, symbol: str) -> Optional[float]:
        raise NotImplementedError

class YahooProvider(BaseProvider):
    name = "yfinance"
    def __init__(self):
        try:
            import yfinance as yf
        except ImportError:
            raise SystemExit("Missing yfinance. Run: pip install yfinance")
        self.yf = yf
    def get_recent_candles(self, symbol: str, interval: str = "1m", limit: int = 300) -> pd.DataFrame:
        # Works for stocks, indices (^GSPC), futures (GC=F), FX (JPY=X), LSE tickers (RR.L/BP.L).
        df = self.yf.download(
            symbol, period="1d", interval=interval, progress=False, auto_adjust=False
        )
        if df is None or df.empty:
            return pd.DataFrame(columns=["open","high","low","close","volume"])
        df = df.rename(columns=str.lower)
        df = df[["open","high","low","close","volume"]]
        if len(df) > limit:
            df = df.iloc[-limit:]
        return df
    def get_last_price(self, symbol: str) -> Optional[float]:
        try:
            t = self.yf.Ticker(symbol)
            px = t.fast_info.last_price
            return float(px) if px is not None else None
        except Exception:
            return None

class BinanceProvider(BaseProvider):
    name = "binance"
    BASE = "https://api.binance.com"
    def get_recent_candles(self, symbol: str, interval: str = "1m", limit: int = 500) -> pd.DataFrame:
        url = f"{self.BASE}/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        cols = [
            "open_time","open","high","low","close","volume",
            "close_time","quote_asset_volume","number_of_trades",
            "taker_buy_base","taker_buy_quote","ignore"
        ]
        df = pd.DataFrame(data, columns=cols)
        if df.empty:
            return pd.DataFrame(columns=["open","high","low","close","volume"])
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df.set_index("open_time")
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
        return df[["open","high","low","close","volume"]]
    def get_last_price(self, symbol: str) -> Optional[float]:
        url = f"{self.BASE}/api/v3/ticker/price"
        r = requests.get(url, params={"symbol": symbol}, timeout=10)
        if r.status_code != 200:
            return None
        try:
            return float(r.json().get("price"))
        except Exception:
            return None

PROVIDERS = {
    "stocks": YahooProvider(),
    "crypto": BinanceProvider(),
}

# -------------------- Technicals --------------------
@dataclass
class Levels:
    support: List[float]
    resistance: List[float]

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def find_sr_levels(df: pd.DataFrame, lookback: int = 200, window: int = 3, tol_pct: float = 0.004) -> Levels:
    if df.empty:
        return Levels(support=[], resistance=[])
    highs, lows = [], []
    hi = df["high"].values
    lo = df["low"].values
    n = len(df)
    start = max(0, n - lookback)
    for i in range(start + window, n - window):
        seg_h = hi[i - window:i + window + 1]
        seg_l = lo[i - window:i + window + 1]
        if hi[i] == seg_h.max():
            highs.append(hi[i])
        if lo[i] == seg_l.min():
            lows.append(lo[i])
    def dedupe(vals: List[float], tol: float) -> List[float]:
        vals = sorted(vals)
        out: List[float] = []
        for v in vals:
            if not out or abs(v - out[-1]) > tol:
                out.append(v)
        return out
    px = as_float(df["close"].iloc[-1])
    tol = max(px * tol_pct, 1e-9)
    return Levels(
        support=dedupe(lows, tol),
        resistance=dedupe(highs, tol),
    )

def nearest_level(levels: List[float], price: float, below: bool) -> Optional[float]:
    if not levels:
        return None
    if below:
        cands = [lv for lv in levels if lv <= price]
        return max(cands) if cands else None
    else:
        cands = [lv for lv in levels if lv >= price]
        return min(cands) if cands else None

# -------------------- Signals --------------------
@dataclass
class Signals:
    short: str
    long: str
    optimal_buy: Optional[float]
    stop_suggestion: Optional[float]
    take_profit: Optional[float]
    rsi_value: Optional[float]

def make_signals(df: pd.DataFrame) -> Signals:
    if df.empty or len(df) < max(EMA_SLOW + 2, RSI_PERIOD + 2, ATR_PERIOD + 2):
        return Signals(short="NO DATA", long="NO DATA",
                       optimal_buy=None, stop_suggestion=None,
                       take_profit=None, rsi_value=None)

    close = df["close"]
    ema_fast_ser = ema(close, EMA_FAST)
    ema_slow_ser = ema(close, EMA_SLOW)
    rsi_series = rsi(close, RSI_PERIOD)
    atr_series = atr(df, ATR_PERIOD)
    lvls = find_sr_levels(df, tol_pct=LEVEL_TOL_PCT)

    last = as_float(close.iloc[-1])
    last_rsi = as_float(rsi_series.iloc[-1])
    last_atr = as_float(atr_series.iloc[-1])

    # ensure real boolean (not a pandas Series)
    uptrend = bool(as_float(ema_fast_ser.iloc[-1]) > as_float(ema_slow_ser.iloc[-1]))

    # nearest support below, resistance above
    s = nearest_level(lvls.support, last, below=True)
    r = nearest_level(lvls.resistance, last, below=False)

    # Optimal buy = nearest support; fallback EMA slow
    optimal_buy = s if s is not None else as_float(ema_slow_ser.iloc[-1])

    # Short-term signal using RSI + proximity to S/R within 0.5 * ATR
    short_sig = "HOLD"
    if s is not None and (last - s) <= 0.5 * last_atr and last_rsi <= RSI_OVERSOLD + 5:
        short_sig = "BUY"
    elif r is not None and (r - last) <= 0.5 * last_atr and last_rsi >= RSI_OVERBOUGHT - 5:
        short_sig = "SELL"

    # Long-term trend
    long_sig = "ACCUMULATE" if uptrend else "AVOID/REDUCE"

    # ATR trailing stop
    stop = last - ATR_MULTIPLIER * last_atr if uptrend else last + ATR_MULTIPLIER * last_atr

    # Take-profit = nearest resistance
    take_profit = r

    return Signals(
        short=short_sig,
        long=long_sig,
        optimal_buy=float(optimal_buy) if optimal_buy else None,
        stop_suggestion=float(stop),
        take_profit=float(take_profit) if take_profit else None,
        rsi_value=last_rsi
    )

# -------------------- Charting --------------------
def make_chart(symbol: str, df: pd.DataFrame, path: str) -> None:
    if df.empty:
        fig = plt.figure(figsize=(8, 3))
        plt.title(f"{symbol} — no data")
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)
        return

    plot_df = df.iloc[-150:] if len(df) > 150 else df
    clos = plot_df["close"]
    ema_f_span = EMA_FAST if EMA_FAST < len(clos) else max(2, len(clos)//3)
    ema_s_span = EMA_SLOW if EMA_SLOW < len(clos) else max(4, len(clos)//2)
    ema_f = ema(clos, ema_f_span)
    ema_s = ema(clos, ema_s_span)

    fig = plt.figure(figsize=(9, 4))
    ax = plt.gca()
    ax.plot(clos.index, clos.values, label="Close")
    ax.plot(ema_f.index, ema_f.values, label=f"EMA{ema_f_span}")
    ax.plot(ema_s.index, ema_s.values, label=f"EMA{ema_s_span}")

    lvls = find_sr_levels(df)
    last = as_float(df["close"].iloc[-1])
    s = nearest_level(lvls.support, last, below=True)
    r = nearest_level(lvls.resistance, last, below=False)
    if s is not None:
        ax.axhline(s, linestyle='--', alpha=0.6, label=f"Support {s:.6g}")
    if r is not None:
        ax.axhline(r, linestyle='--', alpha=0.6, label=f"Resistance {r:.6g}")

    ax.set_title(f"{symbol} — {len(clos)} bars ({INTERVAL})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)

    fig.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)

# -------------------- Discord --------------------
def send_discord_with_image(webhook: str, content: str, embed: Dict, image_path: str) -> bool:
    try:
        with open(image_path, 'rb') as f:
            files = {'file': (os.path.basename(image_path), f, 'image/png')}
            embed = dict(embed)
            embed['image'] = {"url": f"attachment://{os.path.basename(image_path)}"}
            payload = {
                "content": content,
                "embeds": [embed],
                "username": "Watchlist Bot",
            }
            r = requests.post(webhook, data={"payload_json": json.dumps(payload)}, files=files, timeout=20)
            if 200 <= r.status_code < 300:
                return True
            logging.warning("Discord response %s: %s", r.status_code, r.text[:200])
            return False
    except Exception as e:
        logging.exception("send_discord_with_image failed: %s", e)
        return False

# -------------------- State (to avoid spam) --------------------
def load_state() -> Dict:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state: Dict) -> None:
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception:
        logging.warning("Could not write state file.")

# -------------------- Processing --------------------
def format_price(x: float) -> str:
    if x >= 1000:
        return f"{x:,.2f}"
    elif x >= 1:
        return f"{x:.2f}"
    else:
        return f"{x:.6f}".rstrip('0').rstrip('.')

def build_embed(symbol: str, market: str, price: float, sig: Signals) -> Dict:
    fields = []
    fields.append({"name": "Last Price", "value": f"`{format_price(price)}`", "inline": True})
    if sig.rsi_value is not None:
        fields.append({"name": "RSI", "value": f"`{sig.rsi_value:.1f}`", "inline": True})
    if sig.optimal_buy is not None:
        fields.append({"name": "Optimal Buy-in (support)", "value": f"`{format_price(sig.optimal_buy)}`", "inline": True})
    if sig.take_profit is not None:
        fields.append({"name": "Take Profit (resistance)", "value": f"`{format_price(sig.take_profit)}`", "inline": True})
    if sig.stop_suggestion is not None:
        fields.append({"name": "ATR Stop Suggestion", "value": f"`{format_price(sig.stop_suggestion)}`", "inline": True})

    fields.append({"name": "Short-term", "value": f"**{sig.short}**", "inline": True})
    fields.append({"name": "Long-term", "value": f"**{sig.long}**", "inline": True})

    embed = {
        "title": f"{symbol} ({'Crypto' if market=='crypto' else 'Yahoo feed'})",
        "description": "Dynamic levels & signals recalculated every run. Not financial advice.",
        "fields": fields,
        "footer": {"text": f"Interval {INTERVAL} | Provider: {'Binance' if market=='crypto' else 'Yahoo Finance'}"},
    }
    return embed

def process_symbol(market: str, symbol: str, state: Dict) -> None:
    provider = PROVIDERS["crypto" if market == "crypto" else "stocks"]
    try:
        if market == 'crypto':
            candles = provider.get_recent_candles(symbol, interval=INTERVAL if INTERVAL in {"1m","3m","5m","15m","30m"} else "1m")
        else:
            # yfinance behaves best with 1m here
            candles = provider.get_recent_candles(symbol, interval="1m")
    except Exception:
        logging.exception("Failed to fetch candles for %s %s", market, symbol)
        return

    sig = make_signals(candles)
    last_px = provider.get_last_price(symbol)
    if last_px is None:
        if not candles.empty:
            last_px = as_float(candles["close"].iloc[-1])
        else:
            logging.warning("No price for %s", symbol)
            return

    # Chart file
    safe_name = symbol.replace('^','').replace('=','').replace('/','-')
    img_path = os.path.join(CHART_DIR, f"{market}_{safe_name}.png")
    try:
        make_chart(symbol, candles, img_path)
    except Exception:
        logging.exception("Chart failed for %s", symbol)
        return

    # Compare signature to avoid spam
    new_signature = {
        "short": sig.short,
        "long": sig.long,
        "optimal": round(sig.optimal_buy or 0, 8),
        "stop": round(sig.stop_suggestion or 0, 8),
        "tp": round(sig.take_profit or 0, 8),
    }
    key = f"{market}:{symbol}"
    old_signature = state.get(key)
    should_post = (POST_MODE == "always") or (new_signature != old_signature)

    if not should_post:
        logging.info("No change for %s — skipping post.", key)
        return

    embed = build_embed(symbol, market, float(last_px), sig)
    content = f"**{symbol}** update — {time.strftime('%Y-%m-%d %H:%M:%S')}"

    if DISCORD_WEBHOOK_URL:
        ok = send_discord_with_image(DISCORD_WEBHOOK_URL, content, embed, img_path)
        if ok:
            logging.info("Posted %s", key)
            state[key] = new_signature
            save_state(state)
        else:
            logging.warning("Post failed for %s", key)
    else:
        logging.error("No DISCORD_WEBHOOK_URL set. Set it in Railway → Variables as a Secret.")

# -------------------- Main loop --------------------
def run_once(symbols_stocks: List[str], symbols_crypto: List[str]) -> None:
    state = load_state()
    for sym in symbols_stocks:
        process_symbol("stock", sym, state)
        time.sleep(1)  # gentle pacing
    for sym in symbols_crypto:
        process_symbol("crypto", sym, state)
        time.sleep(1)

def main():
    parser = argparse.ArgumentParser(description="Discord Watchlist Webhook Bot")
    parser.add_argument("--once", action="store_true", help="Run a single pass and exit")
    args = parser.parse_args()

    if not WATCHLIST_STOCKS and not WATCHLIST_CRYPTO:
        logging.error("Your watchlists are empty. Set WATCHLIST_STOCKS and/or WATCHLIST_CRYPTO in env")
        return

    logging.info("Starting — stocks=%s | crypto=%s | interval=%s | mode=%s",
                 WATCHLIST_STOCKS, WATCHLIST_CRYPTO, INTERVAL, POST_MODE)

    if args.once:
        run_once(WATCHLIST_STOCKS, WATCHLIST_CRYPTO)
        return

    while True:
        start = time.time()
        try:
            run_once(WATCHLIST_STOCKS, WATCHLIST_CRYPTO)
        except Exception:
            logging.exception("Run failed (continuing)")
        elapsed = time.time() - start
        sleep_for = max(1, POLL_SECONDS - int(elapsed))
        logging.info("Sleeping %ss", sleep_for)
        time.sleep(sleep_for)

# expose run_once for serverless reuse if needed
__all__ = ["run_once"]

if __name__ == "__main__":
    main()
