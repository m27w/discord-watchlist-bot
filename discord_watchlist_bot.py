#!/usr/bin/env python3
"""
Discord Watchlist Webhook Bot — Stocks + Crypto

- Polls your watchlist on a schedule (default per-minute)
- Builds a chart and posts to a Discord channel via Webhook
- Calculates RSI/EMA/ATR + support/resistance; suggests optimal buy (support), take-profit (resistance), and ATR stop
- Posts text-only if a chart fails (won’t abort posting)
- Yahoo (stocks) falls back to 5m data if 1m is sparse/closed

ENV (Railway → Variables)
  DISCORD_WEBHOOK_URL  (required)
  WATCHLIST_STOCKS     e.g. AMD,PLTR,BABA,GOOG,META,MSFT,AAPL,NVDA,TSLA,AMZN,RR.L,BP.L,^NDX,^GSPC,^DJI,GC=F,CL=F,JPY=X
  WATCHLIST_CRYPTO     e.g. XLMUSDT,TRXUSDT,SHIBUSDT,DOGEUSDT,XRPUSDT,ETHUSDT,FLRUSDT,SOLUSDT,ADAUSDT,BTCUSDT
  POLL_SECONDS=60  POST_MODE=changed|always  INTERVAL=1m
  RSI_PERIOD=14  RSI_OVERBOUGHT=70  RSI_OVERSOLD=30
  EMA_FAST=50  EMA_SLOW=200  ATR_PERIOD=14  ATR_MULTIPLIER=2.0
  LEVEL_TOL_PCT=0.004
"""
import os, json, time, logging, argparse
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
WATCHLIST_STOCKS   = [s.strip() for s in os.getenv("WATCHLIST_STOCKS", "AAPL,MSFT,TSLA").split(",") if s.strip()]
WATCHLIST_CRYPTO   = [s.strip().upper() for s in os.getenv("WATCHLIST_CRYPTO", "ETHUSDT,BTCUSDT").split(",") if s.strip()]
POLL_SECONDS       = int(os.getenv("POLL_SECONDS", "60"))
POST_MODE          = os.getenv("POST_MODE", "changed").lower()            # changed | always
INTERVAL           = os.getenv("INTERVAL", "1m").lower()
STATE_FILE         = os.getenv("STATE_FILE", ".bot_state.json")

RSI_PERIOD     = int(os.getenv("RSI_PERIOD", "14"))
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", "70"))
RSI_OVERSOLD   = float(os.getenv("RSI_OVERSOLD", "30"))
EMA_FAST       = int(os.getenv("EMA_FAST", "50"))
EMA_SLOW       = int(os.getenv("EMA_SLOW", "200"))
ATR_PERIOD     = int(os.getenv("ATR_PERIOD", "14"))
ATR_MULTIPLIER = float(os.getenv("ATR_MULTIPLIER", "2.0"))
LEVEL_TOL_PCT  = float(os.getenv("LEVEL_TOL_PCT", "0.004"))  # 0.4%

CHART_DIR = os.path.join(os.getcwd(), "charts")
os.makedirs(CHART_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logging.info("Bot version: 2025-08-29-fixC")

# -------------------- Helpers --------------------
def as_float(x):
    """Return a plain Python float from numpy/pandas scalars or arrays."""
    if x is None:
        return None
    # If it's a numpy array with a single element, unwrap
    if isinstance(x, (np.ndarray, list, tuple)):
        if len(x) == 0:
            return None
        x = x[0]
    try:
        return float(getattr(x, "item", lambda: x)())
    except Exception:
        return float(x)

def safe_span(series: pd.Series, span: int) -> int:
    n = len(series)
    if n <= 2:
        return max(2, n)
    return min(max(2, span), n - 1)

# -------------------- Providers --------------------
class BaseProvider:
    name = "base"
    def get_recent_candles(self, symbol: str, interval: str = "1m", limit: int = 300) -> pd.DataFrame:
        raise NotImplementedError
    def get_last_price(self, symbol: str) -> Optional[float]:
        raise NotImplementedError

class YahooProvider(BaseProvider):
    name = "yfinance"
    def __init__(self):
        try:
            import yfinance as yf
        except ImportError:
            raise SystemExit("Missing yfinance. pip install yfinance")
        self.yf = yf

    def _dl(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        df = self.yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        if df is None or df.empty:
            return pd.DataFrame()
        df = df.rename(columns=str.lower)
        return df[["open","high","low","close","volume"]]

    def get_recent_candles(self, symbol: str, interval: str = "1m", limit: int = 300) -> pd.DataFrame:
        # Try 1m for today; if empty/sparse (closed market / some indices), fall back to 5m (last 5 days)
        df = self._dl(symbol, "1d", "1m")
        if df.empty or len(df) < 30:
            df = self._dl(symbol, "5d", "5m")
        if df.empty:
            return pd.DataFrame(columns=["open","high","low","close","volume"])
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
        cols = ["open_time","open","high","low","close","volume","close_time",
                "quote","trades","tbb","tbq","ignore"]
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

PROVIDERS = {"stocks": YahooProvider(), "crypto": BinanceProvider()}

# -------------------- Technicals --------------------
@dataclass
class Levels:
    support: List[float]
    resistance: List[float]

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=safe_span(series, span), adjust=False).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    period = max(2, period)
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    period = max(2, period)
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def find_sr_levels(df: pd.DataFrame, lookback: int = 200, window: int = 3, tol_pct: float = 0.004) -> Levels:
    if df.empty:
        return Levels([], [])
    hi, lo = df["high"].values, df["low"].values
    highs, lows = [], []
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
            v = as_float(v)
            if not out or abs(v - out[-1]) > tol:
                out.append(v)
        return out
    px = as_float(df["close"].iloc[-1])
    tol = max(px * tol_pct, 1e-9)
    return Levels(dedupe(lows, tol), dedupe(highs, tol))

def nearest_level(levels: List[float], price: float, below: bool) -> Optional[float]:
    if not levels:
        return None
    if below:
        cands = [as_float(lv) for lv in levels if as_float(lv) <= price]
        val = max(cands) if cands else None
    else:
        cands = [as_float(lv) for lv in levels if as_float(lv) >= price]
        val = min(cands) if cands else None
    return as_float(val) if val is not None else None

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
        return Signals("NO DATA", "NO DATA", None, None, None, None)

    close = df["close"]
    ema_fast_ser = ema(close, EMA_FAST)
    ema_slow_ser = ema(close, EMA_SLOW)
    rsi_series  = rsi(close, RSI_PERIOD)
    atr_series  = atr(df, ATR_PERIOD)
    lvls        = find_sr_levels(df, tol_pct=LEVEL_TOL_PCT)

    last     = as_float(close.iloc[-1])
    last_rsi = as_float(rsi_series.iloc[-1])
    last_atr = as_float(atr_series.iloc[-1])

    uptrend = bool(as_float(ema_fast_ser.iloc[-1]) > as_float(ema_slow_ser.iloc[-1]))

    s = nearest_level(lvls.support, last, below=True)
    r = nearest_level(lvls.resistance, last, below=False)

    optimal_buy = s if s is not None else as_float(ema_slow_ser.iloc[-1])

    short_sig = "HOLD"
    if s is not None and (last - s) <= 0.5 * last_atr and last_rsi <= RSI_OVERSOLD + 5:
        short_sig = "BUY"
    elif r is not None and (r - last) <= 0.5 * last_atr and last_rsi >= RSI_OVERBOUGHT - 5:
        short_sig = "SELL"

    long_sig = "ACCUMULATE" if uptrend else "AVOID/REDUCE"
    stop = last - ATR_MULTIPLIER * last_atr if uptrend else last + ATR_MULTIPLIER * last_atr
    take_profit = r

    return Signals(
        short=short_sig,
        long=long_sig,
        optimal_buy=as_float(optimal_buy) if optimal_buy is not None else None,
        stop_suggestion=as_float(stop),
        take_profit=as_float(take_profit) if take_profit is not None else None,
        rsi_value=last_rsi
    )

# -------------------- Charting --------------------
def make_chart(symbol: str, df: pd.DataFrame, path: str) -> None:
    # Always create *some* image so message can include an attachment.
    if df.empty:
        fig = plt.figure(figsize=(8, 3))
        plt.title(f"{symbol} — no intraday data")
        plt.savefig(path, bbox_inches='tight')
        plt.close(fig)
        return

    plot_df = df.iloc[-150:] if len(df) > 150 else df
    clos = plot_df["close"]
    ema_f = ema(clos, EMA_FAST)
    ema_s = ema(clos, EMA_SLOW)

    fig = plt.figure(figsize=(9, 4))
    ax = plt.gca()
    ax.plot(clos.index, clos.values, label="Close")
    ax.plot(ema_f.index, ema_f.values, label=f"EMA{safe_span(clos, EMA_FAST)}")
    ax.plot(ema_s.index, ema_s.values, label=f"EMA{safe_span(clos, EMA_SLOW)}")

    lvls = find_sr_levels(df)
    last = as_float(df["close"].iloc[-1])
    s = nearest_level(lvls.support, last, below=True)
    r = nearest_level(lvls.resistance, last, below=False)
    if s is not None:
        s = as_float(s)
        ax.axhline(s, linestyle='--', alpha=0.6, label=f"Support {s:.6g}")
    if r is not None:
        r = as_float(r)
        ax.axhline(r, linestyle='--', alpha=0.6, label=f"Resistance {r:.6g}")

    ax.set_title(f"{symbol} — {len(clos)} bars ({INTERVAL})")
    ax.set_xlabel("Time"); ax.set_ylabel("Price")
    ax.legend(loc="best"); ax.grid(True, alpha=0.2)

    fig.tight_layout()
    plt.savefig(path, bbox_inches='tight')
    plt.close(fig)

# -------------------- Discord --------------------
def send_discord(webhook: str, content: str, embed: Dict, image_path: Optional[str]) -> bool:
    """Send with optional image. If image_path is None/missing, post without file."""
    try:
        payload = {"content": content, "embeds": [dict(embed)], "username": "Watchlist Bot"}
        files = None
        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                files = {'file': (os.path.basename(image_path), f, 'image/png')}
                payload["embeds"][0]["image"] = {"url": f"attachment://{os.path.basename(image_path)}"}
                r = requests.post(webhook, data={"payload_json": json.dumps(payload)}, files=files, timeout=20)
        else:
            r = requests.post(webhook, json=payload, timeout=20)

        if 200 <= r.status_code < 300:
            return True
        logging.warning("Discord response %s: %s", r.status_code, r.text[:200])
        return False
    except Exception as e:
        logging.exception("send_discord failed: %s", e)
        return False

# -------------------- State --------------------
def load_state() -> Dict:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_state(state: Dict) -> None:
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=2)
    except Exception:
        logging.warning("Could not write state file.")

# -------------------- Messaging --------------------
def format_price(x: float) -> str:
    if x is None:
        return "—"
    if x >= 1000: return f"{x:,.2f}"
    if x >= 1:    return f"{x:.2f}"
    return f"{x:.6f}".rstrip('0').rstrip('.')

def build_embed(symbol: str, market: str, price: float, sig: Signals) -> Dict:
    fields = [
        {"name": "Last Price", "value": f"`{format_price(price)}`", "inline": True},
        {"name": "RSI", "value": f"`{sig.rsi_value:.1f}`" if sig.rsi_value is not None else "`—`", "inline": True},
        {"name": "Optimal Buy-in (support)", "value": f"`{format_price(sig.optimal_buy)}`", "inline": True},
        {"name": "Take Profit (resistance)", "value": f"`{format_price(sig.take_profit)}`", "inline": True},
        {"name": "ATR Stop Suggestion", "value": f"`{format_price(sig.stop_suggestion)}`", "inline": True},
        {"name": "Short-term", "value": f"**{sig.short}**", "inline": True},
        {"name": "Long-term", "value": f"**{sig.long}**", "inline": True},
    ]
    return {
        "title": f"{symbol} ({'Crypto' if market=='crypto' else 'Yahoo feed'})",
        "description": "Dynamic levels & signals recalculated every run. Not financial advice.",
        "fields": fields,
        "footer": {"text": f"Interval {INTERVAL} | Provider: {'Binance' if market=='crypto' else 'Yahoo Finance (1m→5m fallback)'}"},
    }

# -------------------- Orchestration --------------------
def process_symbol(market: str, symbol: str, state: Dict) -> None:
    provider = PROVIDERS["crypto" if market == "crypto" else "stocks"]

    # 1) Get candles
    try:
        if market == "crypto":
            candles = provider.get_recent_candles(symbol, interval=INTERVAL if INTERVAL in {"1m","3m","5m","15m","30m"} else "1m")
        else:
            candles = provider.get_recent_candles(symbol, interval="1m")  # Yahoo handles fallback internally
    except Exception:
        logging.exception("Failed to fetch candles for %s %s", market, symbol)
        candles = pd.DataFrame(columns=["open","high","low","close","volume"])

    # 2) Signals & price
    sig = make_signals(candles) if not candles.empty else Signals("NO DATA","NO DATA",None,None,None,None)
    last_px = provider.get_last_price(symbol)
    if last_px is None:
        last_px = as_float(candles["close"].iloc[-1]) if not candles.empty else None

    # 3) Chart (do NOT abort posting if chart fails)
    safe_name = symbol.replace("^","").replace("=","").replace("/","-")
    img_path = os.path.join(CHART_DIR, f"{market}_{safe_name}.png")
    try:
        make_chart(symbol, candles, img_path)
        have_img = True
    except Exception:
        logging.exception("Chart failed for %s", symbol)
        have_img = False
        img_path = None

    # 4) Anti-spam signature
    new_signature = {
        "short": sig.short, "long": sig.long,
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

    embed = build_embed(symbol, market, float(last_px) if last_px is not None else None, sig)
    content = f"**{symbol}** update — {time.strftime('%Y-%m-%d %H:%M:%S')}"

    if DISCORD_WEBHOOK_URL:
        ok = send_discord(DISCORD_WEBHOOK_URL, content, embed, img_path if have_img else None)
        if ok:
            logging.info("Posted %s", key)
            state[key] = new_signature
            save_state(state)
        else:
            logging.warning("Post failed for %s", key)
    else:
        logging.error("No DISCORD_WEBHOOK_URL set.")

def run_once(symbols_stocks: List[str], symbols_crypto: List[str]) -> None:
    state = load_state()
    for sym in symbols_stocks:
        process_symbol("stock", sym, state)
        time.sleep(1)
    for sym in symbols_crypto:
        process_symbol("crypto", sym, state)
        time.sleep(1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true")
    args = p.parse_args()

    if not WATCHLIST_STOCKS and not WATCHLIST_CRYPTO:
        logging.error("Watchlists empty. Set WATCHLIST_STOCKS / WATCHLIST_CRYPTO.")
        return

    logging.info("Starting — stocks=%s | crypto=%s | interval=%s | mode=%s",
                 WATCHLIST_STOCKS, WATCHLIST_CRYPTO, INTERVAL, POST_MODE)

    if args.once:
        run_once(WATCHLIST_STOCKS, WATCHLIST_CRYPTO); return

    while True:
        t0 = time.time()
        try:
            run_once(WATCHLIST_STOCKS, WATCHLIST_CRYPTO)
        except Exception:
            logging.exception("Run failed (continuing)")
        elapsed = int(time.time() - t0)
        slp = max(1, POLL_SECONDS - elapsed)
        logging.info("Sleeping %ss", slp)
        time.sleep(slp)

__all__ = ["run_once"]

if __name__ == "__main__":
    main()
