#!/usr/bin/env python3
"""
Discord Watchlist Webhook Bot — Stocks + Crypto
Build: 2025-09-01-ALERTS-STYLED

Adds in this build:
- Holiday closure notices (US/GB by default; overridable per symbol)
- Weekend closure (configurable for indices/futures/FX)
- Primary & Secondary support ("best buy-in" levels) on embeds + charts
- Global alert channel that pings @everyone when price taps best buy-ins
- 🚨 Styled alert embed (emoji title, red color, concise fields, NFA footer)
- Alert cooldowns/anti-spam
- Keeps strict routing, per-symbol posting, charts, RSI/EMA/ATR, batching, Discord 429 handling

IMPORTANT: Educational info only. Not financial advice.
"""

import os, json, time, logging, argparse, random, requests
import numpy as np, pandas as pd
from datetime import datetime, timezone, date, timedelta
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

# matplotlib headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dotenv import load_dotenv
import holidays as pyholidays  # holiday calendars

# -------------------- Config --------------------
load_dotenv()

DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
POLL_SECONDS       = int(os.getenv("POLL_SECONDS", "60"))
POST_MODE          = os.getenv("POST_MODE", "changed").lower()  # changed|always
INTERVAL           = os.getenv("INTERVAL", "1m").lower()
STATE_FILE         = os.getenv("STATE_FILE", ".bot_state.json")

STRICT_ROUTING     = os.getenv("STRICT_ROUTING", "false").lower() == "true"
CLOSE_NON_EQUITY_ON_WEEKEND = os.getenv("CLOSE_NON_EQUITY_ON_WEEKEND", "true").lower() == "true"
HOLIDAY_CLOSED     = os.getenv("HOLIDAY_CLOSED", "true").lower() == "true"

WATCHLIST_STOCKS   = [s.strip() for s in os.getenv("WATCHLIST_STOCKS", "AAPL,MSFT,TSLA").split(",") if s.strip()]
WATCHLIST_CRYPTO   = [s.strip().upper() for s in os.getenv("WATCHLIST_CRYPTO", "ETHUSDT,BTCUSDT").split(",") if s.strip()]

RSI_PERIOD     = int(os.getenv("RSI_PERIOD", "14"))
RSI_OVERBOUGHT = float(os.getenv("RSI_OVERBOUGHT", "70"))
RSI_OVERSOLD   = float(os.getenv("RSI_OVERSOLD", "30"))
EMA_FAST       = int(os.getenv("EMA_FAST", "50"))
EMA_SLOW       = int(os.getenv("EMA_SLOW", "200"))
ATR_PERIOD     = int(os.getenv("ATR_PERIOD", "14"))
ATR_MULTIPLIER = float(os.getenv("ATR_MULTIPLIER", "2.0"))
LEVEL_TOL_PCT  = float(os.getenv("LEVEL_TOL_PCT", "0.004"))  # for deduping SR clusters

# Alerts
ALERT_ENABLED       = os.getenv("ALERT_ENABLED", "true").lower() == "true"
ALERT_WEBHOOK_URL   = os.getenv("ALERT_WEBHOOK_URL", "").strip()
ALERT_EVERYONE      = os.getenv("ALERT_EVERYONE", "true").lower() == "true"
ALERT_TOL_PCT       = float(os.getenv("ALERT_TOL_PCT", "0.001"))   # price within 0.1% of support
ALERT_COOLDOWN_MIN  = int(os.getenv("ALERT_COOLDOWN_MIN", "30"))   # minutes per (symbol, level)
ALERT_MARKETS       = os.getenv("ALERT_MARKETS", "stock").lower()  # stock|crypto|both

# Routing map: {"stock": {...}, "crypto": {...}}
try:
    SYMBOL_WEBHOOK_MAP: Dict[str, Dict[str, str]] = json.loads(os.getenv("SYMBOL_WEBHOOK_MAP", "{}"))
except Exception:
    SYMBOL_WEBHOOK_MAP = {}

# Optional per-symbol country overrides (e.g. {"RR.L":"GB"})
try:
    SYMBOL_COUNTRY_MAP: Dict[str, str] = json.loads(os.getenv("SYMBOL_COUNTRY_MAP", "{}"))
except Exception:
    SYMBOL_COUNTRY_MAP = {}

CHART_DIR = os.path.join(os.getcwd(), "charts")
os.makedirs(CHART_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logging.info("Bot version: 2025-09-01-ALERTS-STYLED")

# -------------------- Helpers --------------------
def as_float(x):
    if x is None: return None
    if isinstance(x, (np.ndarray, list, tuple)):
        if len(x) == 0: return None
        x = x[0]
    try:
        return float(getattr(x, "item", lambda: x)())
    except Exception:
        return float(x)

def safe_span(series: pd.Series, span: int) -> int:
    n = len(series)
    if n <= 2: return max(2, n)
    return min(max(2, span), n - 1)

def rand_sleep(lo=0.25, hi=0.8):
    time.sleep(random.uniform(lo, hi))

def resolve_webhook(market: str, symbol: str) -> Optional[str]:
    key = "crypto" if market == "crypto" else "stock"
    try:
        url = SYMBOL_WEBHOOK_MAP.get(key, {}).get(symbol)
        if url: return url.strip()
    except Exception:
        pass
    if STRICT_ROUTING:
        return None
    return DISCORD_WEBHOOK_URL or None

# --- Market-closed logic ---
def infer_country_code(symbol: str) -> Optional[str]:
    """
    Lightweight country inference for holidays:
    - ".L" → GB (London)
    - "JPY=X" → None (FX ignored)
    - else → US
    Overridable with SYMBOL_COUNTRY_MAP.
    """
    if symbol in SYMBOL_COUNTRY_MAP:
        return SYMBOL_COUNTRY_MAP[symbol].upper()
    if symbol.upper() == "JPY=X":
        return None
    if symbol.endswith(".L"):
        return "GB"
    return "US"

def is_weekend_closed(market: str, symbol: str) -> bool:
    if market == "crypto":
        return False
    wd = datetime.now(timezone.utc).weekday()  # 0=Mon ... 6=Sun
    if wd < 5:
        return False
    if CLOSE_NON_EQUITY_ON_WEEKEND:
        return True
    # legacy: only plain equities closed
    if symbol.startswith("^") or "=F" in symbol or symbol.upper() == "JPY=X":
        return False
    return True

def is_holiday_closed(market: str, symbol: str, now_utc: datetime) -> Optional[str]:
    if not HOLIDAY_CLOSED or market == "crypto":
        return None
    cc = infer_country_code(symbol)
    if not cc:
        return None
    d: date = now_utc.date()
    try:
        cal = pyholidays.country_holidays(cc, years=[d.year])
        name = cal.get(d)
        if name:
            if isinstance(name, (list, tuple)):
                name = ", ".join(map(str, name))
            return str(name)
    except Exception:
        return None
    return None

# -------------------- Providers --------------------
class BaseProvider:
    def get_recent_candles(self, symbol: str, interval: str = "1m", limit: int = 300) -> pd.DataFrame: ...
    def get_last_price(self, symbol: str) -> Optional[float]: ...

class YahooProvider(BaseProvider):
    def __init__(self):
        import yfinance as yf
        self.yf = yf
    def _dl(self, symbol: str, period: str, interval: str):
        df = self.yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        if df is None or df.empty: return pd.DataFrame()
        return df.rename(columns=str.lower)[["open","high","low","close","volume"]]
    def get_recent_candles(self, symbol: str, interval: str = "1m", limit: int = 300):
        df = self._dl(symbol, "1d", "1m")
        if df.empty or len(df) < 30:
            df = self._dl(symbol, "5d", "5m")
        if df.empty:
            return pd.DataFrame(columns=["open","high","low","close","volume"])
        return df.iloc[-limit:]
    def get_last_price(self, symbol: str) -> Optional[float]:
        try:
            t = self.yf.Ticker(symbol)
            px = t.fast_info.last_price
            return float(px) if px is not None else None
        except Exception:
            return None

class BinanceProvider(BaseProvider):
    BASE = "https://api.binance.com"
    def get_recent_candles(self, symbol: str, interval: str = "1m", limit: int = 500):
        r = requests.get(
            f"{self.BASE}/api/v3/klines",
            params={"symbol": symbol, "interval": interval, "limit": min(limit, 1000)},
            timeout=10
        )
        r.raise_for_status()
        data = r.json()
        cols = ["ot","open","high","low","close","volume","ct","q","t","tb","tbq","i"]
        df = pd.DataFrame(data, columns=cols)
        if df.empty:
            return pd.DataFrame(columns=["open","high","low","close","volume"])
        df["ot"] = pd.to_datetime(df["ot"], unit="ms")
        df = df.set_index("ot")
        for c in ["open","high","low","close","volume"]:
            df[c] = df[c].astype(float)
        return df[["open","high","low","close","volume"]]
    def get_all_prices(self, symbols: List[str]) -> Dict[str, float]:
        r = requests.get(f"{self.BASE}/api/v3/ticker/price", timeout=10)
        if r.status_code != 200:
            return {}
        out = {}
        try:
            for row in r.json():
                s = row.get("symbol")
                if s in symbols:
                    out[s] = float(row.get("price"))
        except Exception:
            pass
        return out
    def get_last_price(self, symbol: str) -> Optional[float]:
        r = requests.get(f"{self.BASE}/api/v3/ticker/price", params={"symbol":symbol}, timeout=10)
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
    delta = close.diff()
    up    = delta.clip(lower=0)
    down  = -delta.clip(upper=0)
    ma_up   = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h,l,c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def find_sr_levels(df: pd.DataFrame, lookback=200, window=3, tol_pct=0.004) -> Levels:
    if df.empty: return Levels([], [])
    hi, lo = df["high"].values, df["low"].values
    highs, lows = [], []
    n = len(df)
    start = max(0, n - lookback)
    for i in range(start + window, n - window):
        if hi[i] == max(hi[i-window:i+window+1]): highs.append(hi[i])
        if lo[i] == min(lo[i-window:i+window+1]): lows.append(lo[i])
    def dedupe(vals: List[float], tol: float) -> List[float]:
        vals = sorted(vals); out = []
        for v in vals:
            v = as_float(v)
            if not out or abs(v - out[-1]) > tol:
                out.append(v)
        return out
    px = as_float(df["close"].iloc[-1])
    tol = max(px * tol_pct, 1e-9)
    return Levels(dedupe(lows, tol), dedupe(highs, tol))

def nearest_level(levels: List[float], price: float, below: bool) -> Optional[float]:
    if not levels: return None
    if below:
        cands = [as_float(lv) for lv in levels if as_float(lv) <= price]
        val = max(cands) if cands else None
    else:
        cands = [as_float(lv) for lv in levels if as_float(lv) >= price]
        val = min(cands) if cands else None
    return as_float(val) if val is not None else None

def primary_secondary_support(levels: List[float], price: float) -> Tuple[Optional[float], Optional[float]]:
    """
    Primary: nearest support <= price
    Secondary: next-lower distinct support below primary (if any)
    """
    if not levels:
        return None, None
    below = sorted([as_float(x) for x in levels if as_float(x) <= price], reverse=True)
    if not below:
        return None, None
    primary = below[0]
    secondary = below[1] if len(below) > 1 else None
    return primary, secondary

# -------------------- Signals --------------------
@dataclass
class Signals:
    short: str
    long: str
    optimal_buy: Optional[float]
    stop_suggestion: Optional[float]
    take_profit: Optional[float]
    rsi_value: Optional[float]

def make_signals(df: pd.DataFrame)->Signals:
    if df.empty or len(df)<max(EMA_SLOW+2,RSI_PERIOD+2,ATR_PERIOD+2):
        return Signals("NO DATA","NO DATA",None,None,None,None)
    close=df["close"]; ema_fast=ema(close,EMA_FAST); ema_slow=ema(close,EMA_SLOW)
    rsi_series=rsi(close,RSI_PERIOD); atr_series=atr(df,ATR_PERIOD)
    lvls=find_sr_levels(df,tol_pct=LEVEL_TOL_PCT)
    last=as_float(close.iloc[-1]); last_rsi=as_float(rsi_series.iloc[-1]); last_atr=as_float(atr_series.iloc[-1])
    uptrend=bool(as_float(ema_fast.iloc[-1])>as_float(ema_slow.iloc[-1]))
    s=nearest_level(lvls.support,last,True); r=nearest_level(lvls.resistance,last,False)
    optimal_buy=s if s is not None else as_float(ema_slow.iloc[-1])
    short_sig="HOLD"
    if s is not None and (last-s)<=0.5*last_atr and last_rsi<=RSI_OVERSOLD+5: short_sig="BUY"
    elif r is not None and (r-last)<=0.5*last_atr and last_rsi>=RSI_OVERBOUGHT-5: short_sig="SELL"
    long_sig="ACCUMULATE" if uptrend else "AVOID/REDUCE"
    stop=last-ATR_MULTIPLIER*last_atr if uptrend else last+ATR_MULTIPLIER*last_atr
    return Signals(short_sig,long_sig,as_float(optimal_buy),as_float(stop),as_float(r),last_rsi)

# -------------------- Charting --------------------
def make_chart(symbol:str,df:pd.DataFrame,path:str,primary:Optional[float]=None,secondary:Optional[float]=None)->None:
    if df.empty or len(df)<20:
        fig=plt.figure(figsize=(8,3)); plt.title(f"{symbol} — no intraday data")
        plt.savefig(path,bbox_inches='tight',dpi=110); plt.close(fig); return
    clos=df.iloc[-150:]["close"]; ema_f=ema(clos,EMA_FAST); ema_s=ema(clos,EMA_SLOW)
    fig=plt.figure(figsize=(9,4)); ax=plt.gca()
    ax.plot(clos.index,clos.values,label="Close"); ax.plot(ema_f.index,ema_f.values,label=f"EMA{EMA_FAST}")
    ax.plot(ema_s.index,ema_s.values,label=f"EMA{EMA_SLOW}")
    if primary is not None:
        p=as_float(primary); ax.axhline(p,ls='--',alpha=0.7,label=f"Primary S {p:.6g}")
    if secondary is not None:
        s=as_float(secondary); ax.axhline(s,ls='--',alpha=0.5,label=f"Secondary S {s:.6g}")
    ax.set_title(f"{symbol} — {len(clos)} bars ({INTERVAL})"); ax.legend(loc="best"); ax.grid(True,alpha=0.2)
    fig.tight_layout(); plt.savefig(path,bbox_inches='tight',dpi=110); plt.close(fig)

# -------------------- Discord --------------------
def send_discord(webhook:str,content:str,embed:Dict,image_path:Optional[str])->bool:
    payload={"content":content,"embeds":[dict(embed)],"username":"Watchlist Bot"}
    attempt,backoff=0,1.2
    while attempt<4:
        attempt+=1
        try:
            if image_path and os.path.exists(image_path):
                with open(image_path,"rb") as f:
                    files={'file':(os.path.basename(image_path),f,'image/png')}
                    payload["embeds"][0]["image"]={"url":f"attachment://{os.path.basename(image_path)}"}
                    r=requests.post(webhook,data={"payload_json":json.dumps(payload)},files=files,timeout=20)
            else:
                r=requests.post(webhook,json=payload,timeout=20)
            if 200<=r.status_code<300: return True
            if r.status_code==429:
                try: retry=float(r.json().get("retry_after",1.0))
                except Exception: retry=1.0
                time.sleep(min(10.0,retry*backoff)); backoff*=1.5; continue
            logging.warning("Discord response %s: %s", r.status_code, r.text[:200])
            return False
        except Exception as e:
            logging.warning("send_discord attempt %d failed: %s", attempt, e)
            time.sleep(backoff); backoff*=1.5
    return False

# -------------------- State --------------------
def load_state()->Dict:
    if not os.path.exists(STATE_FILE): return {}
    try:
        with open(STATE_FILE,"r") as f: return json.load(f)
    except Exception:
        return {}
def save_state(state:Dict)->None:
    try:
        with open(STATE_FILE,"w") as f: json.dump(state,f,indent=2)
    except Exception:
        logging.warning("Could not write state file.")

# -------------------- Messaging --------------------
def format_price(x:float)->str:
    if x is None: return "—"
    if x>=1000: return f"{x:,.2f}"
    if x>=1: return f"{x:.2f}"
    return f"{x:.6f}".rstrip('0').rstrip('.')

def build_embed(symbol:str,market:str,price:float,sig:Signals,
                primary:Optional[float]=None,secondary:Optional[float]=None)->Dict:
    fields=[
        {"name":"Last Price","value":f"`{format_price(price)}`","inline":True},
        {"name":"RSI","value":f"`{sig.rsi_value:.1f}`" if sig.rsi_value is not None else "`—`","inline":True},
        {"name":"Optimal Buy-in (EMA/Support)","value":f"`{format_price(sig.optimal_buy)}`","inline":True},
        {"name":"Take Profit (nearest R)","value":f"`{format_price(sig.take_profit)}`","inline":True},
        {"name":"ATR Stop Suggestion","value":f"`{format_price(sig.stop_suggestion)}`","inline":True},
        {"name":"Short-term","value":f"**{sig.short}**","inline":True},
        {"name":"Long-term","value":f"**{sig.long}**","inline":True},
    ]
    if primary is not None:
        fields.insert(2, {"name":"Primary Support (best)","value":f"`{format_price(primary)}`","inline":True})
    if secondary is not None:
        fields.insert(3, {"name":"Secondary Support","value":f"`{format_price(secondary)}`","inline":True})
    provider = 'Binance' if market=='crypto' else 'Yahoo Finance (1m→5m fallback)'
    return {
        "title": f"{symbol} ({'Crypto' if market=='crypto' else 'Stock/Index/FX/Future'})",
        "description": "Dynamic levels & signals recalculated every run. Not financial advice.",
        "fields": fields,
        "footer": {"text": f"Interval {INTERVAL} | Provider: {provider}"},
    }

# -------------------- Alerts (styled) --------------------
def within_buy_zone(price: float, level: float, last_atr: Optional[float]) -> bool:
    """
    Consider it a 'tap' if:
      - relative distance <= ALERT_TOL_PCT, OR
      - absolute distance <= 0.25 * ATR (if ATR available)
    """
    if level is None or price is None: return False
    rel = abs(price - level) / max(1e-9, level)
    if rel <= ALERT_TOL_PCT:
        return True
    if last_atr is not None and abs(price - level) <= 0.25 * last_atr:
        return True
    return False

def cooldown_ok(state: Dict, symbol: str, tag: str) -> bool:
    key = f"alert:{symbol}:{tag}"
    last_iso = state.get(key)
    if not last_iso:
        return True
    try:
        last = datetime.fromisoformat(last_iso)
    except Exception:
        return True
    return datetime.utcnow() - last >= timedelta(minutes=ALERT_COOLDOWN_MIN)

def mark_cooldown(state: Dict, symbol: str, tag: str) -> None:
    key = f"alert:{symbol}:{tag}"
    state[key] = datetime.utcnow().isoformat()

def send_buy_alert(symbol: str, market: str, price: float, level: float, level_tag: str,
                   primary: Optional[float], secondary: Optional[float],
                   atr_stop: Optional[float]) -> bool:
    """Styled alert embed for the #alerts channel."""
    if not ALERT_ENABLED or not ALERT_WEBHOOK_URL:
        return False
    if ALERT_MARKETS != "both" and ALERT_MARKETS != market:
        return False

    mention = "@everyone " if ALERT_EVERYONE else ""
    title = f"🚨 ALERT – {symbol} at Support!"
    desc = f"Price has tapped **{level_tag} support**."

    fields = [
        {"name": "Price", "value": f"`{format_price(price)}`", "inline": True},
        {"name": f"{level_tag} Support", "value": f"`{format_price(level)}`", "inline": True},
    ]
    if primary is not None and (level_tag != "Primary"):
        fields.append({"name":"Primary Support","value":f"`{format_price(primary)}`","inline":True})
    if secondary is not None and (level_tag != "Secondary"):
        fields.append({"name":"Secondary Support","value":f"`{format_price(secondary)}`","inline":True})
    if atr_stop is not None:
        fields.append({"name":"ATR Stop (suggestion)","value":f"`{format_price(atr_stop)}`","inline":True})

    embed = {
        "title": title,
        "description": desc,
        "color": 15158332,  # red accent
        "fields": fields,
        "footer": {"text": f"Market: {market} | Interval {INTERVAL} | ⚠️ Not financial advice"}
    }
    content = f"{mention}**{symbol}** best buy-in opportunity detected."

    return send_discord(ALERT_WEBHOOK_URL, content, embed, None)

# -------------------- Orchestration --------------------
def process_symbol(market:str,symbol:str,state:Dict,price_cache:Optional[Dict[str,float]]=None)->None:
    target=resolve_webhook(market,symbol)
    if not target:
        logging.info("STRICT_ROUTING: no webhook for %s:%s — skipping.", market, symbol)
        return

    now=datetime.now(timezone.utc)

    # Holiday-closed notice (stocks/indices/futures/FX), once per day
    holiday_name = is_holiday_closed(market, symbol, now)
    if holiday_name:
        today = now.strftime("%Y-%m-%d")
        hk = f"holiday:{market}:{symbol}:{today}"
        if state.get(hk) is not True:
            embed = {
                "title": f"{symbol} — Market Closed (Holiday)",
                "description": f"Today is **{holiday_name}**. This market is closed.",
                "fields": [],
                "footer": {"text": "Status: holiday closure"}
            }
            content = f"**{symbol}** update — {now.strftime('%Y-%m-%d %H:%M:%S')} UTC"
            if send_discord(target, content, embed, None):
                logging.info("Holiday notice posted for %s", symbol)
                state[hk] = True
                save_state(state)
        return

    # Weekend-closed notice, once per day
    if is_weekend_closed(market,symbol):
        today = now.strftime("%Y-%m-%d")
        closed_key=f"closed:{market}:{symbol}:{today}"
        if state.get(closed_key) is not True:
            embed={"title":f"{symbol} — Market Closed",
                   "description":"This market is currently closed for the weekend. Signals and charts will resume when it reopens.",
                   "fields":[],
                   "footer":{"text":"Status: weekend closure"}}
            content=f"**{symbol}** update — {now.strftime('%Y-%m-%d %H:%M:%S')} UTC"
            if send_discord(target,content,embed,None):
                logging.info("Weekend notice posted for %s", symbol)
                state[closed_key]=True; save_state(state)
        return

    provider=PROVIDERS["crypto" if market=="crypto" else "stocks"]
    # Fetch candles
    try:
        if market == "crypto":
            candles=provider.get_recent_candles(symbol, interval=INTERVAL if INTERVAL in {"1m","3m","5m","15m","30m"} else "1m")
        else:
            candles=provider.get_recent_candles(symbol, interval="1m")
    except Exception:
        logging.exception("Failed to fetch candles for %s %s", market, symbol)
        candles=pd.DataFrame(columns=["open","high","low","close","volume"])

    # Signals & price
    sig=make_signals(candles) if not candles.empty else Signals("NO DATA","NO DATA",None,None,None,None)
    last_px=None
    if market=="crypto" and price_cache:
        last_px=price_cache.get(symbol)
    if last_px is None:
        last_px=provider.get_last_price(symbol)
    if last_px is None and not candles.empty:
        last_px=as_float(candles["close"].iloc[-1])

    # Compute primary/secondary supports for embeds & alerts
    lvls = find_sr_levels(candles) if not candles.empty else Levels([],[])
    primary, secondary = (None, None)
    if lvls.support and last_px is not None:
        primary, secondary = primary_secondary_support(lvls.support, last_px)

    # ALERTS: fire regardless of signature change, with cooldown
    try:
        last_atr = as_float(atr(candles, ATR_PERIOD).iloc[-1]) if not candles.empty else None
    except Exception:
        last_atr = None

    if ALERT_ENABLED and last_px is not None:
        if primary is not None and within_buy_zone(last_px, primary, last_atr) and cooldown_ok(state, symbol, "primary"):
            if send_buy_alert(symbol, "crypto" if market=="crypto" else "stock",
                              last_px, primary, "Primary", primary, secondary, sig.stop_suggestion):
                mark_cooldown(state, symbol, "primary"); save_state(state)
        if secondary is not None and within_buy_zone(last_px, secondary, last_atr) and cooldown_ok(state, symbol, "secondary"):
            if send_buy_alert(symbol, "crypto" if market=="crypto" else "stock",
                              last_px, secondary, "Secondary", primary, secondary, sig.stop_suggestion):
                mark_cooldown(state, symbol, "secondary"); save_state(state)

    # Anti-spam signature BEFORE chart work (for per-symbol channel)
    new_signature={"short":sig.short,"long":sig.long,
                   "optimal":round(sig.optimal_buy or 0,8),
                   "stop":round(sig.stop_suggestion or 0,8),
                   "tp":round(sig.take_profit or 0,8),
                   "ps":round(primary or 0,8) if primary is not None else 0.0,
                   "ss":round(secondary or 0,8) if secondary is not None else 0.0}
    key=f"{market}:{symbol}"
    old_signature=state.get(key)
    should_post=(POST_MODE=="always") or (new_signature!=old_signature)
    if not should_post:
        logging.info("No change for %s — skipping post.", key)
        return

    # Build chart ONLY if we're posting
    safe_name=symbol.replace("^","").replace("=","").replace("/","-")
    img_path=os.path.join(CHART_DIR,f"{market}_{safe_name}.png")
    have_img=False
    try:
        make_chart(symbol,candles,img_path,primary,secondary); have_img=True
    except Exception:
        logging.exception("Chart failed for %s (posting text-only)", symbol)
        img_path=None

    # Send per-symbol update
    embed=build_embed(symbol,market,float(last_px) if last_px is not None else None,sig,primary,secondary)
    content=f"**{symbol}** update — {now.strftime('%Y-%m-%d %H:%M:%S')} UTC"
    ok=send_discord(target,content,embed,img_path if have_img else None)
    if ok:
        logging.info("Posted %s → %s", key, target[:60]+"...")
        state[key]=new_signature; save_state(state)
    else:
        logging.warning("Post failed for %s", key)

def run_once(symbols_stocks: List[str], symbols_crypto: List[str]) -> None:
    state=load_state()
    # Batch crypto prices
    crypto_price_cache={}
    try:
        if symbols_crypto:
            crypto_price_cache=PROVIDERS["crypto"].get_all_prices(symbols_crypto)
    except Exception:
        logging.exception("Fetching batch crypto prices failed")

    processed=0
    for sym in symbols_stocks:
        process_symbol("stock",sym,state,None)
        processed+=1; rand_sleep()
    for sym in symbols_crypto:
        process_symbol("crypto",sym,state,crypto_price_cache)
        processed+=1; rand_sleep()
    logging.info("Cycle complete: processed=%d", processed)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--once",action="store_true")
    args=p.parse_args()

    if not WATCHLIST_STOCKS and not WATCHLIST_CRYPTO:
        logging.error("Watchlists empty. Set WATCHLIST_STOCKS / WATCHLIST_CRYPTO."); return

    logging.info("Starting — stocks=%s | crypto=%s | interval=%s | mode=%s | strict_routing=%s | holidays=%s | alerts=%s (%s)",
                 WATCHLIST_STOCKS, WATCHLIST_CRYPTO, INTERVAL, POST_MODE, STRICT_ROUTING, HOLIDAY_CLOSED,
                 ALERT_ENABLED, ALERT_MARKETS)

    if args.once:
        run_once(WATCHLIST_STOCKS, WATCHLIST_CRYPTO); return

    while True:
        t0=time.time()
        try:
            run_once(WATCHLIST_STOCKS, WATCHLIST_CRYPTO)
        except Exception:
            logging.exception("Run failed (continuing)")
        elapsed=int(time.time()-t0)
        slp=max(1,POLL_SECONDS-elapsed)
        logging.info("Sleeping %ss", slp)
        time.sleep(slp)

__all__=["run_once"]

if __name__=="__main__":
    main()
