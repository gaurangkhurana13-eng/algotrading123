"""
TRI-ENGINE AUTONOMOUS LIVE TELEGRAM FORWARD-TESTING SYSTEM
==========================================================
Engine 1: Bhavishya V43 Titan (15-Minute Institutional Engine, 30 Certified Stocks)
          Exact parity with strategies/V43/run_v43.py (closed 15m bars only).
Engine 2: Predator V3 Bull Squad (1-Minute Long Sniper, 5 Certified High-Beta Stocks)
          Exact parity with strategies/V3/v3_long_bull_squad.pine (closed 1m bars only).
Engine 3: Predator V3 Bear Squad (1-Minute Short Sniper, 3 Certified Distribution Stocks)
          Exact parity with strategies/V3/v3_short_bear_squad.pine (closed 1m bars only).

KEY ARCHITECTURE GUARANTEES:
- Real-time zero-delay NSE data via yfinance batch downloads normalized to Asia/Kolkata (+05:30).
- Closed-candle-only signal evaluation (eliminates intrabar wick fakeouts & duplicate re-entries).
- Holiday / stale-data guard (rejects any bar not matching today's IST session date).
- Strict compartmentalized engine states (zero trigger or capital bleed across engines).
- Clean automatic EOD shutdown at 15:25 IST with daily scorecard summary.
"""

import os
import sys
import time
import json
import datetime
from pathlib import Path
from typing import Dict, Optional, List, Set, Tuple
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import numpy as np
import yfinance as yf

# Force UTF-8 and unbuffered stdout
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

from strategies.framework import calc_dhan_charges, calculate_atr

# ==============================================================================
# TIMEZONE HELPER (STRICT IST = Asia/Kolkata)
# ==============================================================================
IST = ZoneInfo("Asia/Kolkata")


def get_ist_now() -> datetime.datetime:
    """Returns current timezone-aware datetime in Indian Standard Time (IST)."""
    return datetime.datetime.now(IST)


# ==============================================================================
# TELEGRAM CONFIGURATION & ALERTS
# ==============================================================================
CONFIG_FILE = ROOT / "config" / "telegram_config.json"


def load_saved_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


_saved_cfg = load_saved_config()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", _saved_cfg.get("bot_token", ""))
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", str(_saved_cfg.get("chat_id", "")))


def send_telegram_alert(message: str, bot_token: str = None, chat_id: str = None, dry_run: bool = False) -> bool:
    """Sends a formatted HTML alert directly to Telegram (or prints in dry_run)."""
    if dry_run:
        print("[DRY-RUN TELEGRAM ALERT]:\n" + message, flush=True)
        return True

    token = bot_token or TELEGRAM_BOT_TOKEN
    c_id = chat_id or TELEGRAM_CHAT_ID

    if not token or not c_id:
        print(f"[TELEGRAM WARNING] Missing credentials! Token: {bool(token)}, Chat ID: {bool(c_id)}", flush=True)
        print("[TELEGRAM ALERT (OFFLINE DUMP)]:\n" + message, flush=True)
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": c_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            print(f"[TELEGRAM] Alert sent successfully to chat {c_id}", flush=True)
            return True
        else:
            print(f"[TELEGRAM ERROR] API returned status {resp.status_code}: {resp.text}", flush=True)
            return False
    except Exception as e:
        print(f"[ERROR] Failed to send Telegram alert: {e}", flush=True)
        return False


# ==============================================================================
# STOCK UNIVERSES & SECTOR MAP (Canonical V43 & V3 Universes)
# ==============================================================================
V43_30_UNIVERSE = [
    "CENTRALBK", "DELTACORP", "ELECTCAST", "GSFC", "IDEA",
    "IFCI", "IOB", "IRCON", "J&KBANK", "JYOTHYLAB",
    "MANALIPETC", "MTNL", "NBCC", "NCC", "NIACL",
    "NMDC", "PCJEWELLER", "PNB", "RBLBANK", "RCF",
    "RENUKA", "RITES", "RPOWER", "SUZLON", "SWSOLAR",
    "TATASTEEL", "TEXRAIL", "TTML", "UTKARSHBNK", "ZEEL",
]

V3_BULL_UNIVERSE = [
    "PCJEWELLER", "TTML", "SOUTHBANK", "WEBELSOLAR", "SJVN",
]

V3_BEAR_UNIVERSE = [
    "NMDC", "IDFCFIRSTB", "SJVN",
]

SECTOR_MAP_103 = {
    "FIN": {
        "PNB", "CENTRALBK", "IOB", "J&KBANK", "UTKARSHBNK", "RBLBANK", "CANBK", "BANKBARODA",
        "BANKINDIA", "SBIN", "ICICIBANK", "AXISBANK", "FEDERALBNK", "IDFCFIRSTB", "UNIONBANK",
        "YESBANK", "EQUITASBNK", "UCOBANK", "UJJIVANSFB", "MAHABANK", "INDIANB", "IDBI",
        "SOUTHBANK", "BANDHANBNK", "IFCI", "PFC", "RECLTD", "POONAWALLA", "MANAPPURAM",
        "IRFC", "HUDCO", "NIACL",
    },
    "INFRA": {
        "IRCON", "RITES", "TEXRAIL", "RVNL", "NBCC", "NCC", "HCC", "GMRINFRA", "GMRAIRPORT",
        "JAICORPLTD", "LLOYDSENGG", "DLF", "BEL", "MOTHERSON",
    },
    "COMMODITIES": {
        "TATASTEEL", "NMDC", "ELECTCAST", "VEDL", "HINDALCO", "JINDALSTEL", "JSWSTEEL", "SAIL",
        "NATIONALUM", "RCF", "GSFC", "MANALIPETC", "TATACHEM", "NFL", "SPIC", "BCLIND",
        "RENUKA", "PATANJALI", "TRIDENT", "ALOKINDS", "MMTC",
    },
    "ENERGY": {
        "SUZLON", "RPOWER", "SWSOLAR", "TATAPOWER", "ADANIPOWER", "NTPC", "NHPC", "SJVN",
        "JPPOWER", "PTC", "IOC", "BPCL", "GAIL", "COALINDIA", "ONGC", "MRPL", "RELIANCE",
        "IDEA", "MTNL", "TTML",
    },
    "CONSUMER": {
        "PCJEWELLER", "JYOTHYLAB", "ITC", "ZEEL", "DELTACORP", "NETWORK18", "HATHWAY",
    },
    "TECH": {"INFY", "TECHM", "SUBEXLTD"},
    "PHARMA": {"SUNPHARMA", "CIPLA", "MOREPENLAB"},
    "AUTO": {"TATAMOTORS", "ASHOKLEY"},
    "CONGLOMERATE": {"ADANIENT"},
}

SYM_TO_SECTOR_103: Dict[str, str] = {}
for _sec_name, _syms in SECTOR_MAP_103.items():
    for _s in _syms:
        SYM_TO_SECTOR_103[_s] = _sec_name


# ==============================================================================
# MESSAGE FORMATTERS
# ==============================================================================
def format_v43_entry(symbol: str, price: float, sl: float, qty: int, capital: float, charges: float, pattern: str, atr: float, bar_ts: datetime.datetime) -> str:
    trade_val = round(price * qty, 2)
    be_price = price + 0.45 * atr
    return (
        f"🏛️ <b>[BHAVISHYA V43 TITAN 15M — ENTRY]</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 <b>Stock:</b> <code>{symbol}</code> (Core 30 Universe)\n"
        f"⚡ <b>Action:</b> BUY (Institutional MIS 5x)\n"
        f"💵 <b>Entry Price:</b> ₹{price:.2f}\n"
        f"🛑 <b>Hard SL (-0.90 ATR):</b> ₹{sl:.2f}\n"
        f"🛡️ <b>Breakeven Lock (+0.45 ATR):</b> ₹{be_price:.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💼 <b>Wallet Capital:</b> ₹{capital:.2f}\n"
        f"🪙 <b>5x Trade Value:</b> ₹{trade_val:.2f} ({qty} shares)\n"
        f"🧾 <b>Est. Dhan Charges:</b> ₹{charges:.2f}\n"
        f"🎯 <b>Alpha Trigger:</b> <code>{pattern}</code>\n"
        f"🕯️ <b>Closed 15m Bar:</b> {bar_ts.strftime('%H:%M IST')}\n"
        f"⏰ <b>Alert Time:</b> {get_ist_now().strftime('%H:%M:%S IST')}\n"
        f"💡 <i>Asymmetric Ratchet: 0.45 BE, 0.85 Lock, 1.25/2.25/3.50 Runner</i>"
    )


def format_v43_exit(symbol: str, entry_p: float, exit_p: float, qty: int, reason: str, charges: float, wallet_after: float) -> str:
    gross = round((exit_p - entry_p) * qty, 2)
    net = round(gross - charges, 2)
    icon = "💰" if net > 0 else "🛑"
    return (
        f"{icon} <b>[BHAVISHYA V43 TITAN — EXIT COMPLETED]</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 <b>Stock:</b> <code>{symbol}</code> (Core Universe)\n"
        f"📢 <b>Exit Reason:</b> <b>{reason}</b>\n"
        f"💵 <b>Exit Fill:</b> ₹{exit_p:.2f} (Entry: ₹{entry_p:.2f})\n"
        f"🔢 <b>Shares:</b> {qty}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Gross PnL:</b> ₹{gross:+.2f}\n"
        f"🧾 <b>Dhan Statutory Fees:</b> ₹{charges:.2f}\n"
        f"💎 <b>Net PnL:</b> <b>₹{net:+.2f}</b>\n"
        f"💼 <b>Updated Wallet:</b> ₹{wallet_after:.2f}\n"
        f"⏰ <b>Exit Time:</b> {get_ist_now().strftime('%H:%M:%S IST')}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


def format_v3_bull_entry(symbol: str, price: float, sl: float, tp: float, qty: int, capital: float, charges: float, entry_num: int, bar_ts: datetime.datetime) -> str:
    trade_val = round(price * qty, 2)
    return (
        f"🟢 <b>[PREDATOR V3 BULL SQUAD 1M — ENTRY #{entry_num}]</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 <b>Stock:</b> <code>{symbol}</code> (Bull Squad)\n"
        f"⚡ <b>Action:</b> BUY (Sniper MIS 5x)\n"
        f"💵 <b>Entry Price:</b> ₹{price:.2f}\n"
        f"🛑 <b>Hard SL (-1.2%):</b> ₹{sl:.2f}\n"
        f"🎯 <b>Target (+2.52% / 2.1R):</b> ₹{tp:.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💼 <b>Wallet Capital:</b> ₹{capital:.2f}\n"
        f"🪙 <b>5x Trade Value:</b> ₹{trade_val:.2f} ({qty} shares)\n"
        f"🧾 <b>Est. Dhan Charges:</b> ₹{charges:.2f}\n"
        f"🕯️ <b>Closed 1m Bar:</b> {bar_ts.strftime('%H:%M IST')}\n"
        f"⏰ <b>Alert Time:</b> {get_ist_now().strftime('%H:%M:%S IST')}\n"
        f"🏹 <i>Rule: 30m ORB High Breakout + Above VWAP + Daily 20 EMA</i>\n"
        f"🔒 <i>Trailing Stop: +1.2% Gain locks 0.6% Trail</i>"
    )


def format_v3_bull_exit(symbol: str, entry_p: float, exit_p: float, qty: int, reason: str, charges: float, wallet_after: float) -> str:
    gross = round((exit_p - entry_p) * qty, 2)
    net = round(gross - charges, 2)
    icon = "💰" if net > 0 else "🛑"
    return (
        f"{icon} <b>[PREDATOR V3 BULL SQUAD — EXIT COMPLETED]</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 <b>Stock:</b> <code>{symbol}</code> (Bull Squad)\n"
        f"📢 <b>Exit Reason:</b> <b>{reason}</b>\n"
        f"💵 <b>Exit Fill:</b> ₹{exit_p:.2f} (Entry: ₹{entry_p:.2f})\n"
        f"🔢 <b>Shares:</b> {qty}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Gross PnL:</b> ₹{gross:+.2f}\n"
        f"🧾 <b>Dhan Statutory Fees:</b> ₹{charges:.2f}\n"
        f"💎 <b>Net PnL:</b> <b>₹{net:+.2f}</b>\n"
        f"💼 <b>Updated Wallet:</b> ₹{wallet_after:.2f}\n"
        f"⏰ <b>Exit Time:</b> {get_ist_now().strftime('%H:%M:%S IST')}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


def format_v3_bear_entry(symbol: str, price: float, sl: float, tp: float, qty: int, capital: float, charges: float, entry_num: int, bar_ts: datetime.datetime) -> str:
    trade_val = round(price * qty, 2)
    return (
        f"🔴 <b>[PREDATOR V3 BEAR SQUAD 1M — SHORT #{entry_num}]</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 <b>Stock:</b> <code>{symbol}</code> (Bear Squad)\n"
        f"⚡ <b>Action:</b> SHORT SELL (Dhan 5x MIS Sniper)\n"
        f"💵 <b>Short Entry:</b> ₹{price:.2f}\n"
        f"🛑 <b>Hard SL (+1.2% Above):</b> ₹{sl:.2f}\n"
        f"🎯 <b>Target Drop (-2.52% / 2.1R):</b> ₹{tp:.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💼 <b>Wallet Capital:</b> ₹{capital:.2f}\n"
        f"🪙 <b>5x Trade Value:</b> ₹{trade_val:.2f} ({qty} shares)\n"
        f"🧾 <b>Est. Dhan Charges:</b> ₹{charges:.2f}\n"
        f"🕯️ <b>Closed 1m Bar:</b> {bar_ts.strftime('%H:%M IST')}\n"
        f"⏰ <b>Alert Time:</b> {get_ist_now().strftime('%H:%M:%S IST')}\n"
        f"🏹 <i>Rule: 30m ORB Low Breakdown + Below VWAP + Daily 20 EMA</i>\n"
        f"🔒 <i>Trailing Stop: -1.2% Drop locks 0.6% Trail</i>"
    )


def format_v3_bear_exit(symbol: str, entry_p: float, exit_p: float, qty: int, reason: str, charges: float, wallet_after: float) -> str:
    gross = round((entry_p - exit_p) * qty, 2)
    net = round(gross - charges, 2)
    icon = "💰" if net > 0 else "🛑"
    return (
        f"{icon} <b>[PREDATOR V3 BEAR SQUAD — COVER COMPLETED]</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 <b>Stock:</b> <code>{symbol}</code> (Bear Squad)\n"
        f"📢 <b>Cover Reason:</b> <b>{reason}</b>\n"
        f"💵 <b>Cover Fill:</b> ₹{exit_p:.2f} (Short: ₹{entry_p:.2f})\n"
        f"🔢 <b>Shares:</b> {qty}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 <b>Gross PnL:</b> ₹{gross:+.2f}\n"
        f"🧾 <b>Dhan Statutory Fees:</b> ₹{charges:.2f}\n"
        f"💎 <b>Net PnL:</b> <b>₹{net:+.2f}</b>\n"
        f"💼 <b>Updated Wallet:</b> ₹{wallet_after:.2f}\n"
        f"⏰ <b>Exit Time:</b> {get_ist_now().strftime('%H:%M:%S IST')}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


# ==============================================================================
# REAL-TIME YFINANCE BATCH DATA FETCHER (IST + CLOSED-BAR GUARDS)
# ==============================================================================
def _normalize_yf_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalizes columns to lowercase OHLCV and index to Asia/Kolkata."""
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()
    df = raw_df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    needed = ["open", "high", "low", "close", "volume"]
    if not all(col in df.columns for col in needed):
        return pd.DataFrame()
    df = df[needed].dropna(subset=["open", "high", "low", "close"])
    if df.empty:
        return pd.DataFrame()
    if df.index.tz is None:
        df.index = df.index.tz_localize(IST)
    else:
        df.index = df.index.tz_convert(IST)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def fetch_yfinance_batch(symbols: List[str], period: str, interval: str, max_retries: int = 3) -> Dict[str, pd.DataFrame]:
    """Downloads real-time NSE bars in a single vectorized batch call."""
    if not symbols:
        return {}
    tickers = [f"{s}.NS" for s in symbols]
    for attempt in range(max_retries):
        try:
            raw = yf.download(
                tickers=tickers,
                period=period,
                interval=interval,
                group_by="ticker",
                progress=False,
                threads=True,
                auto_adjust=False,
            )
            if raw is None or raw.empty:
                time.sleep(1.0)
                continue

            out: Dict[str, pd.DataFrame] = {}
            if len(symbols) == 1 and not isinstance(raw.columns, pd.MultiIndex):
                norm = _normalize_yf_df(raw)
                if not norm.empty:
                    out[symbols[0]] = norm
                return out

            for s in symbols:
                t = f"{s}.NS"
                try:
                    if t in raw.columns.get_level_values(0):
                        norm = _normalize_yf_df(raw[t])
                        if not norm.empty:
                            out[s] = norm
                except Exception:
                    pass
            if out:
                return out
        except Exception as e:
            print(f"[DATA WARN] yfinance batch ({interval}) attempt {attempt + 1} failed: {e}", flush=True)
            time.sleep(1.5)
    return {}


def filter_closed_bars(df: pd.DataFrame, bar_minutes: int, now_ist: datetime.datetime) -> pd.DataFrame:
    """Strictly removes currently forming (unclosed) bars to prevent lookahead/wick entries."""
    if df is None or df.empty:
        return pd.DataFrame()
    delta = datetime.timedelta(minutes=bar_minutes)
    return df[df.index + delta <= now_ist]


# ==============================================================================
# ENGINE 1: BHAVISHYA V43 TITAN (EXACT PARITY WITH strategies/V43/run_v43.py)
# ==============================================================================
def compute_vwap_shifted(df: pd.DataFrame) -> pd.Series:
    df_temp = df.copy()
    df_temp["_date"] = df_temp.index.date
    tp = (df_temp["high"] + df_temp["low"] + df_temp["close"]) / 3.0
    df_temp["_tpv"] = tp * df_temp["volume"]
    cum_tpv = df_temp.groupby("_date")["_tpv"].cumsum()
    cum_vol = df_temp.groupby("_date")["volume"].cumsum()
    return (cum_tpv / (cum_vol + 1e-9)).shift(1)


def compute_er10(series_close: pd.Series) -> pd.Series:
    change = (series_close - series_close.shift(10)).abs()
    volatility = series_close.diff().abs().rolling(10).sum()
    return change / (volatility + 1e-9)


def compute_adx_di(df: pd.DataFrame, n: int = 10):
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff()
    dn = -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
    atr_s = tr.rolling(n).mean()
    pdi = 100 * pd.Series(pdm, index=df.index).rolling(n).mean() / (atr_s + 1e-9)
    mdi = 100 * pd.Series(mdm, index=df.index).rolling(n).mean() / (atr_s + 1e-9)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    adx = dx.rolling(n).mean()
    return adx, pdi, mdi


def build_v43_data_map(
    pool: Dict[str, pd.DataFrame],
    include_ma20_reclaim: bool = True,
) -> Tuple[dict, pd.Series, pd.Series, pd.Series, List[str], np.ndarray]:
    """Computes exact V43 indicators and cross-universe benchmarks identically to run_v43.py."""
    data_map = {}
    for sym, df_in in pool.items():
        if sym not in V43_30_UNIVERSE or df_in is None or len(df_in) < 150:
            continue
        df_full = df_in.copy()
        c, h, l, v, o = df_full["close"], df_full["high"], df_full["low"], df_full["volume"], df_full["open"]
        atr14 = calculate_atr(df_full, 14)
        atr_pct = (atr14 / c) * 100.0

        ema9 = c.ewm(span=9).mean()
        ema20 = c.ewm(span=20).mean()
        ema50 = c.ewm(span=50).mean()
        ema110 = c.ewm(span=110).mean()

        delta = c.diff()
        gain = delta.clip(lower=0).rolling(7).mean()
        loss = (-delta.clip(upper=0)).rolling(7).mean()
        rsi7 = 100.0 - (100.0 / (1.0 + gain / (loss + 1e-9)))

        macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
        sig = macd.ewm(span=9).mean()
        macd_cross_up = (macd > sig) & (macd.shift(1) <= sig.shift(1))

        adx, pdi, mdi = compute_adx_di(df_full, 10)
        vwap_lag = compute_vwap_shifted(df_full)
        v_sma20 = v.rolling(20).mean()
        rvol = v / (v_sma20 + 1e-9)
        er10 = compute_er10(c)

        high_5 = h.rolling(5).max().shift(1)
        lookback = min(300, len(c) - 1)
        c_300 = c.shift(lookback)
        mom_cond = (c > c_300)

        df_full["_date"] = df_full.index.date
        day_open = df_full.groupby("_date")["open"].transform("first")
        daily_close = df_full.groupby("_date")["close"].last()
        prev_close_map = daily_close.shift(1).to_dict()
        df_full["prev_close"] = df_full["_date"].map(prev_close_map)
        df_full["open_gap_pct"] = (day_open - df_full["prev_close"]) / (df_full["prev_close"] + 1e-9) * 100.0
        df_full["day_ret"] = (df_full["close"] / day_open) - 1.0
        df_full["1h_ret"] = (df_full["close"] / df_full["close"].shift(4)) - 1.0

        univ_cond = (
            (c > ema50) & (c > ema110) &
            (c > vwap_lag) &
            (atr_pct >= 0.50) &
            mom_cond &
            (c <= 450.0) & (c >= 10.0)
        )

        # Trigger 1: Core MACD Cross
        t1_base = (
            macd_cross_up & (macd > 0) &
            (c > ema20) &
            (rsi7 > 50.0) & (adx >= 18.0) & (pdi > mdi) &
            (er10 >= 0.35) &
            (rvol >= 1.30)
        )
        t1_tqs = (t1_base & (rvol >= 1.35) & (c > high_5))

        # Trigger 2: EMA Pullback
        sup_prev = np.maximum(ema9.shift(1), vwap_lag.shift(1)) + (0.15 * atr14)
        t2_pullback = (
            (macd > sig) & (macd > 0) & (pdi > mdi) &
            (ema9 > ema20) & (ema20 > ema50) &
            (c > ema9) &
            (l.shift(1) <= sup_prev) &
            (v.shift(1) < v.shift(2)) & (v >= 1.15 * v.shift(1)) & (rvol >= 1.30) &
            (c > h.shift(1)) & (c.shift(1) <= h.shift(2)) & (c > o) &
            (rsi7 >= 52.0) & (rsi7 <= 74.0) &
            (((c - ema20) / (atr14 + 1e-9)) <= 2.10)
        )

        # Trigger 3: MA20 Reclaim
        trig_reclaim_ma20_15m = (
            (l.shift(1) < ema20.shift(1)) &
            (c.shift(1) <= ema20.shift(1)) &
            (c > ema20) &
            (rvol >= 1.30) &
            (rsi7 > 50.0) & (macd > sig) & (macd > 0) &
            (c > vwap_lag)
        )

        data_map[sym] = {
            "df": df_full, "c": c, "h": h, "l": l, "v": v, "o": o, "atr14": atr14,
            "ema9": ema9, "ema20": ema20, "rsi7": rsi7, "adx": adx, "vwap_lag": vwap_lag,
            "rvol": rvol, "er10": er10,
            "univ_cond": univ_cond, "t1_base": t1_base, "t1_tqs": t1_tqs, "t2_pullback": t2_pullback,
            "trig_reclaim_ma20": trig_reclaim_ma20_15m,
            "day_ret": df_full["day_ret"], "1h_ret": df_full["1h_ret"],
            "c_300": c_300, "day_open": day_open, "open_gap_pct": df_full["open_gap_pct"],
        }

    if not data_map:
        empty_s = pd.Series(dtype=float)
        return {}, empty_s, empty_s, empty_s, [], np.empty((0, 0), dtype=bool)

    bench_day = pd.concat([d["day_ret"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
    bench_1h = pd.concat([d["1h_ret"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
    c300_breadth = pd.concat([(d["c"] > d["c_300"]).astype(float) for d in data_map.values()], axis=1, sort=False).mean(axis=1)

    for sym, d in data_map.items():
        b_d = bench_day.reindex(d["day_ret"].index)
        b_1 = bench_1h.reindex(d["1h_ret"].index)
        d["g1_cond"] = (d["day_ret"] > b_d)
        d["g2_cond"] = (d["1h_ret"] > b_1)
        d["excess_1h"] = d["1h_ret"] - b_1

    all_ts = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))
    ticker_list = sorted(list(data_map.keys()))
    ticker_to_idx = {sym: i for i, sym in enumerate(ticker_list)}
    n_stocks = len(ticker_list)

    bullish_matrix = np.zeros((len(all_ts), n_stocks), dtype=bool)
    for i, s in enumerate(ticker_list):
        s_c = data_map[s]["c"].reindex(all_ts).ffill().to_numpy()
        s_vwap = data_map[s]["vwap_lag"].reindex(all_ts).ffill().to_numpy()
        s_ex = data_map[s]["excess_1h"].reindex(all_ts).fillna(0.0).to_numpy()
        bullish_matrix[:, i] = (s_c > s_vwap) & (s_ex > 0)

    return data_map, bench_day, bench_1h, c300_breadth, ticker_list, bullish_matrix


def create_v43_state(initial_equity: float = 500.0) -> dict:
    return {
        "initial_equity": initial_equity,
        "hwm": initial_equity,
        "positions": {},
        "trades": [],
        "daily_traded": {},
        "daily_losses": {},
        "symbol_lockouts": {},
        "last_exit_times": set(),
    }


def step_v43_bar(
    ts: pd.Timestamp,
    state: dict,
    data_map: dict,
    bench_day: pd.Series,
    c300_breadth: pd.Series,
    ticker_list: List[str],
    bullish_matrix: np.ndarray,
    all_ts_index: Dict[pd.Timestamp, int],
    allow_entries: bool = True,
    send_alerts: bool = False,
    dry_run: bool = False,
    include_ma20_reclaim: bool = True,
    base_risk: float = 25.0,
    max_slots: int = 1,
    slot_unlock: float = 1000.0,
):
    """Executes one closed 15m bar step identically to simulate_v43_era in run_v43.py."""
    if ts.weekday() >= 5 or not data_map:
        return
    cur_date = ts.date()
    state["daily_traded"].setdefault(cur_date, set())
    state["daily_losses"].setdefault(cur_date, 0)
    t_time = ts.time()

    b_c300 = float(c300_breadth.get(ts, 0.50))
    b_day_now = float(bench_day.get(ts, 0.0))
    is_bear_env = (b_c300 < 0.40) or (b_day_now < -0.0060)
    is_severe_bleed_day = (b_day_now < -0.0120)

    slippage = 0.0005
    hard_sl_mult = 0.90

    # 1. Manage Active V43 Positions (Full 5-Tier Asymmetric Ratchet + Scratch + EOD)
    for ticker in list(state["positions"].keys()):
        p = state["positions"][ticker]
        t_data = data_map.get(ticker)
        if not t_data or ts not in t_data["df"].index:
            continue
        row = t_data["df"].loc[ts]
        cur_c = float(row["close"])
        cur_h = float(row["high"])
        cur_l = float(row["low"])
        cur_rsi = float(t_data["rsi7"].loc[ts])
        cur_vwap = float(t_data["vwap_lag"].loc[ts])
        cur_ema9 = float(t_data["ema9"].loc[ts])
        cur_ema20 = float(t_data["ema20"].loc[ts])
        cur_adx = float(t_data["adx"].loc[ts])
        cur_rvol = float(t_data["rvol"].loc[ts])

        p["bars_held"] += 1
        p["high_seen"] = max(p["high_seen"], cur_h)
        mfe_atr = (p["high_seen"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0

        shooting_star = (cur_rvol >= 2.60) and (((cur_h - cur_c) / (cur_h - cur_l + 1e-6)) >= 0.45)
        extreme_rsi = (cur_rsi >= 86.0)
        break_ema9 = (cur_c < p.get("prev_l", cur_l)) and (cur_c < cur_ema9)
        circuit_prox = (mfe_atr >= 8.00)
        psi_climax = shooting_star or extreme_rsi or break_ema9 or circuit_prox

        theta_expand = (cur_c > cur_ema9 > cur_ema20) and (cur_adx >= 24.0 or cur_rvol >= 1.80) and (cur_rsi >= 56.0) and (not psi_climax)
        prev_l = p.get("prev_l", cur_l)

        if mfe_atr >= 3.50:
            if psi_climax:
                stop_ladder = p["high_seen"] - (0.10 * p["entry_atr"])
            elif theta_expand:
                stop_ladder = max(p["entry_p"] + (2.00 * p["entry_atr"]), prev_l - (0.08 * p["entry_atr"]), p["high_seen"] - (0.35 * p["entry_atr"]))
            else:
                stop_ladder = p["high_seen"] - (0.18 * p["entry_atr"])
        elif mfe_atr >= 2.25:
            if psi_climax:
                stop_ladder = p["high_seen"] - (0.10 * p["entry_atr"])
            elif theta_expand:
                stop_ladder = max(p["entry_p"] + (1.20 * p["entry_atr"]), prev_l - (0.10 * p["entry_atr"]), p["high_seen"] - (0.35 * p["entry_atr"]))
            else:
                stop_ladder = p["high_seen"] - (0.20 * p["entry_atr"])
        elif mfe_atr >= 1.25:
            if theta_expand:
                stop_ladder = max(p["entry_p"] + (0.55 * p["entry_atr"]), prev_l - (0.12 * p["entry_atr"]), p["high_seen"] - (0.38 * p["entry_atr"]))
            else:
                stop_ladder = p["high_seen"] - (0.28 * p["entry_atr"])
        elif mfe_atr >= 0.85:
            stop_ladder = p["entry_p"] + (0.40 * p["entry_atr"])
        elif mfe_atr >= 0.45:
            stop_ladder = p["entry_p"] + (0.05 * p["entry_atr"])
        else:
            stop_ladder = p["entry_p"] - (0.90 * p["entry_atr"])

        if mfe_atr >= 0.85 and cur_c < cur_ema9 and cur_rsi < 55.0:
            stop_ladder += 0.05 * p["entry_atr"]

        p["active_sl"] = max(p["active_sl"], round(stop_ladder, 2))
        p["prev_l"] = cur_l

        def finalize_trade(exit_fill: float, exit_reason: str):
            tot_qty = p["qty"]
            gross = round((exit_fill - p["entry_p"]) * tot_qty, 2)
            chg = round(calc_dhan_charges(p["entry_p"], exit_fill, tot_qty, is_intraday=True), 2)
            net = round(gross - chg, 2)
            state["trades"].append({
                "symbol": ticker,
                "direction": "LONG",
                "entry_time": p["entry_t"].strftime("%Y-%m-%d %H:%M:%S"),
                "exit_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "entry_price": p["entry_p"],
                "exit_price": exit_fill,
                "shares": tot_qty,
                "sl": p["initial_sl"],
                "gross_pnl": gross,
                "charges": chg,
                "net_pnl": net,
                "exit_reason": exit_reason,
                "pattern": p.get("pattern", "UNKNOWN"),
            })
            state["last_exit_times"].add(ts)
            if net < 0:
                state["daily_losses"][cur_date] += 1
            if exit_reason == "STOP" and exit_fill <= p["entry_p"]:
                state["symbol_lockouts"][ticker] = cur_date + datetime.timedelta(days=2)
            del state["positions"][ticker]

            if send_alerts:
                ek_now = state["initial_equity"] + sum(t["net_pnl"] for t in state["trades"])
                reason_lbl = "RATCHET STOP" if exit_reason == "STOP" else ("SCRATCH PROTECT" if exit_reason == "SCRATCH" else "14:45 EOD SQUARE OFF")
                send_telegram_alert(format_v43_exit(ticker, p["entry_p"], exit_fill, tot_qty, reason_lbl, chg, ek_now), dry_run=dry_run)

        if cur_l <= p["active_sl"]:
            raw_fill = min(cur_h, p["active_sl"]) * (1.0 - slippage)
            exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
            finalize_trade(exit_fill, "STOP")
            continue
        if p["bars_held"] >= 4 and mfe_atr < 0.40 and cur_c < p["entry_p"] and (cur_rsi < 46.0 or cur_c < cur_vwap):
            raw_fill = cur_c * (1.0 - slippage)
            exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
            finalize_trade(exit_fill, "SCRATCH")
            continue
        if t_time >= datetime.time(14, 45):
            raw_fill = cur_c * (1.0 - slippage)
            exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
            finalize_trade(exit_fill, "EOD")
            continue

    if not allow_entries:
        return

    # 2. Equity, DD, Circuit Breakers & Window Gates
    ek = state["initial_equity"] + sum(t["net_pnl"] for t in state["trades"])
    state["hwm"] = max(state["hwm"], ek)
    dd_pct = (state["hwm"] - ek) / max(state["hwm"], state["initial_equity"])
    current_max_slots = 2 if (ek >= slot_unlock and max_slots >= 2) else 1

    if (
        len(state["positions"]) >= current_max_slots
        or ts in state["last_exit_times"]
        or t_time < datetime.time(9, 20)
        or t_time > datetime.time(14, 15)
    ):
        return
    if datetime.time(11, 30) <= t_time <= datetime.time(12, 30):
        return
    if datetime.time(10, 0) <= t_time <= datetime.time(10, 5):
        return
    if state["daily_losses"].get(cur_date, 0) >= 2:
        return

    avail_margin_total = (0.95 * ek) - sum((p["entry_p"] * p["qty"]) / 5.0 for p in state["positions"].values())
    if avail_margin_total < 10.0:
        return

    consec_losses = 0
    for t in reversed(state["trades"]):
        if t["net_pnl"] < 0:
            consec_losses += 1
        else:
            break

    if consec_losses >= 2 or (ek > 2500.0 and dd_pct > 0.16):
        alpha, beta = 0.015, 0.010
    elif consec_losses == 1:
        alpha, beta = 0.035, 0.020
    else:
        alpha, beta = 0.085, 0.038

    if ek <= 2500.0:
        raw_r = base_risk + (alpha * max(0.0, ek - 500.0))
    else:
        raw_r = base_risk + (alpha * 2000.0) + (beta * (ek - 2500.0))

    if current_max_slots == 1:
        gate_v2_cap = min(0.090 * ek, 258.0)
        if dd_pct > 0.15:
            gate_v2_cap *= 0.50
        target_risk_slot = min(raw_r, gate_v2_cap)
        margin_limit_slot = avail_margin_total
    else:
        gate_v2_cap_slot = min(0.045 * ek, 129.0)
        if dd_pct > 0.15:
            gate_v2_cap_slot *= 0.50
        target_risk_slot = min(raw_r, gate_v2_cap_slot)
        margin_limit_slot = min(avail_margin_total, (0.95 * ek) * 0.50)

    # 3. Cross-Universe Sector Clustering
    ts_idx = all_ts_index.get(ts)
    ticker_to_idx = {sym: i for i, sym in enumerate(ticker_list)}
    n_stocks = len(ticker_list)
    k_cluster: Dict[str, float] = {}
    if ts_idx is not None and n_stocks > 0:
        t_bullish = bullish_matrix[ts_idx, :]
        sum_bullish = int(np.sum(t_bullish))
        for clean in data_map.keys():
            sec = SYM_TO_SECTOR_103.get(clean)
            tick_i = ticker_to_idx[clean]
            if sec and sec in SECTOR_MAP_103:
                p_idxs = [ticker_to_idx[p_sym] for p_sym in SECTOR_MAP_103[sec] if p_sym != clean and p_sym in ticker_to_idx]
                k_cluster[clean] = float(np.mean(t_bullish[p_idxs])) if p_idxs else 0.0
            else:
                k_cluster[clean] = float(sum_bullish - t_bullish[tick_i]) / max(1, n_stocks - 1)

    # 4. Scan Candidates
    t2_start_t = datetime.time(9, 45)
    t2_end_t = datetime.time(13, 15)
    candidates = []

    for ticker, t_data in data_map.items():
        if ticker in state["positions"] or ticker in state["daily_traded"][cur_date]:
            continue
        if ticker in state["symbol_lockouts"] and cur_date <= state["symbol_lockouts"][ticker]:
            continue
        if ts not in t_data["df"].index:
            continue
        idx = t_data["df"].index.get_loc(ts)
        if idx < 100:
            continue
        if not (bool(t_data["univ_cond"].iloc[idx]) and bool(t_data["g1_cond"].iloc[idx]) and bool(t_data["g2_cond"].iloc[idx])):
            continue

        gap_pct = t_data["open_gap_pct"].iloc[idx]
        if not np.isnan(gap_pct) and gap_pct > 1.50:
            continue

        if is_bear_env:
            k_cl = k_cluster.get(ticker, 0.0)
            decoupling = ((float(t_data["c"].iloc[idx]) - float(t_data["day_open"].iloc[idx])) / (float(t_data["day_open"].iloc[idx]) + 1e-9)) - b_day_now
            auth_pass = (k_cl >= 0.25) or (decoupling >= 0.0120 and float(t_data["rvol"].iloc[idx]) >= 1.50)
            if not auth_pass:
                continue

        pattern_matched = None
        if is_severe_bleed_day:
            if bool(t_data["t1_tqs"].iloc[idx]):
                pattern_matched = "V42_TQS"
        else:
            if bool(t_data["t1_base"].iloc[idx]):
                pattern_matched = "V42_MACD"
            elif bool(t_data["t2_pullback"].iloc[idx]) and (b_day_now >= -0.0040) and (t2_start_t <= t_time <= t2_end_t):
                pattern_matched = "V42_PULLBACK"
            elif include_ma20_reclaim and bool(t_data["trig_reclaim_ma20"].iloc[idx]) and (b_day_now >= -0.0020):
                pattern_matched = "MA20_RECLAIM"

        if pattern_matched is None:
            continue

        z_mom = float(t_data["excess_1h"].iloc[idx])
        z_vol = min(3.0, float(t_data["rvol"].iloc[idx])) / 3.0
        z_er = min(1.0, float(t_data["er10"].iloc[idx]))
        bar_h = float(t_data["h"].iloc[idx])
        bar_l = float(t_data["l"].iloc[idx])
        bar_c = float(t_data["c"].iloc[idx])
        z_spread = (bar_c - bar_l) / (bar_h - bar_l + 1e-6)
        base_score = (0.45 * z_mom) + (0.25 * z_vol) + (0.15 * z_er) + (0.15 * z_spread)
        final_alpha = base_score * (1.0 + 0.35 * k_cluster.get(ticker, 0.0))

        candidates.append((
            ticker, bar_c, bar_h, bar_l,
            float(t_data["v"].iloc[idx]),
            max(float(t_data["atr14"].iloc[idx]), 0.05),
            final_alpha,
            pattern_matched,
        ))

    if not candidates:
        return

    candidates.sort(key=lambda x: x[6], reverse=True)
    open_sectors = {SYM_TO_SECTOR_103.get(p, p) for p in state["positions"].keys()}

    for cand in candidates:
        if len(state["positions"]) >= current_max_slots:
            break
        ticker, cur_c, cur_h, cur_l, cur_v, entry_atr, _, cand_pattern = cand
        cand_sector = SYM_TO_SECTOR_103.get(ticker, ticker)
        if cand_sector in open_sectors:
            continue

        cur_avail_margin = (0.95 * ek) - sum((p["entry_p"] * p["qty"]) / 5.0 for p in state["positions"].values())
        if cur_avail_margin < 10.0:
            break

        margin_for_trade = min(cur_avail_margin, margin_limit_slot)
        avail_notional = margin_for_trade * 5.0

        raw_entry = cur_c * (1.0 + slippage)
        entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
        active_sl = round(entry_p - (hard_sl_mult * entry_atr), 2)
        sl_dist = max(entry_p - active_sl, 0.05)
        friction = 0.002067 * entry_p
        denom = sl_dist + friction

        qty_r = int(target_risk_slot / denom)
        qty_m = int((avail_notional * 0.90) / entry_p)
        qty_v = int(cur_v * 0.05)
        qty = min(qty_r, qty_m, qty_v)
        if qty < 1:
            continue

        state["positions"][ticker] = {
            "entry_t": ts,
            "entry_p": entry_p,
            "initial_sl": active_sl,
            "active_sl": active_sl,
            "entry_atr": entry_atr,
            "qty": qty,
            "high_seen": cur_h,
            "prev_l": cur_l,
            "bars_held": 0,
            "sector": cand_sector,
            "pattern": cand_pattern,
        }
        state["daily_traded"][cur_date].add(ticker)
        open_sectors.add(cand_sector)

        if send_alerts:
            est_chg = round(calc_dhan_charges(entry_p, entry_p + entry_atr, qty, is_intraday=True), 2)
            send_telegram_alert(format_v43_entry(ticker, entry_p, active_sl, qty, ek, est_chg, cand_pattern, entry_atr, ts), dry_run=dry_run)


# ==============================================================================
# ENGINES 2 & 3: PREDATOR V3 BULL & BEAR (EXACT PINE SCRIPT PARITY ON CLOSED 1M)
# ==============================================================================
def create_v3_state(initial_equity: float = 500.0) -> dict:
    return {
        "wallet": initial_equity,
        "positions": {},
        "trades_today": {},
        "trades": [],
    }


def step_v3_symbol_bar(
    side: str,  # "BULL" or "BEAR"
    sym: str,
    df_1m: pd.DataFrame,
    bar_idx: int,
    daily_ema: float,
    orb_high: float,
    orb_low: float,
    vwap_val: float,
    vol_sma20: float,
    state: dict,
    allow_entries: bool = True,
    send_alerts: bool = False,
    dry_run: bool = False,
):
    """Processes one closed 1-minute bar for Predator V3 Bull or Bear."""
    ts = df_1m.index[bar_idx]
    cur_date = ts.date()
    bar_time = ts.time()
    c_now = float(df_1m["close"].iloc[bar_idx])
    h_now = float(df_1m["high"].iloc[bar_idx])
    l_now = float(df_1m["low"].iloc[bar_idx])
    v_now = float(df_1m["volume"].iloc[bar_idx])
    c_prev = float(df_1m["close"].iloc[bar_idx - 1]) if bar_idx > 0 else c_now

    state["trades_today"].setdefault(cur_date, {})
    trades_count = state["trades_today"][cur_date].get(sym, 0)

    # 1. Position Management (isolated per engine)
    if sym in state["positions"]:
        pos = state["positions"][sym]

        if side == "BULL":
            pos["high_seen"] = max(pos["high_seen"], h_now)
            gain_pct = (pos["high_seen"] - pos["entry_p"]) / pos["entry_p"]
            if gain_pct >= 0.012:
                new_trail = pos["high_seen"] - (pos["entry_p"] * 0.006)
                pos["sl"] = max(pos["sl"], round(new_trail, 2))

            exit_fill, reason = None, None
            if bar_time >= datetime.time(15, 15):
                exit_fill, reason = c_now, "15:15 EOD SQUARE OFF"
            elif l_now <= pos["sl"]:
                exit_fill = pos["sl"]
                reason = "TRAIL STOP LOCK" if pos["sl"] > pos["entry_p"] else "HARD STOP LOSS (-1.2%)"
            elif h_now >= pos["tp"]:
                exit_fill, reason = pos["tp"], "TARGET HIT (+2.52% / 2.1R)"

            if exit_fill is not None:
                chg = round(calc_dhan_charges(pos["entry_p"], exit_fill, pos["qty"], is_intraday=True), 2)
                gross = round((exit_fill - pos["entry_p"]) * pos["qty"], 2)
                net = round(gross - chg, 2)
                state["wallet"] = round(state["wallet"] + net, 2)
                state["trades"].append({"symbol": sym, "side": side, "entry_time": str(pos["entry_t"]), "exit_time": str(ts), "entry": pos["entry_p"], "exit": exit_fill, "qty": pos["qty"], "gross": gross, "chg": chg, "net": net, "reason": reason})
                del state["positions"][sym]
                if send_alerts:
                    send_telegram_alert(format_v3_bull_exit(sym, pos["entry_p"], exit_fill, pos["qty"], reason, chg, state["wallet"]), dry_run=dry_run)
                return

        else:  # BEAR
            pos["low_seen"] = min(pos["low_seen"], l_now)
            drop_pct = (pos["entry_p"] - pos["low_seen"]) / pos["entry_p"]
            if drop_pct >= 0.012:
                new_trail = pos["low_seen"] + (pos["entry_p"] * 0.006)
                pos["sl"] = min(pos["sl"], round(new_trail, 2))

            exit_fill, reason = None, None
            if bar_time >= datetime.time(15, 15):
                exit_fill, reason = c_now, "15:15 EOD SHORT COVER"
            elif h_now >= pos["sl"]:
                exit_fill = pos["sl"]
                reason = "TRAIL COVER LOCK" if pos["sl"] < pos["entry_p"] else "HARD STOP LOSS (+1.2%)"
            elif l_now <= pos["tp"]:
                exit_fill, reason = pos["tp"], "SHORT TARGET HIT (-2.52% / 2.1R)"

            if exit_fill is not None:
                chg = round(calc_dhan_charges(exit_fill, pos["entry_p"], pos["qty"], is_intraday=True), 2)
                gross = round((pos["entry_p"] - exit_fill) * pos["qty"], 2)
                net = round(gross - chg, 2)
                state["wallet"] = round(state["wallet"] + net, 2)
                state["trades"].append({"symbol": sym, "side": side, "entry_time": str(pos["entry_t"]), "exit_time": str(ts), "entry": pos["entry_p"], "exit": exit_fill, "qty": pos["qty"], "gross": gross, "chg": chg, "net": net, "reason": reason})
                del state["positions"][sym]
                if send_alerts:
                    send_telegram_alert(format_v3_bear_exit(sym, pos["entry_p"], exit_fill, pos["qty"], reason, chg, state["wallet"]), dry_run=dry_run)
                return

    # 2. Entries: strictly 09:45 <= bar_time < 14:45, closed bar, position_size == 0, < 2 shots/day
    if not allow_entries or sym in state["positions"] or trades_count >= 2:
        return
    if not (datetime.time(9, 45) <= bar_time < datetime.time(14, 45)):
        return
    if np.isnan(vol_sma20) or vol_sma20 <= 0 or daily_ema <= 0:
        return

    vol_spike = v_now > (vol_sma20 * 1.05)

    if side == "BULL":
        bull_filter = (c_now > vwap_val) and (c_now > daily_ema) and vol_spike
        trig_e1 = (trades_count == 0) and (c_prev <= orb_high) and (c_now > orb_high) and bull_filter
        trig_e2 = False
        if trades_count == 1 and bar_idx >= 15:
            same_day_slice = df_1m.iloc[max(0, bar_idx - 15):bar_idx]
            highest_15m = float(same_day_slice["high"].max())
            trig_e2 = (c_prev <= highest_15m) and (c_now > highest_15m) and bull_filter

        if trig_e1 or trig_e2:
            trades_count += 1
            state["trades_today"][cur_date][sym] = trades_count
            qty = max(1, int((state["wallet"] * 5.0) / c_now))
            sl_dist = c_now * 0.012
            sl_p = round(c_now - sl_dist, 2)
            tp_p = round(c_now + (sl_dist * 2.1), 2)
            state["positions"][sym] = {"entry_p": c_now, "sl": sl_p, "tp": tp_p, "qty": qty, "high_seen": h_now, "entry_t": ts}
            if send_alerts:
                chg = round(calc_dhan_charges(c_now, tp_p, qty, is_intraday=True), 2)
                send_telegram_alert(format_v3_bull_entry(sym, c_now, sl_p, tp_p, qty, state["wallet"], chg, trades_count, ts), dry_run=dry_run)

    else:  # BEAR
        bear_filter = (c_now < vwap_val) and (c_now < daily_ema) and vol_spike
        trig_e1 = (trades_count == 0) and (c_prev >= orb_low) and (c_now < orb_low) and bear_filter
        trig_e2 = False
        if trades_count == 1 and bar_idx >= 15:
            same_day_slice = df_1m.iloc[max(0, bar_idx - 15):bar_idx]
            lowest_15m = float(same_day_slice["low"].min())
            trig_e2 = (c_prev >= lowest_15m) and (c_now < lowest_15m) and bear_filter

        if trig_e1 or trig_e2:
            trades_count += 1
            state["trades_today"][cur_date][sym] = trades_count
            qty = max(1, int((state["wallet"] * 5.0) / c_now))
            sl_dist = c_now * 0.012
            sl_p = round(c_now + sl_dist, 2)
            tp_p = round(c_now - (sl_dist * 2.1), 2)
            state["positions"][sym] = {"entry_p": c_now, "sl": sl_p, "tp": tp_p, "qty": qty, "low_seen": l_now, "entry_t": ts}
            if send_alerts:
                chg = round(calc_dhan_charges(tp_p, c_now, qty, is_intraday=True), 2)
                send_telegram_alert(format_v3_bear_entry(sym, c_now, sl_p, tp_p, qty, state["wallet"], chg, trades_count, ts), dry_run=dry_run)


# ==============================================================================
# MAIN LIVE SCANNER LOOP
# ==============================================================================
def run_tri_engine_scanner(initial_wallet: float = 500.0, dry_run: bool = False):
    now_ist = get_ist_now()
    boot_ts = now_ist
    print("\n=======================================================", flush=True)
    print("  TRI-ENGINE AUTONOMOUS CLOUD SCANNER ACTIVATED (V2 IST)", flush=True)
    print(f"  Current Time (IST): {now_ist.strftime('%Y-%m-%d %H:%M:%S IST')}", flush=True)
    print(f"  Bot Token Injected: {'YES' if TELEGRAM_BOT_TOKEN else 'NO'}", flush=True)
    print(f"  Chat ID Injected:   {'YES (' + str(TELEGRAM_CHAT_ID) + ')' if TELEGRAM_CHAT_ID else 'NO'}", flush=True)
    print("  Feed: yfinance Real-Time Batch (Asia/Kolkata, Closed Bars Only)", flush=True)
    print(f"  Engine 1: Bhavishya V43 Titan (15M, {len(V43_30_UNIVERSE)} Stocks, Exact run_v43 Parity)", flush=True)
    print(f"  Engine 2: Predator V3 Bull Squad (1M, {len(V3_BULL_UNIVERSE)} Stocks)", flush=True)
    print(f"  Engine 3: Predator V3 Bear Squad (1M, {len(V3_BEAR_UNIVERSE)} Stocks)", flush=True)
    print("=======================================================\n", flush=True)

    send_telegram_alert(
        "🔱 <b>TRI-ENGINE LIVE QUANT SCANNER ONLINE!</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Platform: <b>GitHub Actions Cloud Runner (IST Fixed)</b>\n"
        f"Start Time: <b>{now_ist.strftime('%d-%b-%Y %H:%M:%S IST')}</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "1️⃣ 🏛️ <b>Bhavishya V43 Titan (15M):</b> 30 Stocks | Closed Bars Only\n"
        "2️⃣ 🟢 <b>Predator V3 Bull Squad (1M):</b> 5 Stocks | 30m ORB High\n"
        "3️⃣ 🔴 <b>Predator V3 Bear Squad (1M):</b> 3 Stocks | 30m ORB Low\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Strict Isolation | Zero-Delay Feed | Holiday & Closed-Bar Guard Active.",
        dry_run=dry_run,
    )

    state_v43 = create_v43_state(initial_equity=initial_wallet)
    state_v3_bull = create_v3_state(initial_equity=initial_wallet)
    state_v3_bear = create_v3_state(initial_equity=initial_wallet)

    daily_ema_cache: Dict[str, float] = {}
    orb_cache: Dict[str, Dict[str, float]] = {}
    last_cache_date: Optional[datetime.date] = None

    last_v43_bar_ts: Optional[pd.Timestamp] = None
    last_v3_bull_bar_ts: Dict[str, pd.Timestamp] = {}
    last_v3_bear_bar_ts: Dict[str, pd.Timestamp] = {}

    # Avoid firing stale backlog signals if manually started mid-day after 09:50 IST
    backfill_cutoff: Optional[datetime.datetime] = None
    if boot_ts.time() > datetime.time(9, 50):
        backfill_cutoff = boot_ts - datetime.timedelta(minutes=2)

    all_v3_syms = sorted(set(V3_BULL_UNIVERSE + V3_BEAR_UNIVERSE))
    scan_iteration = 0

    while True:
        scan_iteration += 1
        now = get_ist_now()
        cur_time = now.time()
        cur_date = now.date()

        # Reset daily caches cleanly on new calendar day
        if last_cache_date != cur_date:
            daily_ema_cache.clear()
            orb_cache.clear()
            last_v3_bull_bar_ts.clear()
            last_v3_bear_bar_ts.clear()
            last_cache_date = cur_date

        # Pre-market wait (08:30 - 09:15 IST)
        if cur_time < datetime.time(9, 15):
            if scan_iteration % 5 == 1:
                print(f"[{now.strftime('%H:%M:%S IST')}] Pre-market wait before 09:15 IST...", flush=True)
            time.sleep(30)
            continue

        # Clean EOD shutdown at 15:25 IST (only after open positions closed)
        if cur_time >= datetime.time(15, 25):
            total_trades = len(state_v43["trades"]) + len(state_v3_bull["trades"]) + len(state_v3_bear["trades"])
            v43_net = round(sum(t["net_pnl"] for t in state_v43["trades"]), 2)
            v3b_net = round(sum(t["net"] for t in state_v3_bull["trades"]), 2)
            v3s_net = round(sum(t["net"] for t in state_v3_bear["trades"]), 2)
            print(f"[{now.strftime('%H:%M:%S IST')}] Session complete (15:25 IST). Trades: {total_trades}. Shutting down cleanly.", flush=True)
            send_telegram_alert(
                "📊 <b>TRI-ENGINE EOD SESSION SUMMARY</b>\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"📅 Date: <b>{cur_date.strftime('%d-%b-%Y')}</b>\n"
                f"🏛️ V43 Titan Net: <b>₹{v43_net:+.2f}</b> ({len(state_v43['trades'])} trades)\n"
                f"🟢 V3 Bull Net: <b>₹{v3b_net:+.2f}</b> ({len(state_v3_bull['trades'])} trades)\n"
                f"🔴 V3 Bear Net: <b>₹{v3s_net:+.2f}</b> ({len(state_v3_bear['trades'])} trades)\n"
                "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                "Scanner offline cleanly until next NSE session.",
                dry_run=dry_run,
            )
            break

        # Cache yesterday's completed Daily 20 EMA once per day
        if len(daily_ema_cache) < len(all_v3_syms):
            daily_pool = fetch_yfinance_batch(all_v3_syms, period="6mo", interval="1d")
            for s in all_v3_syms:
                if s not in daily_ema_cache and s in daily_pool:
                    dfd = daily_pool[s]
                    completed = dfd[dfd.index.date < cur_date]
                    if len(completed) >= 20:
                        daily_ema_cache[s] = float(completed["close"].ewm(span=20, adjust=False).mean().iloc[-1])

        loop_start = time.time()

        # ======================================================================
        # ENGINE 1: BHAVISHYA V43 TITAN (15-Minute Closed-Bar Batch Scan)
        # ======================================================================
        try:
            pool_15m = fetch_yfinance_batch(V43_30_UNIVERSE, period="60d", interval="15m")
            closed_pool_15m = {}
            today_closed_ts: Set[pd.Timestamp] = set()
            for sym, df15 in pool_15m.items():
                cdf = filter_closed_bars(df15, 15, now)
                if not cdf.empty:
                    closed_pool_15m[sym] = cdf
                    today_bars = cdf[cdf.index.date == cur_date]
                    today_closed_ts.update(today_bars.index.tolist())

            if today_closed_ts:
                latest_15m_ts = max(today_closed_ts)
                if last_v43_bar_ts is None or latest_15m_ts > last_v43_bar_ts:
                    data_map, bench_day, _, c300_breadth, ticker_list, bullish_matrix = build_v43_data_map(closed_pool_15m)
                    if data_map:
                        all_ts_list = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))
                        all_ts_index = {t: i for i, t in enumerate(all_ts_list)}

                        pending_ts = sorted(t for t in today_closed_ts if (last_v43_bar_ts is None or t > last_v43_bar_ts))
                        for bar_ts in pending_ts:
                            allow_entry = True
                            if backfill_cutoff is not None and (bar_ts + datetime.timedelta(minutes=15)) < backfill_cutoff:
                                allow_entry = False
                            step_v43_bar(
                                ts=bar_ts,
                                state=state_v43,
                                data_map=data_map,
                                bench_day=bench_day,
                                c300_breadth=c300_breadth,
                                ticker_list=ticker_list,
                                bullish_matrix=bullish_matrix,
                                all_ts_index=all_ts_index,
                                allow_entries=allow_entry,
                                send_alerts=True,
                                dry_run=dry_run,
                            )
                            last_v43_bar_ts = bar_ts
                        print(f"[{now.strftime('%H:%M:%S IST')}] V43 closed 15m eval @ {last_v43_bar_ts.strftime('%H:%M')} | Open Pos: {len(state_v43['positions'])}", flush=True)
        except Exception as e:
            print(f"[ERROR V43 Engine]: {e}", flush=True)

        # ======================================================================
        # ENGINES 2 & 3: PREDATOR V3 BULL & BEAR (1-Minute Closed-Bar Batch Scan)
        # ======================================================================
        try:
            pool_1m = fetch_yfinance_batch(all_v3_syms, period="5d", interval="1m")
            for sym, df1 in pool_1m.items():
                cdf1 = filter_closed_bars(df1, 1, now)
                if cdf1.empty:
                    continue
                today_1m = cdf1[cdf1.index.date == cur_date]
                if today_1m.empty:
                    continue

                # Build ORB strictly after 09:45 IST on exact 09:15..09:44 IST closed candles
                if sym not in orb_cache and cur_time >= datetime.time(9, 45):
                    orb_bars = today_1m[(today_1m.index.time >= datetime.time(9, 15)) & (today_1m.index.time < datetime.time(9, 45))]
                    if not orb_bars.empty:
                        orb_cache[sym] = {
                            "high": float(orb_bars["high"].max()),
                            "low": float(orb_bars["low"].min()),
                        }
                if sym not in orb_cache or cur_time < datetime.time(9, 45):
                    continue

                orb_h = orb_cache[sym]["high"]
                orb_l = orb_cache[sym]["low"]
                d_ema = daily_ema_cache.get(sym, 0.0)

                # Daily cumulative VWAP on today's closed 1m bars + 20 SMA volume across cdf1
                tp_day = (today_1m["high"] + today_1m["low"] + today_1m["close"]) / 3.0
                vwap_day = (tp_day * today_1m["volume"]).cumsum() / (today_1m["volume"].cumsum() + 1e-9)
                vol_sma20_full = cdf1["volume"].rolling(20).mean().reindex(today_1m.index)

                # Process Bull if eligible
                if sym in V3_BULL_UNIVERSE:
                    prev_ts = last_v3_bull_bar_ts.get(sym)
                    for i, ts in enumerate(today_1m.index):
                        if prev_ts is not None and ts <= prev_ts:
                            continue
                        allow_entry = True
                        if backfill_cutoff is not None and (ts + datetime.timedelta(minutes=1)) < backfill_cutoff:
                            allow_entry = False
                        step_v3_symbol_bar(
                            side="BULL",
                            sym=sym,
                            df_1m=today_1m,
                            bar_idx=i,
                            daily_ema=d_ema,
                            orb_high=orb_h,
                            orb_low=orb_l,
                            vwap_val=float(vwap_day.iloc[i]),
                            vol_sma20=float(vol_sma20_full.iloc[i]),
                            state=state_v3_bull,
                            allow_entries=allow_entry,
                            send_alerts=True,
                            dry_run=dry_run,
                        )
                        last_v3_bull_bar_ts[sym] = ts

                # Process Bear if eligible
                if sym in V3_BEAR_UNIVERSE:
                    prev_ts = last_v3_bear_bar_ts.get(sym)
                    for i, ts in enumerate(today_1m.index):
                        if prev_ts is not None and ts <= prev_ts:
                            continue
                        allow_entry = True
                        if backfill_cutoff is not None and (ts + datetime.timedelta(minutes=1)) < backfill_cutoff:
                            allow_entry = False
                        step_v3_symbol_bar(
                            side="BEAR",
                            sym=sym,
                            df_1m=today_1m,
                            bar_idx=i,
                            daily_ema=d_ema,
                            orb_high=orb_h,
                            orb_low=orb_l,
                            vwap_val=float(vwap_day.iloc[i]),
                            vol_sma20=float(vol_sma20_full.iloc[i]),
                            state=state_v3_bear,
                            allow_entries=allow_entry,
                            send_alerts=True,
                            dry_run=dry_run,
                        )
                        last_v3_bear_bar_ts[sym] = ts

        except Exception as e:
            print(f"[ERROR V3 Engines]: {e}", flush=True)

        if scan_iteration % 3 == 1:
            print(
                f"[{now.strftime('%H:%M:%S IST')}] Cycle #{scan_iteration} | Open -> V43: {len(state_v43['positions'])}, "
                f"V3 Bull: {len(state_v3_bull['positions'])}, V3 Bear: {len(state_v3_bear['positions'])}",
                flush=True,
            )

        elapsed = time.time() - loop_start
        time.sleep(max(5.0, 25.0 - elapsed))


# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tri-Engine Autonomous Live Telegram Forward-Tester")
    parser.add_argument("--test", action="store_true", help="Send test alert to verify Telegram connection")
    parser.add_argument("--live", action="store_true", help="Start continuous live market scanner")
    parser.add_argument("--dry-run", action="store_true", help="Run live loop without posting to Telegram")
    parser.add_argument("--token", type=str, default="", help="Telegram Bot Token")
    parser.add_argument("--chat_id", type=str, default="", help="Telegram Chat ID")
    args = parser.parse_args()

    token = args.token or TELEGRAM_BOT_TOKEN
    chat_id = args.chat_id or TELEGRAM_CHAT_ID

    if args.test:
        now_ist = get_ist_now()
        init_msg = (
            "🔱 <b>TRI-ENGINE QUANT DISPATCHER ONLINE!</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Status: <b>ONLINE & LIVE (Cloud Verified — IST & Closed-Bar Fixed)</b>\n"
            f"Server Time: <b>{now_ist.strftime('%d-%b-%Y %H:%M:%S IST')}</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"1️⃣ 🏛️ <b>Bhavishya V43 Titan (15M):</b> {len(V43_30_UNIVERSE)} Verified Core Stocks\n"
            f"2️⃣ 🟢 <b>Predator V3 Bull Squad (1M Long):</b> {len(V3_BULL_UNIVERSE)} High-Beta Stocks ({', '.join(V3_BULL_UNIVERSE)})\n"
            f"3️⃣ 🔴 <b>Predator V3 Bear Squad (1M Short):</b> {len(V3_BEAR_UNIVERSE)} Distribution Stocks ({', '.join(V3_BEAR_UNIVERSE)})\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Strict Isolation | Zero-Delay Feed | Closed-Bar Confirmation Active."
        )
        send_telegram_alert(init_msg, token, chat_id)
        print("✅ SUCCESS! Tri-Engine test alert sent.", flush=True)
    elif args.live:
        run_tri_engine_scanner(initial_wallet=500.0, dry_run=args.dry_run)
    else:
        print("Tri-Engine Live Forward-Testing System Ready.", flush=True)
        print("Run with --test or --live.", flush=True)
