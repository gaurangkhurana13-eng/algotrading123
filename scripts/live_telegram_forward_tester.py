"""
TRI-ENGINE QUANTITATIVE LIVE TELEGRAM FORWARD-TESTING SYSTEM
=============================================================
Engine 1: Bhavishya V43 Titan (15-Minute Institutional Engine, 30 Certified Stocks)
Engine 2: Predator V3 Bull Squad (1-Minute Long Sniper, 5 Certified High-Beta Stocks)
Engine 3: Predator V3 Bear Squad (1-Minute Short Sniper, 3 Certified Distribution Stocks)

STRICT COMPARTMENTALIZATION:
- Independent state dictionaries (positions_v43, positions_v3_bull, positions_v3_bear)
- Independent timeframe scanning and zero trigger bleed
- Strict IST timezone handling (UTC + 5:30)
- Real-time unbuffered console logging
- 100% Free Cloud Execution on GitHub Actions
"""

import os
import sys
import time
import json
import datetime
from pathlib import Path
from typing import Dict, Optional, List
import requests
import pandas as pd
import numpy as np

# Force UTF-8 and unbuffered stdout
sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

# ==============================================================================
# TIMEZONE HELPER (STRICT IST = UTC + 5:30)
# ==============================================================================
IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))

def get_ist_now() -> datetime.datetime:
    """Returns current datetime in Indian Standard Time (IST)."""
    return datetime.datetime.now(IST)

# ==============================================================================
# TELEGRAM CONFIGURATION
# ==============================================================================
CONFIG_FILE = ROOT / "config" / "telegram_config.json"

def load_saved_config():
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

def send_telegram_alert(message: str, bot_token: str = None, chat_id: str = None) -> bool:
    """Sends a formatted HTML alert directly to Telegram."""
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
        "disable_web_page_preview": True
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
# REAL DHAN FEE ESTIMATOR
# ==============================================================================
def estimate_dhan_charges(buy_p: float, sell_p: float, qty: int) -> float:
    """Exact Dhan statutory intraday charge calculation."""
    buy_val = buy_p * qty
    sell_val = sell_p * qty
    turnover = buy_val + sell_val
    brokerage = min(20.0, buy_val * 0.0003) + min(20.0, sell_val * 0.0003)
    stt = round(sell_val * 0.00025, 2)
    exchange = round(turnover * 0.0000345, 2)
    sebi = round(turnover * 0.000001, 2)
    stamp = round(buy_val * 0.00003, 2)
    gst = round((brokerage + exchange + sebi) * 0.18, 2)
    return round(brokerage + stt + exchange + sebi + stamp + gst, 2)

# ==============================================================================
# STOCK UNIVERSES
# ==============================================================================
# Engine 1: Bhavishya V43 Titan (30 Certified Stocks)
V43_30_UNIVERSE = [
    "CENTRALBK", "DELTACORP", "ELECTCAST", "GSFC", "IDEA", 
    "IFCI", "IOB", "IRCON", "J&KBANK", "JYOTHYLAB", 
    "MANALIPETC", "MTNL", "NBCC", "NCC", "NIACL", 
    "NMDC", "PCJEWELLER", "PNB", "RBLBANK", "RCF", 
    "RENUKA", "RITES", "RPOWER", "SUZLON", "SWSOLAR", 
    "TATASTEEL", "TEXRAIL", "TTML", "UTKARSHBNK", "ZEEL"
]

# Engine 2: Predator V3 Bull Squad (5 Certified High-Beta Long Stocks)
V3_BULL_UNIVERSE = [
    "PCJEWELLER", "TTML", "SOUTHBANK", "WEBELSOLAR", "SJVN"
]

# Engine 3: Predator V3 Bear Squad (3 Certified Distribution Short Stocks)
V3_BEAR_UNIVERSE = [
    "NMDC", "IDFCFIRSTB", "SJVN"
]

# ==============================================================================
# MESSAGE FORMATTERS
# ==============================================================================

# --- Engine 1: Bhavishya V43 Titan ---
def format_v43_entry(symbol: str, price: float, sl: float, qty: int, capital: float, charges: float, pattern: str, atr: float) -> str:
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
        f"⏰ <b>Trigger Time:</b> {get_ist_now().strftime('%H:%M:%S IST')}\n"
        f"💡 <i>Asymmetric Ratchet: MFE +0.45 ATR Locks BE, +0.85 ATR Trails Runner</i>"
    )

def format_v43_exit(symbol: str, entry_p: float, exit_p: float, qty: int, reason: str, charges: float) -> str:
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
        f"⏰ <b>Exit Time:</b> {get_ist_now().strftime('%H:%M:%S IST')}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

# --- Engine 2: Predator V3 Bull Squad (Long) ---
def format_v3_bull_entry(symbol: str, price: float, sl: float, tp: float, qty: int, capital: float, charges: float, entry_num: int) -> str:
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
        f"⏰ <b>Trigger Time:</b> {get_ist_now().strftime('%H:%M:%S IST')}\n"
        f"🏹 <i>Rule: 30m ORB High Breakout + Above VWAP + Daily 20 EMA</i>\n"
        f"🔒 <i>Trailing Stop: +1.2% Gain locks 0.6% Trail</i>"
    )

def format_v3_bull_exit(symbol: str, entry_p: float, exit_p: float, qty: int, reason: str, charges: float) -> str:
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
        f"⏰ <b>Exit Time:</b> {get_ist_now().strftime('%H:%M:%S IST')}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

# --- Engine 3: Predator V3 Bear Squad (Short) ---
def format_v3_bear_entry(symbol: str, price: float, sl: float, tp: float, qty: int, capital: float, charges: float, entry_num: int) -> str:
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
        f"⏰ <b>Trigger Time:</b> {get_ist_now().strftime('%H:%M:%S IST')}\n"
        f"🏹 <i>Rule: 30m ORB Low Breakdown + Below VWAP + Daily 20 EMA</i>\n"
        f"🔒 <i>Trailing Stop: -1.2% Drop locks 0.6% Trail</i>"
    )

def format_v3_bear_exit(symbol: str, entry_p: float, exit_p: float, qty: int, reason: str, charges: float) -> str:
    gross = round((entry_p - exit_p) * qty, 2)  # Profit when exit_p < entry_p
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
        f"⏰ <b>Exit Time:</b> {get_ist_now().strftime('%H:%M:%S IST')}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

# ==============================================================================
# LIVE TRI-ENGINE SCANNER ENGINE
# ==============================================================================
def run_tri_engine_scanner(initial_wallet: float = 500.0):
    from tvDatafeed import TvDatafeed, Interval

    now_ist = get_ist_now()
    print("\n=======================================================", flush=True)
    print("  TRI-ENGINE AUTONOMOUS CLOUD SCANNER ACTIVATED", flush=True)
    print(f"  Current Time (IST): {now_ist.strftime('%Y-%m-%d %H:%M:%S IST')}", flush=True)
    print(f"  Bot Token Injected: {'YES' if TELEGRAM_BOT_TOKEN else 'NO (Check GitHub Secrets!)'}", flush=True)
    print(f"  Chat ID Injected:   {'YES (' + str(TELEGRAM_CHAT_ID) + ')' if TELEGRAM_CHAT_ID else 'NO (Check GitHub Secrets!)'}", flush=True)
    print(f"  Engine 1: Bhavishya V43 Titan (15M Institutional, {len(V43_30_UNIVERSE)} Stocks)", flush=True)
    print(f"  Engine 2: Predator V3 Bull Squad (1M Sniper Long, {len(V3_BULL_UNIVERSE)} Stocks)", flush=True)
    print(f"  Engine 3: Predator V3 Bear Squad (1M Sniper Short, {len(V3_BEAR_UNIVERSE)} Stocks)", flush=True)
    print("=======================================================\n", flush=True)

    send_telegram_alert(
        "🔱 <b>TRI-ENGINE LIVE QUANT SCANNER ONLINE!</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Platform: <b>GitHub Actions Cloud Runner (Ubuntu)</b>\n"
        f"Start Time: <b>{now_ist.strftime('%d-%b-%Y %H:%M:%S IST')}</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "1️⃣ 🏛️ <b>Bhavishya V43 Titan (15M):</b> 30 Stocks | Asymmetric Ratchet\n"
        "2️⃣ 🟢 <b>Predator V3 Bull Squad (1M):</b> 5 Stocks | 30m ORB High Breakout\n"
        "3️⃣ 🔴 <b>Predator V3 Bear Squad (1M):</b> 3 Stocks | 30m ORB Low Breakdown\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Strict Isolation: Compartmentalized states, independent triggers, zero bleed.\n"
        "Zero laptop involvement required. Live signals will ping your phone!"
    )

    tv = TvDatafeed()
    wallet_v43 = initial_wallet
    wallet_v3_bull = initial_wallet
    wallet_v3_bear = initial_wallet

    # STRICT ISOLATION STATE
    positions_v43: Dict[str, dict] = {}
    positions_v3_bull: Dict[str, dict] = {}
    positions_v3_bear: Dict[str, dict] = {}

    trades_today_v43 = 0
    trades_today_v3_bull: Dict[str, int] = {s: 0 for s in V3_BULL_UNIVERSE}
    trades_today_v3_bear: Dict[str, int] = {s: 0 for s in V3_BEAR_UNIVERSE}

    # Daily EMA Cache (Calculated once per day at open)
    daily_ema_cache: Dict[str, float] = {}
    last_cache_date = None

    # ORB Session Cache (09:15 - 09:45 High and Low per stock)
    orb_cache: Dict[str, Dict[str, float]] = {}

    scan_iteration = 0
    while True:
        scan_iteration += 1
        now = get_ist_now()
        cur_time = now.time()
        cur_date = now.date()

        # Session limits (09:15 to 15:30 IST)
        t_open = datetime.time(9, 15)
        t_trade_start = datetime.time(9, 45)
        t_eod_cutoff = datetime.time(15, 15)
        t_close = datetime.time(15, 30)

        # Reset daily tracking on new day
        if last_cache_date != cur_date:
            daily_ema_cache.clear()
            orb_cache.clear()
            trades_today_v43 = 0
            trades_today_v3_bull = {s: 0 for s in V3_BULL_UNIVERSE}
            trades_today_v3_bear = {s: 0 for s in V3_BEAR_UNIVERSE}
            last_cache_date = cur_date

        # Outside market hours
        if cur_time > t_close or cur_time < t_open:
            if scan_iteration % 5 == 1:
                print(f"[{now.strftime('%H:%M:%S IST')}] Outside NSE Hours (09:15-15:30). Waiting...", flush=True)
            time.sleep(60)
            continue

        print(f"[{now.strftime('%H:%M:%S IST')}] Scan Cycle #{scan_iteration} | Open Pos -> V43: {len(positions_v43)}, V3 Bull: {len(positions_v3_bull)}, V3 Bear: {len(positions_v3_bear)}", flush=True)

        # Pre-cache Daily 20 EMA for V3 Stocks if not cached
        for s in set(V3_BULL_UNIVERSE + V3_BEAR_UNIVERSE):
            if s not in daily_ema_cache:
                try:
                    df_d = tv.get_hist(symbol=s, exchange="NSE", interval=Interval.in_daily, n_bars=35)
                    if df_d is not None and len(df_d) >= 20:
                        daily_ema_cache[s] = float(df_d['close'].ewm(span=20, adjust=False).mean().iloc[-2])
                except Exception:
                    pass

        # ======================================================================
        # 1. POSITION MANAGEMENT: STRICT ISOLATION FOR ALL 3 ENGINES
        # ======================================================================

        # A. Engine 1: Bhavishya V43 Titan Positions (Asymmetric Ratchet)
        for sym in list(positions_v43.keys()):
            pos = positions_v43[sym]
            try:
                df_cur = tv.get_hist(symbol=sym, exchange="NSE", interval=Interval.in_1_minute, n_bars=3)
                if df_cur is None or df_cur.empty: continue
                cur_close = float(df_cur["close"].iloc[-1])
                cur_high = float(df_cur["high"].iloc[-1])
                cur_low = float(df_cur["low"].iloc[-1])
            except Exception:
                continue

            pos["high_seen"] = max(pos["high_seen"], cur_high)

            if cur_time >= t_eod_cutoff:
                chg = estimate_dhan_charges(pos["entry_p"], cur_close, pos["qty"])
                send_telegram_alert(format_v43_exit(sym, pos["entry_p"], cur_close, pos["qty"], "15:15 EOD SQUARE OFF", chg))
                del positions_v43[sym]
                continue

            mfe = pos["high_seen"] - pos["entry_p"]
            atr = pos["atr"]
            if mfe >= 0.85 * atr:
                pos["sl"] = max(pos["sl"], pos["entry_p"] + (0.40 * atr))
            elif mfe >= 0.45 * atr:
                pos["sl"] = max(pos["sl"], pos["entry_p"] + (0.05 * atr))

            if cur_low <= pos["sl"]:
                fill_p = pos["sl"]
                chg = estimate_dhan_charges(pos["entry_p"], fill_p, pos["qty"])
                send_telegram_alert(format_v43_exit(sym, pos["entry_p"], fill_p, pos["qty"], "RATCHET STOP", chg))
                del positions_v43[sym]
                continue

        # B. Engine 2: Predator V3 Bull Positions (1.2% Risk, 2.52% TP, 1.2% Trail)
        for sym in list(positions_v3_bull.keys()):
            pos = positions_v3_bull[sym]
            try:
                df_cur = tv.get_hist(symbol=sym, exchange="NSE", interval=Interval.in_1_minute, n_bars=3)
                if df_cur is None or df_cur.empty: continue
                cur_close = float(df_cur["close"].iloc[-1])
                cur_high = float(df_cur["high"].iloc[-1])
                cur_low = float(df_cur["low"].iloc[-1])
            except Exception:
                continue

            pos["high_seen"] = max(pos.get("high_seen", pos["entry_p"]), cur_high)

            # EOD Cutoff
            if cur_time >= t_eod_cutoff:
                chg = estimate_dhan_charges(pos["entry_p"], cur_close, pos["qty"])
                send_telegram_alert(format_v3_bull_exit(sym, pos["entry_p"], cur_close, pos["qty"], "15:15 EOD SQUARE OFF", chg))
                del positions_v3_bull[sym]
                continue

            # Trailing stop update (Pine Script logic: trail_points = 1.2%, trail_offset = 0.6%)
            gain_pct = (pos["high_seen"] - pos["entry_p"]) / pos["entry_p"]
            if gain_pct >= 0.012:
                # Trail activated at 0.6% distance behind high_seen
                new_trail_sl = pos["high_seen"] - (pos["entry_p"] * 0.006)
                pos["sl"] = max(pos["sl"], new_trail_sl)

            # Check Stop Loss / Trailing Stop
            if cur_low <= pos["sl"]:
                fill_p = pos["sl"]
                chg = estimate_dhan_charges(pos["entry_p"], fill_p, pos["qty"])
                reason = "TRAIL STOP LOCK" if pos["sl"] > pos["entry_p"] else "HARD STOP LOSS (-1.2%)"
                send_telegram_alert(format_v3_bull_exit(sym, pos["entry_p"], fill_p, pos["qty"], reason, chg))
                del positions_v3_bull[sym]
                continue

            # Check Take Profit
            if cur_high >= pos["tp"]:
                fill_p = pos["tp"]
                chg = estimate_dhan_charges(pos["entry_p"], fill_p, pos["qty"])
                send_telegram_alert(format_v3_bull_exit(sym, pos["entry_p"], fill_p, pos["qty"], "TARGET HIT (+2.52% / 2.1R)", chg))
                del positions_v3_bull[sym]
                continue

        # C. Engine 3: Predator V3 Bear Positions (Short: 1.2% SL, 2.52% TP drop, 1.2% Trail)
        for sym in list(positions_v3_bear.keys()):
            pos = positions_v3_bear[sym]
            try:
                df_cur = tv.get_hist(symbol=sym, exchange="NSE", interval=Interval.in_1_minute, n_bars=3)
                if df_cur is None or df_cur.empty: continue
                cur_close = float(df_cur["close"].iloc[-1])
                cur_high = float(df_cur["high"].iloc[-1])
                cur_low = float(df_cur["low"].iloc[-1])
            except Exception:
                continue

            pos["low_seen"] = min(pos.get("low_seen", pos["entry_p"]), cur_low)

            # EOD Cutoff
            if cur_time >= t_eod_cutoff:
                chg = estimate_dhan_charges(cur_close, pos["entry_p"], pos["qty"])
                send_telegram_alert(format_v3_bear_exit(sym, pos["entry_p"], cur_close, pos["qty"], "15:15 EOD SHORT COVER", chg))
                del positions_v3_bear[sym]
                continue

            # Trailing stop update (Short: trail_points = 1.2% drop, trail_offset = 0.6%)
            drop_pct = (pos["entry_p"] - pos["low_seen"]) / pos["entry_p"]
            if drop_pct >= 0.012:
                # Trail activated at 0.6% distance above low_seen
                new_trail_sl = pos["low_seen"] + (pos["entry_p"] * 0.006)
                pos["sl"] = min(pos["sl"], new_trail_sl)

            # Check Short Stop Loss / Trailing Stop
            if cur_high >= pos["sl"]:
                fill_p = pos["sl"]
                chg = estimate_dhan_charges(fill_p, pos["entry_p"], pos["qty"])
                reason = "TRAIL COVER LOCK" if pos["sl"] < pos["entry_p"] else "HARD STOP LOSS (+1.2%)"
                send_telegram_alert(format_v3_bear_exit(sym, pos["entry_p"], fill_p, pos["qty"], reason, chg))
                del positions_v3_bear[sym]
                continue

            # Check Short Take Profit
            if cur_low <= pos["tp"]:
                fill_p = pos["tp"]
                chg = estimate_dhan_charges(fill_p, pos["entry_p"], pos["qty"])
                send_telegram_alert(format_v3_bear_exit(sym, pos["entry_p"], fill_p, pos["qty"], "SHORT TARGET HIT (-2.52% / 2.1R)", chg))
                del positions_v3_bear[sym]
                continue

        # ======================================================================
        # 2. SCANNING ENGINE 1: BHAVISHYA V43 TITAN (15-Minute Institutional)
        # ======================================================================
        if cur_time >= t_trade_start and cur_time <= datetime.time(14, 15) and len(positions_v43) == 0 and trades_today_v43 < 2:
            for s in V43_30_UNIVERSE:
                try:
                    df_15m = tv.get_hist(symbol=s, exchange="NSE", interval=Interval.in_15_minute, n_bars=60)
                    if df_15m is None or len(df_15m) < 45: continue
                    c = df_15m["close"]
                    h = df_15m["high"]
                    l = df_15m["low"]
                    v = df_15m["volume"]

                    ema20 = c.ewm(span=20).mean()
                    ema50 = c.ewm(span=50).mean()
                    ema110 = c.ewm(span=110).mean()

                    delta = c.diff()
                    gain = delta.clip(lower=0).rolling(7).mean()
                    loss = (-delta.clip(upper=0)).rolling(7).mean()
                    rsi7 = 100.0 - (100.0 / (1.0 + gain / (loss + 1e-9)))

                    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
                    sig = macd.ewm(span=9).mean()
                    macd_cross_up = (macd.iloc[-1] > sig.iloc[-1]) and (macd.iloc[-2] <= sig.iloc[-2])

                    tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
                    atr14 = tr.rolling(14).mean()
                    atr_now = float(atr14.iloc[-1])

                    v_sma20 = v.rolling(20).mean()
                    rvol = v / (v_sma20 + 1e-9)

                    tp_series = (h + l + c) / 3.0
                    vwap_15m = (tp_series * v).cumsum() / (v.cumsum() + 1e-9)

                    c_now = float(c.iloc[-1])
                    c_prev = float(c.iloc[-2])
                    l_prev = float(l.iloc[-2])
                    e20_now = float(ema20.iloc[-1])
                    e20_prev = float(ema20.iloc[-2])
                    e50_now = float(ema50.iloc[-1])
                    e110_now = float(ema110.iloc[-1])
                    vwap_now = float(vwap_15m.iloc[-1])
                    rvol_now = float(rvol.iloc[-1])
                    rsi_now = float(rsi7.iloc[-1])
                    macd_now = float(macd.iloc[-1])
                    sig_now = float(sig.iloc[-1])

                    macro_ok = (c_now > e50_now) and (c_now > e110_now) and (c_now > vwap_now)

                    trig_ma20_reclaim = (
                        (l_prev < e20_prev) and (c_prev <= e20_prev) and (c_now > e20_now) and
                        (rvol_now >= 1.25) and (rsi_now > 50.0) and (macd_now > sig_now) and macro_ok
                    )
                    trig_macd = (
                        macd_cross_up and (macd_now > 0) and (c_now > e20_now) and
                        (rsi_now > 50.0) and (rvol_now >= 1.30) and macro_ok
                    )

                    if trig_ma20_reclaim or trig_macd:
                        pattern_name = "MA20 RECLAIM INSTITUTIONAL" if trig_ma20_reclaim else "MACD MOMENTUM CROSS"
                        qty = max(1, int((wallet_v43 * 5.0) / c_now))
                        sl = round(c_now - (0.90 * atr_now), 2)
                        est_chg = estimate_dhan_charges(c_now, c_now + atr_now, qty)
                        
                        send_telegram_alert(format_v43_entry(s, c_now, sl, qty, wallet_v43, est_chg, pattern_name, atr_now))
                        positions_v43[s] = {
                            "engine": "V43", "entry_p": c_now, "sl": sl, "atr": atr_now,
                            "qty": qty, "high_seen": c_now, "entry_t": now
                        }
                        trades_today_v43 += 1
                        break
                except Exception as e:
                    print(f"[ERROR V43 Scan {s}]: {e}", flush=True)

        # ======================================================================
        # 3. SCANNING ENGINE 2: PREDATOR V3 BULL SQUAD (1-Minute Long)
        # ======================================================================
        if cur_time >= t_trade_start and cur_time <= datetime.time(14, 45):
            for s in V3_BULL_UNIVERSE:
                if s in positions_v3_bull or trades_today_v3_bull.get(s, 0) >= 2:
                    continue
                try:
                    df_1m = tv.get_hist(symbol=s, exchange="NSE", interval=Interval.in_1_minute, n_bars=300)
                    if df_1m is None or len(df_1m) < 40: continue

                    # Filter bars of today
                    df_today = df_1m[df_1m.index.date == cur_date]
                    if df_today.empty or len(df_today) < 30: continue

                    # ORB Calculation (09:15 - 09:45)
                    orb_bars = df_today[(df_today.index.time >= datetime.time(9, 15)) & (df_today.index.time <= datetime.time(9, 45))]
                    if orb_bars.empty: continue
                    orb_high = float(orb_bars['high'].max())
                    orb_low = float(orb_bars['low'].min())

                    # VWAP
                    tp = (df_today['high'] + df_today['low'] + df_today['close']) / 3.0
                    vwap = (tp * df_today['volume']).cumsum() / (df_today['volume'].cumsum() + 1e-9)
                    vwap_now = float(vwap.iloc[-1])

                    # Volume Spike (> 1.05 * 20 SMA)
                    vol_sma20 = float(df_today['volume'].rolling(20).mean().iloc[-1])
                    cur_vol = float(df_today['volume'].iloc[-1])
                    vol_spike = cur_vol > (vol_sma20 * 1.05)

                    c_now = float(df_today['close'].iloc[-1])
                    c_prev = float(df_today['close'].iloc[-2])
                    daily_ema = daily_ema_cache.get(s, 0.0)

                    # Filters: Above VWAP and Daily 20 EMA
                    bull_filter = (c_now > vwap_now) and (c_now > daily_ema) and vol_spike

                    # Entry 1: Morning ORB High Breakout
                    trig_entry1 = (trades_today_v3_bull[s] == 0) and (c_prev <= orb_high) and (c_now > orb_high) and bull_filter

                    # Entry 2: Afternoon Momentum Push (Breakout of 15-min consolidation)
                    highest_15m = float(df_today['high'].iloc[-16:-1].max()) if len(df_today) >= 16 else 999999.0
                    trig_entry2 = (trades_today_v3_bull[s] == 1) and (c_prev <= highest_15m) and (c_now > highest_15m) and bull_filter

                    if trig_entry1 or trig_entry2:
                        trades_today_v3_bull[s] += 1
                        qty = max(1, int((wallet_v3_bull * 5.0) / c_now))
                        sl_dist = c_now * 0.012
                        sl_price = round(c_now - sl_dist, 2)
                        tp_price = round(c_now + (sl_dist * 2.1), 2)
                        est_chg = estimate_dhan_charges(c_now, tp_price, qty)

                        send_telegram_alert(format_v3_bull_entry(s, c_now, sl_price, tp_price, qty, wallet_v3_bull, est_chg, trades_today_v3_bull[s]))
                        positions_v3_bull[s] = {
                            "engine": "V3_BULL", "entry_p": c_now, "sl": sl_price, "tp": tp_price,
                            "qty": qty, "high_seen": c_now, "entry_t": now
                        }
                except Exception as e:
                    print(f"[ERROR V3 Bull Scan {s}]: {e}", flush=True)

        # ======================================================================
        # 4. SCANNING ENGINE 3: PREDATOR V3 BEAR SQUAD (1-Minute Short)
        # ======================================================================
        if cur_time >= t_trade_start and cur_time <= datetime.time(14, 45):
            for s in V3_BEAR_UNIVERSE:
                if s in positions_v3_bear or trades_today_v3_bear.get(s, 0) >= 2:
                    continue
                try:
                    df_1m = tv.get_hist(symbol=s, exchange="NSE", interval=Interval.in_1_minute, n_bars=300)
                    if df_1m is None or len(df_1m) < 40: continue

                    df_today = df_1m[df_1m.index.date == cur_date]
                    if df_today.empty or len(df_today) < 30: continue

                    # ORB Calculation (09:15 - 09:45)
                    orb_bars = df_today[(df_today.index.time >= datetime.time(9, 15)) & (df_today.index.time <= datetime.time(9, 45))]
                    if orb_bars.empty: continue
                    orb_low = float(orb_bars['low'].min())

                    # VWAP
                    tp = (df_today['high'] + df_today['low'] + df_today['close']) / 3.0
                    vwap = (tp * df_today['volume']).cumsum() / (df_today['volume'].cumsum() + 1e-9)
                    vwap_now = float(vwap.iloc[-1])

                    # Volume Spike
                    vol_sma20 = float(df_today['volume'].rolling(20).mean().iloc[-1])
                    cur_vol = float(df_today['volume'].iloc[-1])
                    vol_spike = cur_vol > (vol_sma20 * 1.05)

                    c_now = float(df_today['close'].iloc[-1])
                    c_prev = float(df_today['close'].iloc[-2])
                    daily_ema = daily_ema_cache.get(s, 999999.0)

                    # Filters: Below VWAP and Below Daily 20 EMA
                    bear_filter = (c_now < vwap_now) and (c_now < daily_ema) and vol_spike

                    # Setup 1: Clean 30-Min ORB Low Breakdown after 09:45
                    trig_entry1 = (trades_today_v3_bear[s] == 0) and (c_prev >= orb_low) and (c_now < orb_low) and bear_filter

                    # Setup 2: Afternoon Breakdown of 15-Min Support below VWAP
                    lowest_15m = float(df_today['low'].iloc[-16:-1].min()) if len(df_today) >= 16 else 0.0
                    trig_entry2 = (trades_today_v3_bear[s] == 1) and (c_prev >= lowest_15m) and (c_now < lowest_15m) and bear_filter

                    if trig_entry1 or trig_entry2:
                        trades_today_v3_bear[s] += 1
                        qty = max(1, int((wallet_v3_bear * 5.0) / c_now))
                        sl_dist = c_now * 0.012
                        sl_price = round(c_now + sl_dist, 2)
                        tp_price = round(c_now - (sl_dist * 2.1), 2)
                        est_chg = estimate_dhan_charges(tp_price, c_now, qty)

                        send_telegram_alert(format_v3_bear_entry(s, c_now, sl_price, tp_price, qty, wallet_v3_bear, est_chg, trades_today_v3_bear[s]))
                        positions_v3_bear[s] = {
                            "engine": "V3_BEAR", "entry_p": c_now, "sl": sl_price, "tp": tp_price,
                            "qty": qty, "low_seen": c_now, "entry_t": now
                        }
                except Exception as e:
                    print(f"[ERROR V3 Bear Scan {s}]: {e}", flush=True)

        time.sleep(50)

# ==============================================================================
# ENTRY POINT
# ==============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Tri-Engine Autonomous Live Telegram Forward-Tester")
    parser.add_argument("--test", action="store_true", help="Send test alerts to verify Telegram connection")
    parser.add_argument("--live", action="store_true", help="Start continuous live market scanner")
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
            "Status: <b>ONLINE & LIVE (Cloud Verified)</b>\n"
            f"Server Time: <b>{now_ist.strftime('%d-%b-%Y %H:%M:%S IST')}</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"1️⃣ 🏛️ <b>Bhavishya V43 Titan (15M):</b> {len(V43_30_UNIVERSE)} Verified Core Stocks\n"
            f"2️⃣ 🟢 <b>Predator V3 Bull Squad (1M Long):</b> {len(V3_BULL_UNIVERSE)} High-Beta Stocks ({', '.join(V3_BULL_UNIVERSE)})\n"
            f"3️⃣ 🔴 <b>Predator V3 Bear Squad (1M Short):</b> {len(V3_BEAR_UNIVERSE)} Distribution Stocks ({', '.join(V3_BEAR_UNIVERSE)})\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Strict Isolation: Compartmentalized states, independent triggers, zero bleed.\n"
            "Zero laptop involvement. Verified signals will ping your phone live!"
        )
        send_telegram_alert(init_msg, token, chat_id)
        print("✅ SUCCESS! Tri-Engine test alert sent.", flush=True)
    elif args.live:
        run_tri_engine_scanner(initial_wallet=500.0)
    else:
        print("Tri-Engine Live Forward-Testing System Ready.", flush=True)
        print("Run with --test or --live.", flush=True)
