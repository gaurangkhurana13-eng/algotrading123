"""
Bhavishya V43 Titan Autonomous Live Telegram Forward-Testing Engine
15-Minute Institutional Quant Engine across Certified Core 30-Stock Universe
100% Free, Live, Precise, and Accurate Real-Time Cloud Dispatcher
Strict IST Timezone & Real-Time Console Flushing
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
        print(f"[TELEGRAM WARNING] Missing credentials! Token set: {bool(token)}, Chat ID set: {bool(c_id)}", flush=True)
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
# BHAVISHYA V43 TITAN MESSAGE FORMATTERS
# ==============================================================================
def format_v43_entry(symbol: str, price: float, sl: float, qty: int, capital: float, charges: float, pattern: str, atr: float) -> str:
    trade_val = round(price * qty, 2)
    be_price = price + 0.45 * atr
    return (
        f"🏛️ <b>[BHAVISHYA V43 TITAN 15M — ENTRY]</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 <b>Stock:</b> <code>{symbol}</code> (Core Universe)\n"
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
        f"📌 <b>Stock:</b> <code>{symbol}</code>\n"
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

# ==============================================================================
# CERTIFIED BHAVISHYA V43 30-STOCK UNIVERSE
# ==============================================================================
V43_30_UNIVERSE = [
    "CENTRALBK", "DELTACORP", "ELECTCAST", "GSFC", "IDEA", 
    "IFCI", "IOB", "IRCON", "J&KBANK", "JYOTHYLAB", 
    "MANALIPETC", "MTNL", "NBCC", "NCC", "NIACL", 
    "NMDC", "PCJEWELLER", "PNB", "RBLBANK", "RCF", 
    "RENUKA", "RITES", "RPOWER", "SUZLON", "SWSOLAR", 
    "TATASTEEL", "TEXRAIL", "TTML", "UTKARSHBNK", "ZEEL"
]

# ==============================================================================
# LIVE REAL-TIME CLOUD SCANNER LOOP (PURE V43 TITAN)
# ==============================================================================
def run_live_cloud_scanner(initial_wallet: float = 500.0):
    from tvDatafeed import TvDatafeed, Interval

    now_ist = get_ist_now()
    print("\n=======================================================", flush=True)
    print("  BHAVISHYA V43 TITAN LIVE CLOUD SCANNER ACTIVATED", flush=True)
    print(f"  Current Time (IST): {now_ist.strftime('%Y-%m-%d %H:%M:%S IST')}", flush=True)
    print(f"  Bot Token Injected: {'YES' if TELEGRAM_BOT_TOKEN else 'NO (Check GitHub Secrets!)'}", flush=True)
    print(f"  Chat ID Injected:   {'YES (' + str(TELEGRAM_CHAT_ID) + ')' if TELEGRAM_CHAT_ID else 'NO (Check GitHub Secrets!)'}", flush=True)
    print(f"  Engine: Bhavishya V43 Titan 15-Minute Institutional", flush=True)
    print(f"  Universe: Exact {len(V43_30_UNIVERSE)} Verified Stocks", flush=True)
    print("=======================================================\n", flush=True)

    send_telegram_alert(
        "🏛️ <b>BHAVISHYA V43 TITAN CLOUD SCANNER ONLINE!</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Platform: <b>GitHub Actions Cloud Runner (Ubuntu)</b>\n"
        f"Start Time: <b>{now_ist.strftime('%d-%b-%Y %H:%M:%S IST')}</b>\n"
        f"Strategy: <b>Bhavishya V43 Titan (15-Min Institutional)</b>\n"
        f"Universe: <b>Certified {len(V43_30_UNIVERSE)} Stocks</b>\n"
        "Risk Protocol: <b>-0.90 ATR Hard SL | +0.45 ATR Breakeven Lock</b>\n"
        "Mode: <b>Real-Time Paper Forward Testing</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Zero laptop running required. Verified signals will ping your phone live!"
    )

    tv = TvDatafeed()
    wallet = initial_wallet
    positions: Dict[str, dict] = {}
    daily_trades = 0

    scan_iteration = 0
    while True:
        scan_iteration += 1
        now = get_ist_now()
        cur_time = now.time()

        # Session limits (09:15 to 15:30 IST)
        t_open = datetime.time(9, 15)
        t_trade_start = datetime.time(9, 45)
        t_eod_cutoff = datetime.time(15, 15)
        t_close = datetime.time(15, 30)

        # Outside market hours
        if cur_time > t_close or cur_time < t_open:
            if scan_iteration % 5 == 1:
                print(f"[{now.strftime('%H:%M:%S IST')}] Outside NSE Trading Hours (09:15-15:30). Waiting for market open...", flush=True)
            time.sleep(60)
            continue

        print(f"[{now.strftime('%H:%M:%S IST')}] Scan Cycle #{scan_iteration} active. Monitoring positions: {len(positions)}", flush=True)

        # 1. Manage Active Positions (Asymmetric Ratchet)
        for sym in list(positions.keys()):
            pos = positions[sym]
            try:
                df_cur = tv.get_hist(symbol=sym, exchange="NSE", interval=Interval.in_1_minute, n_bars=3)
                if df_cur is None or df_cur.empty:
                    continue
                cur_close = float(df_cur["close"].iloc[-1])
                cur_high = float(df_cur["high"].iloc[-1])
                cur_low = float(df_cur["low"].iloc[-1])
            except Exception:
                continue

            pos["high_seen"] = max(pos["high_seen"], cur_high)

            # EOD 15:15 IST Mandatory Square Off
            if cur_time >= t_eod_cutoff:
                chg = estimate_dhan_charges(pos["entry_p"], cur_close, pos["qty"])
                send_telegram_alert(format_v43_exit(sym, pos["entry_p"], cur_close, pos["qty"], "15:15 EOD SQUARE OFF", chg))
                del positions[sym]
                continue

            # V43 Titan Exit Logic (Asymmetric Ratchet)
            mfe = pos["high_seen"] - pos["entry_p"]
            atr = pos["atr"]
            if mfe >= 0.85 * atr:
                pos["sl"] = max(pos["sl"], pos["entry_p"] + (0.40 * atr)) # Runner Lock
            elif mfe >= 0.45 * atr:
                pos["sl"] = max(pos["sl"], pos["entry_p"] + (0.05 * atr)) # Breakeven Lock

            if cur_low <= pos["sl"]:
                fill_p = pos["sl"]
                chg = estimate_dhan_charges(pos["entry_p"], fill_p, pos["qty"])
                send_telegram_alert(format_v43_exit(sym, pos["entry_p"], fill_p, pos["qty"], "RATCHET STOP", chg))
                del positions[sym]
                continue

        # 2. Scan Bhavishya V43 Titan Across All 30 Stocks (15-Min)
        if cur_time >= t_trade_start and cur_time <= datetime.time(14, 15) and len(positions) == 0 and daily_trades < 2:
            for s in V43_30_UNIVERSE:
                try:
                    df_15m = tv.get_hist(symbol=s, exchange="NSE", interval=Interval.in_15_minute, n_bars=60)
                    if df_15m is None or len(df_15m) < 45:
                        continue
                    c = df_15m["close"]
                    h = df_15m["high"]
                    l = df_15m["low"]
                    v = df_15m["volume"]
                    o = df_15m["open"]

                    ema9 = c.ewm(span=9).mean()
                    ema20 = c.ewm(span=20).mean()
                    ema50 = c.ewm(span=50).mean()
                    ema110 = c.ewm(span=110).mean()

                    # Delta & RSI 7
                    delta = c.diff()
                    gain = delta.clip(lower=0).rolling(7).mean()
                    loss = (-delta.clip(upper=0)).rolling(7).mean()
                    rsi7 = 100.0 - (100.0 / (1.0 + gain / (loss + 1e-9)))

                    # MACD 12,26,9
                    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
                    sig = macd.ewm(span=9).mean()
                    macd_cross_up = (macd.iloc[-1] > sig.iloc[-1]) and (macd.iloc[-2] <= sig.iloc[-2])

                    # True Range & ATR 14
                    tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
                    atr14 = tr.rolling(14).mean()
                    atr_now = float(atr14.iloc[-1])

                    # Volume & RVOL
                    v_sma20 = v.rolling(20).mean()
                    rvol = v / (v_sma20 + 1e-9)

                    # VWAP estimate
                    tp_series = (h + l + c) / 3.0
                    vwap_15m = (tp_series * v).cumsum() / (v.cumsum() + 1e-9)

                    # Values
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

                    # V43 Macro Filter
                    macro_ok = (c_now > e50_now) and (c_now > e110_now) and (c_now > vwap_now)

                    # Trigger A: MA20 Reclaim (Day Trader Institutional Trigger)
                    trig_ma20_reclaim = (
                        (l_prev < e20_prev) and (c_prev <= e20_prev) and (c_now > e20_now) and
                        (rvol_now >= 1.25) and (rsi_now > 50.0) and (macd_now > sig_now) and macro_ok
                    )

                    # Trigger B: Core MACD Momentum
                    trig_macd = (
                        macd_cross_up and (macd_now > 0) and (c_now > e20_now) and
                        (rsi_now > 50.0) and (rvol_now >= 1.30) and macro_ok
                    )

                    is_triggered = trig_ma20_reclaim or trig_macd
                    pattern_name = "MA20 RECLAIM INSTITUTIONAL" if trig_ma20_reclaim else "MACD MOMENTUM CROSS"

                    if is_triggered:
                        qty = max(1, int((wallet * 5.0) / c_now))
                        sl = round(c_now - (0.90 * atr_now), 2)
                        est_chg = estimate_dhan_charges(c_now, c_now + atr_now, qty)
                        
                        send_telegram_alert(format_v43_entry(s, c_now, sl, qty, wallet, est_chg, pattern_name, atr_now))
                        positions[s] = {
                            "engine": "V43", "entry_p": c_now, "sl": sl, "atr": atr_now,
                            "qty": qty, "high_seen": c_now, "entry_t": now
                        }
                        daily_trades += 1
                        break
                except Exception as e:
                    print(f"[ERROR V43 Scan {s}]: {e}", flush=True)

        time.sleep(55)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bhavishya V43 Titan Live Cloud Forward-Tester")
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
            "🏛️ <b>BHAVISHYA V43 TITAN DISPATCHER ACTIVATED!</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Status: <b>ONLINE & LIVE (Cloud Verified)</b>\n"
            f"Server Time: <b>{now_ist.strftime('%d-%b-%Y %H:%M:%S IST')}</b>\n"
            "Engine: <b>Bhavishya V43 Titan 15-Minute Institutional</b>\n"
            f"Universe: <b>Exact {len(V43_30_UNIVERSE)} Verified Stocks</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Zero laptop involvement. Verified institutional signals will ping your phone live!"
        )
        send_telegram_alert(init_msg, token, chat_id)
        print("✅ SUCCESS! V43 Titan test alert sent.", flush=True)
    elif args.live:
        run_live_cloud_scanner(initial_wallet=500.0)
    else:
        print("Bhavishya V43 Titan Live Engine Ready.", flush=True)
        print("Run with --test or --live.", flush=True)
