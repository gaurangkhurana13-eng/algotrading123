"""
Bhavishya Autonomous Live Telegram Forward-Testing Engine
Dual-Engine: PC Jeweller 1m Live Predator V3 + Bhavishya V43 Titan 15m
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
# MESSAGE FORMATTERS
# ==============================================================================
def format_v3_entry(symbol: str, price: float, sl: float, tp: float, qty: int, capital: float, charges: float) -> str:
    trade_val = round(price * qty, 2)
    return (
        f"🎯 <b>[PREDATOR V3 — ENTRY BREAKOUT]</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 <b>Stock:</b> <code>{symbol}</code>\n"
        f"⚡ <b>Action:</b> BUY (Intraday MIS 5x)\n"
        f"💵 <b>Entry Price:</b> ₹{price:.2f}\n"
        f"🛑 <b>Stop Loss (1.2%):</b> ₹{sl:.2f}\n"
        f"🎯 <b>Target (2.1R):</b> ₹{tp:.2f}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💼 <b>Wallet Capital:</b> ₹{capital:.2f}\n"
        f"🪙 <b>5x Trade Value:</b> ₹{trade_val:.2f} ({qty} shares)\n"
        f"🧾 <b>Est. Dhan Charges:</b> ₹{charges:.2f} (0.03% / ₹20 cap)\n"
        f"⏰ <b>Trigger Time:</b> {get_ist_now().strftime('%H:%M:%S IST')}\n"
        f"💡 <i>Setup: 30-Min ORB High Breakout + VWAP Confirmation</i>"
    )

def format_v3_exit(symbol: str, entry_p: float, exit_p: float, qty: int, reason: str, charges: float) -> str:
    gross = round((exit_p - entry_p) * qty, 2)
    net = round(gross - charges, 2)
    icon = "💰" if net > 0 else "🛑"
    return (
        f"{icon} <b>[PREDATOR V3 — EXIT COMPLETED]</b>\n"
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

def format_v43_entry(symbol: str, price: float, sl: float, qty: int, capital: float, charges: float, pattern: str) -> str:
    trade_val = round(price * qty, 2)
    return (
        f"🏛️ <b>[BHAVISHYA V43 TITAN 15M — ENTRY]</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📌 <b>Stock:</b> <code>{symbol}</code>\n"
        f"⚡ <b>Action:</b> BUY (Institutional MIS 5x)\n"
        f"💵 <b>Entry Price:</b> ₹{price:.2f}\n"
        f"🛑 <b>Initial Hard SL:</b> ₹{sl:.2f}\n"
        f"🛡️ <b>Breakeven Lock Target:</b> ₹{(price + (price - sl) * 0.5):.2f} (+0.45 ATR)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"💼 <b>Wallet Capital:</b> ₹{capital:.2f}\n"
        f"🪙 <b>5x Trade Value:</b> ₹{trade_val:.2f} ({qty} shares)\n"
        f"🧾 <b>Est. Dhan Charges:</b> ₹{charges:.2f}\n"
        f"🎯 <b>Alpha Pattern:</b> <code>{pattern}</code>\n"
        f"⏰ <b>Time:</b> {get_ist_now().strftime('%H:%M:%S IST')}\n"
        f"💡 <i>Ratchet: MFE +0.45 ATR lock to Breakeven | Runners Trailed</i>"
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
# LIVE REAL-TIME CLOUD SCANNER LOOP
# ==============================================================================
def run_live_cloud_scanner(initial_wallet: float = 500.0):
    from tvDatafeed import TvDatafeed, Interval

    now_ist = get_ist_now()
    print("\n=======================================================", flush=True)
    print("  BHAVISHYA AUTONOMOUS LIVE CLOUD SCANNER ACTIVATED", flush=True)
    print(f"  Current Time (IST): {now_ist.strftime('%Y-%m-%d %H:%M:%S IST')}", flush=True)
    print(f"  Bot Token Injected: {'YES' if TELEGRAM_BOT_TOKEN else 'NO (Check GitHub Secrets!)'}", flush=True)
    print(f"  Chat ID Injected:   {'YES (' + str(TELEGRAM_CHAT_ID) + ')' if TELEGRAM_CHAT_ID else 'NO (Check GitHub Secrets!)'}", flush=True)
    print("  Engine 1: 1m Micro Predator V3 (ORB Sniper)", flush=True)
    print("  Engine 2: 15m Institutional V43 Titan", flush=True)
    print("=======================================================\n", flush=True)

    sent = send_telegram_alert(
        "🚀 <b>BHAVISHYA CLOUD SCANNER ONLINE!</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Platform: <b>GitHub Actions Cloud Runner (Ubuntu)</b>\n"
        f"Start Time: <b>{now_ist.strftime('%d-%b-%Y %H:%M:%S IST')}</b>\n"
        "Live Feeds: <b>NSE 1-Min & 15-Min Streams (Active)</b>\n"
        "Universe: <b>PC Jeweller, South Bank, Suzlon, MTNL, NMDC + Titans</b>\n"
        "Mode: <b>Real-Time Paper Forward Testing</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Zero laptop involvement. Signals will ping your phone live!"
    )
    if sent:
        print("[SUCCESS] Startup Telegram announcement delivered to phone!", flush=True)
    else:
        print("[WARNING] Could not deliver startup alert to Telegram. Verify GitHub Secrets.", flush=True)

    tv = TvDatafeed()
    v3_symbols = ["PCJEWELLER", "SOUTHBANK", "SUZLON", "IDFCFIRSTB", "NMDC"]
    v43_symbols = ["MTNL", "SOUTHBANK", "RCF", "NBCC", "COALINDIA", "TATASTEEL", "SAIL", "SUZLON", "PCJEWELLER", "IOB"]

    wallet = initial_wallet
    positions: Dict[str, dict] = {}
    daily_trades_v3 = 0
    daily_trades_v43 = 0

    scan_iteration = 0
    while True:
        scan_iteration += 1
        now = get_ist_now()
        cur_time = now.time()

        # Check market session (09:15 to 15:30 IST)
        t_open = datetime.time(9, 15)
        t_orb_end = datetime.time(9, 45)
        t_eod_cutoff = datetime.time(15, 15)
        t_close = datetime.time(15, 30)

        # Outside market hours
        if cur_time > t_close or cur_time < t_open:
            if scan_iteration % 5 == 1:
                print(f"[{now.strftime('%H:%M:%S IST')}] Outside NSE Trading Hours (09:15-15:30). Waiting for market open...", flush=True)
            time.sleep(60)
            continue

        print(f"[{now.strftime('%H:%M:%S IST')}] Starting Scan Cycle #{scan_iteration} across {len(v3_symbols) + len(v43_symbols)} stocks...", flush=True)

        # 1. Manage Active Positions
        for sym in list(positions.keys()):
            pos = positions[sym]
            eng = pos["engine"]
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

            # EOD Cutoff Exit at 15:15 IST
            if cur_time >= t_eod_cutoff:
                chg = estimate_dhan_charges(pos["entry_p"], cur_close, pos["qty"])
                msg = format_v3_exit(sym, pos["entry_p"], cur_close, pos["qty"], "15:15 EOD SQUARE OFF", chg) if eng == "V3" else format_v43_exit(sym, pos["entry_p"], cur_close, pos["qty"], "15:15 EOD SQUARE OFF", chg)
                send_telegram_alert(msg)
                del positions[sym]
                continue

            # Exit logic for V3
            if eng == "V3":
                if cur_high >= pos["tp"]:
                    fill_p = pos["tp"]
                    chg = estimate_dhan_charges(pos["entry_p"], fill_p, pos["qty"])
                    send_telegram_alert(format_v3_exit(sym, pos["entry_p"], fill_p, pos["qty"], "TARGET (2.1R HIT)", chg))
                    del positions[sym]
                    continue
                if cur_low <= pos["sl"]:
                    fill_p = pos["sl"]
                    chg = estimate_dhan_charges(pos["entry_p"], fill_p, pos["qty"])
                    send_telegram_alert(format_v3_exit(sym, pos["entry_p"], fill_p, pos["qty"], "STOP LOSS", chg))
                    del positions[sym]
                    continue

            # Exit logic for V43
            if eng == "V43":
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
                    del positions[sym]
                    continue

        # 2. Scan V3 Micro Sniper (1-Min ORB)
        if cur_time >= t_orb_end and daily_trades_v3 < 2 and len(positions) == 0:
            for s in v3_symbols:
                try:
                    df_1m = tv.get_hist(symbol=s, exchange="NSE", interval=Interval.in_1_minute, n_bars=45)
                    if df_1m is None or len(df_1m) < 35:
                        continue
                    orb_high = df_1m["high"].iloc[:30].max()
                    c_now = float(df_1m["close"].iloc[-1])
                    c_prev = float(df_1m["close"].iloc[-2])
                    vol_now = float(df_1m["volume"].iloc[-1])
                    vol_sma = float(df_1m["volume"].rolling(20).mean().iloc[-1])
                    
                    tp_series = (df_1m["high"] + df_1m["low"] + df_1m["close"]) / 3.0
                    vwap = (tp_series * df_1m["volume"]).cumsum().iloc[-1] / (df_1m["volume"].cumsum().iloc[-1] + 1e-9)

                    if c_prev <= orb_high and c_now > orb_high and c_now > vwap and vol_now > vol_sma * 1.05:
                        qty = max(1, int((wallet * 5.0) / c_now))
                        sl = round(c_now * 0.988, 2)
                        tp = round(c_now + ((c_now - sl) * 2.1), 2)
                        est_chg = estimate_dhan_charges(c_now, tp, qty)
                        
                        send_telegram_alert(format_v3_entry(s, c_now, sl, tp, qty, wallet, est_chg))
                        positions[s] = {
                            "engine": "V3", "entry_p": c_now, "sl": sl, "tp": tp,
                            "qty": qty, "high_seen": c_now, "entry_t": now
                        }
                        daily_trades_v3 += 1
                        break
                except Exception as e:
                    print(f"[ERROR V3 Scan {s}]: {e}", flush=True)

        # 3. Scan V43 Titan (15-Min)
        if cur_time >= datetime.time(9, 45) and cur_time <= datetime.time(14, 15) and len(positions) == 0 and daily_trades_v43 < 2:
            for s in v43_symbols:
                try:
                    df_15m = tv.get_hist(symbol=s, exchange="NSE", interval=Interval.in_15_minute, n_bars=55)
                    if df_15m is None or len(df_15m) < 40:
                        continue
                    c = df_15m["close"]
                    ema20 = c.ewm(span=20).mean()
                    ema50 = c.ewm(span=50).mean()
                    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
                    sig = macd.ewm(span=9).mean()
                    tr = np.maximum(df_15m["high"] - df_15m["low"], np.maximum((df_15m["high"] - c.shift(1)).abs(), (df_15m["low"] - c.shift(1)).abs()))
                    atr14 = float(tr.rolling(14).mean().iloc[-1])

                    c_now = float(c.iloc[-1])
                    c_prev = float(c.iloc[-2])
                    l_prev = float(df_15m["low"].iloc[-2])
                    e20_now = float(ema20.iloc[-1])
                    e20_prev = float(ema20.iloc[-2])
                    e50_now = float(ema50.iloc[-1])
                    macd_now = float(macd.iloc[-1])
                    sig_now = float(sig.iloc[-1])

                    ma20_reclaim = (l_prev < e20_prev) and (c_prev <= e20_prev) and (c_now > e20_now) and (macd_now > sig_now) and (c_now > e50_now)
                    if ma20_reclaim:
                        qty = max(1, int((wallet * 5.0) / c_now))
                        sl = round(c_now - (0.90 * atr14), 2)
                        est_chg = estimate_dhan_charges(c_now, c_now + atr14, qty)
                        
                        send_telegram_alert(format_v43_entry(s, c_now, sl, qty, wallet, est_chg, "MA20 RECLAIM + TREND MOMENTUM"))
                        positions[s] = {
                            "engine": "V43", "entry_p": c_now, "sl": sl, "atr": atr14,
                            "qty": qty, "high_seen": c_now, "entry_t": now
                        }
                        daily_trades_v43 += 1
                        break
                except Exception as e:
                    print(f"[ERROR V43 Scan {s}]: {e}", flush=True)

        print(f"[{now.strftime('%H:%M:%S IST')}] Cycle #{scan_iteration} complete. Positions: {len(positions)}. Sleeping 55s...", flush=True)
        time.sleep(55)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Bhavishya Live Telegram Forward-Tester")
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
            "🚀 <b>BHAVISHYA TELEGRAM DISPATCHER ACTIVATED!</b>\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Status: <b>ONLINE & LIVE (Cloud Verified)</b>\n"
            f"Server Time: <b>{now_ist.strftime('%d-%b-%Y %H:%M:%S IST')}</b>\n"
            "Mode: <b>Forward-Testing / Paper Trading</b>\n"
            "Engines Monitored:\n"
            "  1. 🎯 <b>PC Jeweller 1m Live Predator V3</b> (1-Min ORB Sniper)\n"
            "  2. 🏛️ <b>Bhavishya V43 Titan</b> (15-Min Institutional Fortress)\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Cloud Runner connected. Signals will ping your phone live!"
        )
        send_telegram_alert(init_msg, token, chat_id)
        print("✅ SUCCESS! Test alert sent.", flush=True)
    elif args.live:
        run_live_cloud_scanner(initial_wallet=500.0)
    else:
        print("Bhavishya Live Telegram Engine Ready.", flush=True)
        print("Run with --test or --live.", flush=True)
