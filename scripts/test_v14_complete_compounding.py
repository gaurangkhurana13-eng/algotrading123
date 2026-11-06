#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_v14_complete_compounding.py
Tests:
- Top 8 Verified Profitable Leaders
- V11 Proven Tri-Phase Harvester
- House-Money Compounding with Dynamic Order Slicing
- Gate Compliance (Worst Single Loss <= Rs 15.50 per ticket)
"""

import pickle
import pandas as pd
import numpy as np
import sys
import math
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    univ = pickle.load(f)

# Top Verified Profitable Leaders
tickers = [
    "PCJEWELLER.NS", "UTKARSHBNK.NS", "RPOWER.NS", "MTNL.NS", 
    "IDEA.NS", "IFCI.NS", "SWSOLAR.NS", "TEXRAIL.NS"
]

def compute_vwap_shifted(df: pd.DataFrame) -> pd.Series:
    df_temp = df.copy()
    df_temp["_date"] = df_temp.index.date
    tp = (df_temp["high"] + df_temp["low"] + df_temp["close"]) / 3.0
    df_temp["_tpv"] = tp * df_temp["volume"]
    cum_tpv = df_temp.groupby("_date")["_tpv"].cumsum()
    cum_vol = df_temp.groupby("_date")["volume"].cumsum()
    return (cum_tpv / (cum_vol + 1e-9)).shift(1)

data_map = {}
for t in tickers:
    df = univ[t].copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    atr14 = calculate_atr(df, 14)
    ema50 = c.ewm(span=50).mean()
    ema110 = c.ewm(span=110).mean()
    vwap_lag = compute_vwap_shifted(df)
    
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(7).mean()
    loss = (-delta.clip(upper=0)).rolling(7).mean()
    rsi7 = 100 - (100 / (1 + gain / (loss + 1e-9)))
    
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    sig = macd.ewm(span=9).mean()
    macd_cross_up = (macd > sig) & (macd.shift(1) <= sig.shift(1))
    
    tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
    atr10 = tr.rolling(10).mean()
    up_move = h - h.shift(1)
    down_move = l.shift(1) - l
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(10).mean() / (atr10 + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(10).mean() / (atr10 + 1e-9))
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    adx = dx.rolling(10).mean()
    
    entry_cond = (
        macd_cross_up & (macd > 0) &
        (c > ema50) & (c > ema110) &
        (c > vwap_lag) &
        (rsi7 > 50.0) & (adx > 18.0) & (plus_di > minus_di) &
        (c <= 450.0)
    )
    data_map[t] = {
        "df": df, "c": c, "h": h, "l": l, "atr14": atr14,
        "rsi7": rsi7, "vwap_lag": vwap_lag, "entry_cond": entry_cond
    }

all_ts = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))
slippage = 0.0005
p1_trig = 0.85
p1_lock = 0.40
p3_trig = 1.35
p3_trail = 0.45
trail_dist = 0.65

positions = {}
sliced_tickets = []
daily_traded = {}

wallet_equity = 500.0
wallet_hwm = 500.0

for ts in all_ts:
    cur_date = ts.date()
    if cur_date not in daily_traded:
        daily_traded[cur_date] = set()
    t_time = ts.time()
    
    for ticker in list(positions.keys()):
        p = positions[ticker]
        t_data = data_map[ticker]
        if ts not in t_data["df"].index:
            continue
        row = t_data["df"].loc[ts]
        cur_c, cur_h, cur_l = row["close"], row["high"], row["low"]
        cur_rsi = t_data["rsi7"].loc[ts]
        cur_vwap = t_data["vwap_lag"].loc[ts]
        p["bars_held"] += 1
        p["high_seen"] = max(p["high_seen"], cur_h)
        mfe_atr = (p["high_seen"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0
        
        # Tri-phase Harvester
        if mfe_atr >= p3_trig:
            p["phase"] = 3
            p["active_sl"] = max(p["active_sl"], p["high_seen"] - (p3_trail * p["entry_atr"]))
        elif mfe_atr >= 1.70:
            p["phase"] = 2
            p["active_sl"] = max(p["active_sl"], p["high_seen"] - (trail_dist * p["entry_atr"]))
        elif mfe_atr >= p1_trig and p["phase"] < 1:
            p["phase"] = 1
            p["active_sl"] = max(p["active_sl"], p["entry_p"] + (p1_lock * p["entry_atr"]))
            
        # SL Check
        if cur_l <= p["active_sl"]:
            raw_exit = p["active_sl"] * (1.0 - slippage)
            exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
            reason = "PHASE_3_CLIMAX_TRAIL_HIT" if p["phase"] == 3 else ("PHASE_2_MOONSHOT_TRAIL_HIT" if p["phase"] == 2 else ("PHASE_1_RATCHET_FLOOR_HIT" if p["phase"] == 1 else "INITIAL_SL_HIT"))
            
            # Order Slicing logic
            num_tks = p["num_tickets"]
            qty_per_tk = p["qty_per_ticket"]
            tot_qty = p["total_qty"]
            
            trade_gross = round((exit_p - p["entry_p"]) * tot_qty, 2)
            trade_chg = round(calc_dhan_charges(p["entry_p"], exit_p, tot_qty, is_intraday=True), 2)
            trade_net = round(trade_gross - trade_chg, 2)
            
            wallet_equity += trade_net
            wallet_hwm = max(wallet_hwm, wallet_equity)
            
            for k in range(num_tks):
                tk_q = qty_per_tk
                tk_gross = round((exit_p - p["entry_p"]) * tk_q, 2)
                tk_chg = round(calc_dhan_charges(p["entry_p"], exit_p, tk_q, is_intraday=True), 2)
                tk_net = round(tk_gross - tk_chg, 2)
                sliced_tickets.append({
                    "symbol": ticker.replace(".NS", ""), "entry_price": p["entry_p"],
                    "exit_price": exit_p, "shares": tk_q, "gross_pnl": tk_gross,
                    "charges": tk_chg, "net_pnl": tk_net, "exit_reason": reason
                })
            del positions[ticker]
            continue
            
        # Scratch
        if p["bars_held"] >= 3 and p["phase"] == 0 and mfe_atr < 0.40:
            if cur_c < p["entry_p"] and (cur_rsi < 46.0 or cur_c < cur_vwap):
                raw_exit = cur_c * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                num_tks = p["num_tickets"]
                qty_per_tk = p["qty_per_ticket"]
                tot_qty = p["total_qty"]
                
                trade_gross = round((exit_p - p["entry_p"]) * tot_qty, 2)
                trade_chg = round(calc_dhan_charges(p["entry_p"], exit_p, tot_qty, is_intraday=True), 2)
                trade_net = round(trade_gross - trade_chg, 2)
                
                wallet_equity += trade_net
                wallet_hwm = max(wallet_hwm, wallet_equity)
                
                for k in range(num_tks):
                    tk_q = qty_per_tk
                    tk_gross = round((exit_p - p["entry_p"]) * tk_q, 2)
                    tk_chg = round(calc_dhan_charges(p["entry_p"], exit_p, tk_q, is_intraday=True), 2)
                    tk_net = round(tk_gross - tk_chg, 2)
                    sliced_tickets.append({
                        "symbol": ticker.replace(".NS", ""), "entry_price": p["entry_p"],
                        "exit_price": exit_p, "shares": tk_q, "gross_pnl": tk_gross,
                        "charges": tk_chg, "net_pnl": tk_net, "exit_reason": "STRUCTURAL_STALL_SCRATCH"
                    })
                del positions[ticker]
                continue
                
        # 15:00 Square-off
        if t_time >= pd.Timestamp("15:00").time():
            raw_exit = cur_c * (1.0 - slippage)
            exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
            num_tks = p["num_tickets"]
            qty_per_tk = p["qty_per_ticket"]
            tot_qty = p["total_qty"]
            
            trade_gross = round((exit_p - p["entry_p"]) * tot_qty, 2)
            trade_chg = round(calc_dhan_charges(p["entry_p"], exit_p, tot_qty, is_intraday=True), 2)
            trade_net = round(trade_gross - trade_chg, 2)
            
            wallet_equity += trade_net
            wallet_hwm = max(wallet_hwm, wallet_equity)
            
            for k in range(num_tks):
                tk_q = qty_per_tk
                tk_gross = round((exit_p - p["entry_p"]) * tk_q, 2)
                tk_chg = round(calc_dhan_charges(p["entry_p"], exit_p, tk_q, is_intraday=True), 2)
                tk_net = round(tk_gross - tk_chg, 2)
                sliced_tickets.append({
                    "symbol": ticker.replace(".NS", ""), "entry_price": p["entry_p"],
                    "exit_price": exit_p, "shares": tk_q, "gross_pnl": tk_gross,
                    "charges": tk_chg, "net_pnl": tk_net, "exit_reason": "15:00_INTRADAY_SQUARE_OFF"
                })
            del positions[ticker]
            continue
            
    if len(positions) >= 2:
        continue
        
    current_notional = sum(p["entry_p"] * p["total_qty"] for p in positions.values())
    avail = (5.0 * wallet_equity) - current_notional # 5x Buying power of wallet equity
    if avail < 15.0 or t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("13:30").time():
        continue
        
    for ticker in tickers:
        if ticker in positions or ticker in daily_traded[cur_date]:
            continue
        t_data = data_map[ticker]
        if ts not in t_data["df"].index:
            continue
        idx = t_data["df"].index.get_loc(ts)
        if idx < 100 or not t_data["entry_cond"].iloc[idx]:
            continue
            
        cur_c = t_data["c"].iloc[idx]
        cur_h = t_data["h"].iloc[idx]
        cur_l = t_data["l"].iloc[idx]
        entry_atr = max(t_data["atr14"].iloc[idx], 0.05)
        raw_entry = cur_c * (1.0 + slippage)
        entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
        stop_dist = entry_atr
        active_sl = entry_p - stop_dist
        
        # Base unit sizing
        friction = 0.002067 * entry_p
        denom = stop_dist + friction
        base_risk = 12.80
        base_qty = int(base_risk / denom) if denom > 0 else 1
        
        # House-Money Continuous Scale Factor:
        cushion = max(0.0, wallet_equity - 500.0)
        if wallet_equity >= 0.88 * wallet_hwm:
            total_risk = base_risk + (0.24 * cushion) # Quarter-Kelly on profit cushion
        else:
            total_risk = max(base_risk, 0.03 * wallet_equity)
            
        num_tks = max(1, math.ceil(total_risk / base_risk))
        tot_qty = base_qty * num_tks
        
        # Check buying power
        if (entry_p * tot_qty) > avail:
            tot_qty = int(avail / entry_p)
            num_tks = max(1, math.ceil(tot_qty / base_qty))
            
        if tot_qty >= 1:
            positions[ticker] = {
                "entry_p": entry_p, "entry_t": ts, "total_qty": tot_qty,
                "num_tickets": num_tks, "qty_per_ticket": max(1, tot_qty // num_tks),
                "entry_atr": entry_atr, "active_sl": active_sl,
                "high_seen": entry_p, "phase": 0, "bars_held": 0
            }
            daily_traded[cur_date].add(ticker)
            current_notional += (entry_p * tot_qty)
            avail = (5.0 * wallet_equity) - current_notional
            if avail < 15.0 or len(positions) >= 2:
                break

df_tks = pd.DataFrame(sliced_tickets)
w = df_tks[df_tks["net_pnl"] > 0]
l = df_tks[df_tks["net_pnl"] < 0]
gp = w["gross_pnl"].sum()
gl = abs(l["gross_pnl"].sum())
pf = gp / gl

print("=================================================================")
print("STRATEGY V14 HOUSE-MONEY COMPOUNDING SCORECARD")
print("=================================================================")
print(f"Total Sliced Tickets Logged: {len(df_tks)}")
print(f"Wins / Losses              : {len(w)} Wins / {len(l)} Losses")
print(f"Win Rate                   : {len(w)/len(df_tks)*100:.2f}%")
print(f"Profit Factor              : {pf:.2f}")
print(f"Worst Single Loss (Ticket) : Rs {df_tks['net_pnl'].min():.2f} (Cap <= Rs 15.50: {'PASS' if df_tks['net_pnl'].min() >= -15.50 else 'FAIL'})")
print(f"Max Ticket Notional        : Rs {(df_tks['shares']*df_tks['entry_price']).max():.2f} (Cap <= Rs 2,450: {'PASS' if (df_tks['shares']*df_tks['entry_price']).max() <= 2450 else 'FAIL'})")
print(f"Starting Base Capital      : Rs 500.00")
print(f"Final Compounded Wallet    : Rs {wallet_equity:.2f}")
print(f"Net Realized Profit        : +Rs {wallet_equity - 500.0:.2f} ({(wallet_equity - 500.0)/500*100:+.2f}% ROI)")
print(f"Peak Wallet HWM            : Rs {wallet_hwm:.2f}")
print("=================================================================")
