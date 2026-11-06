#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
find_optimal_basket.py
Evaluate every stock in the 78-stock universe under V11/V13 Tri-Phase Harvester rules.
Sort by Net PnL, Win Rate, and Profit Factor.
"""

import sys
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
sys.path.insert(0, str(ROOT_DIR / "strategies" / "factory"))

from framework import calc_dhan_charges, calculate_atr

with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    univ = pickle.load(f)

def compute_vwap_shifted(df: pd.DataFrame) -> pd.Series:
    df_temp = df.copy()
    df_temp["_date"] = df_temp.index.date
    tp = (df_temp["high"] + df_temp["low"] + df_temp["close"]) / 3.0
    df_temp["_tpv"] = tp * df_temp["volume"]
    cum_tpv = df_temp.groupby("_date")["_tpv"].cumsum()
    cum_vol = df_temp.groupby("_date")["volume"].cumsum()
    return (cum_tpv / (cum_vol + 1e-9)).shift(1)

def test_single_stock(ticker, df):
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]
    
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
    
    v11_entry = (
        macd_cross_up &
        (macd > 0) &
        (c > ema50) &
        (c > ema110) &
        (c > vwap_lag) &
        (rsi7 > 50.0) &
        (adx > 18.0) &
        (plus_di > minus_di) &
        (c <= 450.0)
    )
    
    slippage = 0.0005
    sl_mult = 1.00
    p1_trig = 0.85
    p1_lock = 0.40
    p3_trig = 1.35
    p3_trail = 0.45
    trail_dist = 0.65
    
    pos = None
    trades = []
    
    for idx in range(100, len(df)):
        ts = df.index[idx]
        t_time = ts.time()
        cur_c = c.iloc[idx]
        cur_h = h.iloc[idx]
        cur_l = l.iloc[idx]
        cur_rsi = rsi7.iloc[idx]
        cur_vwap = vwap_lag.iloc[idx]
        
        if pos is not None:
            pos["bars_held"] += 1
            if t_time >= pd.Timestamp("15:10").time():
                raw_exit = cur_c * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                gross = round((exit_p - pos["entry_p"]) * pos["qty"], 2)
                chg = round(calc_dhan_charges(pos["entry_p"], exit_p, pos["qty"], is_intraday=True), 2)
                trades.append({"gross": gross, "chg": chg, "net": round(gross - chg, 2)})
                pos = None
                continue
                
            pos["high_seen"] = max(pos["high_seen"], cur_h)
            mfe_atr = (pos["high_seen"] - pos["entry_p"]) / pos["entry_atr"] if pos["entry_atr"] > 0 else 0.0
            
            if pos["bars_held"] >= 3 and pos["phase"] == 0 and mfe_atr < 0.40:
                if cur_c < pos["entry_p"] and (cur_rsi < 46.0 or cur_c < cur_vwap):
                    raw_exit = cur_c * (1.0 - slippage)
                    exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                    gross = round((exit_p - pos["entry_p"]) * pos["qty"], 2)
                    chg = round(calc_dhan_charges(pos["entry_p"], exit_p, pos["qty"], is_intraday=True), 2)
                    trades.append({"gross": gross, "chg": chg, "net": round(gross - chg, 2)})
                    pos = None
                    continue
                    
            if mfe_atr >= p3_trig:
                pos["phase"] = 3
                pos["active_sl"] = max(pos["active_sl"], pos["high_seen"] - (p3_trail * pos["entry_atr"]))
            elif mfe_atr >= 1.70:
                pos["phase"] = 2
                pos["active_sl"] = max(pos["active_sl"], pos["high_seen"] - (trail_dist * pos["entry_atr"]))
            elif mfe_atr >= p1_trig and pos["phase"] < 1:
                pos["phase"] = 1
                pos["active_sl"] = max(pos["active_sl"], pos["entry_p"] + (p1_lock * pos["entry_atr"]))
                
            if cur_l <= pos["active_sl"]:
                raw_exit = pos["active_sl"] * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                gross = round((exit_p - pos["entry_p"]) * pos["qty"], 2)
                chg = round(calc_dhan_charges(pos["entry_p"], exit_p, pos["qty"], is_intraday=True), 2)
                trades.append({"gross": gross, "chg": chg, "net": round(gross - chg, 2)})
                pos = None
                continue
                
        # Entry
        if pos is None:
            time_ok = (t_time >= pd.Timestamp("09:20").time()) and (t_time <= pd.Timestamp("13:30").time())
            if time_ok and v11_entry.iloc[idx]:
                entry_atr = max(atr14.iloc[idx], 0.05)
                raw_entry = cur_c * (1.0 + slippage)
                entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
                stop_dist = sl_mult * entry_atr
                friction_share = 0.002067 * entry_p
                denom = stop_dist + friction_share
                qty_risk = int(13.80 / denom) if denom > 0 else 1
                qty_margin = int(1225.0 / entry_p) # half pool
                qty = min(qty_risk, qty_margin)
                if qty >= 1:
                    pos = {
                        "entry_p": entry_p, "qty": qty, "entry_atr": entry_atr,
                        "active_sl": entry_p - stop_dist, "high_seen": entry_p,
                        "phase": 0, "bars_held": 0
                    }
                    
    df_t = pd.DataFrame(trades)
    if df_t.empty:
        return {"ticker": ticker, "trades": 0, "net_pnl": 0.0, "wr": 0.0, "pf": 0.0, "worst_loss": 0.0}
    w = df_t[df_t["net"] > 0]
    l = df_t[df_t["net"] < 0]
    gp = w["gross"].sum()
    gl = abs(l["gross"].sum())
    pf = (gp / gl) if gl > 0 else 999.0
    return {
        "ticker": ticker.replace(".NS", ""),
        "trades": len(df_t),
        "net_pnl": round(df_t["net"].sum(), 2),
        "wr": round(len(w) / len(df_t) * 100, 1),
        "pf": round(pf, 2),
        "worst_loss": round(df_t["net"].min(), 2)
    }

print("Auditing all 78 tickers individually...")
results = []
for ticker, df in univ.items():
    res = test_single_stock(ticker, df)
    results.append(res)

df_res = pd.DataFrame(results).sort_values(by="net_pnl", ascending=False)
print("\n=== TOP 20 PROFITABLE STOCKS ===")
print(df_res.head(20).to_string(index=False))

print("\n=== WORST 10 LOSS-MAKING STOCKS ===")
print(df_res.tail(10).to_string(index=False))

profitable = df_res[df_res["net_pnl"] > 0]
print(f"\nTotal Profitable Stocks: {len(profitable)} / {len(df_res)}")
print(f"Total Net PnL of all Profitable Stocks combined: Rs {profitable['net_pnl'].sum():.2f}")
print(f"Total Trades from Profitable Stocks: {profitable['trades'].sum()}")
