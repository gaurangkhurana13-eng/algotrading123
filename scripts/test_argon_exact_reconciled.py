#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_argon_exact_reconciled.py
==============================
Tests Argon's EXACT reconciled sizing function:
- Win streak: alpha=0.075, pct_cap=0.0425 (4.25%)
- 1-loss brake: alpha=0.032, pct_cap=0.0310 (3.10%)
- 2-loss / DD>15%: alpha=0.015, pct_cap=0.0175 (1.75%)
- Stock weights: 1.12 for Tier-S, 1.08/0.85 for Tier-A, 1.00 for Tier-B
- 65% / 35% concurrent buying power sharing
"""
import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr
import strategy_evaluate_gate_v2 as gate

# 1. Load Data for Top 10 Verified Basket
with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    univ = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_high_beta_10_15m_1y.pkl", "rb") as f:
    d_beta = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_10_15m_1y.pkl", "rb") as f:
    d_unseen = pickle.load(f)

tickers_10 = [
    "PCJEWELLER.NS", "RPOWER.NS", "DELTACORP.NS", "RITES.NS",
    "IDEA.NS", "MTNL.NS", "ZEEL.NS",
    "UTKARSHBNK.NS", "IFCI.NS", "NCC.NS"
]

all_dfs = {}
for t in ["PCJEWELLER.NS", "RPOWER.NS", "IDEA.NS", "MTNL.NS", "UTKARSHBNK.NS", "IFCI.NS"]:
    all_dfs[t] = univ[t].copy()
all_dfs["DELTACORP.NS"] = d_beta["DELTACORP.NS"].copy()
all_dfs["RITES.NS"] = d_beta["RITES.NS"].copy()
all_dfs["NCC.NS"] = d_beta["NCC.NS"].copy()
all_dfs["ZEEL.NS"] = d_unseen["ZEEL.NS"].copy()

def compute_vwap_shifted(df: pd.DataFrame) -> pd.Series:
    df_temp = df.copy()
    df_temp["_date"] = df_temp.index.date
    tp = (df_temp["high"] + df_temp["low"] + df_temp["close"]) / 3.0
    df_temp["_tpv"] = tp * df_temp["volume"]
    cum_tpv = df_temp.groupby("_date")["_tpv"].cumsum()
    cum_vol = df_temp.groupby("_date")["volume"].cumsum()
    return (cum_tpv / (cum_vol + 1e-9)).shift(1)

data_map = {}
for t in tickers_10:
    df = all_dfs[t]
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]
    atr14 = calculate_atr(df, 14)
    ema50 = c.ewm(span=50).mean()
    ema110 = c.ewm(span=110).mean()
    vwap_lag = compute_vwap_shifted(df)
    v_sma20 = v.rolling(20).mean()
    rvol = v / (v_sma20 + 1e-9)
    
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
        "df": df, "c": c, "h": h, "l": l, "v": v, "atr14": atr14,
        "rsi7": rsi7, "vwap_lag": vwap_lag, "rvol": rvol, "entry_cond": entry_cond
    }

all_ts = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))
slippage = 0.0005
p1_trig = 0.85
p1_lock = 0.40
p3_trig = 1.35
p3_trail = 0.45
trail_dist = 0.65

# Argon's exact reconciled risk function
def get_v15_reconciled_risk(
    wallet_equity: float,
    hwm_equity: float,
    consecutive_losses: int,
    stock: str,
    rvol: float
) -> float:
    cushion = max(0.0, wallet_equity - 500.0)
    current_dd = (hwm_equity - wallet_equity) / max(hwm_equity, 500.0)

    # 1. Drawdown Override + Streak-Modulated House-Money Alpha
    if current_dd > 0.15 or consecutive_losses >= 2:
        alpha = 0.015
        pct_cap = 0.0175
    elif consecutive_losses == 1:
        alpha = 0.032
        pct_cap = 0.0310
    else:
        alpha = 0.075
        pct_cap = 0.0425

    raw_risk = 14.0 + (alpha * cushion)

    # 2. Stock-Specific Conviction Weighting
    if stock in ("PCJEWELLER.NS", "RPOWER.NS", "DELTACORP.NS"):
        weight = 1.12
    elif stock in ("UTKARSHBNK.NS", "IFCI.NS", "RITES.NS", "NCC.NS"):
        weight = 1.08 if rvol >= 2.2 else 0.85
    else:
        weight = 1.00

    weighted_risk = raw_risk * weight

    # 3. Institutional Ceiling
    max_allowed_risk = min(max(14.0, wallet_equity * pct_cap), 250.0)
    return round(min(weighted_risk, max_allowed_risk), 2)

positions = {}
trades = []
daily_traded = {}
last_exit_times = set()

for ts in all_ts:
    cur_date = ts.date()
    if cur_date not in daily_traded:
        daily_traded[cur_date] = set()
    t_time = ts.time()
    
    # 1. Manage Active Positions
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
        
        # Harvester
        if mfe_atr >= p3_trig:
            p["phase"] = 3
            p["active_sl"] = max(p["active_sl"], round(p["high_seen"] - (p3_trail * p["entry_atr"]), 2))
        elif mfe_atr >= 1.70:
            p["phase"] = 2
            p["active_sl"] = max(p["active_sl"], round(p["high_seen"] - (trail_dist * p["entry_atr"]), 2))
        elif mfe_atr >= p1_trig and p["phase"] < 1:
            p["phase"] = 1
            p["active_sl"] = max(p["entry_p"] + (p1_lock * p["entry_atr"]), round(p["active_sl"], 2))
            
        # SL Hit Check
        if cur_l <= p["active_sl"]:
            raw_exit = p["active_sl"] * (1.0 - slippage)
            exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
            gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
            chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
            net = round(gross - chg, 2)
            reason = "PHASE_3_TRAILING_STOP_HIT" if p["phase"] == 3 else ("PHASE_2_TRAILING_STOP_HIT" if p["phase"] == 2 else ("PHASE_1_RATCHET_STOP_HIT" if p["phase"] == 1 else "STOP_LOSS_HIT"))
            trades.append({
                "symbol": ticker.replace(".NS", ""), "direction": "long",
                "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                "entry_price": p["entry_p"], "exit_price": exit_p,
                "shares": p["qty"], "sl": p["initial_sl"],
                "gross_pnl": gross, "charges": chg, "net_pnl": net,
                "exit_reason": reason
            })
            last_exit_times.add(ts)
            del positions[ticker]
            continue
            
        # Scratch
        if p["bars_held"] >= 3 and p["phase"] == 0 and mfe_atr < 0.40:
            if cur_c < p["entry_p"] and (cur_rsi < 46.0 or cur_c < cur_vwap):
                raw_exit = cur_c * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
                net = round(gross - chg, 2)
                trades.append({
                    "symbol": ticker.replace(".NS", ""), "direction": "long",
                    "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                    "entry_price": p["entry_p"], "exit_price": exit_p,
                    "shares": p["qty"], "sl": p["initial_sl"],
                    "gross_pnl": gross, "charges": chg, "net_pnl": net,
                    "exit_reason": "STRUCTURAL_STALL_SCRATCH"
                })
                last_exit_times.add(ts)
                del positions[ticker]
                continue
                
        # 14:45 Square-off
        if t_time >= pd.Timestamp("14:45").time():
            raw_exit = cur_c * (1.0 - slippage)
            exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
            gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
            chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
            net = round(gross - chg, 2)
            trades.append({
                "symbol": ticker.replace(".NS", ""), "direction": "long",
                "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                "entry_price": p["entry_p"], "exit_price": exit_p,
                "shares": p["qty"], "sl": p["initial_sl"],
                "gross_pnl": gross, "charges": chg, "net_pnl": net,
                "exit_reason": "INTRADAY_SQUARE_OFF"
            })
            last_exit_times.add(ts)
            del positions[ticker]
            continue
            
    if len(positions) >= 2:
        continue
    if ts in last_exit_times:
        continue
    if t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:15").time():
        continue
        
    closed_before = [t for t in trades if pd.Timestamp(t["exit_time"]) < ts]
    ek = 500.0 + sum(t["net_pnl"] for t in closed_before)
    
    cum = 500.0
    hwm = 500.0
    sorted_closed = sorted(closed_before, key=lambda x: pd.Timestamp(x["exit_time"]))
    for t in sorted_closed:
        cum += t["net_pnl"]
        if cum > hwm:
            hwm = cum
            
    consec_losses = 0
    for t in reversed(sorted_closed):
        if t["net_pnl"] < 0:
            consec_losses += 1
        else:
            break
            
    # Margin Buying Power
    current_notional = sum(p["entry_p"] * p["qty"] for p in positions.values())
    current_margin_used = current_notional / 5.0
    avail_margin = (0.95 * ek) - current_margin_used
    if avail_margin < 10.0:
        continue
        
    # If this is the 1st position of the candle, allow up to 65% of available notional; if 2nd position, use remaining
    if len(positions) == 0:
        avail_notional = avail_margin * 5.0 * 0.65
    else:
        avail_notional = avail_margin * 5.0
        
    for ticker in tickers_10:
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
        bar_vol = t_data["v"].iloc[idx]
        bar_rvol = t_data["rvol"].iloc[idx]
        entry_atr = max(t_data["atr14"].iloc[idx], 0.05)
        raw_entry = cur_c * (1.0 + slippage)
        entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
        stop_dist = entry_atr
        active_sl = round(entry_p - stop_dist, 2)
        
        target_risk = get_v15_reconciled_risk(ek, hwm, consec_losses, ticker, bar_rvol) * 0.90
        
        friction = 0.002067 * entry_p
        risk_per_share = entry_p - active_sl + friction
        if risk_per_share <= 0.001:
            continue
        qty_risk = int(target_risk / risk_per_share)
        
        qty_margin = int((avail_notional * 0.90) / entry_p)
        qty = min(qty_risk, qty_margin)
        
        if bar_vol > 0:
            qty_vol = int(0.045 * bar_vol)
            qty = min(qty, qty_vol)
            
        if qty >= 1:
            positions[ticker] = {
                "entry_p": entry_p, "entry_t": ts, "qty": qty,
                "entry_atr": entry_atr, "active_sl": active_sl,
                "initial_sl": active_sl, "high_seen": entry_p, "phase": 0, "bars_held": 0
            }
            daily_traded[cur_date].add(ticker)
            current_notional += (entry_p * qty)
            current_margin_used = current_notional / 5.0
            avail_margin = (0.95 * ek) - current_margin_used
            avail_notional = avail_margin * 5.0
            if avail_margin < 10.0 or len(positions) >= 2:
                break

df_res = pd.DataFrame(trades)
m = gate.metrics(df_res)
path = 500.0 + np.cumsum(df_res["net_pnl"].to_numpy(float))
peak = np.maximum.accumulate(np.maximum(path, 500.0))
mdd = float(((peak - path) / peak).max()) * 100.0 if len(path) else 0.0

print("\n=== RESULTS OF ARGON'S EXACT RECONCILED SIZING ===")
print(f"Total Trades: {m['n']}")
print(f"Wins: {m['wins']} | Losses: {m['losses']} | Win Rate: {m['win_rate']:.2f}%")
print(f"Net Realized Profit: Rs {m['net']:.2f}")
print(f"Profit Factor: {m['pf']:.2f}")
print(f"Starting Capital: Rs 500.00")
print(f"Final Realized Equity: Rs {path[-1]:.2f} ({(path[-1]-500)/500*100:.2f}% ROI / {path[-1]/500:.2f}x)")
print(f"Peak HWM: Rs {float(peak.max()):.2f}")
print(f"Max Drawdown: {mdd:.2f}%")
print(f"Worst Single Loss: Rs {df_res['net_pnl'].min():.2f}")
print(f"Largest Notional: Rs {(df_res['entry_price'] * df_res['shares']).max():.2f}")
