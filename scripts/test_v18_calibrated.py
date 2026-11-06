#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_v18_calibrated.py
======================
Simulates V18 with realistically calibrated microstructure physics:
- Price >= Rs 10.0, tick_to_atr <= 0.15 (blocks GTLINFRA/FCSSOFT, admits liquid leaders)
- 15m NATR >= 0.45% (healthy momentum above 0.18% Dhan friction)
- 15m median turnover >= Rs 10 Lakhs (daily turnover > Rs 2.5 Crore)
- Decoupled 0.75x ATR stop + 2-bar / 3-bar fast scratches
- 3-Gear Structural Corridor (rides 2-bar completed low)
- 3-Stage Concave Multi-Slot Spline (Max single trade risk capped at Rs 135)
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
import strategy_evaluate_gate_v2 as gate

# 1. Load Data for 20 Candidate Stocks
with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    univ = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_high_beta_10_15m_1y.pkl", "rb") as f:
    d_beta = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_10_15m_1y.pkl", "rb") as f:
    d_unseen = pickle.load(f)

UNIVERSE_20 = [
    "PCJEWELLER.NS", "RPOWER.NS", "DELTACORP.NS", "RITES.NS", "IDEA.NS",
    "MTNL.NS", "ZEEL.NS", "UTKARSHBNK.NS", "IFCI.NS", "NCC.NS",
    "NBCC.NS", "HFCL.NS", "MANAPPURAM.NS", "NATIONALUM.NS", "HUDCO.NS",
    "BHEL.NS", "SAIL.NS", "IRFC.NS", "NLCINDIA.NS", "CESC.NS"
]

all_dfs = {}
for sym in UNIVERSE_20:
    clean_sym = sym.replace(".NS", "").lower()
    indiv_file = ROOT_DIR / "data" / f"{clean_sym}_15m_1y.pkl"
    if sym in univ:
        all_dfs[sym] = univ[sym].copy()
    elif sym in d_beta:
        all_dfs[sym] = d_beta[sym].copy()
    elif sym in d_unseen:
        all_dfs[sym] = d_unseen[sym].copy()
    elif indiv_file.exists():
        with open(indiv_file, "rb") as f:
            all_dfs[sym] = pickle.load(f).copy()
    else:
        raise FileNotFoundError(f"Cannot find data for {sym}")

def compute_vwap_shifted(df: pd.DataFrame) -> pd.Series:
    df_temp = df.copy()
    df_temp["_date"] = df_temp.index.date
    tp = (df_temp["high"] + df_temp["low"] + df_temp["close"]) / 3.0
    df_temp["_tpv"] = tp * df_temp["volume"]
    cum_tpv = df_temp.groupby("_date")["_tpv"].cumsum()
    cum_vol = df_temp.groupby("_date")["volume"].cumsum()
    return (cum_tpv / (cum_vol + 1e-9)).shift(1)

def compute_er10(c: pd.Series) -> pd.Series:
    direction = (c - c.shift(10)).abs()
    volatility = c.diff().abs().rolling(10).sum()
    return direction / (volatility + 1e-9)

data_map = {}
for t in UNIVERSE_20:
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
    turnover20 = (c * v).rolling(20).median()
    er10 = compute_er10(c)
    
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
    
    # Base entry
    base_entry = (
        macd_cross_up & (macd > 0) &
        (c > ema50) & (c > ema110) &
        (c > vwap_lag) &
        (rsi7 > 50.0) & (adx > 18.0) & (plus_di > minus_di) &
        (c <= 450.0)
    )
    
    # Calibrated Microstructure Laws:
    law1 = (c >= 10.0) & (atr14 > 0) & ((0.05 / (atr14 + 1e-9)) <= 0.15)
    law2 = ((atr14 / (c + 1e-9)) * 100.0) >= 0.45
    law3 = turnover20 >= 1_000_000.0
    law4 = er10 >= 0.30
    
    admissions_gate = law1 & law2 & law3 & law4
    entry_cond = base_entry & admissions_gate
    
    er_bonus = 0.25 * np.maximum(0.0, er10 - 0.30)
    rvol_bonus = 0.05 * np.minimum(np.maximum(0.0, rvol - 1.5), 3.0)
    dynamic_weight = np.clip(0.85 + er_bonus + rvol_bonus, 0.85, 1.20)
    
    data_map[t] = {
        "df": df, "c": c, "h": h, "l": l, "v": v, "atr14": atr14,
        "rsi7": rsi7, "vwap_lag": vwap_lag, "rvol": rvol, "entry_cond": entry_cond,
        "dynamic_weight": dynamic_weight
    }

all_ts = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))
slippage = 0.0005

def get_v18_spline_risk_and_slots(
    wallet_equity: float,
    hwm_equity: float,
    consecutive_losses: int,
    conviction_weight: float
) -> tuple[float, int]:
    dd_pct = (hwm_equity - wallet_equity) / max(hwm_equity, 500.0)

    # Dynamic Slots: 1 slot below 2200, 2 slots up to 6500, 3 slots above 6500
    if wallet_equity < 2200.0 or consecutive_losses >= 2 or dd_pct > 0.14:
        max_slots = 1
    elif wallet_equity < 6500.0:
        max_slots = 2
    else:
        max_slots = 3

    # Streak-Modulated Spline Slopes
    if dd_pct > 0.14 or consecutive_losses >= 2:
        alpha, beta, gamma = 0.010, 0.0025, 0.0015
    elif consecutive_losses == 1:
        alpha, beta, gamma = 0.020, 0.0045, 0.0025
    else:
        alpha, beta, gamma = 0.034, 0.0075, 0.0040

    if wallet_equity <= 2500.0:
        raw_risk = 14.0 + alpha * max(0.0, wallet_equity - 500.0)
    elif wallet_equity <= 7500.0:
        r_2500 = 14.0 + alpha * 2000.0
        raw_risk = r_2500 + beta * (wallet_equity - 2500.0)
    else:
        r_2500 = 14.0 + alpha * 2000.0
        r_7500 = r_2500 + beta * 5000.0
        raw_risk = r_7500 + gamma * (wallet_equity - 7500.0)

    final_risk = min(raw_risk * conviction_weight, 135.0)
    return round(final_risk, 2), max_slots

positions = {}
trades = []
daily_traded = {}
last_exit_times = set()

print("\nRunning Calibrated V18 Simulation across 20 Stocks...")

for ts in all_ts:
    cur_date = ts.date()
    if cur_date not in daily_traded:
        daily_traded[cur_date] = set()
    t_time = ts.time()
    t_time_int = t_time.hour * 100 + t_time.minute
    
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
        cur_atr = t_data["atr14"].loc[ts]
        
        idx_now = t_data["df"].index.get_loc(ts)
        low_t1 = t_data["l"].iloc[idx_now - 1] if idx_now >= 1 else cur_l
        low_t2 = t_data["l"].iloc[idx_now - 2] if idx_now >= 2 else low_t1
        
        p["bars_held"] += 1
        p["high_seen"] = max(p["high_seen"], cur_h)
        mfe_atr = (p["high_seen"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0
        
        # --- STAGE 0: FAST TIME-DECAY SCRATCHES ---
        # Bar 2 VWAP failure scratch (~ -0.20R)
        if p["bars_held"] == 2 and mfe_atr < 0.35:
            if cur_c < cur_vwap or (cur_c < (p["entry_p"] - 0.15 * p["entry_atr"]) and cur_rsi < 49.0):
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
                    "exit_reason": "BAR2_VWAP_FAIL_SCRATCH"
                })
                last_exit_times.add(ts)
                del positions[ticker]
                continue
                
        # Bar 3 Stagnation scratch
        if p["bars_held"] >= 3 and mfe_atr < 0.45:
            if cur_c < p["entry_p"] and (cur_rsi < 47.0 or cur_c < cur_vwap):
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
                    "exit_reason": "BAR3_STAGNATION_SCRATCH"
                })
                last_exit_times.add(ts)
                del positions[ticker]
                continue
                
        # --- 3-GEAR STRUCTURAL CHANDELIER TRAIL ---
        if mfe_atr >= 0.75:
            p["active_sl"] = max(p["active_sl"], round(p["entry_p"] + (0.30 * p["entry_atr"]), 2))
            
        if mfe_atr >= 1.35:
            profit_floor = p["entry_p"] + (0.85 * p["entry_atr"])
            structural_2bar_low = min(low_t1, low_t2) - (0.08 * cur_atr)
            wide_chandelier = p["high_seen"] - (0.80 * cur_atr)
            p["active_sl"] = max(p["active_sl"], round(max(profit_floor, structural_2bar_low, wide_chandelier), 2))
            
        if mfe_atr >= 2.40 or (mfe_atr >= 1.35 and t_time_int >= 1415):
            climax_chandelier = p["high_seen"] - (0.38 * cur_atr)
            retention_floor = p["entry_p"] + 0.72 * (p["high_seen"] - p["entry_p"])
            p["active_sl"] = max(p["active_sl"], round(max(climax_chandelier, retention_floor), 2))
            
        # Hard Stop Hit Check
        if cur_l <= p["active_sl"]:
            raw_exit = p["active_sl"] * (1.0 - slippage)
            exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
            gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
            chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
            net = round(gross - chg, 2)
            reason = "GEAR_3_CLIMAX_STOP_HIT" if mfe_atr >= 2.40 else ("GEAR_2_STRUCTURAL_STOP_HIT" if mfe_atr >= 1.35 else ("GEAR_1_RATCHET_STOP_HIT" if mfe_atr >= 0.75 else "HARD_STOP_LOSS_HIT"))
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
            
        # 14:45 Intraday Hard Flatten
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
            
    _, max_allowed_slots = get_v18_spline_risk_and_slots(ek, hwm, consec_losses, 1.0)
    
    if len(positions) >= max_allowed_slots:
        continue
    if ts in last_exit_times:
        continue
    if t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:15").time():
        continue
        
    current_notional = sum(p["entry_p"] * p["qty"] for p in positions.values())
    current_margin_used = current_notional / 5.0
    avail_margin = (0.95 * ek) - current_margin_used
    if avail_margin < 10.0:
        continue
    avail_notional = avail_margin * 5.0
    
    for ticker in UNIVERSE_20:
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
        entry_atr = max(t_data["atr14"].iloc[idx], 0.05)
        raw_entry = cur_c * (1.0 + slippage)
        entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
        
        # Physical Execution Hard Stop: 0.75x ATR
        hard_sl = round(entry_p - (0.75 * entry_atr), 2)
        
        # Decoupled Sizing: Denominator uses 1.00x ATR
        conv_weight = t_data["dynamic_weight"].iloc[idx]
        target_risk, _ = get_v18_spline_risk_and_slots(ek, hwm, consec_losses, conv_weight)
        
        friction = 0.002067 * entry_p
        sizing_denominator = entry_atr + friction
        qty_risk = int(target_risk / sizing_denominator)
        
        qty_margin = int((avail_notional * 0.90) / entry_p)
        qty = min(qty_risk, qty_margin)
        
        if bar_vol > 0:
            qty_vol = int(0.042 * bar_vol)
            qty = min(qty, qty_vol)
            
        if qty >= 1:
            positions[ticker] = {
                "entry_p": entry_p, "entry_t": ts, "qty": qty,
                "entry_atr": entry_atr, "active_sl": hard_sl,
                "initial_sl": hard_sl, "high_seen": entry_p, "phase": 0, "bars_held": 0
            }
            daily_traded[cur_date].add(ticker)
            current_notional += (entry_p * qty)
            current_margin_used = current_notional / 5.0
            avail_margin = (0.95 * ek) - current_margin_used
            avail_notional = avail_margin * 5.0
            if avail_margin < 10.0 or len(positions) >= max_allowed_slots:
                break

df_v18 = pd.DataFrame(trades)
m = gate.metrics(df_v18)
path = 500.0 + np.cumsum(df_v18["net_pnl"].to_numpy(float))
peak = np.maximum.accumulate(np.maximum(path, 500.0))
mdd = float(((peak - path) / peak).max()) * 100.0 if len(path) else 0.0

total_gross_wins = df_v18[df_v18["gross_pnl"] > 0]["gross_pnl"].sum()
total_gross_losses = df_v18[df_v18["gross_pnl"] < 0]["gross_pnl"].sum()
total_charges = df_v18["charges"].sum()

print("\n=======================================================")
print("=== CALIBRATED V18 BACKTEST RESULTS (20 STOCKS) ===")
print("=======================================================")
print(f"Total Trades: {m['n']}")
print(f"Wins: {m['wins']} | Losses: {m['losses']} | Win Rate: {m['win_rate']:.2f}%")
print(f"Total Gross Wins: Rs {total_gross_wins:.2f}")
print(f"Total Gross Losses: Rs {total_gross_losses:.2f}")
print(f"Total Dhan Taxes & Charges: Rs {total_charges:.2f}")
print(f"Net Realized Profit: Rs {m['net']:.2f}")
print(f"Profit Factor: {m['pf']:.2f}")
print(f"Starting Capital: Rs 500.00")
print(f"Final Realized Equity: Rs {path[-1]:.2f} ({(path[-1]-500)/500*100:.2f}% ROI / {path[-1]/500:.2f}x)")
print(f"Peak HWM: Rs {float(peak.max()):.2f}")
print(f"Max Drawdown: {mdd:.2f}%")
print(f"Average Win: Rs {m['avg_win']:.2f}")
print(f"Average Loss: Rs {m['avg_loss']:.2f}")
print(f"Worst Single Loss: Rs {df_v18['net_pnl'].min():.2f}")
print(f"Largest Notional: Rs {(df_v18['entry_price'] * df_v18['shares']).max():.2f}")
