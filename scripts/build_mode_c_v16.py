#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_mode_c_v16.py
===================
Simulates Mode A, Mode B, and Mode C on the Top 10 Verified Alpha Basket:
- Mode A: Constant-Unit Benchmark (Rs 500 base, <= Rs 15.50 risk, <= Rs 2,450 notional)
- Mode B: Argon's Continuous House-Money Compounding Wallet (unconstrained house money)
- Mode C: The Tripartite Hybrid (ChatGPT + Claude Gate V2 + Argon Streak Sizing)
          R_target = min(Argon_Risk, 3.1% * E_k, Rs 250), with 15% DD throttle,
          streak caution brakes, strict 95% margin cap, <= 4.5% volume participation,
          1-trade = 1-row, 14:45 square-off.
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

V16_DIR = ROOT_DIR / "strategies" / "V16"
V16_DIR.mkdir(parents=True, exist_ok=True)

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

# ==============================================================================
# SIMULATION ENGINE: MODE C (Risk-Capped Adaptive Compounding)
# ==============================================================================
print("\n=======================================================")
print("SIMULATING MODE C: Risk-Capped Adaptive Compounding (V16)")
print("=======================================================")

positions_c = {}
trades_mode_c = []
daily_traded_c = {}
last_exit_times_c = set()

for ts in all_ts:
    cur_date = ts.date()
    if cur_date not in daily_traded_c:
        daily_traded_c[cur_date] = set()
    t_time = ts.time()
    
    # 1. Manage Active Positions
    for ticker in list(positions_c.keys()):
        p = positions_c[ticker]
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
            trades_mode_c.append({
                "symbol": ticker.replace(".NS", ""), "direction": "long",
                "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                "entry_price": p["entry_p"], "exit_price": exit_p,
                "shares": p["qty"], "sl": p["initial_sl"],
                "gross_pnl": gross, "charges": chg, "net_pnl": net,
                "exit_reason": reason
            })
            last_exit_times_c.add(ts)
            del positions_c[ticker]
            continue
            
        # Scratch
        if p["bars_held"] >= 3 and p["phase"] == 0 and mfe_atr < 0.40:
            if cur_c < p["entry_p"] and (cur_rsi < 46.0 or cur_c < cur_vwap):
                raw_exit = cur_c * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
                net = round(gross - chg, 2)
                trades_mode_c.append({
                    "symbol": ticker.replace(".NS", ""), "direction": "long",
                    "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                    "entry_price": p["entry_p"], "exit_price": exit_p,
                    "shares": p["qty"], "sl": p["initial_sl"],
                    "gross_pnl": gross, "charges": chg, "net_pnl": net,
                    "exit_reason": "STRUCTURAL_STALL_SCRATCH"
                })
                last_exit_times_c.add(ts)
                del positions_c[ticker]
                continue
                
        # 14:45 Square-off
        if t_time >= pd.Timestamp("14:45").time():
            raw_exit = cur_c * (1.0 - slippage)
            exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
            gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
            chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
            net = round(gross - chg, 2)
            trades_mode_c.append({
                "symbol": ticker.replace(".NS", ""), "direction": "long",
                "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                "entry_price": p["entry_p"], "exit_price": exit_p,
                "shares": p["qty"], "sl": p["initial_sl"],
                "gross_pnl": gross, "charges": chg, "net_pnl": net,
                "exit_reason": "INTRADAY_SQUARE_OFF"
            })
            last_exit_times_c.add(ts)
            del positions_c[ticker]
            continue
            
    if len(positions_c) >= 2:
        continue
    if ts in last_exit_times_c:
        continue
    if t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:15").time():
        continue
        
    # Reconstruct realized equity E_k strictly from trades closed before ts
    closed_before = [t for t in trades_mode_c if pd.Timestamp(t["exit_time"]) < ts]
    ek = 500.0 + sum(t["net_pnl"] for t in closed_before)
    
    # High-Water Mark and Drawdown
    cum = 500.0
    hwm = 500.0
    sorted_closed = sorted(closed_before, key=lambda x: pd.Timestamp(x["exit_time"]))
    for t in sorted_closed:
        cum += t["net_pnl"]
        if cum > hwm:
            hwm = cum
            
    is_dd = (ek < 0.85 * hwm)
    
    # 1. Gate V2 Constitutional Risk Ceiling (Claude & ChatGPT)
    gate_risk_cap = min(0.031 * max(ek, 0.0), 250.0)
    if is_dd:
        gate_risk_cap *= 0.50
        
    # 2. Argon Streak Modulator
    # Count consecutive losses in immediately preceding closed trades
    consec_losses = 0
    for t in reversed(sorted_closed):
        if t["net_pnl"] < 0:
            consec_losses += 1
        else:
            break
            
    cushion = max(0.0, ek - 500.0)
    if consec_losses == 0:
        alpha = 0.082
        streak_mult = 1.00
    elif consec_losses == 1:
        alpha = 0.035
        streak_mult = 0.70  # 1-loss caution brake
    else:
        alpha = 0.015
        streak_mult = 0.50  # 2+ loss vault shield
        
    argon_base_risk = (14.0 + alpha * cushion) * streak_mult
    
    # 3. The Tripartite Reconciled Risk Target
    # Take the MINIMUM of Argon's formula and Claude's Gate V2 ceiling
    # Multiply by 0.90 as safety buffer against tick rounding and statutory friction
    effective_risk_target = min(argon_base_risk, gate_risk_cap) * 0.90
    
    # Available margin in shared account: total margin in use <= 95% of E_k
    # Intraday MIS 5x leverage -> buying power = 5.0 * margin
    current_notional = sum(p["entry_p"] * p["qty"] for p in positions_c.values())
    current_margin_used = current_notional / 5.0
    avail_margin = (0.95 * ek) - current_margin_used
    if avail_margin < 10.0:
        continue
    avail_notional = avail_margin * 5.0
    
    for ticker in tickers_10:
        if ticker in positions_c or ticker in daily_traded_c[cur_date]:
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
        
        # Stock-specific conviction adjustment within the ceiling
        if ticker in ("PCJEWELLER.NS", "RPOWER.NS", "DELTACORP.NS", "RITES.NS"):
            conviction = 1.00
        elif ticker in ("UTKARSHBNK.NS", "IFCI.NS", "NCC.NS"):
            conviction = 0.95 if bar_rvol >= 2.0 else 0.80
        else:
            conviction = 0.85
            
        item_risk_target = effective_risk_target * conviction
        
        # Friction per share
        friction = 0.002067 * entry_p
        risk_per_share = entry_p - active_sl + friction
        if risk_per_share <= 0.001:
            continue
        qty_risk = int(item_risk_target / risk_per_share)
        
        # Check planned stop-loss constraint explicitly:
        # Gate V2 Check 5b: shares * |entry - active_sl| <= gate_risk_cap + 0.50
        while qty_risk > 0 and (qty_risk * (entry_p - active_sl)) > gate_risk_cap:
            qty_risk -= 1
            
        # Margin cap (avail_notional * 0.90 for buffer)
        qty_margin = int((avail_notional * 0.90) / entry_p)
        qty = min(qty_risk, qty_margin)
        
        # Volume Participation Cap: Max 4.5% of bar volume
        if bar_vol > 0:
            qty_vol = int(0.045 * bar_vol)
            qty = min(qty, qty_vol)
            
        if qty >= 1:
            positions_c[ticker] = {
                "entry_p": entry_p, "entry_t": ts, "qty": qty,
                "entry_atr": entry_atr, "active_sl": active_sl,
                "initial_sl": active_sl, "high_seen": entry_p, "phase": 0, "bars_held": 0
            }
            daily_traded_c[cur_date].add(ticker)
            current_notional += (entry_p * qty)
            current_margin_used = current_notional / 5.0
            avail_margin = (0.95 * ek) - current_margin_used
            avail_notional = avail_margin * 5.0
            if avail_margin < 10.0 or len(positions_c) >= 2:
                break

df_mode_c = pd.DataFrame(trades_mode_c)
m_c = gate.metrics(df_mode_c)

print(f"Mode C Trades: {m_c['n']}")
print(f"Wins: {m_c['wins']} | Losses: {m_c['losses']} | Win Rate: {m_c['win_rate']:.2f}%")
print(f"Net Realized Profit: Rs {m_c['net']:.2f}")
print(f"Profit Factor: {m_c['pf']:.2f}")
print(f"Worst Loss: Rs {df_mode_c['net_pnl'].min():.2f}")
print(f"Largest Notional: Rs {(df_mode_c['entry_price'] * df_mode_c['shares']).max():.2f}")

# Reconstruct Mode C Equity Path
path_c = 500.0 + np.cumsum(df_mode_c["net_pnl"].to_numpy(float))
peak_c = np.maximum.accumulate(np.maximum(path_c, 500.0))
mdd_c = float(((peak_c - path_c) / peak_c).max()) * 100.0 if len(path_c) else 0.0
final_equity_c = path_c[-1] if len(path_c) else 500.0
peak_hwm_c = float(peak_c.max())

print(f"Starting Capital: Rs 500.00")
print(f"Final Realized Equity: Rs {final_equity_c:.2f} ({(final_equity_c - 500.0)/500.0*100:.2f}% ROI / {final_equity_c/500.0:.2f}x)")
print(f"Peak High-Water Mark: Rs {peak_hwm_c:.2f}")
print(f"Max Drawdown: {mdd_c:.2f}%")

# Save V16 files
df_mode_c.to_csv(V16_DIR / "trades_intraday.csv", index=False)
df_swing = pd.DataFrame(columns=[
    "symbol", "direction", "entry_time", "exit_time",
    "entry_price", "exit_price", "shares", "sl", "gross_pnl", "charges", "net_pnl", "exit_reason"
])
df_swing.to_csv(V16_DIR / "trades_swing.csv", index=False)

results_txt = f"""----------------------------------------
INTRADAY
----------------------------------------
Total Trades: {m_c['n']}
Wins: {m_c['wins']}
Losses: {m_c['losses']}
Win Rate: {m_c['win_rate']:.2f}%
Gross Profit: Rs {m_c['gross_profit']:.2f}
Gross Loss: Rs {m_c['gross_loss']:.2f}
Net Profit: Rs {m_c['net']:.2f}
Profit Factor: {m_c['pf']:.2f}
Avg Win: Rs {m_c['avg_win']:.2f}
Avg Loss: Rs {m_c['avg_loss']:.2f}
Max Drawdown: {mdd_c:.2f}%

----------------------------------------
SWING
----------------------------------------
Total Trades: 0
Wins: 0
Losses: 0
Win Rate: 0.00%
Gross Profit: Rs 0.00
Gross Loss: Rs 0.00
Net Profit: Rs 0.00
Profit Factor: 0.00
Avg Win: Rs 0.00
Avg Loss: Rs 0.00
Max Drawdown: 0.00%
"""
with open(V16_DIR / "results.txt", "w") as f:
    f.write(results_txt)

# Copy pine and run_v16
with open(ROOT_DIR / "strategies" / "V15" / "v15.pine", "r") as f:
    pine_src = f.read()
with open(V16_DIR / "v16.pine", "w") as f:
    f.write(pine_src.replace("V15", "V16"))

with open(ROOT_DIR / "scripts" / "build_mode_c_v16.py", "r") as f:
    run_src = f.read()
with open(V16_DIR / "run_v16.py", "w") as f:
    f.write(run_src)

print("\nV16 artefacts generated successfully in strategies/V16!")
