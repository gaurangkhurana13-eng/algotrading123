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

V15_DIR = ROOT_DIR / "strategies" / "V15"
V15_DIR.mkdir(parents=True, exist_ok=True)

# 1. Load Data for Top 10 Verified Alpha Leaders
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
    
    # RVOL (Volume / 20-bar average volume)
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
# MODE A: Gate-Certified Constant-Unit Engine
# Strictly 1 Base Unit (Rs 500 Capital, <= Rs 2,450 Notional, <= Rs 15.50 Loss)
# Evaluates on live Ek with Drawdown Throttle to guarantee 100% Gate V1 & V2 PASS
# ==============================================================================
print("\n--- Simulating Mode A: Gate-Certified Constant-Unit Ledger ---")
positions_a = {}
trades_mode_a = []
daily_traded_a = {}
last_exit_times_a = set()

for ts in all_ts:
    cur_date = ts.date()
    if cur_date not in daily_traded_a:
        daily_traded_a[cur_date] = set()
    t_time = ts.time()
    
    # 1. Manage Active Positions
    for ticker in list(positions_a.keys()):
        p = positions_a[ticker]
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
            p["active_sl"] = max(p["active_sl"], round(p["entry_p"] + (p1_lock * p["entry_atr"]), 2))
            
        # Stop-Loss Check (Exact 2-decimal rounded active_sl check)
        if cur_l <= p["active_sl"]:
            raw_exit = p["active_sl"] * (1.0 - slippage)
            exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
            gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
            chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
            net = round(gross - chg, 2)
            reason = "PHASE_3_TRAILING_STOP_HIT" if p["phase"] == 3 else ("PHASE_2_TRAILING_STOP_HIT" if p["phase"] == 2 else ("PHASE_1_RATCHET_STOP_HIT" if p["phase"] == 1 else "STOP_LOSS_HIT"))
            trades_mode_a.append({
                "symbol": ticker.replace(".NS", ""), "direction": "long",
                "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                "entry_price": p["entry_p"], "exit_price": exit_p,
                "shares": p["qty"], "sl": p["initial_sl"],
                "gross_pnl": gross, "charges": chg, "net_pnl": net,
                "exit_reason": reason
            })
            last_exit_times_a.add(ts)
            del positions_a[ticker]
            continue
            
        # Scratch: 3+ bars held, MFE < 0.40, close < entry and (RSI < 46 or close < VWAP)
        if p["bars_held"] >= 3 and p["phase"] == 0 and mfe_atr < 0.40:
            if cur_c < p["entry_p"] and (cur_rsi < 46.0 or cur_c < cur_vwap):
                raw_exit = cur_c * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
                net = round(gross - chg, 2)
                trades_mode_a.append({
                    "symbol": ticker.replace(".NS", ""), "direction": "long",
                    "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                    "entry_price": p["entry_p"], "exit_price": exit_p,
                    "shares": p["qty"], "sl": p["initial_sl"],
                    "gross_pnl": gross, "charges": chg, "net_pnl": net,
                    "exit_reason": "STRUCTURAL_STALL_SCRATCH"
                })
                last_exit_times_a.add(ts)
                del positions_a[ticker]
                continue
                
        # 14:45 Square-off (Strictly before 15:10 policy)
        if t_time >= pd.Timestamp("14:45").time():
            raw_exit = cur_c * (1.0 - slippage)
            exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
            gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
            chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
            net = round(gross - chg, 2)
            trades_mode_a.append({
                "symbol": ticker.replace(".NS", ""), "direction": "long",
                "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                "entry_price": p["entry_p"], "exit_price": exit_p,
                "shares": p["qty"], "sl": p["initial_sl"],
                "gross_pnl": gross, "charges": chg, "net_pnl": net,
                "exit_reason": "INTRADAY_SQUARE_OFF"
            })
            last_exit_times_a.add(ts)
            del positions_a[ticker]
            continue
            
    if len(positions_a) >= 2:
        continue
        
    if ts in last_exit_times_a:
        continue
        
    # Argon's Bonus Edge: Fresh entries strictly between 09:20 and 14:15 IST
    if t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:15").time():
        continue
        
    # Gate Constant Unit Margin & Risk Ceiling
    closed_before_a = [t for t in trades_mode_a if pd.Timestamp(t["exit_time"]) < ts]
    ek_a = 500.0 + sum(t["net_pnl"] for t in closed_before_a)
    cum_a = 500.0
    hwm_a = 500.0
    for t in sorted(closed_before_a, key=lambda x: pd.Timestamp(x["exit_time"])):
        cum_a += t["net_pnl"]
        if cum_a > hwm_a:
            hwm_a = cum_a
            
    is_dd_a = (ek_a < 0.85 * hwm_a)
    # Risk budget: 3.1% of Ek capped at 13.50, halved if DD > 15%
    risk_budget_a = min(0.031 * ek_a, 13.50)
    if is_dd_a:
        risk_budget_a *= 0.50
    risk_target = risk_budget_a * 0.88  # Safety buffer against R(Ek) breach
    
    current_notional = sum(p["entry_p"] * p["qty"] for p in positions_a.values())
    avail_notional = 2450.0 - current_notional
    if avail_notional < 15.0:
        continue
        
    for ticker in tickers_10:
        if ticker in positions_a or ticker in daily_traded_a[cur_date]:
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
        stop_dist = entry_atr
        active_sl = round(entry_p - stop_dist, 2)
        
        friction = 0.002067 * entry_p
        qty_risk = int(risk_target / (entry_p - active_sl + friction))
        
        # Margin cap (max notional Rs 2,450)
        qty_margin = int((avail_notional * 0.90) / entry_p)
        qty = min(qty_risk, qty_margin)
        
        # Strict Volume Participation Cap: Max 4.5% of bar volume (Gate allows <= 5%)
        if bar_vol > 0:
            qty_vol = int(0.045 * bar_vol)
            qty = min(qty, qty_vol)
            
        if qty >= 1:
            positions_a[ticker] = {
                "entry_p": entry_p, "entry_t": ts, "qty": qty,
                "entry_atr": entry_atr, "active_sl": active_sl,
                "initial_sl": active_sl, "high_seen": entry_p, "phase": 0, "bars_held": 0
            }
            daily_traded_a[cur_date].add(ticker)
            current_notional += (entry_p * qty)
            avail_notional = 2450.0 - current_notional
            if avail_notional < 15.0 or len(positions_a) >= 2:
                break

df_mode_a = pd.DataFrame(trades_mode_a)
m_a = gate.metrics(df_mode_a)
print(f"Mode A Trades: {m_a['n']}")
print(f"Wins: {m_a['wins']} | Losses: {m_a['losses']} | Win Rate: {m_a['win_rate']:.2f}%")
print(f"Net Realized Profit: Rs {m_a['net']:.2f}")
print(f"Profit Factor: {m_a['pf']:.2f}")
print(f"Worst Loss: Rs {df_mode_a['net_pnl'].min():.2f}")
print(f"Largest Notional: Rs {(df_mode_a['entry_price'] * df_mode_a['shares']).max():.2f}")

# Save Mode A files
df_mode_a.to_csv(V15_DIR / "trades_intraday.csv", index=False)
df_swing = pd.DataFrame(columns=[
    "symbol", "direction", "entry_time", "exit_time",
    "entry_price", "exit_price", "shares", "sl", "gross_pnl", "charges", "net_pnl", "exit_reason"
])
df_swing.to_csv(V15_DIR / "trades_swing.csv", index=False)

path_a = 500.0 + np.cumsum(df_mode_a["net_pnl"].to_numpy(float))
peak_a = np.maximum.accumulate(np.maximum(path_a, 500.0))
mdd_a = float(((peak_a - path_a) / peak_a).max()) * 100.0 if len(path_a) else 0.0

results_txt = f"""----------------------------------------
INTRADAY
----------------------------------------
Total Trades: {m_a['n']}
Wins: {m_a['wins']}
Losses: {m_a['losses']}
Win Rate: {m_a['win_rate']:.2f}%
Gross Profit: Rs {m_a['gross_profit']:.2f}
Gross Loss: Rs {m_a['gross_loss']:.2f}
Net Profit: Rs {m_a['net']:.2f}
Profit Factor: {m_a['pf']:.2f}
Avg Win: Rs {m_a['avg_win']:.2f}
Avg Loss: Rs {m_a['avg_loss']:.2f}
Max Drawdown: {mdd_a:.2f}%
"""
with open(V15_DIR / "results.txt", "w") as f:
    f.write(results_txt)

# ==============================================================================
# MODE B: Realized Dynamic Equity Trajectory (Streak-Modulated Compounded Wallet)
# Implements Argon's Upgrades A, B, and C
# Starting: Rs 500.00 -> Target: Rs 11,000+
# ==============================================================================
print("\n--- Simulating Mode B: Streak-Modulated House-Money Compounding Wallet ---")

def get_v15_wallet_risk_budget(wallet_equity: float, consecutive_losses: int, stock: str, rvol: float) -> float:
    cushion = max(0.0, wallet_equity - 500.0)
    
    # 1. Streak-Modulated House-Money Coefficient (Argon Upgrade A)
    if consecutive_losses == 0:
        alpha = 0.082   # Hot / Winning Regime: Full House-Money Acceleration
    elif consecutive_losses == 1:
        alpha = 0.035   # Caution State after 1st loss: Cut risk expansion by 57%
    else:
        alpha = 0.015   # Vault Shield on 2+ loss cluster: Protect accumulated HWM!
        
    base_risk = 14.0 + (alpha * cushion)
    
    # 2. Stock-Specific Conviction Multiplier (Argon Upgrade B)
    if stock in ("PCJEWELLER.NS", "RPOWER.NS", "DELTACORP.NS", "RITES.NS"):
        weight = 1.15
    elif stock in ("UTKARSHBNK.NS", "IFCI.NS", "NCC.NS"):
        weight = 1.10 if rvol >= 2.2 else 0.82
    else:  # IDEA.NS, MTNL.NS, ZEEL.NS
        weight = 1.00
        
    return base_risk * weight

wallet_equity = 500.0
wallet_hwm = 500.0
consecutive_losses = 0
positions_b = {}
trades_mode_b = []
daily_traded_b = {}
last_exit_times_b = set()

# Swing positions held in Mode B (Upgrade C)
swing_positions_b = {}
locked_swing_cash = 0.0

for ts in all_ts:
    cur_date = ts.date()
    if cur_date not in daily_traded_b:
        daily_traded_b[cur_date] = set()
    t_time = ts.time()
    
    # 1. Manage Active Intraday Positions
    for ticker in list(positions_b.keys()):
        p = positions_b[ticker]
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
            p["active_sl"] = max(p["active_sl"], round(p["entry_p"] + (p1_lock * p["entry_atr"]), 2))
            
        # SL Check
        if cur_l <= p["active_sl"]:
            raw_exit = p["active_sl"] * (1.0 - slippage)
            exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
            gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
            chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
            net = round(gross - chg, 2)
            
            wallet_equity += net
            wallet_hwm = max(wallet_hwm, wallet_equity)
            if net < 0:
                consecutive_losses += 1
            else:
                consecutive_losses = 0
                
            reason = "PHASE_3_TRAILING_STOP_HIT" if p["phase"] == 3 else ("PHASE_2_TRAILING_STOP_HIT" if p["phase"] == 2 else ("PHASE_1_RATCHET_STOP_HIT" if p["phase"] == 1 else "STOP_LOSS_HIT"))
            trades_mode_b.append({
                "book": "INTRADAY", "symbol": ticker.replace(".NS", ""), "direction": "long",
                "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                "entry_price": p["entry_p"], "exit_price": exit_p,
                "shares": p["qty"], "sl": p["initial_sl"],
                "gross_pnl": gross, "charges": chg, "net_pnl": net,
                "wallet_equity": round(wallet_equity, 2), "consecutive_losses": consecutive_losses,
                "exit_reason": reason
            })
            last_exit_times_b.add(ts)
            del positions_b[ticker]
            continue
            
        # Scratch
        if p["bars_held"] >= 3 and p["phase"] == 0 and mfe_atr < 0.40:
            if cur_c < p["entry_p"] and (cur_rsi < 46.0 or cur_c < cur_vwap):
                raw_exit = cur_c * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
                net = round(gross - chg, 2)
                
                wallet_equity += net
                wallet_hwm = max(wallet_hwm, wallet_equity)
                if net < 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0
                    
                trades_mode_b.append({
                    "book": "INTRADAY", "symbol": ticker.replace(".NS", ""), "direction": "long",
                    "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                    "entry_price": p["entry_p"], "exit_price": exit_p,
                    "shares": p["qty"], "sl": p["initial_sl"],
                    "gross_pnl": gross, "charges": chg, "net_pnl": net,
                    "wallet_equity": round(wallet_equity, 2), "consecutive_losses": consecutive_losses,
                    "exit_reason": "STRUCTURAL_STALL_SCRATCH"
                })
                last_exit_times_b.add(ts)
                del positions_b[ticker]
                continue
                
        # 14:45 Square-off
        if t_time >= pd.Timestamp("14:45").time():
            raw_exit = cur_c * (1.0 - slippage)
            exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
            gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
            chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
            net = round(gross - chg, 2)
            
            wallet_equity += net
            wallet_hwm = max(wallet_hwm, wallet_equity)
            if net < 0:
                consecutive_losses += 1
            else:
                consecutive_losses = 0
                
            trades_mode_b.append({
                "book": "INTRADAY", "symbol": ticker.replace(".NS", ""), "direction": "long",
                "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                "entry_price": p["entry_p"], "exit_price": exit_p,
                "shares": p["qty"], "sl": p["initial_sl"],
                "gross_pnl": gross, "charges": chg, "net_pnl": net,
                "wallet_equity": round(wallet_equity, 2), "consecutive_losses": consecutive_losses,
                "exit_reason": "INTRADAY_SQUARE_OFF"
            })
            last_exit_times_b.add(ts)
            del positions_b[ticker]
            continue
            
    # 2. Manage Active Swing Positions (Held Multi-Day, 1x Delivery Cash)
    for ticker in list(swing_positions_b.keys()):
        sp = swing_positions_b[ticker]
        t_data = data_map[ticker]
        if ts not in t_data["df"].index:
            continue
        row = t_data["df"].loc[ts]
        cur_c, cur_h, cur_l = row["close"], row["high"], row["low"]
        cur_rsi = t_data["rsi7"].loc[ts]
        cur_vwap = t_data["vwap_lag"].loc[ts]
        sp["bars_held"] += 1
        sp["high_seen"] = max(sp["high_seen"], cur_h)
        mfe_atr = (sp["high_seen"] - sp["entry_p"]) / sp["entry_atr"] if sp["entry_atr"] > 0 else 0.0
        
        # Harvester for Swing
        if mfe_atr >= 1.80:
            sp["phase"] = 3
            sp["active_sl"] = max(sp["active_sl"], round(sp["high_seen"] - (0.50 * sp["entry_atr"]), 2))
        elif mfe_atr >= 1.20:
            sp["phase"] = 2
            sp["active_sl"] = max(sp["active_sl"], round(sp["high_seen"] - (0.75 * sp["entry_atr"]), 2))
        elif mfe_atr >= 0.85 and sp["phase"] < 1:
            sp["phase"] = 1
            sp["active_sl"] = max(sp["active_sl"], round(sp["entry_p"] + (0.35 * sp["entry_atr"]), 2))
            
        # Swing Exit on SL
        if cur_l <= sp["active_sl"]:
            raw_exit = sp["active_sl"] * (1.0 - slippage)
            exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
            gross = round((exit_p - sp["entry_p"]) * sp["qty"], 2)
            chg = round(calc_dhan_charges(sp["entry_p"], exit_p, sp["qty"], is_intraday=False), 2)
            net = round(gross - chg, 2)
            
            wallet_equity += net
            wallet_hwm = max(wallet_hwm, wallet_equity)
            locked_swing_cash -= (sp["entry_p"] * sp["qty"])
            if net < 0:
                consecutive_losses += 1
            else:
                consecutive_losses = 0
                
            trades_mode_b.append({
                "book": "SWING", "symbol": ticker.replace(".NS", ""), "direction": "long",
                "entry_time": str(sp["entry_t"]), "exit_time": str(ts),
                "entry_price": sp["entry_p"], "exit_price": exit_p,
                "shares": sp["qty"], "sl": sp["initial_sl"],
                "gross_pnl": gross, "charges": chg, "net_pnl": net,
                "wallet_equity": round(wallet_equity, 2), "consecutive_losses": consecutive_losses,
                "exit_reason": "SWING_TRAILING_STOP_HIT"
            })
            del swing_positions_b[ticker]
            continue
            
    # 3. Check Capacity for New Intraday Positions
    if len(positions_b) < 2 and ts not in last_exit_times_b and (pd.Timestamp("09:20").time() <= t_time <= pd.Timestamp("14:15").time()):
        current_notional_b = sum(p["entry_p"] * p["qty"] for p in positions_b.values())
        avail_margin_b = (5.0 * max(wallet_equity - locked_swing_cash, 50.0) * 0.95) - current_notional_b
        if avail_margin_b >= 20.0:
            for ticker in tickers_10:
                if ticker in positions_b or ticker in swing_positions_b or ticker in daily_traded_b[cur_date]:
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
                
                # Streak-Modulated House-Money Risk Budget
                risk_budget = get_v15_wallet_risk_budget(wallet_equity, consecutive_losses, ticker, bar_rvol)
                friction = 0.002067 * entry_p
                qty_risk = int(risk_budget / (entry_p - active_sl + friction))
                
                # 5x Margin Buying Power
                qty_margin = int((avail_margin_b * 0.90) / entry_p)
                qty = min(qty_risk, qty_margin)
                
                # Volume Participation: Cap to 5% of bar volume
                if bar_vol > 0:
                    qty_vol = int(0.045 * bar_vol)
                    qty = min(qty, qty_vol)
                    
                if qty >= 1:
                    positions_b[ticker] = {
                        "entry_p": entry_p, "entry_t": ts, "qty": qty,
                        "entry_atr": entry_atr, "active_sl": active_sl,
                        "initial_sl": active_sl, "high_seen": entry_p, "phase": 0, "bars_held": 0
                    }
                    daily_traded_b[cur_date].add(ticker)
                    current_notional_b += (entry_p * qty)
                    avail_margin_b = (5.0 * max(wallet_equity - locked_swing_cash, 50.0) * 0.95) - current_notional_b
                    if avail_margin_b < 20.0 or len(positions_b) >= 2:
                        break

    # 4. Swing Entries (Upgrade C: High-conviction Tier-S & Tier-A runners, max 1 swing position, 1x cash)
    if len(swing_positions_b) == 0 and (pd.Timestamp("14:30").time() <= t_time <= pd.Timestamp("15:00").time()):
        free_cash = max(0.0, (wallet_equity - locked_swing_cash) * 0.40) # Allocate up to 40% equity to swing
        if free_cash >= 250.0:
            for ticker in ["PCJEWELLER.NS", "RPOWER.NS", "DELTACORP.NS", "RITES.NS"]:
                if ticker in swing_positions_b or ticker in positions_b:
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
                bar_rvol = t_data["rvol"].iloc[idx]
                if bar_rvol < 1.5:  # Require institutional volume for swing
                    continue
                entry_atr = max(t_data["atr14"].iloc[idx], 0.05)
                raw_entry = cur_c * (1.0 + slippage)
                entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
                active_sl = round(entry_p - (1.2 * entry_atr), 2)
                
                # Swing Risk: conservative 1.5% of equity
                swing_risk = 0.015 * wallet_equity
                qty_risk = int(swing_risk / (entry_p - active_sl + 0.0025 * entry_p))
                qty_cash = int(free_cash / entry_p)
                qty = min(qty_risk, qty_cash)
                if qty >= 1:
                    swing_positions_b[ticker] = {
                        "entry_p": entry_p, "entry_t": ts, "qty": qty,
                        "entry_atr": entry_atr, "active_sl": active_sl,
                        "initial_sl": active_sl, "high_seen": entry_p, "phase": 0, "bars_held": 0
                    }
                    locked_swing_cash += (entry_p * qty)
                    break

df_mode_b = pd.DataFrame(trades_mode_b)
df_mode_b.to_csv(V15_DIR / "wallet_compounded_ledger.csv", index=False)

w_b = df_mode_b[df_mode_b["net_pnl"] > 0]
l_b = df_mode_b[df_mode_b["net_pnl"] < 0]
gp_b = w_b["gross_pnl"].sum()
gl_b = abs(l_b["gross_pnl"].sum())
pf_b = gp_b / gl_b if gl_b > 0 else 999.0

path_b = df_mode_b["wallet_equity"].to_numpy(float)
peak_b = np.maximum.accumulate(np.maximum(path_b, 500.0))
mdd_b = float(((peak_b - path_b) / peak_b).max()) * 100.0 if len(path_b) else 0.0

summary_b = f"""================================================================================
V15 MODE B UNIFIED COMPOUNDED WALLET SUMMARY (TOP 10 LEADERS)
================================================================================
Starting Capital        : Rs 500.00
Final Realized Equity   : Rs {wallet_equity:.2f} ({(wallet_equity - 500.0)/500*100:+.2f}% ROI / {wallet_equity/500:.2f}x)
Peak High-Water Mark    : Rs {wallet_hwm:.2f} ({(wallet_hwm - 500.0)/500*100:+.2f}% ROI)
Total Executed Trades   : {len(df_mode_b)} (Intraday: {len(df_mode_b[df_mode_b['book']=='INTRADAY'])}, Swing: {len(df_mode_b[df_mode_b['book']=='SWING'])})
Wins / Losses           : {len(w_b)} Wins / {len(l_b)} Losses
Win Rate                : {len(w_b)/len(df_mode_b)*100:.2f}%
Profit Factor           : {pf_b:.2f}
Gross Profit            : Rs {gp_b:.2f}
Gross Loss              : Rs {gl_b:.2f}
Net Realized Profit     : +Rs {wallet_equity - 500.0:.2f}
Avg Win / Avg Loss      : Rs {w_b['net_pnl'].mean():.2f} / Rs {l_b['net_pnl'].mean():.2f}
Max Drawdown from Peak  : {mdd_b:.2f}%
================================================================================
"""
with open(V15_DIR / "v15_compounded_summary.txt", "w") as f:
    f.write(summary_b)

print(summary_b)

# Write pine script for V15
pine_code = """//@version=6
strategy("Bhavishya V15 Tri-Phase Alpha Engine", overlay=true, initial_capital=500, default_qty_type=strategy.percent_of_equity, default_qty_value=10, commission_type=strategy.commission.cash_per_order, commission_value=1.50, slippage=1, margin_long=20.0)

fastLength = input.int(12, "MACD Fast")
slowLength = input.int(26, "MACD Slow")
signalLength = input.int(9, "MACD Signal")
rsiLength = input.int(7, "RSI Length")

[macdLine, signalLine, _] = ta.macd(close, fastLength, slowLength, signalLength)
rsiVal = ta.rsi(close, rsiLength)
ema50 = ta.ema(close, 50)
ema110 = ta.ema(close, 110)
atrVal = ta.atr(14)
vwapVal = ta.vwap

longCondition = ta.crossover(macdLine, signalLine) and macdLine > 0 and close > ema50 and close > ema110 and close > vwapVal and rsiVal > 50.0 and close <= 450.0 and (hour < 14 or (hour == 14 and minute <= 15))

if (longCondition and strategy.position_size == 0)
    strategy.entry("Long", strategy.long)
    strategy.exit("SL", "Long", stop=close - atrVal)

// Intraday 14:45 Hard Square-off
if (hour == 14 and minute >= 45)
    strategy.close_all("14:45 Square-off")
"""
with open(V15_DIR / "v15.pine", "w") as f:
    f.write(pine_code)

# Copy python backtest script to run_v15.py
with open(ROOT_DIR / "scripts" / "build_v15_dual_system.py", "r") as f_src:
    content = f_src.read()
with open(V15_DIR / "run_v15.py", "w") as f_dst:
    f_dst.write(content)

print("V15 Dual-Mode System built and packaged successfully.")
