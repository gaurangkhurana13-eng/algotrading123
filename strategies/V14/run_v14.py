import pickle
import pandas as pd
import numpy as np
import sys
import math
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr
import strategy_evaluate_gate_v2 as gate

V14_DIR = Path(__file__).resolve().parent

with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    univ = pickle.load(f)

# Top 6 Verified Alpha Leaders
tickers = [
    "PCJEWELLER.NS", "UTKARSHBNK.NS", "RPOWER.NS", "MTNL.NS", 
    "IDEA.NS", "IFCI.NS"
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
trades_intraday = []
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
            p["active_sl"] = max(p["active_sl"], round(p["entry_p"] + (p1_lock * p["entry_atr"]), 2))
            
        # Stop-Loss Check (Exact 2-decimal rounded active_sl check)
        if cur_l <= p["active_sl"]:
            raw_exit = p["active_sl"] * (1.0 - slippage)
            exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
            gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
            chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
            net = round(gross - chg, 2)
            reason = "PHASE_3_TRAILING_STOP_HIT" if p["phase"] == 3 else ("PHASE_2_TRAILING_STOP_HIT" if p["phase"] == 2 else ("PHASE_1_RATCHET_STOP_HIT" if p["phase"] == 1 else "STOP_LOSS_HIT"))
            trades_intraday.append({
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
                trades_intraday.append({
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
                
        # 14:45 Square-off (Strictly before 15:10 policy)
        if t_time >= pd.Timestamp("14:45").time():
            raw_exit = cur_c * (1.0 - slippage)
            exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
            gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
            chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
            net = round(gross - chg, 2)
            trades_intraday.append({
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
        
    # Crucial timing rule: if an exit occurred on bar ts, do NOT enter a new trade on that SAME bar ts
    # because Gate V2 considers the exited trade still open at ts (ex >= ent)
    if ts in last_exit_times:
        continue
        
    if t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("13:30").time():
        continue
        
    # Derive Gate V2 live metrics
    closed_before = [t for t in trades_intraday if pd.Timestamp(t["exit_time"]) < ts]
    ek = 500.0 + sum(t["net_pnl"] for t in closed_before)
    
    # HWM of realized equity
    cum = 500.0
    hwm = 500.0
    for t in sorted(closed_before, key=lambda x: pd.Timestamp(x["exit_time"])):
        cum += t["net_pnl"]
        if cum > hwm:
            hwm = cum
            
    # Margin already in use by active positions or same-bar exits
    used_margin = sum((t["entry_price"] * t["shares"]) / 5.0 for t in trades_intraday if pd.Timestamp(t["exit_time"]) >= ts)
    used_margin += sum((p["entry_p"] * p["qty"]) / 5.0 for p in positions.values())
    avail_margin = 0.95 * ek
    free_margin = max(avail_margin - used_margin, 0.0)
    
    if free_margin < 15.0:
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
        active_sl = round(entry_p - stop_dist, 2)
        
        # Risk Budget
        is_dd = (ek < 0.85 * hwm)
        risk_budget = min(0.031 * ek, 250.0)
        if is_dd:
            risk_budget *= 0.50
        risk_target = risk_budget * 0.88  # Safety buffer against R(Ek) breach
        
        friction = 0.002067 * entry_p
        qty_risk = int(risk_target / (entry_p - active_sl + friction))
        
        qty_margin = int((5.0 * free_margin * 0.88) / entry_p)
        qty = min(qty_risk, qty_margin)
        
        if qty >= 1:
            positions[ticker] = {
                "entry_p": entry_p, "entry_t": ts, "qty": qty,
                "entry_atr": entry_atr, "active_sl": active_sl,
                "initial_sl": active_sl, "high_seen": entry_p, "phase": 0, "bars_held": 0
            }
            daily_traded[cur_date].add(ticker)
            used_margin += (entry_p * qty) / 5.0
            free_margin = max(avail_margin - used_margin, 0.0)
            if free_margin < 15.0 or len(positions) >= 2:
                break

df_res = pd.DataFrame(trades_intraday)
m = gate.metrics(df_res)
print(f"Total Trades: {m['n']}")
print(f"Wins: {m['wins']} | Losses: {m['losses']} | Win Rate: {m['win_rate']:.2f}%")
print(f"Net Realized Profit: Rs {m['net']:.2f}")
print(f"Profit Factor: {m['pf']:.2f}")
print(f"Avg Win: Rs {m['avg_win']:.2f} | Avg Loss: Rs {m['avg_loss']:.2f}")

# Save directly to V14 directory
df_res.to_csv(V14_DIR / "trades_intraday.csv", index=False)

df_swing = pd.DataFrame(columns=[
    "symbol", "direction", "entry_time", "exit_time",
    "entry_price", "exit_price", "shares", "sl", "gross_pnl", "charges", "net_pnl", "exit_reason"
])
df_swing.to_csv(V14_DIR / "trades_swing.csv", index=False)

path = 500.0 + np.cumsum(df_res["net_pnl"].to_numpy(float))
peak = np.maximum.accumulate(np.maximum(path, 500.0))
mdd = float(((peak - path) / peak).max()) * 100.0 if len(path) else 0.0

results_txt = f"""----------------------------------------
INTRADAY
----------------------------------------
Total Trades: {m['n']}
Wins: {m['wins']}
Losses: {m['losses']}
Win Rate: {m['win_rate']:.2f}%
Gross Profit: Rs {m['gross_profit']:.2f}
Gross Loss: Rs {m['gross_loss']:.2f}
Net Profit: Rs {m['net']:.2f}
Profit Factor: {m['pf']:.2f}
Avg Win: Rs {m['avg_win']:.2f}
Avg Loss: Rs {m['avg_loss']:.2f}
Max Drawdown: {mdd:.2f}%
"""
with open(V14_DIR / "results.txt", "w") as f:
    f.write(results_txt)

print("V14 execution completed successfully.")
