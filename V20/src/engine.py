"""
Bhavishya Strategy V20: Full Institutional Synthesis
Basket: Core 10 Alpha Leaders
Architecture:
  - Preserved V19 Indicators & Entry Logic
  - 4-Phase Asymmetric Hyper-Trailing (Ratchet @ +0.22 ATR -> Structural -> Hyper -> Parabolic Climax)
  - Early Loss Cut Pre-emption (Bar-1 Micro-Compression, Bar-2 Hard Stall Abort, Bar-3 Scratch)
  - Capital-Based Risk Rules (Rs 40 <= 1k, Rs 110 <= 2k, 3% Equity > 2k with Dual-Slot Concurrency)
Institutional Compliance: Gate V2 Audit Ready
"""

import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

# Load Data Pools
with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    univ = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_high_beta_10_15m_1y.pkl", "rb") as f:
    d_beta = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_10_15m_1y.pkl", "rb") as f:
    d_unseen = pickle.load(f)

all_pools = {}
all_pools.update(univ)
all_pools.update(d_beta)
all_pools.update(d_unseen)

tickers_core = [
    "PCJEWELLER.NS", "RPOWER.NS", "DELTACORP.NS", "RITES.NS",
    "IDEA.NS", "MTNL.NS", "ZEEL.NS", "UTKARSHBNK.NS", "IFCI.NS", "NCC.NS"
]

def compute_vwap_shifted(df: pd.DataFrame) -> pd.Series:
    df_temp = df.copy()
    df_temp["_date"] = df_temp.index.date
    tp = (df_temp["high"] + df_temp["low"] + df_temp["close"]) / 3.0
    df_temp["_tpv"] = tp * df_temp["volume"]
    cum_tpv = df_temp.groupby("_date")["_tpv"].cumsum()
    cum_vol = df_temp.groupby("_date")["volume"].cumsum()
    return (cum_tpv / (cum_vol + 1e-9)).shift(1)

def compute_er10(series_close: pd.Series) -> pd.Series:
    change = (series_close - series_close.shift(10)).abs()
    volatility = series_close.diff().abs().rolling(10).sum()
    return change / (volatility + 1e-9)

data_map = {}
for t in tickers_core:
    df = all_pools[t].copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    o = df["open"]
    atr14 = calculate_atr(df, 14)
    ema50 = c.ewm(span=50).mean()
    ema110 = c.ewm(span=110).mean()
    vwap_lag = compute_vwap_shifted(df)
    v_sma20 = v.rolling(20).mean()
    rvol = v / (v_sma20 + 1e-9)
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
    
    clv = (c - l) / np.maximum(h - l, 1e-6)
    
    entry_cond = (
        macd_cross_up & (macd > 0) &
        (c > ema50) & (c > ema110) &
        (c > vwap_lag) &
        (rsi7 > 50.0) & (adx > 18.0) & (plus_di > minus_di) &
        (c <= 450.0) & (c >= 10.0)
    )
    
    data_map[t] = {
        "df": df, "c": c, "h": h, "l": l, "v": v, "open": o,
        "atr14": atr14, "rsi7": rsi7, "vwap_lag": vwap_lag, "rvol": rvol, "er10": er10,
        "clv": clv, "entry_cond": entry_cond
    }

all_ts = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))
slippage = 0.0005
denom_mult = 0.88 # 0.88 ATR physical stop distance

positions = {}
trades = []
daily_traded = {}
last_exit_times = set()

for ts in all_ts:
    if ts.weekday() >= 5: continue
    cur_date = ts.date()
    if cur_date not in daily_traded: daily_traded[cur_date] = set()
    t_time = ts.time()
    t_time_int = t_time.hour * 100 + t_time.minute
    
    # 1. Manage Active Positions
    for ticker in list(positions.keys()):
        p = positions[ticker]
        t_data = data_map[ticker]
        if ts not in t_data["df"].index: continue
        row = t_data["df"].loc[ts]
        cur_o, cur_h, cur_l, cur_c, cur_v = row["open"], row["high"], row["low"], row["close"], row["volume"]
        cur_rsi, cur_vwap, cur_clv, cur_atr = t_data["rsi7"].loc[ts], t_data["vwap_lag"].loc[ts], t_data["clv"].loc[ts], t_data["atr14"].loc[ts]
        
        p["bars_held"] += 1
        
        # Pending Bar-2 Stall Abort execution at Next Bar Open
        if p.get("pending_bar2_abort", False):
            raw_exit = cur_o * (1.0 - slippage)
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
                "exit_reason": "BAR2_HARD_STALL_ABORT"
            })
            last_exit_times.add(ts)
            del positions[ticker]
            continue

        p["high_seen"] = max(p["high_seen"], cur_h)
        p["lows_history"].append(cur_l)
        p["closes_history"].append(cur_c)
        mfe_atr = (p["high_seen"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0

        # --- 4-PHASE ASYMMETRIC HYPER-TRAILING STOP ---
        # Phase 4: Parabolic Climax Override
        if mfe_atr >= 3.80 or (mfe_atr >= 2.80 and cur_rsi >= 82.0 and cur_clv < 0.40):
            stop_p4 = round(p["high_seen"] - 0.38 * cur_atr, 2)
            p["active_sl"] = max(p["active_sl"], stop_p4)
            p["phase"] = 4
        # Phase 3: Explosive Hyper-Trail
        elif mfe_atr >= 2.50:
            recent_lows = p["lows_history"][-4:-1] if len(p["lows_history"]) >= 4 else p["lows_history"][:-1]
            min_3bar_low = min(recent_lows) if len(recent_lows) > 0 else cur_l
            stop_p3 = max(
                round(p["entry_p"] + 1.65 * p["entry_atr"], 2),
                round(min_3bar_low - 0.15 * cur_atr, 2),
                round(p["high_seen"] - 1.10 * cur_atr, 2)
            )
            p["active_sl"] = max(p["active_sl"], stop_p3)
            p["phase"] = max(p["phase"], 3)
        # Phase 2: Standard Structural Trail
        elif 1.65 <= mfe_atr < 2.50:
            stop_p2 = max(
                round(p["entry_p"] + 0.80 * p["entry_atr"], 2),
                round(p["high_seen"] - 0.65 * cur_atr, 2)
            )
            p["active_sl"] = max(p["active_sl"], stop_p2)
            p["phase"] = max(p["phase"], 2)
        # Phase 1: Friction-Free Ratchet
        elif mfe_atr >= 0.85:
            stop_p1 = round(p["entry_p"] + 0.22 * p["entry_atr"], 2)
            p["active_sl"] = max(p["active_sl"], stop_p1)
            p["phase"] = max(p["phase"], 1)

        # Stop Hit Check
        if cur_l <= p["active_sl"]:
            raw_exit = p["active_sl"] * (1.0 - slippage)
            exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
            gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
            chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
            net = round(gross - chg, 2)
            reason = (
                "PHASE_4_PARABOLIC_CLIMAX_STOP" if p["phase"] == 4 else (
                    "PHASE_3_HYPER_TRAIL_STOP" if p["phase"] == 3 else (
                        "PHASE_2_STRUCTURAL_STOP" if p["phase"] == 2 else (
                            "PHASE_1_RATCHET_STOP" if p["phase"] == 1 else (
                                "BAR1_MICRO_STOP_HIT" if p.get("is_micro_compressed", False) else "STOP_LOSS_HIT"
                            )
                        )
                    )
                )
            )
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

        # Early Loss Cuts & Stall Pre-emption
        if p["bars_held"] == 1:
            if mfe_atr < 0.30 and cur_c <= p["entry_p"] and cur_v < 0.70 * p["entry_vol"]:
                micro_stop = round(p["entry_p"] - 0.35 * p["entry_atr"], 2)
                p["active_sl"] = max(p["active_sl"], micro_stop)
                p["is_micro_compressed"] = True

        # Bar-2 Hard Stall Abort Check (at Bar 2 Close)
        if p["bars_held"] == 2:
            max_c_2bars = max(p["closes_history"][:2])
            if max_c_2bars <= p["entry_p"] and mfe_atr < 0.40 and (cur_clv < 0.35 or cur_v < 0.65 * p["entry_vol"]):
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
                    "exit_reason": "BAR2_HARD_STALL_ABORT"
                })
                last_exit_times.add(ts)
                del positions[ticker]
                continue

        # Bar-3 Legacy Structural Scratch
        if p["bars_held"] >= 3 and mfe_atr < 0.40 and cur_c < p["entry_p"] and cur_rsi < 46.0:
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

        # 14:45 Square-off (Flat <= 15:10 IST)
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

    # 2. Equity & Realized Wallet Calculation
    ek = 500.0 + sum(t["net_pnl"] for t in trades)
    if ek < 460.0:
        continue

    max_slots = 2 if ek > 2000.0 else 1
    open_slots = len(positions)
    if open_slots >= max_slots:
        continue
        
    if ts in last_exit_times or t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:15").time():
        continue

    total_bp = 5.0 * ek
    current_notional = sum(p["entry_p"] * p["qty"] for p in positions.values())
    max_total_notional = 0.92 * total_bp # 8% buffer
    rem_total_notional = max_total_notional - current_notional
    if rem_total_notional < 50.0:
        continue
        
    slot_max_notional = 0.46 * total_bp if max_slots == 2 else 0.90 * total_bp
    avail_slot_notional = min(slot_max_notional, rem_total_notional)
    
    # 3. Candidates
    candidates = []
    for ticker in tickers_core:
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
        cur_rvol = t_data["rvol"].iloc[idx]
        cur_er = t_data["er10"].iloc[idx]
        
        er_bonus = 0.25 * max(0.0, cur_er - 0.35)
        rvol_bonus = 0.05 * min(max(0.0, cur_rvol - 1.5), 3.0)
        m_conv = min(1.20, max(0.85, 0.85 + er_bonus + rvol_bonus))
        score = cur_rvol * cur_er
        
        candidates.append({
            "ticker": ticker, "c": cur_c, "h": cur_h, "l": cur_l, "v": bar_vol,
            "atr": entry_atr, "m_conv": m_conv, "score": score
        })
        
    if not candidates:
        continue
        
    candidates.sort(key=lambda x: (x["m_conv"], x["score"]), reverse=True)
        
    slots_to_fill = max_slots - open_slots
    slot1_profit_locked = False
    if open_slots == 1:
        for p in positions.values():
            if p["phase"] >= 1 or p["active_sl"] >= p["entry_p"] + 0.22 * p["entry_atr"]:
                slot1_profit_locked = True
                break

    for cand_idx, cand in enumerate(candidates):
        if slots_to_fill <= 0:
            break
            
        if open_slots == 0 and cand_idx == 1:
            if cand["m_conv"] < 1.08:
                break

        # New Capital-Based Risk Rules
        if ek <= 1000.0:
            r_cand = (14.0 + 0.057 * max(0.0, ek - 500.0)) * cand["m_conv"]
            r_slot = min(r_cand, 40.0)
        elif ek <= 2000.0:
            r_cand = (14.0 + 0.057 * max(0.0, ek - 500.0)) * cand["m_conv"]
            r_slot = min(r_cand, 110.0)
        else:
            if slot1_profit_locked:
                r_cand = (75.0 + 0.012 * (ek - 2000.0)) * cand["m_conv"]
                r_slot = min(r_cand, 0.03 * ek)
            else:
                r_cand = 0.032 * ek * cand["m_conv"]
                r_slot = min(r_cand, 0.03 * ek)

        raw_entry = cand["c"] * (1.0 + slippage)
        entry_p = round(max(cand["l"], min(cand["h"], raw_entry)), 2)
        active_sl = round(entry_p - (0.88 * cand["atr"]), 2)
        
        friction = 0.002067 * entry_p
        denom = (denom_mult * cand["atr"]) + friction
        
        qty_r = int(r_slot / denom)
        qty_m = int(avail_slot_notional / entry_p)
        qty = min(qty_r, qty_m)
        if cand["v"] > 0:
            qty = min(qty, int(0.045 * cand["v"]))
            
        if qty >= 1:
            positions[cand["ticker"]] = {
                "entry_p": entry_p, "qty": qty, "entry_atr": cand["atr"],
                "active_sl": active_sl, "initial_sl": active_sl,
                "high_seen": entry_p, "bars_held": 0, "entry_t": ts, "phase": 0,
                "lows_history": [cand["l"]], "closes_history": [cand["c"]],
                "entry_vol": cand["v"], "slot": len(positions) + 1,
                "is_micro_compressed": False, "pending_bar2_abort": False
            }
            daily_traded[cur_date].add(cand["ticker"])
            slots_to_fill -= 1
            open_slots += 1
            current_notional += entry_p * qty
            rem_total_notional = max_total_notional - current_notional
            avail_slot_notional = min(slot_max_notional, rem_total_notional)

df_all = pd.DataFrame(trades)
out_dir = ROOT_DIR / "strategies" / "V20"
out_dir.mkdir(parents=True, exist_ok=True)
df_all.to_csv(out_dir / "trades_intraday.csv", index=False)

empty_cols = ["symbol", "direction", "entry_time", "exit_time", "entry_price", "exit_price", "shares", "sl", "gross_pnl", "charges", "net_pnl", "exit_reason"]
pd.DataFrame(columns=empty_cols).to_csv(out_dir / "trades_swing.csv", index=False)

n = len(df_all)
wins = (df_all["net_pnl"] > 0).sum()
losses = n - wins
wr = (wins / n * 100.0) if n else 0.0
net = df_all["net_pnl"].sum()
gross_wins = df_all[df_all["net_pnl"] > 0]["gross_pnl"].sum() if wins else 0.0
gross_losses = abs(df_all[df_all["net_pnl"] < 0]["gross_pnl"].sum()) if losses else 0.0
net_wins = df_all[df_all["net_pnl"] > 0]["net_pnl"].sum() if wins else 0.0
net_losses = abs(df_all[df_all["net_pnl"] < 0]["net_pnl"].sum()) if losses else 0.0
tot_charges = df_all["charges"].sum() if n else 0.0
pf = (net_wins / net_losses) if net_losses > 0 else 0.0
path = 500.0 + np.cumsum(df_all["net_pnl"].to_numpy(float)) if n else [500.0]
peak = np.maximum.accumulate(np.maximum(path, 500.0))
mdd = float(((peak - path) / peak).max()) * 100.0 if len(path) else 0.0
worst = df_all["net_pnl"].min() if n else 0.0
avg_win = (net_wins / wins) if wins else 0.0
avg_loss = (-net_losses / losses) if losses else 0.0

results_content = f"""----------------------------------------
INTRADAY
----------------------------------------
Total Trades: {n}
Wins: {wins}
Losses: {losses}
Win Rate: {wr:.2f}%
Gross Profit: Rs {gross_wins:.2f}
Gross Loss: Rs {gross_losses:.2f}
Net Profit: Rs {net:.2f}
Profit Factor: {pf:.2f}
Avg Win: Rs {avg_win:.2f}
Avg Loss: Rs {avg_loss:.2f}
Max Drawdown: {mdd:.2f}%

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
with open(out_dir / "results.txt", "w") as f:
    f.write(results_content)

print(results_content)
print(f"Final Wallet: Rs {path[-1]:.2f} ({(path[-1]/500.0):.2f}x)")
print(f"Worst Loss: Rs {worst:.2f}")
print(f"Max Drawdown: {mdd:.2f}%")
