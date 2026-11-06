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

DATA_FILE = ROOT_DIR / "data" / "unseen_batch3_10_15m_1y.pkl"
with open(DATA_FILE, "rb") as f:
    unseen_data = pickle.load(f)

tickers_unseen = list(unseen_data.keys())
print(f"Loaded {len(tickers_unseen)} 100% UNSEEN Stocks: {tickers_unseen}")

def compute_vwap_shifted(df: pd.DataFrame) -> pd.Series:
    df_temp = df.copy()
    df_temp["_date"] = df_temp.index.date
    tp = (df_temp["high"] + df_temp["low"] + df_temp["close"]) / 3.0
    df_temp["_tpv"] = tp * df_temp["volume"]
    cum_tpv = df_temp.groupby("_date")["_tpv"].cumsum()
    cum_vol = df_temp.groupby("_date")["volume"].cumsum()
    return (cum_tpv / (cum_vol + 1e-9)).shift(1)

data_map = {}
for t in tickers_unseen:
    df = unseen_data[t].copy()
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
    
    # EXACT Untouched V15 Confluence Condition
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
# 1. RUN V15 MODE A ON 10 UNSEEN STOCKS (Constant Unit)
# ==============================================================================
positions_a = {}
trades_mode_a = []
daily_traded_a = {}
last_exit_times_a = set()

for ts in all_ts:
    cur_date = ts.date()
    if cur_date not in daily_traded_a:
        daily_traded_a[cur_date] = set()
    t_time = ts.time()
    
    # Active Positions
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
            
        # SL Check
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
            
        # Scratch
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
                
        # 14:45 Square-off
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
            
    if len(positions_a) >= 2 or ts in last_exit_times_a:
        continue
        
    if t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:15").time():
        continue
        
    closed_before_a = [t for t in trades_mode_a if pd.Timestamp(t["exit_time"]) < ts]
    ek_a = 500.0 + sum(t["net_pnl"] for t in closed_before_a)
    cum_a = 500.0
    hwm_a = 500.0
    for t in sorted(closed_before_a, key=lambda x: pd.Timestamp(x["exit_time"])):
        cum_a += t["net_pnl"]
        if cum_a > hwm_a:
            hwm_a = cum_a
            
    is_dd_a = (ek_a < 0.85 * hwm_a)
    risk_budget_a = min(0.031 * ek_a, 13.50)
    if is_dd_a:
        risk_budget_a *= 0.50
    risk_target = risk_budget_a * 0.88
    
    current_notional = sum(p["entry_p"] * p["qty"] for p in positions_a.values())
    avail_notional = 2450.0 - current_notional
    if avail_notional < 15.0:
        continue
        
    for ticker in tickers_unseen:
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
        qty_margin = int((avail_notional * 0.90) / entry_p)
        qty = min(qty_risk, qty_margin)
        
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

# ==============================================================================
# 2. RUN V15 MODE B ON 10 UNSEEN STOCKS (Streak-Modulated Compounded Wallet)
# ==============================================================================
def get_v15_wallet_risk_budget(wallet_equity: float, consecutive_losses: int, rvol: float) -> float:
    cushion = max(0.0, wallet_equity - 500.0)
    if consecutive_losses == 0:
        alpha = 0.082
    elif consecutive_losses == 1:
        alpha = 0.035
    else:
        alpha = 0.015
    base_risk = 14.0 + (alpha * cushion)
    # Neutral Kelly weight on unseen stocks (1.00 or 1.10 if high rvol)
    weight = 1.10 if rvol >= 2.0 else 0.95
    return base_risk * weight

wallet_equity = 500.0
wallet_hwm = 500.0
consecutive_losses = 0
positions_b = {}
trades_mode_b = []
daily_traded_b = {}
last_exit_times_b = set()

for ts in all_ts:
    cur_date = ts.date()
    if cur_date not in daily_traded_b:
        daily_traded_b[cur_date] = set()
    t_time = ts.time()
    
    # Active Positions
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
                "symbol": ticker.replace(".NS", ""), "direction": "long",
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
                    "symbol": ticker.replace(".NS", ""), "direction": "long",
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
                "symbol": ticker.replace(".NS", ""), "direction": "long",
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
            
    if len(positions_b) >= 2 or ts in last_exit_times_b:
        continue
        
    if t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:15").time():
        continue
        
    current_notional_b = sum(p["entry_p"] * p["qty"] for p in positions_b.values())
    avail_margin_b = (5.0 * wallet_equity * 0.95) - current_notional_b
    if avail_margin_b < 20.0:
        continue
        
    for ticker in tickers_unseen:
        if ticker in positions_b or ticker in daily_traded_b[cur_date]:
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
        
        risk_budget = get_v15_wallet_risk_budget(wallet_equity, consecutive_losses, bar_rvol)
        friction = 0.002067 * entry_p
        qty_risk = int(risk_budget / (entry_p - active_sl + friction))
        
        qty_margin = int((avail_margin_b * 0.90) / entry_p)
        qty = min(qty_risk, qty_margin)
        
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
            avail_margin_b = (5.0 * wallet_equity * 0.95) - current_notional_b
            if avail_margin_b < 20.0 or len(positions_b) >= 2:
                break

df_mode_b = pd.DataFrame(trades_mode_b)

# Save results for inspection
OUT_DIR = ROOT_DIR / "strategies" / "V15_UNSEEN_TEST"
OUT_DIR.mkdir(parents=True, exist_ok=True)
df_mode_a.to_csv(OUT_DIR / "unseen_mode_a_trades.csv", index=False)
df_mode_b.to_csv(OUT_DIR / "unseen_mode_b_trades.csv", index=False)

print("\n" + "="*80)
print("100% UNSEEN OUT-OF-SAMPLE TEST RESULTS (10 NEW HIGH-BETA STOCKS)")
print("="*80)

# Mode A Summary
if len(df_mode_a) > 0:
    m_a = gate.metrics(df_mode_a)
    print("\n--- MODE A: CONSTANT-UNIT GATE-CERTIFIED LEDGER ---")
    print(f"Total Trades        : {m_a['n']}")
    print(f"Wins / Losses       : {m_a['wins']} Wins / {m_a['losses']} Losses")
    print(f"Win Rate            : {m_a['win_rate']:.2f}%")
    print(f"Profit Factor       : {m_a['pf']:.2f}")
    print(f"Gross Profit        : Rs {m_a['gross_profit']:.2f}")
    print(f"Gross Loss          : Rs {m_a['gross_loss']:.2f}")
    print(f"Net Realized Profit : Rs {m_a['net']:.2f} (from Rs 500 base, {m_a['net']/500*100:+.2f}% ROI)")
    print(f"Avg Win / Avg Loss  : Rs {m_a['avg_win']:.2f} / Rs {m_a['avg_loss']:.2f} (Payoff: {abs(m_a['avg_win']/m_a['avg_loss']) if m_a['avg_loss'] else 0:.2f}x)")
    print(f"Worst Single Loss   : Rs {df_mode_a['net_pnl'].min():.2f}")
    print(f"Largest Notional    : Rs {(df_mode_a['entry_price'] * df_mode_a['shares']).max():.2f}")

# Mode B Summary
if len(df_mode_b) > 0:
    w_b = df_mode_b[df_mode_b["net_pnl"] > 0]
    l_b = df_mode_b[df_mode_b["net_pnl"] < 0]
    gp_b = w_b["gross_pnl"].sum() if len(w_b) else 0
    gl_b = abs(l_b["gross_pnl"].sum()) if len(l_b) else 0
    pf_b = gp_b / gl_b if gl_b > 0 else 999.0
    path_b = df_mode_b["wallet_equity"].to_numpy(float)
    peak_b = np.maximum.accumulate(np.maximum(path_b, 500.0))
    mdd_b = float(((peak_b - path_b) / peak_b).max()) * 100.0 if len(path_b) else 0.0

    print("\n--- MODE B: STREAK-MODULATED HOUSE-MONEY COMPOUNDING WALLET ---")
    print(f"Starting Capital    : Rs 500.00")
    print(f"Final Realized Equity: Rs {wallet_equity:.2f} ({(wallet_equity - 500.0)/500*100:+.2f}% ROI / {wallet_equity/500:.2f}x)")
    print(f"Peak High-Water Mark: Rs {wallet_hwm:.2f} ({(wallet_hwm - 500.0)/500*100:+.2f}% ROI)")
    print(f"Total Trades        : {len(df_mode_b)}")
    print(f"Wins / Losses       : {len(w_b)} Wins / {len(l_b)} Losses")
    print(f"Win Rate            : {len(w_b)/len(df_mode_b)*100:.2f}%")
    print(f"Profit Factor       : {pf_b:.2f}")
    print(f"Net Realized Profit : +Rs {wallet_equity - 500.0:.2f}")
    print(f"Max Drawdown        : {mdd_b:.2f}%")

# Stock-by-Stock breakdown for Mode A
print("\n" + "-"*80)
print("STOCK-BY-STOCK PERFORMANCE BREAKDOWN (MODE A):")
print(f"{'Ticker':<12}{'Trades':<8}{'Wins':<6}{'Losses':<8}{'WR%':<8}{'Net PnL':<12}{'PF':<8}{'Worst Loss':<12}")
print("-"*80)
for t in tickers_unseen:
    sym = t.replace(".NS", "")
    sub = df_mode_a[df_mode_a["symbol"] == sym]
    if len(sub) == 0:
        print(f"{sym:<12}{0:<8}{0:<6}{0:<8}{'0.0%':<8}{'Rs 0.00':<12}{'0.00':<8}{'Rs 0.00':<12}")
        continue
    w = sub[sub["net_pnl"] > 0]
    l = sub[sub["net_pnl"] < 0]
    wr = len(w) / len(sub) * 100
    net = sub["net_pnl"].sum()
    gp = w["gross_pnl"].sum() if len(w) else 0
    gl = abs(l["gross_pnl"].sum()) if len(l) else 0
    pf = gp / gl if gl > 0 else 999.0
    wl = sub["net_pnl"].min()
    print(f"{sym:<12}{len(sub):<8}{len(w):<6}{len(l):<8}{wr:<7.1f}%{net:<12.2f}{pf:<8.2f}{wl:<12.2f}")
print("="*80)
