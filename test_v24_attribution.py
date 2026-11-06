import sys
from pathlib import Path
ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR))
import test_v24_dev as dev
import pandas as pd

t_chop_start = pd.Timestamp("2025-06-01 09:15:00")
t_chop_end = pd.Timestamp("2025-12-31 15:30:00")
t_2026_start = pd.Timestamp("2026-01-01 09:15:00")
t_2026_end = pd.Timestamp("2026-10-29 15:30:00")

configs = [
    ("Baseline V23 (No RS, No G1G2)", False, False, False),
    ("V23 + Monthly RS Only", True, False, False),
    ("V23 + G1 Only", False, True, False),
    ("V23 + G2 Only", False, False, True),
    ("V23 + Both G1 and G2", False, True, True),
    ("Full DeepSeek V24 (RS + G1 + G2)", True, True, True),
]

def run_custom(start_date, end_date, use_rs, use_g1, use_g2):
    ts_range = [ts for ts in dev.all_ts if start_date <= ts <= end_date and ts.weekday() < 5]
    positions = {}
    trades = []
    daily_traded = {}
    last_exit_times = set()
    slippage = 0.0005
    hard_sl_mult = 0.90
    alpha_streak = 0.082
    beta_streak = 0.035
    
    for ts in ts_range:
        cur_date = ts.date()
        m_str = ts.strftime("%Y-%m")
        if cur_date not in daily_traded:
            daily_traded[cur_date] = set()
        t_time = ts.time()
        t_time_int = t_time.hour * 100 + t_time.minute
        
        for ticker in list(positions.keys()):
            p = positions[ticker]
            t_data = dev.data_map[ticker]
            if ts not in t_data["df"].index:
                continue
            row = t_data["df"].loc[ts]
            cur_c, cur_h, cur_l = row["close"], row["high"], row["low"]
            cur_rsi = t_data["rsi7"].loc[ts]
            cur_vwap = t_data["vwap_lag"].loc[ts]
            cur_v = row["volume"]
            
            p["bars_held"] += 1
            p["high_seen"] = max(p["high_seen"], cur_h)
            mfe_atr = (p["high_seen"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0
            
            if mfe_atr >= 2.50:
                p["active_sl"] = max(p["active_sl"], round(p["high_seen"] - (0.12 * p["entry_atr"]), 2))
            elif mfe_atr >= 1.50 or (mfe_atr >= 1.15 and t_time_int >= 1415):
                p["active_sl"] = max(p["active_sl"], round(p["high_seen"] - (0.18 * p["entry_atr"]), 2))
            elif mfe_atr >= 1.15:
                p["active_sl"] = max(p["active_sl"], round(p["high_seen"] - (0.28 * p["entry_atr"]), 2))
            elif mfe_atr >= 0.85:
                p["active_sl"] = max(p["active_sl"], round(p["entry_p"] + (0.40 * p["entry_atr"]), 2))
                
            def finalize_trade(exit_price_raw, exit_reason):
                tot_qty = p["qty"]
                if cur_v > 0 and tot_qty > int(0.045 * cur_v):
                    tot_qty = max(1, int(0.045 * cur_v))
                exit_fill = round(max(cur_l, min(cur_h, exit_price_raw)), 2)
                gross = round(tot_qty * (exit_fill - p["entry_p"]), 2)
                chg = round(dev.calc_dhan_charges(p["entry_p"], exit_fill, tot_qty, is_intraday=True), 2)
                net = round(gross - chg, 2)
                trades.append({
                    "symbol": ticker, "entry_p": p["entry_p"], "exit_p": exit_fill,
                    "gross_pnl": gross, "charges": chg, "net_pnl": net, "reason": exit_reason
                })
                last_exit_times.add(ts)
                del positions[ticker]
                
            if cur_l <= p["active_sl"]:
                finalize_trade(p["active_sl"] * (1.0 - slippage), "STOP")
                continue
            if p["bars_held"] >= 3 and mfe_atr < 0.40 and cur_c < p["entry_p"] and (cur_rsi < 46.0 or cur_c < cur_vwap):
                finalize_trade(cur_c * (1.0 - slippage), "SCRATCH")
                continue
            if t_time >= pd.Timestamp("14:45").time():
                finalize_trade(cur_c * (1.0 - slippage), "EOD")
                continue
                
        ek = 500.0 + sum(t["net_pnl"] for t in trades)
        hwm = 500.0
        cum = 500.0
        for t in trades:
            cum += t["net_pnl"]
            if cum > hwm:
                hwm = cum
        dd_pct = (hwm - ek) / max(hwm, 500.0)
        
        if len(positions) >= 1 or ts in last_exit_times or t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:15").time():
            continue
            
        current_notional = sum(p["entry_p"] * p["qty"] for p in positions.values())
        avail_margin = (0.95 * ek) - (current_notional / 5.0)
        if avail_margin < 10.0:
            continue
        avail_notional = avail_margin * 5.0
        
        consec_losses = 0
        for t in reversed(trades):
            if t["net_pnl"] < 0:
                consec_losses += 1
            else:
                break
        if consec_losses >= 2 or (ek > 2500.0 and dd_pct > 0.16):
            alpha, beta = 0.015, 0.010
        elif consec_losses == 1:
            alpha, beta = 0.035, 0.020
        else:
            alpha, beta = alpha_streak, beta_streak
            
        if ek <= 2500.0:
            raw_r = 14.0 + (alpha * max(0.0, ek - 500.0))
        else:
            raw_r = 14.0 + (alpha * 2000.0) + (beta * (ek - 2500.0))
        gate_v2_cap = min(0.09 * ek, 258.0)
        if dd_pct > 0.15:
            gate_v2_cap *= 0.5
        target_risk = min(raw_r, gate_v2_cap)
        
        candidates = []
        allowed = dev.monthly_universe.get(m_str, {}) if use_rs else {t: 1.0 for t in dev.data_map.keys()}
        
        for ticker in dev.tickers_combined:
            if ticker in positions or ticker in daily_traded[cur_date] or ticker not in allowed:
                continue
            t_data = dev.data_map[ticker]
            if ts not in t_data["df"].index:
                continue
            idx = t_data["df"].index.get_loc(ts)
            if idx < 100 or not t_data["g4_cond"].iloc[idx]:
                continue
                
            if use_g1 and not t_data["g1_cond"].iloc[idx]:
                continue
            if use_g2 and not t_data["g2_cond"].iloc[idx]:
                continue
                
            cur_c = t_data["c"].iloc[idx]
            cur_h = t_data["h"].iloc[idx]
            cur_l = t_data["l"].iloc[idx]
            cur_v = t_data["v"].iloc[idx]
            entry_atr = max(t_data["atr14"].iloc[idx], 0.05)
            excess_1h = t_data["excess_1h"].iloc[idx]
            conviction = t_data["rvol"].iloc[idx] * t_data["er10"].iloc[idx]
            
            candidates.append({
                "ticker": ticker, "c": cur_c, "h": cur_h, "l": cur_l, "v": cur_v,
                "atr": entry_atr, "score": excess_1h if use_g2 else conviction
            })
            
        if not candidates:
            continue
            
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]
        ticker = best["ticker"]
        cur_c = best["c"]
        cur_h = best["h"]
        cur_l = best["l"]
        cur_v = best["v"]
        entry_atr = best["atr"]
        
        raw_entry = cur_c * (1.0 + slippage)
        entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
        active_sl = round(entry_p - (hard_sl_mult * entry_atr), 2)
        sl_dist = max(entry_p - active_sl, 0.05)
        friction = 0.002067 * entry_p
        denom = sl_dist + friction
        
        qty_r = int(target_risk / denom)
        qty_m = int((avail_notional * 0.90) / entry_p)
        qty = min(qty_r, qty_m)
        if cur_v > 0:
            qty = min(qty, int(0.045 * cur_v))
            
        if qty >= 1:
            positions[ticker] = {
                "entry_p": entry_p, "qty": qty, "entry_atr": entry_atr,
                "active_sl": active_sl, "initial_sl": active_sl,
                "high_seen": entry_p, "bars_held": 0, "entry_t": ts
            }
            daily_traded[cur_date].add(ticker)
            
    tdf = pd.DataFrame(trades)
    return tdf

def calc_summary(tdf):
    if len(tdf) == 0:
        return "0 trades"
    wins = tdf[tdf["net_pnl"] > 0]
    wr = len(wins) / len(tdf) * 100.0
    net = tdf["net_pnl"].sum()
    cum = 500.0 + tdf["net_pnl"].cumsum()
    hwm = cum.cummax()
    dd = (hwm - cum) / hwm * 100.0
    max_dd = dd.max() if len(dd) > 0 else 0.0
    return f"N={len(tdf)}, WR={wr:.1f}%, Net=Rs.{net:.2f}, DD={max_dd:.1f}%, Final=Rs.{cum.iloc[-1]:.2f}"

print("Running comparative matrix...")
results = []
for name, use_rs, use_g1, use_g2 in configs:
    tdf_chop = run_custom(t_chop_start, t_chop_end, use_rs, use_g1, use_g2)
    tdf_2026 = run_custom(t_2026_start, t_2026_end, use_rs, use_g1, use_g2)
    results.append({
        "Config": name,
        "2025 Chop": calc_summary(tdf_chop),
        "2026 Modern": calc_summary(tdf_2026)
    })

print("\n--- COMPARATIVE FILTER MATRIX ---")
for r in results:
    print(f"\n>> {r['Config']}:")
    print(f"   2025 Monsoon Chop : {r['2025 Chop']}")
    print(f"   2026 Modern Market: {r['2026 Modern']}")
