import sys
from pathlib import Path
ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies" / "V25"))
import run_v25 as v25
import pandas as pd
import numpy as np

def run_variant(start_date, end_date, rank_by="c_score", base_risk_mode="v23_dynamic"):
    ts_range = [ts for ts in v25.all_ts if start_date <= ts <= end_date and ts.weekday() < 5]
    positions = {}
    trades = []
    daily_traded = {}
    last_exit_times = set()
    slippage = 0.0005
    hard_sl_mult = 0.90
    alpha_streak = 0.082
    beta_streak = 0.035
    initial_equity = 500.0
    
    for ts in ts_range:
        cur_date = ts.date()
        cur_ts_date = pd.Timestamp(cur_date)
        if cur_date not in daily_traded:
            daily_traded[cur_date] = set()
        t_time = ts.time()
        t_time_int = t_time.hour * 100 + t_time.minute
        
        # 1. Manage Active Positions
        for ticker in list(positions.keys()):
            p = positions[ticker]
            t_data = v25.data_map[ticker]
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
                chg = round(v25.calc_dhan_charges(p["entry_p"], exit_fill, tot_qty, is_intraday=True), 2)
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
                
        ek = initial_equity + sum(t["net_pnl"] for t in trades)
        hwm = initial_equity
        cum = initial_equity
        for t in trades:
            cum += t["net_pnl"]
            if cum > hwm:
                hwm = cum
        dd_frac = (hwm - ek) / max(hwm, initial_equity)
        
        if len(positions) >= 1 or ts in last_exit_times or t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:15").time():
            continue
            
        current_notional = sum(p["entry_p"] * p["qty"] for p in positions.values())
        avail_margin = (0.95 * ek) - (current_notional / 5.0)
        if avail_margin < 10.0:
            continue
        avail_notional = avail_margin * 5.0
        
        if base_risk_mode == "deepseek_3_5":
            kappa_dd = np.clip((0.15 - dd_frac) / (0.15 - 0.07), 0.0, 1.0)
            cur_day_bench = v25.bench_day_ret_series.loc[ts] if ts in v25.bench_day_ret_series.index else 0.0
            kappa_regime = 1.30 if cur_day_bench > 0 else 0.85
            f_eff = 0.035 * kappa_dd * kappa_regime
            target_risk = min(f_eff * ek, min(0.09 * ek, 258.0))
        else:
            # v23_dynamic
            consec_losses = 0
            for t in reversed(trades):
                if t["net_pnl"] < 0:
                    consec_losses += 1
                else:
                    break
            if consec_losses >= 2 or (ek > 2500.0 and dd_frac > 0.16):
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
            if dd_frac > 0.15:
                gate_v2_cap *= 0.5
            target_risk = min(raw_r, gate_v2_cap)
            
        candidates = []
        for ticker in v25.tickers_combined:
            if ticker in positions or ticker in daily_traded[cur_date]:
                continue
            t_data = v25.data_map[ticker]
            if ts not in t_data["df"].index:
                continue
            idx = t_data["df"].index.get_loc(ts)
            if idx < 100 or not t_data["g4_cond"].iloc[idx]:
                continue
                
            if not (t_data["g1_cond"].iloc[idx] and t_data["g2_cond"].iloc[idx]):
                continue
                
            prev_days = t_data["df_daily"].loc[:cur_ts_date]
            erm_val = 0.0
            if len(prev_days) >= 2:
                yest_row = prev_days.iloc[-2]
                erm_val = yest_row.get("erm", 0.0)
                
            cur_c = t_data["c"].iloc[idx]
            cur_h = t_data["h"].iloc[idx]
            cur_l = t_data["l"].iloc[idx]
            cur_v = t_data["v"].iloc[idx]
            entry_atr = max(t_data["atr14"].iloc[idx], 0.05)
            
            candidates.append({
                "ticker": ticker, "c": cur_c, "h": cur_h, "l": cur_l, "v": cur_v,
                "atr": entry_atr, "g1_mag": t_data["g1_mag"].iloc[idx],
                "g2_mag": t_data["g2_mag"].iloc[idx], "rvol": t_data["rvol"].iloc[idx],
                "erm": erm_val
            })
            
        if not candidates:
            continue
            
        if rank_by == "g2_mag":
            candidates.sort(key=lambda x: x["g2_mag"], reverse=True)
        else:
            if len(candidates) > 1:
                for feat in ["g1_mag", "g2_mag", "rvol", "erm"]:
                    vals = [c[feat] for c in candidates]
                    m = np.mean(vals)
                    s = np.std(vals) + 1e-9
                    for c in candidates:
                        c[f"z_{feat}"] = np.clip((c[feat] - m) / s, -3.0, 3.0)
                for c in candidates:
                    c["score"] = 0.35 * c["z_g1_mag"] + 0.30 * c["z_g2_mag"] + 0.20 * c["z_rvol"] + 0.15 * c["z_erm"]
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

t_chop_start = pd.Timestamp("2025-06-01 09:15:00")
t_chop_end = pd.Timestamp("2025-12-31 15:30:00")
t_2026_start = pd.Timestamp("2026-01-01 09:15:00")
t_2026_end = pd.Timestamp("2026-10-29 15:30:00")

def fmt(tdf):
    if len(tdf) == 0: return "0 trades"
    wins = tdf[tdf["net_pnl"] > 0]
    wr = len(wins)/len(tdf)*100.0
    net = tdf["net_pnl"].sum()
    cum = 500.0 + tdf["net_pnl"].cumsum()
    hwm = cum.cummax()
    dd = (hwm - cum)/hwm * 100.0
    return f"N={len(tdf)}, WR={wr:.1f}%, Net=Rs.{net:.2f}, DD={dd.max():.1f}%, Final=Rs.{cum.iloc[-1]:.2f}"

variants = [
    ("A: DeepSeek 3.5% Risk + C-Score", "c_score", "deepseek_3_5"),
    ("B: DeepSeek 3.5% Risk + G2 Ranking", "g2_mag", "deepseek_3_5"),
    ("C: Dynamic Compounding Risk + G2 Ranking", "g2_mag", "v23_dynamic"),
    ("D: Dynamic Compounding Risk + C-Score", "c_score", "v23_dynamic")
]

print("\n--- VARIANT COMPARISON MATRIX ---")
for label, r_mode, s_mode in variants:
    tdf_c = run_variant(t_chop_start, t_chop_end, rank_by=r_mode, base_risk_mode=s_mode)
    tdf_2 = run_variant(t_2026_start, t_2026_end, rank_by=r_mode, base_risk_mode=s_mode)
    print(f"\n>> {label}")
    print(f"   2025 Monsoon Chop : {fmt(tdf_c)}")
    print(f"   2026 Modern Market: {fmt(tdf_2)}")
