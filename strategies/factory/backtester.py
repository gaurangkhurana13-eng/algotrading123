"""
Agent 4: Backtest Engineer (backtester.py)
Executes deterministic 15-minute backtests with exact Dhan costs, slippage,
and strict real-time portfolio margin pool tracking (max total open notional <= Rs 2,450).
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from framework import calc_dhan_charges, compute_metrics

def run_portfolio_simulation(data_map, basket, signal_fn, params, mode="intraday"):
    # Unified chronological timeline
    all_ts = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))
    
    capital = 500.0
    max_portfolio_notional = 2450.0
    slippage = 0.0005
    sl_mult = params.get("sl_mult", 1.00)
    p1_trig = params.get("p1_trig", 0.85)
    p1_lock = params.get("p1_lock", 0.40)
    p2_trig = params.get("p2_trig", 1.70)
    trail_dist = params.get("trail_dist", 0.65)
    
    positions = {}
    trades = []
    
    for ts in all_ts:
        t_time = ts.time()
        
        # 1. Process EXITS for live positions
        live_tickers = list(positions.keys())
        for ticker in live_tickers:
            p = positions[ticker]
            t_data = data_map[ticker]
            if ts not in t_data["df"].index:
                continue
            row = t_data["df"].loc[ts]
            cur_c = row["close"]
            cur_h = row["high"]
            cur_l = row["low"]
            
            # Intraday 15:10 square-off check
            if mode == "intraday" and t_time >= pd.Timestamp("15:10").time():
                raw_exit = cur_c * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
                net = round(gross - chg, 2)
                clean_sym = ticker.replace(".NS", "").replace("NSE:", "")
                trades.append({
                    "symbol": clean_sym, "direction": "long",
                    "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                    "entry_price": p["entry_p"], "exit_price": exit_p,
                    "shares": p["qty"], "qty": p["qty"], "gross_pnl": gross, "charges": chg,
                    "net_pnl": net, "exit_reason": "15:10_INTRADAY_SQUARE_OFF"
                })
                del positions[ticker]
                continue
                
            # Ratchet & trailing logic
            p["high_seen"] = max(p["high_seen"], cur_h)
            mfe_atr = (p["high_seen"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0
            
            if mfe_atr >= p2_trig:
                p["phase"] = 2
                p["active_sl"] = max(p["active_sl"], p["high_seen"] - (trail_dist * p["entry_atr"]))
            elif mfe_atr >= p1_trig and p["phase"] < 1:
                p["phase"] = 1
                p["active_sl"] = max(p["active_sl"], p["entry_p"] + (p1_lock * p["entry_atr"]))
                
            # Exit check: Trailing stop or floor hit
            if cur_l <= p["active_sl"]:
                raw_exit = p["active_sl"] * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=(mode=="intraday")), 2)
                net = round(gross - chg, 2)
                reason = "PHASE_2_MOONSHOT_TRAIL_HIT" if p["phase"] == 2 else ("PHASE_1_RATCHET_FLOOR_HIT" if p["phase"] == 1 else "INITIAL_SL_HIT")
                clean_sym = ticker.replace(".NS", "").replace("NSE:", "")
                trades.append({
                    "symbol": clean_sym, "direction": "long",
                    "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                    "entry_price": p["entry_p"], "exit_price": exit_p,
                    "shares": p["qty"], "qty": p["qty"], "gross_pnl": gross, "charges": chg,
                    "net_pnl": net, "exit_reason": reason
                })
                del positions[ticker]
                
        # 2. Process ENTRIES with strict portfolio margin cap
        current_live_notional = sum(p["entry_p"] * p["qty"] for p in positions.values())
        avail_notional = max_portfolio_notional - current_live_notional
        
        if avail_notional < 15.0:
            continue
            
        time_ok = (t_time >= pd.Timestamp("09:20").time()) and (t_time <= pd.Timestamp("13:30").time() if mode=="intraday" else True)
        if not time_ok:
            continue
            
        for ticker in basket:
            if ticker in positions:
                continue
            t_data = data_map[ticker]
            if ts not in t_data["df"].index:
                continue
            idx = t_data["df"].index.get_loc(ts)
            if idx < 100:
                continue
            if not t_data["entry_cond"].iloc[idx]:
                continue
                
            cur_c = t_data["df"]["close"].iloc[idx]
            cur_h = t_data["df"]["high"].iloc[idx]
            cur_l = t_data["df"]["low"].iloc[idx]
            
            raw_entry = cur_c * (1.0 + slippage)
            entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
            entry_atr = max(t_data["atr"].iloc[idx], 0.05)
            
            stop_dist = sl_mult * entry_atr
            friction_share = (0.002067 if mode == "intraday" else 0.0045) * entry_p
            denom = stop_dist + friction_share
            
            eff_rcap = 14.00 if mode == "intraday" else 11.00
            qty_risk = int(eff_rcap / denom) if denom > 0 else 1
            qty_margin = int(avail_notional / entry_p)
            qty = min(qty_risk, qty_margin)
            
            if qty >= 1:
                notional_used = entry_p * qty
                if current_live_notional + notional_used <= max_portfolio_notional:
                    positions[ticker] = {
                        "entry_p": entry_p,
                        "entry_t": ts,
                        "qty": qty,
                        "entry_atr": entry_atr,
                        "active_sl": entry_p - stop_dist,
                        "high_seen": entry_p,
                        "phase": 0
                    }
                    current_live_notional += notional_used
                    avail_notional = max_portfolio_notional - current_live_notional
                    if avail_notional < 15.0:
                        break
                        
    return pd.DataFrame(trades)
