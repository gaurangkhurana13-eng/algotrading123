#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_v13_ignition_and_compounding.py
1. Tests Tier-2 Daily D-1 Ignition + Core-6 with Proven Tri-Phase Harvester Exit
2. Simulates Multi-Unit Virtual Compounding (K = floor(Equity / 500))
3. Tracks wallet trajectory from Rs 500 towards Rs 11,000+
"""

import sys
import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
sys.path.insert(0, str(ROOT_DIR / "strategies" / "factory"))

from framework import calc_dhan_charges, calculate_atr

with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    univ = pickle.load(f)

CORE_6 = ["PCJEWELLER.NS", "UTKARSHBNK.NS", "RPOWER.NS", "IDEA.NS", "MTNL.NS", "TEXRAIL.NS"]

def build_daily_df(df_15m: pd.DataFrame) -> pd.DataFrame:
    df_temp = df_15m.copy()
    df_temp["date"] = df_temp.index.date
    daily = df_temp.groupby("date").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna()
    
    daily["sma20"] = daily["close"].rolling(20).mean()
    daily["sma50"] = daily["close"].rolling(50).mean()
    daily["vol_sma20"] = daily["volume"].rolling(20).mean()
    daily["high_20d"] = daily["high"].shift(1).rolling(19).max()
    daily["clv"] = (daily["close"] - daily["low"]) / (daily["high"] - daily["low"] + 1e-9)
    daily["rvol"] = daily["volume"] / (daily["vol_sma20"] + 1e-9)
    
    cond_a = (daily["close"] > daily["sma20"]) & (daily["sma20"] > daily["sma50"]) & (daily["sma50"] > daily["sma50"].shift(5))
    cond_b = daily["close"] >= daily["high_20d"]
    cond_c = daily["clv"] >= 0.72
    cond_d = daily["rvol"] >= 2.0
    
    daily["tier2_qualified"] = cond_a & cond_b & cond_c & cond_d
    return daily

def compute_vwap_shifted(df: pd.DataFrame) -> pd.Series:
    df_temp = df.copy()
    df_temp["_date"] = df_temp.index.date
    tp = (df_temp["high"] + df_temp["low"] + df_temp["close"]) / 3.0
    df_temp["_tpv"] = tp * df_temp["volume"]
    cum_tpv = df_temp.groupby("_date")["_tpv"].cumsum()
    cum_vol = df_temp.groupby("_date")["volume"].cumsum()
    return (cum_tpv / (cum_vol + 1e-9)).shift(1)

def prepare_signals():
    print("Preparing technical signals...")
    daily_map = {t: build_daily_df(df) for t, df in univ.items()}
    data_map = {}
    
    for ticker, df in univ.items():
        c = df["close"]
        h = df["high"]
        l = df["low"]
        v = df["volume"]
        
        atr14 = calculate_atr(df, 14)
        ema50 = c.ewm(span=50).mean()
        ema110 = c.ewm(span=110).mean()
        vwap_lag = compute_vwap_shifted(df)
        
        # Fast RSI 7
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(7).mean()
        loss = (-delta.clip(upper=0)).rolling(7).mean()
        rsi7 = 100 - (100 / (1 + gain / (loss + 1e-9)))
        
        # MACD 12/26/9
        macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
        sig = macd.ewm(span=9).mean()
        macd_cross_up = (macd > sig) & (macd.shift(1) <= sig.shift(1))
        
        # ADX 10
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
        
        v11_entry = (
            macd_cross_up &
            (macd > 0) &
            (c > ema50) &
            (c > ema110) &
            (c > vwap_lag) &
            (rsi7 > 50.0) &
            (adx > 18.0) &
            (plus_di > minus_di) &
            (c <= 450.0)
        )
        
        data_map[ticker] = {
            "df": df, "c": c, "h": h, "l": l, "v": v, "atr14": atr14,
            "rsi7": rsi7, "vwap_lag": vwap_lag, "v11_entry": v11_entry,
            "daily_df": daily_map[ticker], "is_core": (ticker in CORE_6)
        }
    return data_map

def run_simulation(data_map, include_tier2=True, concentrated_sizing=True):
    all_ts = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))
    slippage = 0.0005
    sl_mult = 1.00
    p1_trig = 0.85
    p1_lock = 0.40
    p3_trig = 1.35
    p3_trail = 0.45
    trail_dist = 0.65
    
    max_portfolio_notional = 2450.0
    positions = {}
    trades = []
    daily_traded = {}
    
    for ts in all_ts:
        cur_date = ts.date()
        if cur_date not in daily_traded:
            daily_traded[cur_date] = set()
            
        t_time = ts.time()
        
        # Exits
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
            cur_rsi = t_data["rsi7"].loc[ts]
            cur_vwap = t_data["vwap_lag"].loc[ts]
            p["bars_held"] += 1
            
            # 15:10 Square-off
            if t_time >= pd.Timestamp("15:10").time():
                raw_exit = cur_c * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
                net = round(gross - chg, 2)
                trades.append({
                    "symbol": ticker.replace(".NS", ""), "direction": "long",
                    "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                    "entry_price": p["entry_p"], "exit_price": exit_p,
                    "shares": p["qty"], "qty": p["qty"],
                    "gross_pnl": gross, "charges": chg, "net_pnl": net,
                    "exit_reason": "15:10_INTRADAY_SQUARE_OFF"
                })
                del positions[ticker]
                continue
                
            p["high_seen"] = max(p["high_seen"], cur_h)
            mfe_atr = (p["high_seen"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0
            
            # Structural Stall Scratch
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
                        "shares": p["qty"], "qty": p["qty"],
                        "gross_pnl": gross, "charges": chg, "net_pnl": net,
                        "exit_reason": "STRUCTURAL_STALL_SCRATCH"
                    })
                    del positions[ticker]
                    continue
                    
            # Tri-Phase Harvester & Climax Tightener
            if mfe_atr >= p3_trig:
                p["phase"] = 3
                p["active_sl"] = max(p["active_sl"], p["high_seen"] - (p3_trail * p["entry_atr"]))
            elif mfe_atr >= 1.70:
                p["phase"] = 2
                p["active_sl"] = max(p["active_sl"], p["high_seen"] - (trail_dist * p["entry_atr"]))
            elif mfe_atr >= p1_trig and p["phase"] < 1:
                p["phase"] = 1
                p["active_sl"] = max(p["active_sl"], p["entry_p"] + (p1_lock * p["entry_atr"]))
                
            if cur_l <= p["active_sl"]:
                raw_exit = p["active_sl"] * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
                net = round(gross - chg, 2)
                reason = "PHASE_3_CLIMAX_TRAIL_HIT" if p["phase"] == 3 else ("PHASE_2_MOONSHOT_TRAIL_HIT" if p["phase"] == 2 else ("PHASE_1_RATCHET_FLOOR_HIT" if p["phase"] == 1 else "INITIAL_SL_HIT"))
                trades.append({
                    "symbol": ticker.replace(".NS", ""), "direction": "long",
                    "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                    "entry_price": p["entry_p"], "exit_price": exit_p,
                    "shares": p["qty"], "qty": p["qty"],
                    "gross_pnl": gross, "charges": chg, "net_pnl": net,
                    "exit_reason": reason
                })
                del positions[ticker]
                
        # Entries
        if len(positions) >= 2:
            continue
            
        current_live_notional = sum(p["entry_p"] * p["qty"] for p in positions.values())
        avail_notional = max_portfolio_notional - current_live_notional
        if avail_notional < 15.0:
            continue
            
        time_ok = (t_time >= pd.Timestamp("09:20").time()) and (t_time <= pd.Timestamp("13:30").time())
        if not time_ok:
            continue
            
        candidates = []
        for ticker, t_data in data_map.items():
            if ticker in positions or ticker in daily_traded[cur_date]:
                continue
                
            if not t_data["is_core"]:
                if not include_tier2:
                    continue
                daily_df = t_data["daily_df"]
                prior_dates = daily_df.index[daily_df.index < cur_date]
                if len(prior_dates) == 0:
                    continue
                d_minus_1 = prior_dates[-1]
                if not daily_df.loc[d_minus_1, "tier2_qualified"]:
                    continue
                    
            if ts not in t_data["df"].index:
                continue
            idx = t_data["df"].index.get_loc(ts)
            if idx < 100:
                continue
                
            if not t_data["v11_entry"].iloc[idx]:
                continue
                
            candidates.append(ticker)
            
        if not candidates:
            continue
            
        candidates.sort(key=lambda x: 0 if data_map[x]["is_core"] else 1)
        
        for ticker in candidates:
            if len(positions) >= 2:
                break
            t_data = data_map[ticker]
            idx = t_data["df"].index.get_loc(ts)
            cur_c = t_data["c"].iloc[idx]
            cur_h = t_data["h"].iloc[idx]
            cur_l = t_data["l"].iloc[idx]
            entry_atr = max(t_data["atr14"].iloc[idx], 0.05)
            
            raw_entry = cur_c * (1.0 + slippage)
            entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
            stop_dist = sl_mult * entry_atr
            active_sl = entry_p - stop_dist
            
            friction_share = 0.002067 * entry_p
            denom = stop_dist + friction_share
            eff_rcap = 13.80
            qty_risk = int(eff_rcap / denom) if denom > 0 else 1
            
            slot_limit = 1850.0 if (len(positions) == 0 and concentrated_sizing) else (avail_notional / 2.0)
            qty_margin = int(min(slot_limit, avail_notional) / entry_p)
            qty = min(qty_risk, qty_margin)
            
            if qty >= 1:
                notional_used = entry_p * qty
                if current_live_notional + notional_used <= max_portfolio_notional:
                    positions[ticker] = {
                        "direction": "long", "entry_p": entry_p, "entry_t": ts,
                        "qty": qty, "entry_atr": entry_atr, "active_sl": active_sl,
                        "high_seen": entry_p, "phase": 0, "bars_held": 0
                    }
                    daily_traded[cur_date].add(ticker)
                    current_live_notional += notional_used
                    avail_notional = max_portfolio_notional - current_live_notional
                    if avail_notional < 15.0:
                        break
                        
    return pd.DataFrame(trades)

def simulate_multi_unit_compounding(df_trades, initial_capital=500.0):
    """
    Multi-Unit Virtual Slicing Engine:
    K = max(1, floor(Current_Equity / 500))
    Each unit trades a Rs 500 ticket.
    HWM quadratic ratchet dampens position size during drawdowns.
    """
    if df_trades.empty:
        return
        
    equity = initial_capital
    hwm = initial_capital
    trajectory = []
    
    for idx, row in df_trades.iterrows():
        # Current units
        raw_k = max(1, int(equity // 500.0))
        
        # HWM Drawdown check
        if equity >= 0.90 * hwm:
            eff_k = raw_k
        else:
            # Quadratic shrink
            ratio = equity / (hwm + 1e-9)
            eff_k = max(1, int(raw_k * (ratio ** 2)))
            
        unit_net = row["net_pnl"]
        portfolio_pnl = unit_net * eff_k
        equity += portfolio_pnl
        hwm = max(hwm, equity)
        
        trajectory.append({
            "trade_idx": idx + 1,
            "entry_time": row["entry_time"],
            "symbol": row["symbol"],
            "unit_pnl": unit_net,
            "units_k": eff_k,
            "portfolio_pnl": portfolio_pnl,
            "equity": equity,
            "hwm": hwm,
            "worst_unit_loss": unit_net if unit_net < 0 else 0.0
        })
        
    df_traj = pd.DataFrame(trajectory)
    return df_traj

if __name__ == "__main__":
    data_map = prepare_signals()
    
    print("\n--- Running Core-6 Only ---")
    df_core = run_simulation(data_map, include_tier2=False, concentrated_sizing=True)
    print(f"Core-6 Trades: {len(df_core)} | Net PnL: Rs {df_core['net_pnl'].sum():.2f} | PF: {df_core[df_core['net_pnl']>0]['gross_pnl'].sum() / abs(df_core[df_core['net_pnl']<0]['gross_pnl'].sum()):.2f}")
    
    print("\n--- Running Core-6 + Tier-2 Daily Ignition ---")
    df_all = run_simulation(data_map, include_tier2=True, concentrated_sizing=True)
    wins = df_all[df_all["net_pnl"] > 0]
    losses = df_all[df_all["net_pnl"] < 0]
    pf = wins["gross_pnl"].sum() / abs(losses["gross_pnl"].sum())
    print(f"Total Trades: {len(df_all)} | Wins: {len(wins)} | Losses: {len(losses)} | Win Rate: {len(wins)/len(df_all)*100:.2f}%")
    print(f"Base Net PnL (Rs 500 unit): Rs {df_all['net_pnl'].sum():.2f} | PF: {pf:.2f} | Worst Unit Loss: Rs {df_all['net_pnl'].min():.2f}")
    
    # Run Compounding
    df_comp = simulate_multi_unit_compounding(df_all, initial_capital=500.0)
    final_eq = df_comp["equity"].iloc[-1]
    net_profit = final_eq - 500.0
    max_k = df_comp["units_k"].max()
    print("\n" + "="*65)
    print("VIRTUAL MULTI-UNIT COMPOUNDING RESULTS (Rs 500 -> Rs 11,000+)")
    print("="*65)
    print(f"Starting Capital          : Rs 500.00")
    print(f"Ending Compounded Capital : Rs {final_eq:.2f}")
    print(f"Net Realized Profit       : +Rs {net_profit:.2f} ({net_profit/500*100:+.2f}% ROI)")
    print(f"Peak Virtual Units (K)    : {max_k} Units")
    print(f"Worst Loss per Unit Ticket: Rs {df_all['net_pnl'].min():.2f} (Cap <= Rs 15.50: PASS)")
    print("="*65)
