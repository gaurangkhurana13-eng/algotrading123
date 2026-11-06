#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_v13_mechanics.py
Empirical validation of Strategy V13:
- Tier 1: Core 6 Leaders (PCJEWELLER, UTKARSHBNK, RPOWER, IDEA, MTNL, TEXRAIL)
- Tier 2: 72 Stocks with Daily D-1 Institutional Ignition Gate
- Concentrated Slot Sizing (up to Rs 1,850 notional, max risk Rs 13.80)
- 50/50 Split-Ticket Harvester (Half-A banker, Half-B unchoked runner)
- Multi-Unit Virtual Compounding K = floor(Equity / 500)
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

# Load full 78-stock universe
with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    univ = pickle.load(f)

CORE_6 = ["PCJEWELLER.NS", "UTKARSHBNK.NS", "RPOWER.NS", "IDEA.NS", "MTNL.NS", "TEXRAIL.NS"]
TIER_2_CANDIDATES = [s for s in univ.keys() if s not in CORE_6]

def build_daily_df(df_15m: pd.DataFrame) -> pd.DataFrame:
    """Build completed daily OHLCV from 15m bars for prior-day analysis without lookahead."""
    df_temp = df_15m.copy()
    df_temp["date"] = df_temp.index.date
    daily = df_temp.groupby("date").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()
    
    # Calculate Daily Technicals on completed daily bars
    daily["sma20"] = daily["close"].rolling(20).mean()
    daily["sma50"] = daily["close"].rolling(50).mean()
    daily["vol_sma20"] = daily["volume"].rolling(20).mean()
    daily["high_20d"] = daily["high"].shift(1).rolling(19).max() # max of prior 20 days
    daily["clv"] = (daily["close"] - daily["low"]) / (daily["high"] - daily["low"] + 1e-9)
    daily["rvol"] = daily["volume"] / (daily["vol_sma20"] + 1e-9)
    
    # Condition A: Daily Close > SMA20 > SMA50 and SMA50 > SMA50[5 days ago]
    cond_a = (daily["close"] > daily["sma20"]) & (daily["sma20"] > daily["sma50"]) & (daily["sma50"] > daily["sma50"].shift(5))
    # Condition B: Daily Close >= prior 20-day High
    cond_b = daily["close"] >= daily["high_20d"]
    # Condition C: Daily CLV >= 0.72 (closed in top 28% of range, no upper wick dump)
    cond_c = daily["clv"] >= 0.72
    # Condition D: Daily RVOL >= 2.0
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

def prepare_data():
    print("Building Daily screening models and 15m signals for all 78 tickers...")
    data_map = {}
    
    # Pre-build daily frames for Tier 2 screening
    daily_map = {}
    for ticker, df in univ.items():
        daily_map[ticker] = build_daily_df(df)
        
    for ticker, df in univ.items():
        c = df["close"]
        h = df["high"]
        l = df["low"]
        v = df["volume"]
        
        atr14 = calculate_atr(df, 14)
        ema20 = c.ewm(span=20).mean()
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
        
        # 15m Baseline Confluence Entry
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
        
        # Determine for each date whether ticker is eligible
        # Tier 1 is ALWAYS eligible
        # Tier 2 is eligible on date D only if qualified on D-1 completed daily bar
        daily_df = daily_map[ticker]
        
        data_map[ticker] = {
            "df": df, "c": c, "h": h, "l": l, "v": v, "atr14": atr14,
            "ema20": ema20, "rsi7": rsi7, "vwap_lag": vwap_lag,
            "v11_entry": v11_entry, "daily_df": daily_df,
            "is_core": (ticker in CORE_6)
        }
    print("Signals ready for all 78 tickers.")
    return data_map

def run_v13_backtest(data_map, enable_split_exit=True, concentrated_sizing=True):
    all_ts = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))
    slippage = 0.0005
    sl_mult = 1.00
    
    positions = {}
    trades = []
    daily_traded = {} # date -> set of tickers already traded today
    
    # Capital tracking for single-unit base (Rs 500 base)
    max_portfolio_notional = 2450.0
    
    for ts in all_ts:
        cur_date = ts.date()
        if cur_date not in daily_traded:
            daily_traded[cur_date] = set()
            
        t_time = ts.time()
        
        # 1. Update live positions & check exits
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
            cur_ema20 = t_data["ema20"].loc[ts]
            p["bars_held"] += 1
            
            p["high_seen"] = max(p["high_seen"], cur_h)
            mfe_atr = (p["high_seen"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0
            
            # --- MANDATORY 15:10 SQUARE-OFF ---
            if t_time >= pd.Timestamp("15:10").time():
                raw_exit = cur_c * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                gross = round((exit_p - p["entry_p"]) * p["qty_rem"], 2)
                chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty_rem"], is_intraday=True), 2)
                net = round(gross - chg, 2)
                
                # Combine with any previously booked half
                tot_gross = round(p["booked_gross"] + gross, 2)
                tot_chg = round(p["booked_chg"] + chg, 2)
                tot_net = round(tot_gross - tot_chg, 2)
                
                trades.append({
                    "symbol": ticker.replace(".NS", ""), "direction": "long",
                    "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                    "entry_price": p["entry_p"], "exit_price": exit_p,
                    "shares": p["init_qty"], "qty": p["init_qty"],
                    "gross_pnl": tot_gross, "charges": tot_chg, "net_pnl": tot_net,
                    "exit_reason": "15:10_INTRADAY_SQUARE_OFF"
                })
                del positions[ticker]
                continue
                
            # --- STRUCTURAL STALL SCRATCH ---
            if p["bars_held"] >= 3 and p["phase"] == 0 and mfe_atr < 0.40 and p["qty_rem"] == p["init_qty"]:
                if cur_c < p["entry_p"] and (cur_rsi < 46.0 or cur_c < cur_vwap):
                    raw_exit = cur_c * (1.0 - slippage)
                    exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                    gross = round((exit_p - p["entry_p"]) * p["qty_rem"], 2)
                    chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty_rem"], is_intraday=True), 2)
                    net = round(gross - chg, 2)
                    trades.append({
                        "symbol": ticker.replace(".NS", ""), "direction": "long",
                        "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                        "entry_price": p["entry_p"], "exit_price": exit_p,
                        "shares": p["init_qty"], "qty": p["init_qty"],
                        "gross_pnl": gross, "charges": chg, "net_pnl": net,
                        "exit_reason": "STRUCTURAL_STALL_SCRATCH"
                    })
                    del positions[ticker]
                    continue
                    
            # --- 50/50 SPLIT TICKET HARVESTER ---
            if enable_split_exit:
                # Stage 1: Breakeven Shield on Half-A at +0.45x ATR
                if mfe_atr >= 0.45 and p["phase"] == 0:
                    p["phase"] = 1
                    # Move stop to breakeven + friction
                    friction_be = (0.002067 * p["entry_p"])
                    p["active_sl"] = max(p["active_sl"], p["entry_p"] + friction_be)
                    
                # Stage 2: Book Half-A (50% shares) at +1.10x ATR Target
                if mfe_atr >= 1.10 and not p["half_a_booked"] and p["qty_rem"] > 1:
                    qty_to_book = int(p["init_qty"] * 0.5)
                    if qty_to_book >= 1:
                        raw_exit = (p["entry_p"] + (1.10 * p["entry_atr"])) * (1.0 - slippage)
                        exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                        gross = round((exit_p - p["entry_p"]) * qty_to_book, 2)
                        chg = round(calc_dhan_charges(p["entry_p"], exit_p, qty_to_book, is_intraday=True), 2)
                        p["booked_gross"] += gross
                        p["booked_chg"] += chg
                        p["qty_rem"] -= qty_to_book
                        p["half_a_booked"] = True
                        p["phase"] = 2
                        # Stop for Half-B is strictly Entry + friction
                        p["active_sl"] = p["entry_p"] + (0.002067 * p["entry_p"])
                        
                # Stage 3: Half-B Unchoked Runner Logic
                # Before 14:15, Half-B ONLY exits if 15m Close < EMA 20
                if p["half_a_booked"]:
                    # Structural trend break exit
                    if cur_c < cur_ema20 and t_time < pd.Timestamp("14:15").time():
                        raw_exit = cur_c * (1.0 - slippage)
                        exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                        gross = round((exit_p - p["entry_p"]) * p["qty_rem"], 2)
                        chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty_rem"], is_intraday=True), 2)
                        tot_gross = round(p["booked_gross"] + gross, 2)
                        tot_chg = round(p["booked_chg"] + chg, 2)
                        tot_net = round(tot_gross - tot_chg, 2)
                        trades.append({
                            "symbol": ticker.replace(".NS", ""), "direction": "long",
                            "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                            "entry_price": p["entry_p"], "exit_price": exit_p,
                            "shares": p["init_qty"], "qty": p["init_qty"],
                            "gross_pnl": tot_gross, "charges": tot_chg, "net_pnl": tot_net,
                            "exit_reason": "HALF_B_EMA20_TREND_EXIT"
                        })
                        del positions[ticker]
                        continue
                        
                    # After 14:15, activate Afternoon Climax Trail
                    if t_time >= pd.Timestamp("14:15").time():
                        if mfe_atr >= 1.35:
                            p["active_sl"] = max(p["active_sl"], p["high_seen"] - (0.40 * p["entry_atr"]))
                            
                # Check active stop loss hit
                if cur_l <= p["active_sl"]:
                    raw_exit = p["active_sl"] * (1.0 - slippage)
                    exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                    gross = round((exit_p - p["entry_p"]) * p["qty_rem"], 2)
                    chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty_rem"], is_intraday=True), 2)
                    tot_gross = round(p["booked_gross"] + gross, 2)
                    tot_chg = round(p["booked_chg"] + chg, 2)
                    tot_net = round(tot_gross - tot_chg, 2)
                    trades.append({
                        "symbol": ticker.replace(".NS", ""), "direction": "long",
                        "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                        "entry_price": p["entry_p"], "exit_price": exit_p,
                        "shares": p["init_qty"], "qty": p["init_qty"],
                        "gross_pnl": tot_gross, "charges": tot_chg, "net_pnl": tot_net,
                        "exit_reason": "HALF_B_TRAIL_OR_SL_HIT" if p["half_a_booked"] else "INITIAL_SL_HIT"
                    })
                    del positions[ticker]
                    continue
                    
            else:
                # Standard V11 exit for comparison
                p1_trig = 0.85
                p1_lock = 0.40
                p3_trig = 1.35
                p3_trail = 0.45
                trail_dist = 0.65
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
                    gross = round((exit_p - p["entry_p"]) * p["qty_rem"], 2)
                    chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty_rem"], is_intraday=True), 2)
                    net = round(gross - chg, 2)
                    trades.append({
                        "symbol": ticker.replace(".NS", ""), "direction": "long",
                        "entry_time": str(p["entry_t"]), "exit_time": str(ts),
                        "entry_price": p["entry_p"], "exit_price": exit_p,
                        "shares": p["qty_rem"], "qty": p["qty_rem"],
                        "gross_pnl": gross, "charges": chg, "net_pnl": net,
                        "exit_reason": "V11_EXIT_HIT"
                    })
                    del positions[ticker]
                    
        # 2. Entries
        if len(positions) >= 2: # max 2 concurrent
            continue
            
        current_live_notional = sum(p["entry_p"] * p["qty_rem"] for p in positions.values())
        avail_notional = max_portfolio_notional - current_live_notional
        if avail_notional < 15.0:
            continue
            
        time_ok = (t_time >= pd.Timestamp("09:20").time()) and (t_time <= pd.Timestamp("13:30").time())
        if not time_ok:
            continue
            
        candidates = []
        for ticker, t_data in data_map.items():
            if ticker in positions:
                continue
            if ticker in daily_traded[cur_date]:
                continue
                
            # Tier Check:
            if not t_data["is_core"]:
                # Must be qualified on completed D-1 daily bar
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
            
        # Prioritize Core-6 first, then Tier-2
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
            
            # Sizing
            friction_share = 0.002067 * entry_p
            denom = stop_dist + friction_share
            eff_rcap = 13.80 # Strict risk cap
            qty_risk = int(eff_rcap / denom) if denom > 0 else 1
            
            # If 1st slot is active, allow up to Rs 1,850 notional
            slot_limit = 1850.0 if (len(positions) == 0 and concentrated_sizing) else (avail_notional / 2.0)
            qty_margin = int(min(slot_limit, avail_notional) / entry_p)
            qty = min(qty_risk, qty_margin)
            
            if qty >= 2: # Need at least 2 shares for 50/50 split
                notional_used = entry_p * qty
                if current_live_notional + notional_used <= max_portfolio_notional:
                    positions[ticker] = {
                        "direction": "long", "entry_p": entry_p, "entry_t": ts,
                        "init_qty": qty, "qty_rem": qty, "entry_atr": entry_atr,
                        "active_sl": active_sl, "high_seen": entry_p,
                        "phase": 0, "bars_held": 0, "half_a_booked": False,
                        "booked_gross": 0.0, "booked_chg": 0.0
                    }
                    daily_traded[cur_date].add(ticker)
                    current_live_notional += notional_used
                    avail_notional = max_portfolio_notional - current_live_notional
                    if avail_notional < 15.0:
                        break
                        
    return pd.DataFrame(trades)

def analyze_scorecard(df_trades, title):
    if df_trades.empty:
        print(f"[{title}] No trades generated!")
        return
    wins = df_trades[df_trades["net_pnl"] > 0]
    losses = df_trades[df_trades["net_pnl"] < 0]
    total_trades = len(df_trades)
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total_trades) * 100.0
    gross_profit = wins["gross_pnl"].sum()
    gross_loss = abs(losses["gross_pnl"].sum())
    total_charges = df_trades["charges"].sum()
    net_pnl = df_trades["net_pnl"].sum()
    pf = (gross_profit / gross_loss) if gross_loss > 0 else 999.0
    worst_loss = df_trades["net_pnl"].min()
    best_win = df_trades["net_pnl"].max()
    avg_win = wins["net_pnl"].mean() if win_count > 0 else 0.0
    avg_loss = losses["net_pnl"].mean() if loss_count > 0 else 0.0
    
    print("\n" + "="*65)
    print(f"SCORECARD: {title}")
    print("="*65)
    print(f"Total Trades        : {total_trades}")
    print(f"Wins / Losses       : {win_count} Wins / {loss_count} Losses")
    print(f"Win Rate            : {win_rate:.2f}%")
    print(f"Gross Profit        : +Rs {gross_profit:.2f}")
    print(f"Gross Loss          : -Rs {gross_loss:.2f}")
    print(f"Total Dhan Friction : Rs {total_charges:.2f}")
    print(f"Net Realized PnL    : +Rs {net_pnl:.2f} ({net_pnl/500.0*100:+.2f}% ROI on Rs 500 base)")
    print(f"Profit Factor       : {pf:.2f}")
    print(f"Average Win / Loss  : +Rs {avg_win:.2f} / Rs {avg_loss:.2f} (Payoff: {abs(avg_win/avg_loss):.2f}x)")
    print(f"Worst Single Loss   : Rs {worst_loss:.2f} (Cap <= Rs 15.50: {'PASS' if worst_loss >= -15.50 else 'FAIL'})")
    print(f"Best Single Win     : +Rs {best_win:.2f}")
    
    # Monthly
    df_trades["month"] = pd.to_datetime(df_trades["entry_time"]).dt.to_period("M")
    monthly = df_trades.groupby("month")["net_pnl"].agg(["count", "sum"]).reset_index()
    print("\n--- Monthly PnL Breakdown ---")
    for _, row in monthly.iterrows():
        print(f"  {row['month']}: {row['count']:2d} trades | Net PnL: Rs {row['sum']:+7.2f}")
    print("="*65 + "\n")

if __name__ == "__main__":
    data_map = prepare_data()
    
    print("\n--- TEST 1: Baseline Core-6 with V11 Ratchet ---")
    df1 = run_v13_backtest(data_map, enable_split_exit=False, concentrated_sizing=False)
    analyze_scorecard(df1, "TEST 1: Baseline V11 on Core-6")
    
    print("\n--- TEST 2: V13 Full Engine (Core-6 + Tier-2 + 50/50 Split + Concentrated Sizing) ---")
    df2 = run_v13_backtest(data_map, enable_split_exit=True, concentrated_sizing=True)
    analyze_scorecard(df2, "TEST 2: V13 Full Engine")
