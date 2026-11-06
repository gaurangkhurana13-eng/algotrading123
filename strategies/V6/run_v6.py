"""
Strategy V6: Multi-Asset High-Beta Retail Momentum Engine
Executed across High-Beta Retail Leaders with strict portfolio margin tracking (max total notional <= Rs 2,450)
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from framework import calc_dhan_charges, calculate_atr, compute_metrics

def prepare_ticker_data(df):
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]
    atr14 = calculate_atr(df, 14)
    ema50 = c.ewm(span=50).mean()
    
    # VWAP shifted
    df_temp = df.copy()
    df_temp["_date"] = df_temp.index.date
    tp = (df_temp["high"] + df_temp["low"] + df_temp["close"]) / 3.0
    df_temp["_tpv"] = tp * df_temp["volume"]
    cum_tpv = df_temp.groupby("_date")["_tpv"].cumsum()
    cum_vol = df_temp.groupby("_date")["volume"].cumsum()
    vwap_lag = (cum_tpv / (cum_vol + 1e-9)).shift(1)
    
    # Fast RSI 7
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(7).mean()
    loss = (-delta.clip(upper=0)).rolling(7).mean()
    rsi7 = 100 - (100 / (1 + gain / (loss + 1e-9)))
    
    # MACD 12/26/9
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    sig = macd.ewm(span=9).mean()
    
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
    
    macd_cross = (macd > sig) & (macd.shift(1) <= sig.shift(1))
    entry_cond = macd_cross & (macd > 0) & (c > ema50) & (rsi7 > 50.0) & (adx > 18.0) & (plus_di > minus_di) & (c > vwap_lag)
    
    return {
        "df": df, "c": c, "h": h, "l": l, "atr14": atr14, "entry_cond": entry_cond
    }

def run_portfolio_simulation(data_map, basket, mode="intraday"):
    # Unified timeline
    all_ts = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))
    
    # Portfolio constraints
    capital = 500.0
    max_portfolio_notional = 2450.0
    r_cap = 15.00
    slippage = 0.0005
    sl_mult = 1.00
    p1_trig = 0.85
    p1_lock = 0.40
    p2_trig = 1.70
    trail_dist = 0.65
    
    positions = {} # ticker -> pos_info
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
            
            # Intraday square-off check
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
            continue # Margin full
            
        time_ok = (t_time >= pd.Timestamp("09:20").time()) and (t_time <= pd.Timestamp("13:30").time() if mode=="intraday" else True)
        if not time_ok:
            continue
            
        for ticker in basket:
            if ticker in positions:
                continue # Already in this stock
            t_data = data_map[ticker]
            if ts not in t_data["df"].index:
                continue
            idx = t_data["df"].index.get_loc(ts)
            if idx < 100:
                continue
            if not t_data["entry_cond"].iloc[idx]:
                continue
                
            cur_c = t_data["c"].iloc[idx]
            cur_h = t_data["h"].iloc[idx]
            cur_l = t_data["l"].iloc[idx]
            
            raw_entry = cur_c * (1.0 + slippage)
            entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
            entry_atr = max(t_data["atr14"].iloc[idx], 0.05)
            
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
                        break # No more margin for other stocks on this bar
                        
    return pd.DataFrame(trades)

def main():
    v_dir = Path(__file__).parent
    data_dir = Path("C:/Users/admin/.gemini/antigravity/scratch/new_trade/data")
    
    basket = ["PCJEWELLER.NS", "RPOWER.NS", "IDEA.NS", "MTNL.NS", "HCC.NS", "UJJIVANSFB.NS"]
    
    data_map = {}
    for t in basket:
        clean = t.replace(".NS", "").lower()
        pkl_path = data_dir / f"{clean}_15m_1y.pkl"
        if pkl_path.exists():
            df_s = pd.read_pickle(pkl_path)
            data_map[t] = prepare_ticker_data(df_s)
        else:
            print(f"Warning: {pkl_path} not found")
            
    print(f"Loaded {len(data_map)} stocks for Portfolio Simulation.")
    
    df_intra = run_portfolio_simulation(data_map, basket, mode="intraday")
    df_swing = run_portfolio_simulation(data_map, basket, mode="swing")
    
    # Save CSVs
    intra_csv = v_dir / "trades_intraday.csv"
    swing_csv = v_dir / "trades_swing.csv"
    df_intra.to_csv(intra_csv, index=False)
    df_swing.to_csv(swing_csv, index=False)
    
    m_intra = compute_metrics(df_intra, capital=500.0)
    m_swing = compute_metrics(df_swing, capital=500.0)
    
    # Generate results.txt
    res_path = v_dir / "results.txt"
    report = f"""================================================================================
  BHAVISHYA STRATEGY V6: HIGH-BETA RETAIL MOMENTUM ENGINE
================================================================================

1. LOGIC SUMMARY
   - Portfolio Basket: High-Beta Retail Leaders (PCJEWELLER, RPOWER, IDEA, MTNL, HCC, UJJIVANSFB)
   - MACD Zero-Line Cross (Fast 12, Slow 26, Sig 9 while MACD > 0)
   - Macro Trend Gate (Close > 50 EMA)
   - Institutional VWAP Anchor (Close > Shifted VWAP.shift(1))
   - Fast RSI Buyer Dominance (RSI 7 > 50.0)
   - ADX Trend Strength Confirmation (ADX 10 > 18.0 and +DI > -DI)
   - Dynamic Moonshot Trailing Ratchet:
     * Phase 1: At +0.85x ATR MFE -> Lock profit at +0.40x ATR floor
     * Phase 2: At +1.70x ATR MFE -> Dynamic Moonshot Trailing (HighSeen - 0.65x ATR)
     * Initial Stop: 1.00x ATR
   - Realistic Execution: Dhan MIS costs + 0.05% slippage, Rs 500 base, 5x leverage max Rs 2,450
   - Multi-Stock Portfolio Concurrency: Real-time margin pool tracking ensuring total open notional <= Rs 2,450

--------------------------------------------------------------------------------
INTRADAY
--------------------------------------------------------------------------------
Trades: {m_intra['total_trades']}
Wins: {m_intra['wins']}
Losses: {m_intra['losses']}
Win Rate: {m_intra['win_rate']}%
Net Profit: Rs {m_intra['net_pnl']}
Profit Factor: {m_intra['profit_factor']}
Avg Win: Rs {m_intra['avg_win']}
Avg Loss: {m_intra['avg_loss']}
Payoff Ratio: {round(abs(m_intra['avg_win']/m_intra['avg_loss']), 2) if m_intra['avg_loss'] != 0 else 'N/A'}x
Avg Dhan Charges: Rs {m_intra['avg_charge']}
Max Drawdown: Rs {m_intra['max_drawdown']}
Final Capital: Rs {m_intra['final_capital']} (ROI: {m_intra['roi_pct']}%)

--------------------------------------------------------------------------------
SWING
--------------------------------------------------------------------------------
Trades: {m_swing['total_trades']}
Wins: {m_swing['wins']}
Losses: {m_swing['losses']}
Win Rate: {m_swing['win_rate']}%
Net Profit: Rs {m_swing['net_pnl']}
Profit Factor: {m_swing['profit_factor']}
Avg Win: Rs {m_swing['avg_win']}
Avg Loss: {m_swing['avg_loss']}
Payoff Ratio: {round(abs(m_swing['avg_win']/m_swing['avg_loss']), 2) if m_swing['avg_loss'] != 0 else 'N/A'}x
Avg Dhan Charges: Rs {m_swing['avg_charge']}
Max Drawdown: Rs {m_swing['max_drawdown']}
Final Capital: Rs {m_swing['final_capital']} (ROI: {m_swing['roi_pct']}%)

================================================================================
"""
    res_path.write_text(report, encoding="utf-8")
    print(report)

if __name__ == "__main__":
    main()
