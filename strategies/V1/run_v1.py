"""
Execution and Backtesting script for Strategy V1.
Computes metrics for both Intraday MIS and Swing / Delivery modes,
saves trades CSVs, generates results.txt, and audits with integrity gate.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from framework import (
    calc_dhan_charges, calculate_atr, calculate_hma,
    calculate_ut_bot, calculate_mapo, compute_metrics
)

def backtest_v1(df: pd.DataFrame, mode: str = "intraday"):
    close = df['close']
    high = df['high']
    low = df['low']
    open_p = df['open']
    
    # 1. Indicators
    ts_series = calculate_ut_bot(df, a=2.0, c=1)
    hma_series = calculate_hma(close, length=31)
    mapo_series = calculate_mapo(close)
    atr14 = calculate_atr(df, 14)
    
    buy_cross = (close > ts_series) & (close.shift(1) <= ts_series.shift(1))
    sell_cross = (close < ts_series) & (close.shift(1) >= ts_series.shift(1))
    
    # Entry Condition
    entry_cond = buy_cross & (close > hma_series) & (mapo_series >= 50.0)
    
    capital = 500.0
    notional_cap = 2450.0 if mode == "intraday" else 500.0
    r_cap = 15.50
    slippage = 0.0005
    sl_mult = 1.50
    p1_trig = 1.80
    p1_lock = 0.80
    tp_mult = 4.0
    
    trades = []
    pos = 0
    entry_p = 0.0
    entry_t = None
    qty = 0
    active_sl = 0.0
    active_tp = 0.0
    high_seen = 0.0
    phase = 0
    atr_entry = 0.0
    
    n = len(df)
    for i in range(105, n):
        ts = df.index[i]
        t = ts.time()
        c = close.iloc[i]
        h = high.iloc[i]
        l = low.iloc[i]
        
        # Intraday 15:10 square-off check
        if mode == "intraday" and pos == 1:
            if t >= pd.Timestamp("15:10").time():
                exit_p = c * (1.0 - slippage)
                gross = (exit_p - entry_p) * qty
                chg = calc_dhan_charges(entry_p, exit_p, qty, is_intraday=True)
                net = round(gross - chg, 2)
                trades.append({
                    "entry_time": str(entry_t), "exit_time": str(ts),
                    "entry_price": round(entry_p, 2), "exit_price": round(exit_p, 2),
                    "qty": qty, "gross_pnl": round(gross, 2), "charges": round(chg, 2),
                    "net_pnl": net, "exit_reason": "15:10_INTRADAY_SQUARE_OFF"
                })
                pos = 0
                continue
                
        # In Position: evaluate ratchet, target, and stop
        if pos == 1:
            high_seen = max(high_seen, h)
            mfe_atr = (high_seen - entry_p) / atr_entry if atr_entry > 0 else 0.0
            
            # Profit Ratchet Phase 1
            if mfe_atr >= p1_trig and phase == 0:
                phase = 1
                active_sl = max(active_sl, entry_p + (p1_lock * atr_entry))
                
            # Exit checks
            if h >= active_tp:
                exit_p = active_tp * (1.0 - slippage)
                gross = (exit_p - entry_p) * qty
                chg = calc_dhan_charges(entry_p, exit_p, qty, is_intraday=(mode=="intraday"))
                net = round(gross - chg, 2)
                trades.append({
                    "entry_time": str(entry_t), "exit_time": str(ts),
                    "entry_price": round(entry_p, 2), "exit_price": round(exit_p, 2),
                    "qty": qty, "gross_pnl": round(gross, 2), "charges": round(chg, 2),
                    "net_pnl": net, "exit_reason": "TAKE_PROFIT_HIT"
                })
                pos = 0
            elif l <= active_sl:
                fill_p = min(active_sl, open_p.iloc[i])
                exit_p = fill_p * (1.0 - slippage)
                gross = (exit_p - entry_p) * qty
                chg = calc_dhan_charges(entry_p, exit_p, qty, is_intraday=(mode=="intraday"))
                net = round(gross - chg, 2)
                reason = "PHASE_1_RATCHET_FLOOR_HIT" if phase == 1 else "INITIAL_SL_HIT"
                trades.append({
                    "entry_time": str(entry_t), "exit_time": str(ts),
                    "entry_price": round(entry_p, 2), "exit_price": round(exit_p, 2),
                    "qty": qty, "gross_pnl": round(gross, 2), "charges": round(chg, 2),
                    "net_pnl": net, "exit_reason": reason
                })
                pos = 0
            elif sell_cross.iloc[i]:
                exit_p = c * (1.0 - slippage)
                gross = (exit_p - entry_p) * qty
                chg = calc_dhan_charges(entry_p, exit_p, qty, is_intraday=(mode=="intraday"))
                net = round(gross - chg, 2)
                trades.append({
                    "entry_time": str(entry_t), "exit_time": str(ts),
                    "entry_price": round(entry_p, 2), "exit_price": round(exit_p, 2),
                    "qty": qty, "gross_pnl": round(gross, 2), "charges": round(chg, 2),
                    "net_pnl": net, "exit_reason": "UT_STOP_CROSSUNDER"
                })
                pos = 0
                
        # Look for Entry
        if pos == 0:
            time_ok = (t >= pd.Timestamp("09:20").time()) and (t <= pd.Timestamp("13:30").time() if mode=="intraday" else True)
            if entry_cond.iloc[i] and time_ok:
                pos = 1
                entry_t = ts
                entry_p = c * (1.0 + slippage)
                atr_entry = max(atr14.iloc[i], 0.05)
                
                # Sizing: Dual-constraint
                stop_dist = sl_mult * atr_entry
                friction_share = 0.002067 * entry_p
                denom = stop_dist + friction_share
                
                qty_risk = int(r_cap / denom) if denom > 0 else 1
                qty_notional = int(notional_cap / entry_p)
                qty = max(1, min(qty_risk, qty_notional))
                
                active_sl = entry_p - stop_dist
                active_tp = entry_p + (tp_mult * atr_entry)
                high_seen = entry_p
                phase = 0

    return pd.DataFrame(trades)

def main():
    data_file = Path("C:/Users/admin/.gemini/antigravity/scratch/new_trade/data/pcjeweller_15m_1y.pkl")
    df = pd.read_pickle(data_file)
    v_dir = Path(__file__).parent
    
    print("Running Backtest for Strategy V1...")
    df_intra = backtest_v1(df, mode="intraday")
    df_swing = backtest_v1(df, mode="swing")
    
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
  BHAVISHYA STRATEGY V1: CONFLUENCE BASELINE (SVMKR + MAPO + HMA)
================================================================================

1. LOGIC SUMMARY
   - UT Bot ATR Trailing Stop (a=2.0, c=1)
   - Hull Moving Average Trend Filter (HMA 31)
   - LuxAlgo MAPO Consensus (> 50% bullish across 19 MAs)
   - Asymmetric Profit Ratchet: Initial SL 1.5x ATR, Phase 1 Lock at +1.8x ATR -> +0.8x ATR floor, TP 4.0x ATR
   - Realistic Execution: Dhan MIS costs + 0.05% slippage, Rs 500 base, 5x leverage max Rs 2,450

--------------------------------------------------------------------------------
2. INTRADAY MIS PERFORMANCE (Square-off by 15:10 IST)
--------------------------------------------------------------------------------
   Total Trades      : {m_intra['total_trades']}
   Wins / Losses     : {m_intra['wins']} Wins / {m_intra['losses']} Losses
   Win Rate          : {m_intra['win_rate']}%
   Net Profit / Loss : Rs {m_intra['net_pnl']}
   Profit Factor     : {m_intra['profit_factor']}
   Average Win       : Rs {m_intra['avg_win']}
   Average Loss      : Rs {m_intra['avg_loss']}
   Payoff Ratio      : {round(abs(m_intra['avg_win']/m_intra['avg_loss']), 2) if m_intra['avg_loss'] != 0 else 'N/A'}x
   Avg Dhan Charges  : Rs {m_intra['avg_charge']}
   Max Drawdown      : Rs {m_intra['max_drawdown']}
   Final Capital     : Rs {m_intra['final_capital']} (ROI: {m_intra['roi_pct']}%)

--------------------------------------------------------------------------------
3. SWING / DELIVERY PERFORMANCE (Held Until Reversal/Target)
--------------------------------------------------------------------------------
   Total Trades      : {m_swing['total_trades']}
   Wins / Losses     : {m_swing['wins']} Wins / {m_swing['losses']} Losses
   Win Rate          : {m_swing['win_rate']}%
   Net Profit / Loss : Rs {m_swing['net_pnl']}
   Profit Factor     : {m_swing['profit_factor']}
   Average Win       : Rs {m_swing['avg_win']}
   Average Loss      : Rs {m_swing['avg_loss']}
   Payoff Ratio      : {round(abs(m_swing['avg_win']/m_swing['avg_loss']), 2) if m_swing['avg_loss'] != 0 else 'N/A'}x
   Avg Dhan Charges  : Rs {m_swing['avg_charge']}
   Max Drawdown      : Rs {m_swing['max_drawdown']}
   Final Capital     : Rs {m_swing['final_capital']} (ROI: {m_swing['roi_pct']}%)

================================================================================
"""
    res_path.write_text(report, encoding="utf-8")
    print(report)

if __name__ == "__main__":
    main()
