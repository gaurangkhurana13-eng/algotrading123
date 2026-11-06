"""
Execution and Backtesting script for Strategy V3:
Momentum Velocity & Asymmetric Ratchet Engine.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from framework import (
    calc_dhan_charges, calculate_atr, compute_metrics
)

def backtest_v3(df: pd.DataFrame, mode: str = "intraday"):
    c = df['close']
    h = df['high']
    l = df['low']
    o = df['open']
    v = df['volume']
    
    atr14 = calculate_atr(df, 14)
    ema50 = c.ewm(span=50).mean()
    
    # RSI 7
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
    
    entry_cond = (
        (macd > sig) & 
        (macd.shift(1) <= sig.shift(1)) & 
        (macd > 0) & 
        (c > ema50) & 
        (rsi7 > 52.0) & 
        (adx > 20.0) & 
        (plus_di > minus_di)
    )
    
    capital = 500.0
    notional_cap = 2450.0 if mode == "intraday" else 500.0
    r_cap = 15.50
    slippage = 0.0005
    sl_mult = 1.00
    p1_trig = 0.90
    p1_lock = 0.30
    p2_trig = 2.00
    p2_lock = 1.20
    tp_mult = 2.20
    
    trades = []
    pos = 0
    entry_p = 0.0
    entry_t = None
    qty = 0
    active_sl = 0.0
    active_tp = 0.0
    high_seen = 0.0
    phase = 0
    entry_atr = 0.0
    
    n = len(df)
    for i in range(100, n):
        ts = df.index[i]
        t = ts.time()
        cur_c = c.iloc[i]
        cur_h = h.iloc[i]
        cur_l = l.iloc[i]
        
        # Intraday 15:10 square-off check
        if mode == "intraday" and pos == 1:
            if t >= pd.Timestamp("15:10").time():
                exit_p = cur_c * (1.0 - slippage)
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
            high_seen = max(high_seen, cur_h)
            mfe_atr = (high_seen - entry_p) / entry_atr if entry_atr > 0 else 0.0
            
            # Dynamic Asymmetric Ratchet
            if mfe_atr >= p2_trig and phase < 2:
                phase = 2
                active_sl = max(active_sl, entry_p + (p2_lock * entry_atr))
            elif mfe_atr >= p1_trig and phase < 1:
                phase = 1
                active_sl = max(active_sl, entry_p + (p1_lock * entry_atr))
                
            # Exit checks
            if cur_h >= active_tp:
                exit_p = active_tp * (1.0 - slippage)
                gross = (exit_p - entry_p) * qty
                chg = calc_dhan_charges(entry_p, exit_p, qty, is_intraday=(mode=="intraday"))
                net = round(gross - chg, 2)
                trades.append({
                    "entry_time": str(entry_t), "exit_time": str(ts),
                    "entry_price": round(entry_p, 2), "exit_price": round(exit_p, 2),
                    "qty": qty, "gross_pnl": round(gross, 2), "charges": round(chg, 2),
                    "net_pnl": net, "exit_reason": "TAKE_PROFIT_HIT (+2.2x ATR)"
                })
                pos = 0
            elif cur_l <= active_sl:
                exit_p = active_sl * (1.0 - slippage)
                gross = (exit_p - entry_p) * qty
                chg = calc_dhan_charges(entry_p, exit_p, qty, is_intraday=(mode=="intraday"))
                net = round(gross - chg, 2)
                reason = "PHASE_2_RATCHET_FLOOR_HIT" if phase == 2 else ("PHASE_1_RATCHET_FLOOR_HIT" if phase == 1 else "INITIAL_SL_HIT")
                trades.append({
                    "entry_time": str(entry_t), "exit_time": str(ts),
                    "entry_price": round(entry_p, 2), "exit_price": round(exit_p, 2),
                    "qty": qty, "gross_pnl": round(gross, 2), "charges": round(chg, 2),
                    "net_pnl": net, "exit_reason": reason
                })
                pos = 0
                
        # Look for Entry
        if pos == 0:
            time_ok = (t >= pd.Timestamp("09:20").time()) and (t <= pd.Timestamp("13:30").time() if mode=="intraday" else True)
            if entry_cond.iloc[i] and time_ok:
                pos = 1
                entry_t = ts
                entry_p = cur_c * (1.0 + slippage)
                entry_atr = max(atr14.iloc[i], 0.05)
                
                # Sizing: Dual-constraint
                stop_dist = sl_mult * entry_atr
                friction_share = 0.002067 * entry_p
                denom = stop_dist + friction_share
                
                qty_risk = int(r_cap / denom) if denom > 0 else 1
                qty_notional = int(notional_cap / entry_p)
                qty = max(1, min(qty_risk, qty_notional))
                
                active_sl = entry_p - stop_dist
                active_tp = entry_p + (tp_mult * entry_atr)
                high_seen = entry_p
                phase = 0

    return pd.DataFrame(trades)

def main():
    data_file = Path("C:/Users/admin/.gemini/antigravity/scratch/new_trade/data/pcjeweller_15m_1y.pkl")
    df = pd.read_pickle(data_file)
    v_dir = Path(__file__).parent
    
    print("Running Backtest for Strategy V3...")
    df_intra = backtest_v3(df, mode="intraday")
    df_swing = backtest_v3(df, mode="swing")
    
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
  BHAVISHYA STRATEGY V3: MOMENTUM VELOCITY & ASYMMETRIC RATCHET ENGINE
================================================================================

1. LOGIC SUMMARY
   - MACD Zero-Line Cross (Fast 12, Slow 26, Sig 9 while MACD > 0)
   - Macro Trend Gate (Close > 50 EMA)
   - Fast RSI Buyer Dominance (RSI 7 > 52.0)
   - ADX Trend Strength Confirmation (ADX 10 > 20.0 and +DI > -DI)
   - Dynamic Asymmetric Ratchet:
     * Phase 1: At +0.90x ATR MFE -> Lock profit at +0.30x ATR floor
     * Phase 2: At +2.00x ATR MFE -> Lock profit at +1.20x ATR floor
     * Take Profit: Target +2.20x ATR ceiling
     * Initial Stop: 1.00x ATR
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
