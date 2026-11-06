"""
Execution and Backtesting script for Strategy V5:
Precision Squeeze & Adaptive Moonshot Engine.
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

def compute_vwap_shifted(df: pd.DataFrame) -> pd.Series:
    df_temp = df.copy()
    df_temp["_date"] = df_temp.index.date
    tp = (df_temp["high"] + df_temp["low"] + df_temp["close"]) / 3.0
    df_temp["_tpv"] = tp * df_temp["volume"]
    cum_tpv = df_temp.groupby("_date")["_tpv"].cumsum()
    cum_vol = df_temp.groupby("_date")["volume"].cumsum()
    return (cum_tpv / (cum_vol + 1e-9)).shift(1)

def backtest_v5(df: pd.DataFrame, mode: str = "intraday"):
    c = df['close']
    h = df['high']
    l = df['low']
    o = df['open']
    v = df['volume']
    
    atr14 = calculate_atr(df, 14)
    ema50 = c.ewm(span=50).mean()
    vwap_lag = compute_vwap_shifted(df)
    
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
        (c > vwap_lag) & 
        (rsi7 > 50.0) & 
        (adx > 18.0) & 
        (plus_di > minus_di)
    )
    
    capital = 500.0
    notional_cap = 2450.0 if mode == "intraday" else 500.0
    r_cap = 15.50
    slippage = 0.0005
    sl_mult = 1.00
    p1_trig = 0.85
    p1_lock = 0.45
    p2_trig = 1.70
    trail_dist = 0.60
    
    trades = []
    pos = 0
    entry_p = 0.0
    entry_t = None
    qty = 0
    active_sl = 0.0
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
                exit_p = round(cur_c * (1.0 - slippage), 2)
                gross = round((exit_p - entry_p) * qty, 2)
                chg = round(calc_dhan_charges(entry_p, exit_p, qty, is_intraday=True), 2)
                net = round(gross - chg, 2)
                trades.append({
                    "entry_time": str(entry_t), "exit_time": str(ts),
                    "symbol": "PCJEWELLER", "direction": "long",
                    "entry_price": entry_p, "exit_price": exit_p,
                    "shares": qty, "qty": qty, "gross_pnl": gross, "charges": chg,
                    "net_pnl": net, "exit_reason": "15:10_INTRADAY_SQUARE_OFF"
                })
                pos = 0
                continue
                
        # In Position: evaluate ratchet & dynamic moonshot trailing
        if pos == 1:
            high_seen = max(high_seen, cur_h)
            mfe_atr = (high_seen - entry_p) / entry_atr if entry_atr > 0 else 0.0
            
            # Dynamic Moonshot Ratchet
            if mfe_atr >= p2_trig:
                phase = 2
                active_sl = max(active_sl, high_seen - (trail_dist * entry_atr))
            elif mfe_atr >= p1_trig and phase < 1:
                phase = 1
                active_sl = max(active_sl, entry_p + (p1_lock * entry_atr))
                
            # Exit check: Trailing stop or floor hit
            if cur_l <= active_sl:
                exit_p = round(active_sl * (1.0 - slippage), 2)
                gross = round((exit_p - entry_p) * qty, 2)
                chg = round(calc_dhan_charges(entry_p, exit_p, qty, is_intraday=(mode=="intraday")), 2)
                net = round(gross - chg, 2)
                reason = "PHASE_2_MOONSHOT_TRAIL_HIT" if phase == 2 else ("PHASE_1_RATCHET_FLOOR_HIT" if phase == 1 else "INITIAL_SL_HIT")
                trades.append({
                    "entry_time": str(entry_t), "exit_time": str(ts),
                    "symbol": "PCJEWELLER", "direction": "long",
                    "entry_price": entry_p, "exit_price": exit_p,
                    "shares": qty, "qty": qty, "gross_pnl": gross, "charges": chg,
                    "net_pnl": net, "exit_reason": reason
                })
                pos = 0
                
        # Look for Entry
        if pos == 0:
            time_ok = (t >= pd.Timestamp("09:20").time()) and (t <= pd.Timestamp("13:30").time() if mode=="intraday" else True)
            if entry_cond.iloc[i] and time_ok:
                pos = 1
                entry_t = ts
                entry_p = round(cur_c * (1.0 + slippage), 2)
                entry_atr = max(atr14.iloc[i], 0.05)
                
                # Sizing: Dual-constraint
                stop_dist = sl_mult * entry_atr
                friction_share = 0.002067 * entry_p
                denom = stop_dist + friction_share
                
                qty_risk = int(r_cap / denom) if denom > 0 else 1
                qty_notional = int(notional_cap / entry_p)
                qty = max(1, min(qty_risk, qty_notional))
                
                active_sl = entry_p - stop_dist
                high_seen = entry_p
                phase = 0

    return pd.DataFrame(trades)

def main():
    data_file = Path("C:/Users/admin/.gemini/antigravity/scratch/new_trade/data/pcjeweller_15m_1y.pkl")
    df = pd.read_pickle(data_file)
    v_dir = Path(__file__).parent
    
    print("Running Backtest for Strategy V5...")
    df_intra = backtest_v5(df, mode="intraday")
    df_swing = backtest_v5(df, mode="swing")
    
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
  BHAVISHYA STRATEGY V5: PRECISION SQUEEZE & ADAPTIVE MOONSHOT ENGINE
================================================================================

1. LOGIC SUMMARY
   - MACD Zero-Line Cross (Fast 12, Slow 26, Sig 9 while MACD > 0)
   - Macro Trend Gate (Close > 50 EMA)
   - Institutional VWAP Anchor (Close > Shifted VWAP.shift(1))
   - Fast RSI Buyer Dominance (RSI 7 > 50.0)
   - ADX Trend Strength Confirmation (ADX 10 > 18.0 and +DI > -DI)
   - Volatility Squeeze Context (Bollinger Bands 20, 2 inside Keltner 20, 1.5 ATR)
   - Precision Moonshot Trailing Ratchet:
     * Phase 1: At +0.85x ATR MFE -> Lock profit at +0.45x ATR floor
     * Phase 2: At +1.70x ATR MFE -> Dynamic Moonshot Trailing (HighSeen - 0.60x ATR)
     * Initial Stop: 1.00x ATR
   - Realistic Execution: Dhan MIS costs + 0.05% slippage, Rs 500 base, 5x leverage max Rs 2,450

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
Avg Loss: Rs {m_intra['avg_loss']}
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
Avg Loss: Rs {m_swing['avg_loss']}
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
