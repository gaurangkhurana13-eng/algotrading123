import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher import DataFetcher
from src.strategy import calculate_atr, calculate_rsi
from fast_engine import compute_ut_trailing_stop

fetcher = DataFetcher(data_dir="data")

def simulate_partial_tp_strategy(df, 
                                 ut_a=2.0, 
                                 ut_c=7, 
                                 tp1_mult=1.5, 
                                 tp2_mult=3.5, 
                                 use_htf_regime=True, 
                                 htf_ema=100,
                                 commission_pct=0.03):
    """
    Advanced Quantitative Engine:
    1. UT Bot Base Trend Line
    2. Macro Regime Filter (Close > HTF EMA)
    3. Milestone Target 1: 50% locked at tp1_mult * ATR -> SL moved to Breakeven
    4. Milestone Target 2: 25% locked at tp2_mult * ATR
    5. Runner (remaining 25%): Trailed with UT Trailing Line
    """
    atr = calculate_atr(df, period=ut_c)
    vol_atr = calculate_atr(df, period=14)
    close = df["close"]
    high = df["high"]
    low = df["low"]
    
    trail_stop, _ = compute_ut_trailing_stop(df, key_value=ut_a, atr_period=ut_c)
    regime_ema = close.ewm(span=htf_ema, adjust=False).mean()
    
    # Buy signal
    base_buy = (close > trail_stop) & (close.shift(1) <= trail_stop.shift(1))
    if use_htf_regime:
        buy_signal = base_buy & (close > regime_ema)
    else:
        buy_signal = base_buy
        
    trades = []
    capital = 100000.0
    in_pos = False
    
    entry_price = 0.0
    entry_idx = 0
    pos_sl = 0.0
    pos_tp1 = 0.0
    pos_tp2 = 0.0
    tp1_hit = False
    tp2_hit = False
    active_shares = 0.0
    initial_shares = 0.0
    
    for i in range(1, len(df)):
        c = close.iloc[i]
        h = high.iloc[i]
        l = low.iloc[i]
        curr_trail = trail_stop.iloc[i]
        
        # Check Entry
        if not in_pos:
            if buy_signal.iloc[i]:
                in_pos = True
                entry_price = c
                entry_idx = i
                initial_shares = capital / entry_price
                active_shares = initial_shares
                
                pos_sl = curr_trail
                pos_tp1 = entry_price + (vol_atr.iloc[i] * tp1_mult)
                pos_tp2 = entry_price + (vol_atr.iloc[i] * tp2_mult)
                tp1_hit = False
                tp2_hit = False
                realized_pnl = 0.0
                
        # Check Active Position
        elif in_pos:
            # Update trailing stop if in profit
            pos_sl = max(pos_sl, curr_trail)
            
            # Check TP1
            if not tp1_hit and h >= pos_tp1:
                tp1_hit = True
                # Scale out 50%
                shares_to_sell = initial_shares * 0.50
                gain = (pos_tp1 - entry_price) * shares_to_sell
                comm = (entry_price + pos_tp1) * shares_to_sell * (commission_pct / 100.0)
                realized_pnl += (gain - comm)
                active_shares -= shares_to_sell
                # MOVE STOP LOSS TO BREAKEVEN!
                pos_sl = max(pos_sl, entry_price)
                
            # Check TP2
            if tp1_hit and not tp2_hit and h >= pos_tp2:
                tp2_hit = True
                # Scale out another 25%
                shares_to_sell = initial_shares * 0.25
                gain = (pos_tp2 - entry_price) * shares_to_sell
                comm = (entry_price + pos_tp2) * shares_to_sell * (commission_pct / 100.0)
                realized_pnl += (gain - comm)
                active_shares -= shares_to_sell
                # Trail SL to TP1 price!
                pos_sl = max(pos_sl, pos_tp1)
                
            # Check Exit on SL or Trailing Stop Hit
            if l <= pos_sl:
                exit_price = min(c, pos_sl)
                gain = (exit_price - entry_price) * active_shares
                comm = (entry_price + exit_price) * active_shares * (commission_pct / 100.0)
                realized_pnl += (gain - comm)
                
                capital += realized_pnl
                total_pnl_pct = (realized_pnl / capital) * 100.0
                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "tp1_hit": tp1_hit,
                    "tp2_hit": tp2_hit,
                    "pnl": realized_pnl,
                    "pnl_pct": total_pnl_pct,
                    "capital": capital
                })
                in_pos = False
                
    if not trades:
        return {"total_trades": 0, "win_rate": 0.0, "net_profit_pct": 0.0, "profit_factor": 0.0, "max_drawdown_pct": 0.0}
        
    tdf = pd.DataFrame(trades)
    wins = tdf[tdf["pnl"] > 0]
    losses = tdf[tdf["pnl"] <= 0]
    
    total_trades = len(tdf)
    win_rate = (len(wins) / total_trades) * 100.0
    net_profit_pct = ((capital - 100000.0) / 100000.0) * 100.0
    
    gp = wins["pnl"].sum()
    gl = abs(losses["pnl"].sum())
    pf = (gp / gl) if gl > 0 else (99.0 if gp > 0 else 0.0)
    
    eq = tdf["capital"]
    peak = eq.cummax()
    dd = (eq - peak) / peak * 100.0
    max_dd = abs(dd.min()) if not dd.empty else 0.0
    
    return {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 1),
        "net_profit_pct": round(net_profit_pct, 2),
        "profit_factor": round(pf, 2),
        "max_drawdown_pct": round(max_dd, 2)
    }

print("Running advanced scaling simulation...")
for sym in ["SUZLON", "PCJEWELLER"]:
    for tf in ["15m", "1h", "1d"]:
        data = fetcher.get_data(sym, tf)
        if data is not None and len(data) > 100:
            print(f"\n--- {sym} [{tf.upper()}] (Bars: {len(data)}) ---")
            # Unfiltered
            m_raw = simulate_partial_tp_strategy(data, ut_a=2.0, ut_c=7, use_htf_regime=False)
            # With Macro Regime
            m_regime = simulate_partial_tp_strategy(data, ut_a=2.0, ut_c=7, use_htf_regime=True, htf_ema=100)
            # Optimized
            m_opt = simulate_partial_tp_strategy(data, ut_a=2.5, ut_c=10, use_htf_regime=True, htf_ema=100, tp1_mult=1.5, tp2_mult=3.0)
            
            print(f"  Raw Scaling:       Trades={m_raw['total_trades']:<3} | WinRate={m_raw['win_rate']:>5.1f}% | Profit={m_raw['net_profit_pct']:>8.2f}% | PF={m_raw['profit_factor']:>5.2f} | MaxDD={m_raw['max_drawdown_pct']:>5.2f}%")
            print(f"  + Regime Filter:   Trades={m_regime['total_trades']:<3} | WinRate={m_regime['win_rate']:>5.1f}% | Profit={m_regime['net_profit_pct']:>8.2f}% | PF={m_regime['profit_factor']:>5.2f} | MaxDD={m_regime['max_drawdown_pct']:>5.2f}%")
            print(f"  + Opt Targets:     Trades={m_opt['total_trades']:<3} | WinRate={m_opt['win_rate']:>5.1f}% | Profit={m_opt['net_profit_pct']:>8.2f}% | PF={m_opt['profit_factor']:>5.2f} | MaxDD={m_opt['max_drawdown_pct']:>5.2f}%")
