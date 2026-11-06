import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher import DataFetcher
from src.strategy import calculate_atr
from fast_engine import compute_ut_trailing_stop

fetcher = DataFetcher(data_dir="data")

def backtest_dual_direction(df, ut_a=2.0, ut_c=7, commission_pct=0.03):
    trail_stop, _ = compute_ut_trailing_stop(df, key_value=ut_a, atr_period=ut_c)
    close = df["close"]
    
    # 1. Long Entry: Close crosses above trailing stop
    # 2. Short Entry: Close crosses below trailing stop
    buy_signals = (close > trail_stop) & (close.shift(1) <= trail_stop.shift(1))
    sell_signals = (close < trail_stop) & (close.shift(1) >= trail_stop.shift(1))
    
    capital = 100000.0
    pos = 0 # 1 = Long, -1 = Short
    entry_price = 0.0
    entry_idx = 0
    trades = []
    
    for i in range(1, len(df)):
        c = close.iloc[i]
        curr_buy = buy_signals.iloc[i]
        curr_sell = sell_signals.iloc[i]
        
        if pos == 0:
            if curr_buy:
                pos = 1
                entry_price = c
                entry_idx = i
            elif curr_sell:
                pos = -1
                entry_price = c
                entry_idx = i
        elif pos == 1:
            if curr_sell:
                # Close Long, open Short
                pnl_pct = ((c - entry_price) / entry_price) * 100.0 - (2 * commission_pct)
                pnl_abs = capital * (pnl_pct / 100.0)
                capital += pnl_abs
                trades.append({"entry_idx": entry_idx, "exit_idx": i, "dir": "LONG", "pnl_pct": pnl_pct, "capital": capital})
                
                pos = -1
                entry_price = c
                entry_idx = i
        elif pos == -1:
            if curr_buy:
                # Close Short, open Long
                pnl_pct = ((entry_price - c) / entry_price) * 100.0 - (2 * commission_pct)
                pnl_abs = capital * (pnl_pct / 100.0)
                capital += pnl_abs
                trades.append({"entry_idx": entry_idx, "exit_idx": i, "dir": "SHORT", "pnl_pct": pnl_pct, "capital": capital})
                
                pos = 1
                entry_price = c
                entry_idx = i
                
    if not trades:
        return {"total_trades": 0, "win_rate": 0.0, "net_profit_pct": 0.0, "profit_factor": 0.0, "max_drawdown_pct": 0.0}
        
    tdf = pd.DataFrame(trades)
    wins = tdf[tdf["pnl_pct"] > 0]
    losses = tdf[tdf["pnl_pct"] <= 0]
    
    total_trades = len(tdf)
    win_rate = (len(wins) / total_trades) * 100.0
    net_profit_pct = ((capital - 100000.0) / 100000.0) * 100.0
    
    gp = wins["pnl_pct"].sum()
    gl = abs(losses["pnl_pct"].sum())
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

print("Dual-Direction (Long + Short) Test across timeframes:")
for sym in ["SUZLON", "PCJEWELLER"]:
    for tf in ["15m", "1h", "1d"]:
        data = fetcher.get_data(sym, tf)
        if data is not None and len(data) > 100:
            print(f"\n=== {sym} [{tf.upper()}] Dual-Direction ===")
            for a in [1.5, 2.0, 2.5, 3.0]:
                for c in [3, 7, 10]:
                    m = backtest_dual_direction(data, ut_a=a, ut_c=c)
                    if m["net_profit_pct"] > 10.0 or (tf == "15m" and m["net_profit_pct"] > 0):
                        print(f"  a={a}, c={c:<2} -> Trades: {m['total_trades']:<3} | WinRate: {m['win_rate']:>5.1f}% | Profit: {m['net_profit_pct']:>8.2f}% | PF: {m['profit_factor']:>5.2f} | MaxDD: {m['max_drawdown_pct']:>5.2f}%")
