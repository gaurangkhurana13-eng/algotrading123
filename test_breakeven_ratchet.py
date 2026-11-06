import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher import DataFetcher
from src.strategy import calculate_atr
from fast_engine import compute_ut_trailing_stop

fetcher = DataFetcher(data_dir="data")

def backtest_ratchet_strategy(df, 
                              symbol="STOCK", 
                              tf="15m", 
                              trade_mode="Both",      # "Both" or "Long Only"
                              ut_a=2.0, 
                              ut_c=7, 
                              be_atr_mult=1.5,       # Reach 1.5 ATR -> Lock Breakeven
                              use_macro_filter=True, # Close > 200 EMA for Longs
                              commission_pct=0.03):
    
    close = df["close"]
    high = df["high"]
    low = df["low"]
    
    trail_stop, _ = compute_ut_trailing_stop(df, key_value=ut_a, atr_period=ut_c)
    vol_atr = calculate_atr(df, period=14)
    ema200 = close.ewm(span=200, adjust=False).mean()
    
    raw_buy = (close > trail_stop) & (close.shift(1) <= trail_stop.shift(1))
    raw_sell = (close < trail_stop) & (close.shift(1) >= trail_stop.shift(1))
    
    if use_macro_filter and trade_mode == "Long Only":
        buy_signal = raw_buy & (close > ema200)
    else:
        buy_signal = raw_buy
    sell_signal = raw_sell
    
    capital = 100000.0
    pos = 0 # 1 or -1
    entry_price = 0.0
    pos_sl = 0.0
    be_target = 0.0
    be_active = False
    trades = []
    
    for i in range(1, len(df)):
        c = close.iloc[i]
        h = high.iloc[i]
        l = low.iloc[i]
        curr_trail = trail_stop.iloc[i]
        curr_buy = buy_signal.iloc[i]
        curr_sell = sell_signal.iloc[i]
        
        # 1. FLAT
        if pos == 0:
            if curr_buy:
                pos = 1
                entry_price = c
                pos_sl = curr_trail
                be_target = entry_price + (vol_atr.iloc[i] * be_atr_mult)
                be_active = False
            elif curr_sell and trade_mode == "Both":
                pos = -1
                entry_price = c
                pos_sl = curr_trail
                be_target = entry_price - (vol_atr.iloc[i] * be_atr_mult)
                be_active = False
                
        # 2. LONG
        elif pos == 1:
            # Check if breakeven threshold reached
            if not be_active and h >= be_target:
                be_active = True
                pos_sl = max(pos_sl, entry_price)
                
            # If breakeven active, floor SL at entry
            if be_active:
                pos_sl = max(pos_sl, max(curr_trail, entry_price))
            else:
                pos_sl = max(pos_sl, curr_trail)
                
            exit_triggered = False
            exit_price = 0.0
            if l <= pos_sl:
                exit_triggered = True
                exit_price = min(c, pos_sl)
            elif curr_sell:
                exit_triggered = True
                exit_price = c
                
            if exit_triggered:
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0 - (2 * commission_pct)
                capital += capital * (pnl_pct / 100.0)
                trades.append({"dir": "LONG", "pnl_pct": pnl_pct, "capital": capital, "be": be_active})
                
                if curr_sell and trade_mode == "Both":
                    pos = -1
                    entry_price = c
                    pos_sl = curr_trail
                    be_target = entry_price - (vol_atr.iloc[i] * be_atr_mult)
                    be_active = False
                else:
                    pos = 0
                    
        # 3. SHORT
        elif pos == -1:
            if not be_active and l <= be_target:
                be_active = True
                pos_sl = min(pos_sl, entry_price)
                
            if be_active:
                pos_sl = min(pos_sl, min(curr_trail, entry_price))
            else:
                pos_sl = min(pos_sl, curr_trail)
                
            exit_triggered = False
            exit_price = 0.0
            if h >= pos_sl:
                exit_triggered = True
                exit_price = max(c, pos_sl)
            elif curr_buy:
                exit_triggered = True
                exit_price = c
                
            if exit_triggered:
                pnl_pct = ((entry_price - exit_price) / entry_price) * 100.0 - (2 * commission_pct)
                capital += capital * (pnl_pct / 100.0)
                trades.append({"dir": "SHORT", "pnl_pct": pnl_pct, "capital": capital, "be": be_active})
                
                if curr_buy:
                    pos = 1
                    entry_price = c
                    pos_sl = curr_trail
                    be_target = entry_price + (vol_atr.iloc[i] * be_atr_mult)
                    be_active = False
                else:
                    pos = 0

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
        "symbol": symbol,
        "tf": tf,
        "mode": trade_mode,
        "total_trades": total_trades,
        "win_rate": round(win_rate, 1),
        "net_profit_pct": round(net_profit_pct, 2),
        "profit_factor": round(pf, 2),
        "max_drawdown_pct": round(max_dd, 2)
    }

print("Running Break-Even Ratchet Test across SUZLON and PCJEWELLER:")
results = []
for sym in ["SUZLON", "PCJEWELLER"]:
    for tf in ["15m", "1h", "1d"]:
        data = fetcher.get_data(sym, tf)
        if data is not None and len(data) > 100:
            # Dual-Direction
            r1 = backtest_ratchet_strategy(data, symbol=sym, tf=tf, trade_mode="Both", ut_a=2.0, ut_c=7, be_atr_mult=1.5)
            results.append(r1)
            # Long Only with 200 EMA
            r2 = backtest_ratchet_strategy(data, symbol=sym, tf=tf, trade_mode="Long Only", ut_a=2.0, ut_c=7, be_atr_mult=1.5, use_macro_filter=True)
            results.append(r2)

rdf = pd.DataFrame(results)
print(rdf.to_string(index=False))
