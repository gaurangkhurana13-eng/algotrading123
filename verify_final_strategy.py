import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher import DataFetcher
from src.strategy import calculate_atr, calculate_rsi
from fast_engine import compute_ut_trailing_stop

fetcher = DataFetcher(data_dir="data")

def full_quant_backtest(df, 
                        symbol="ASSET", 
                        tf="15m", 
                        trade_mode="Both",      # "Both" or "Long Only"
                        ut_a=2.0, 
                        ut_c=7, 
                        use_macro_filter=True,  # 200 EMA
                        use_partial_tp=True,    # 50% at TP1 + Breakeven SL
                        tp1_mult=1.5, 
                        commission_pct=0.03):
    
    close = df["close"]
    high = df["high"]
    low = df["low"]
    
    atr = calculate_atr(df, period=ut_c)
    vol_atr = calculate_atr(df, period=14)
    trail_stop, _ = compute_ut_trailing_stop(df, key_value=ut_a, atr_period=ut_c)
    ema200 = close.ewm(span=200, adjust=False).mean()
    
    # Signals
    raw_buy = (close > trail_stop) & (close.shift(1) <= trail_stop.shift(1))
    raw_sell = (close < trail_stop) & (close.shift(1) >= trail_stop.shift(1))
    
    if use_macro_filter and trade_mode == "Long Only":
        # Only buy if close > 200 EMA
        buy_signal = raw_buy & (close > ema200)
        sell_signal = raw_sell # Always exit on sell signal
    else:
        buy_signal = raw_buy
        sell_signal = raw_sell
        
    capital = 100000.0
    pos = 0 # 1 or -1
    entry_price = 0.0
    pos_sl = 0.0
    pos_tp1 = 0.0
    tp1_hit = False
    active_shares = 0.0
    initial_shares = 0.0
    realized_pnl = 0.0
    trades = []
    
    for i in range(1, len(df)):
        c = close.iloc[i]
        h = high.iloc[i]
        l = low.iloc[i]
        curr_trail = trail_stop.iloc[i]
        curr_buy = buy_signal.iloc[i]
        curr_sell = sell_signal.iloc[i]
        
        # 1. FLAT STATE
        if pos == 0:
            if curr_buy:
                pos = 1
                entry_price = c
                initial_shares = capital / entry_price
                active_shares = initial_shares
                pos_sl = curr_trail
                pos_tp1 = entry_price + (vol_atr.iloc[i] * tp1_mult)
                tp1_hit = False
                realized_pnl = 0.0
            elif curr_sell and trade_mode == "Both":
                pos = -1
                entry_price = c
                initial_shares = capital / entry_price
                active_shares = initial_shares
                pos_sl = curr_trail
                pos_tp1 = entry_price - (vol_atr.iloc[i] * tp1_mult)
                tp1_hit = False
                realized_pnl = 0.0
                
        # 2. LONG STATE
        elif pos == 1:
            # Update trailing stop upwards
            pos_sl = max(pos_sl, curr_trail)
            
            # Check TP1 Partial Book
            if use_partial_tp and not tp1_hit and h >= pos_tp1:
                tp1_hit = True
                shares_sold = initial_shares * 0.50
                gain = (pos_tp1 - entry_price) * shares_sold
                comm = (entry_price + pos_tp1) * shares_sold * (commission_pct / 100.0)
                realized_pnl += (gain - comm)
                active_shares -= shares_sold
                # Breakeven lock
                pos_sl = max(pos_sl, entry_price)
                
            # Check Exit
            exit_triggered = False
            exit_price = 0.0
            
            if l <= pos_sl:
                exit_triggered = True
                exit_price = min(c, pos_sl)
            elif curr_sell:
                exit_triggered = True
                exit_price = c
                
            if exit_triggered:
                gain = (exit_price - entry_price) * active_shares
                comm = (entry_price + exit_price) * active_shares * (commission_pct / 100.0)
                realized_pnl += (gain - comm)
                
                capital += realized_pnl
                pnl_pct = (realized_pnl / capital) * 100.0
                trades.append({"dir": "LONG", "pnl": realized_pnl, "pnl_pct": pnl_pct, "capital": capital, "tp1": tp1_hit})
                
                # Check immediate reversal to short
                if curr_sell and trade_mode == "Both":
                    pos = -1
                    entry_price = c
                    initial_shares = capital / entry_price
                    active_shares = initial_shares
                    pos_sl = curr_trail
                    pos_tp1 = entry_price - (vol_atr.iloc[i] * tp1_mult)
                    tp1_hit = False
                    realized_pnl = 0.0
                else:
                    pos = 0
                    
        # 3. SHORT STATE
        elif pos == -1:
            pos_sl = min(pos_sl, curr_trail)
            
            # Check TP1 Partial Book
            if use_partial_tp and not tp1_hit and l <= pos_tp1:
                tp1_hit = True
                shares_sold = initial_shares * 0.50
                gain = (entry_price - pos_tp1) * shares_sold
                comm = (entry_price + pos_tp1) * shares_sold * (commission_pct / 100.0)
                realized_pnl += (gain - comm)
                active_shares -= shares_sold
                pos_sl = min(pos_sl, entry_price) # Breakeven lock
                
            exit_triggered = False
            exit_price = 0.0
            
            if h >= pos_sl:
                exit_triggered = True
                exit_price = max(c, pos_sl)
            elif curr_buy:
                exit_triggered = True
                exit_price = c
                
            if exit_triggered:
                gain = (entry_price - exit_price) * active_shares
                comm = (entry_price + exit_price) * active_shares * (commission_pct / 100.0)
                realized_pnl += (gain - comm)
                
                capital += realized_pnl
                pnl_pct = (realized_pnl / capital) * 100.0
                trades.append({"dir": "SHORT", "pnl": realized_pnl, "pnl_pct": pnl_pct, "capital": capital, "tp1": tp1_hit})
                
                if curr_buy:
                    pos = 1
                    entry_price = c
                    initial_shares = capital / entry_price
                    active_shares = initial_shares
                    pos_sl = curr_trail
                    pos_tp1 = entry_price + (vol_atr.iloc[i] * tp1_mult)
                    tp1_hit = False
                    realized_pnl = 0.0
                else:
                    pos = 0

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
        "symbol": symbol,
        "tf": tf,
        "mode": trade_mode,
        "total_trades": total_trades,
        "win_rate": round(win_rate, 1),
        "net_profit_pct": round(net_profit_pct, 2),
        "profit_factor": round(pf, 2),
        "max_drawdown_pct": round(max_dd, 2)
    }

print("Running Comprehensive Evaluation:")
results = []
for sym in ["SUZLON", "PCJEWELLER"]:
    for tf in ["15m", "1h", "1d"]:
        data = fetcher.get_data(sym, tf)
        if data is not None and len(data) > 100:
            # 1. Dual Direction Mode (Intraday / F&O)
            r_both = full_quant_backtest(data, symbol=sym, tf=tf, trade_mode="Both", ut_a=2.0, ut_c=7, use_partial_tp=True)
            results.append(r_both)
            # 2. Long Only with 200 EMA Macro Regime (Cash / Delivery)
            r_long = full_quant_backtest(data, symbol=sym, tf=tf, trade_mode="Long Only", ut_a=2.0, ut_c=7, use_macro_filter=True, use_partial_tp=True)
            results.append(r_long)

rdf = pd.DataFrame(results)
print(rdf.to_string(index=False))
