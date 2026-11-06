import sys
import os
import itertools
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher import DataFetcher
from src.strategy import calculate_atr, calculate_rsi

def compute_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    hl2 = (df["high"] + df["low"]) / 2.0
    atr = calculate_atr(df, period=period)
    
    basic_upper = hl2 + (multiplier * atr)
    basic_lower = hl2 - (multiplier * atr)
    
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int) # 1 = Bull, -1 = Bear
    
    close = df["close"]
    
    curr_dir = 1
    for i in range(1, len(df)):
        # Lower band
        if basic_lower.iloc[i] > final_lower.iloc[i-1] or close.iloc[i-1] < final_lower.iloc[i-1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]
            
        # Upper band
        if basic_upper.iloc[i] < final_upper.iloc[i-1] or close.iloc[i-1] > final_upper.iloc[i-1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]
            
        # Direction
        if curr_dir == 1:
            if close.iloc[i] < final_lower.iloc[i]:
                curr_dir = -1
                supertrend.iloc[i] = final_upper.iloc[i]
            else:
                supertrend.iloc[i] = final_lower.iloc[i]
        else:
            if close.iloc[i] > final_upper.iloc[i]:
                curr_dir = 1
                supertrend.iloc[i] = final_lower.iloc[i]
            else:
                supertrend.iloc[i] = final_upper.iloc[i]
        direction.iloc[i] = curr_dir
        
    return supertrend, direction

def compute_ut_trailing_stop(df: pd.DataFrame, key_value: float = 2.0, atr_period: int = 10):
    atr = calculate_atr(df, period=atr_period)
    n_loss = key_value * atr
    close = df["close"]
    
    trailing_stop = np.zeros(len(df))
    pos = np.zeros(len(df), dtype=int)
    
    prev_stop = 0.0
    for i in range(1, len(df)):
        c = close.iloc[i]
        c_prev = close.iloc[i-1]
        loss = n_loss.iloc[i]
        
        if c > prev_stop and c_prev > prev_stop:
            curr_stop = max(prev_stop, c - loss)
        elif c < prev_stop and c_prev < prev_stop:
            curr_stop = min(prev_stop, c + loss)
        elif c > prev_stop:
            curr_stop = c - loss
        else:
            curr_stop = c + loss
            
        trailing_stop[i] = curr_stop
        prev_stop = curr_stop
        
        if c > curr_stop:
            pos[i] = 1
        elif c < curr_stop:
            pos[i] = -1
        else:
            pos[i] = pos[i-1]
            
    return pd.Series(trailing_stop, index=df.index), pd.Series(pos, index=df.index)

def backtest_vectorized(df: pd.DataFrame, initial_capital: float = 100000.0, commission_pct: float = 0.03):
    """Fast simulation of trades based on buy_signal and sell_signal with dynamic exit."""
    capital = initial_capital
    pos = 0 # 0 or 1
    entry_price = 0.0
    trades = []
    
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    buy = df["buy_signal"].values
    sell = df["sell_signal"].values
    n = len(df)
    
    for i in range(n):
        c = close[i]
        if pos == 0:
            if buy[i]:
                pos = 1
                entry_price = c
                entry_idx = i
        elif pos == 1:
            if sell[i]:
                exit_price = c
                # calculate PnL
                comm = (entry_price + exit_price) * (commission_pct / 100.0)
                pnl_pct = ((exit_price - entry_price) / entry_price) * 100.0 - (2 * commission_pct)
                pnl_abs = capital * (pnl_pct / 100.0)
                capital += pnl_abs
                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "capital": capital
                })
                pos = 0
                
    if not trades:
        return {"total_trades": 0, "win_rate": 0.0, "net_profit_pct": 0.0, "profit_factor": 0.0, "max_drawdown_pct": 0.0}
        
    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df["pnl_pct"] > 0]
    losses = trades_df[trades_df["pnl_pct"] <= 0]
    
    total_trades = len(trades_df)
    win_rate = (len(wins) / total_trades) * 100.0
    net_profit_pct = ((capital - initial_capital) / initial_capital) * 100.0
    
    gross_profit = wins["pnl_pct"].sum()
    gross_loss = abs(losses["pnl_pct"].sum())
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
    
    # Calculate Max Drawdown
    equity_series = trades_df["capital"]
    peak = equity_series.cummax()
    dd = (equity_series - peak) / peak * 100.0
    max_dd = abs(dd.min()) if not dd.empty else 0.0
    
    return {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 1),
        "net_profit_pct": round(net_profit_pct, 2),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown_pct": round(max_dd, 2)
    }

print("Backtest module loaded cleanly.")
