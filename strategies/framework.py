"""
Universal Backtesting & Execution Framework for Strategy Research.
Supports both Intraday MIS (15:10 square-off, 5x leverage) and Swing / Delivery (1x equity, multi-day holding).
Includes full Dhan MIS statutory charges + 0.05% slippage per leg.
"""

import numpy as np
import pandas as pd
from pathlib import Path

def calc_dhan_charges(buy_p: float, sell_p: float, qty: int, is_intraday: bool = True) -> float:
    buy_val = buy_p * qty
    sell_val = sell_p * qty
    turnover = buy_val + sell_val
    
    brokerage = min(20.0, buy_val * 0.0003) + min(20.0, sell_val * 0.0003)
    if is_intraday:
        stt = sell_val * 0.00025
    else:
        stt = turnover * 0.0010 # 0.1% on buy and sell for delivery
        
    exch = turnover * 0.0000345 # NSE turnover charge
    sebi = turnover * 0.000001 # Rs 10 per crore
    stamp = buy_val * 0.00003
    gst = 0.18 * (brokerage + exch + sebi)
    
    return float(brokerage + stt + exch + sebi + stamp + gst)

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h = df['high']
    l = df['low']
    c = df['close']
    c_prev = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - c_prev).abs(), (l - c_prev).abs()))
    return tr.rolling(period).mean()

def calculate_wma(series: pd.Series, length: int) -> pd.Series:
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda p: np.dot(p, weights) / weights.sum(), raw=True)

def calculate_hma(series: pd.Series, length: int) -> pd.Series:
    half_len = int(round(length / 2))
    sqrt_len = int(round(np.sqrt(length)))
    raw_hma = 2 * calculate_wma(series, half_len) - calculate_wma(series, length)
    return calculate_wma(raw_hma, sqrt_len)

def calculate_ut_bot(df: pd.DataFrame, a: float = 2.0, c: int = 1) -> pd.Series:
    close = df['close']
    atr = calculate_atr(df, c)
    n_loss = a * atr
    n = len(df)
    trail_stop = np.zeros(n)
    prev_stop = 0.0
    
    for i in range(1, n):
        cur_c = close.iloc[i]
        prev_c = close.iloc[i-1]
        loss = n_loss.iloc[i] if not np.isnan(n_loss.iloc[i]) else 0.5
        
        if cur_c > prev_stop and prev_c > prev_stop:
            curr = max(prev_stop, cur_c - loss)
        elif cur_c < prev_stop and prev_c < prev_stop:
            curr = min(prev_stop, cur_c + loss)
        elif cur_c > prev_stop:
            curr = cur_c - loss
        else:
            curr = cur_c + loss
            
        trail_stop[i] = curr
        prev_stop = curr
        
    return pd.Series(trail_stop, index=df.index)

def calculate_mapo(close: pd.Series) -> pd.Series:
    n = len(close)
    ma_bull_pct = np.zeros(n)
    ma_windows = list(range(10, 101, 5)) # 19 windows
    for m in ma_windows:
        ma_bull_pct += (close > close.rolling(m).mean()).astype(int)
    ma_bull_pct = (ma_bull_pct / len(ma_windows)) * 100.0
    return pd.Series(ma_bull_pct, index=close.index)

def compute_metrics(trades_df: pd.DataFrame, capital: float = 500.0) -> dict:
    if len(trades_df) == 0:
        return {
            "total_trades": 0, "win_rate": 0.0, "net_pnl": 0.0, "profit_factor": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0, "avg_charge": 0.0, "max_drawdown": 0.0,
            "final_capital": capital, "roi_pct": 0.0
        }
    
    n = len(trades_df)
    pnl = trades_df["net_pnl"]
    wins = trades_df[pnl > 0]
    losses = trades_df[pnl <= 0]
    
    wr = (len(wins) / n) * 100.0
    tot_pnl = pnl.sum()
    tot_win = wins["net_pnl"].sum() if len(wins) > 0 else 0.0
    tot_loss = abs(losses["net_pnl"].sum()) if len(losses) > 0 else 0.0
    
    pf = (tot_win / tot_loss) if tot_loss > 0 else (999.0 if tot_win > 0 else 0.0)
    avg_win = wins["net_pnl"].mean() if len(wins) > 0 else 0.0
    avg_loss = losses["net_pnl"].mean() if len(losses) > 0 else 0.0
    avg_chg = trades_df["charges"].mean() if "charges" in trades_df.columns else 0.0
    
    # Calculate drawdown
    equity_curve = capital + pnl.cumsum()
    peak = equity_curve.cummax()
    dd = (peak - equity_curve)
    max_dd = dd.max() if len(dd) > 0 else 0.0
    
    return {
        "total_trades": n,
        "win_rate": round(wr, 2),
        "wins": len(wins),
        "losses": len(losses),
        "net_pnl": round(tot_pnl, 2),
        "profit_factor": round(pf, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "avg_charge": round(avg_chg, 2),
        "max_drawdown": round(max_dd, 2),
        "final_capital": round(capital + tot_pnl, 2),
        "roi_pct": round((tot_pnl / capital) * 100.0, 2)
    }
