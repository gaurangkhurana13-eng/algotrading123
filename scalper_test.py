import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher import DataFetcher
from src.strategy import calculate_atr, calculate_rsi

fetcher = DataFetcher(data_dir="data")

def test_pullback_scalper(df, ema_fast=9, ema_slow=21, ema_trend=50, rr_ratio=1.8, sl_atr_mult=1.2, commission_pct=0.03):
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_p = df["open"]
    
    ema9 = close.ewm(span=ema_fast, adjust=False).mean()
    ema21 = close.ewm(span=ema_slow, adjust=False).mean()
    ema50 = close.ewm(span=ema_trend, adjust=False).mean()
    atr = calculate_atr(df, period=10)
    rsi = calculate_rsi(close, period=7)
    
    # Uptrend: EMA 9 > EMA 21 and Close > EMA 50
    uptrend = (ema9 > ema21) & (close > ema50)
    # Downtrend: EMA 9 < EMA 21 and Close < EMA 50
    downtrend = (ema9 < ema21) & (close < ema50)
    
    # Long Pullback: Low touched or penetrated EMA 21, but candle closed green above EMA 9, RSI < 55
    long_pullback = (low <= ema21 * 1.002) & (close > ema9) & (close > open_p) & (rsi < 55) & uptrend
    
    # Short Pullback: High touched or penetrated EMA 21, but candle closed red below EMA 9, RSI > 45
    short_pullback = (high >= ema21 * 0.998) & (close < ema9) & (close < open_p) & (rsi > 45) & downtrend
    
    trades = []
    capital = 100000.0
    pos = 0 # 1 or -1
    entry_price = 0.0
    sl = 0.0
    tp = 0.0
    
    for i in range(1, len(df)):
        c = close.iloc[i]
        h = high.iloc[i]
        l = low.iloc[i]
        curr_atr = atr.iloc[i]
        
        if pos == 0:
            if long_pullback.iloc[i]:
                pos = 1
                entry_price = c
                sl = entry_price - (curr_atr * sl_atr_mult)
                risk = entry_price - sl
                tp = entry_price + (risk * rr_ratio)
            elif short_pullback.iloc[i]:
                pos = -1
                entry_price = c
                sl = entry_price + (curr_atr * sl_atr_mult)
                risk = sl - entry_price
                tp = entry_price - (risk * rr_ratio)
                
        elif pos == 1:
            # Check TP / SL
            if h >= tp:
                pnl_pct = ((tp - entry_price) / entry_price) * 100.0 - (2 * commission_pct)
                capital += capital * (pnl_pct / 100.0)
                trades.append({"dir": "LONG", "result": "WIN", "pnl_pct": pnl_pct, "capital": capital})
                pos = 0
            elif l <= sl:
                pnl_pct = ((sl - entry_price) / entry_price) * 100.0 - (2 * commission_pct)
                capital += capital * (pnl_pct / 100.0)
                trades.append({"dir": "LONG", "result": "LOSS", "pnl_pct": pnl_pct, "capital": capital})
                pos = 0
                
        elif pos == -1:
            if l <= tp:
                pnl_pct = ((entry_price - tp) / entry_price) * 100.0 - (2 * commission_pct)
                capital += capital * (pnl_pct / 100.0)
                trades.append({"dir": "SHORT", "result": "WIN", "pnl_pct": pnl_pct, "capital": capital})
                pos = 0
            elif h >= sl:
                pnl_pct = ((entry_price - sl) / entry_price) * 100.0 - (2 * commission_pct)
                capital += capital * (pnl_pct / 100.0)
                trades.append({"dir": "SHORT", "result": "LOSS", "pnl_pct": pnl_pct, "capital": capital})
                pos = 0
                
    if not trades:
        return {"total_trades": 0, "win_rate": 0.0, "net_profit_pct": 0.0, "profit_factor": 0.0, "max_drawdown_pct": 0.0}
        
    tdf = pd.DataFrame(trades)
    wins = tdf[tdf["result"] == "WIN"]
    losses = tdf[tdf["result"] == "LOSS"]
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

print("Testing Pullback Sniper Scalper (High Win Rate & Trade Frequency):")
for sym in ["SUZLON", "PCJEWELLER"]:
    for tf in ["15m", "1h"]:
        data = fetcher.get_data(sym, tf)
        if data is not None and len(data) > 100:
            print(f"\n=== {sym} [{tf.upper()}] Pullback Scalper ===")
            for rr in [1.2, 1.5, 1.8, 2.0]:
                for sl_mult in [1.0, 1.2, 1.5]:
                    m = test_pullback_scalper(data, rr_ratio=rr, sl_atr_mult=sl_mult)
                    if m["total_trades"] >= 10:
                        print(f"  RR={rr:<3} SL={sl_mult}x -> Trades: {m['total_trades']:<3} | WinRate: {m['win_rate']:>5.1f}% | Profit: {m['net_profit_pct']:>8.2f}% | PF: {m['profit_factor']:>5.2f} | MaxDD: {m['max_drawdown_pct']:>5.2f}%")
