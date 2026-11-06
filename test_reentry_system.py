import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher import DataFetcher
from src.strategy import calculate_atr, calculate_rsi
from fast_engine import compute_ut_trailing_stop

fetcher = DataFetcher(data_dir="data")

def test_reentry_performance(df, ut_a=2.0, ut_c=7, commission_pct=0.03):
    """
    Tests UT Bot + Pullback Re-entry Signals.
    When in a Bull Trend (Close > Trailing Stop), every time RSI7 dips < 40 and re-crosses EMA9,
    a secondary 'RE-BUY' trade is taken with quick ATR target!
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    
    trail_stop, _ = compute_ut_trailing_stop(df, key_value=ut_a, atr_period=ut_c)
    vol_atr = calculate_atr(df, period=14)
    ema9 = close.ewm(span=9, adjust=False).mean()
    rsi7 = calculate_rsi(close, period=7)
    
    # 1. Primary Trend Crossovers
    primary_buy = (close > trail_stop) & (close.shift(1) <= trail_stop.shift(1))
    primary_sell = (close < trail_stop) & (close.shift(1) >= trail_stop.shift(1))
    
    # 2. Re-entry Pullback Signals in Bull Trend
    in_bull_trend = close > trail_stop
    rsi_dip = (rsi7 < 40) | (rsi7.shift(1) < 40)
    rebound = (close > ema9) & (close.shift(1) <= ema9.shift(1))
    reentry_buy = in_bull_trend & rsi_dip & rebound & (~primary_buy)
    
    # Let's count potential signals
    print(f"Primary Buys: {primary_buy.sum()} | Re-entry Dip Buys: {reentry_buy.sum()} | Primary Sells: {primary_sell.sum()}")

for sym in ["SUZLON", "PCJEWELLER"]:
    for tf in ["15m", "1h"]:
        d = fetcher.get_data(sym, tf)
        if d is not None:
            print(f"--- {sym} [{tf}] ---")
            test_reentry_performance(d)
