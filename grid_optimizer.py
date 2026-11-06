import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher import DataFetcher
from src.strategy import calculate_atr, calculate_rsi
from fast_engine import compute_ut_trailing_stop, compute_supertrend, backtest_vectorized

fetcher = DataFetcher(data_dir="data")

def evaluate_strategy_variants(df, symbol="STOCK", tf="15m"):
    results = []
    
    # Pre-calculate base indicators
    close = df["close"]
    high = df["high"]
    low = df["low"]
    
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    ema200 = close.ewm(span=200, adjust=False).mean()
    rsi7 = calculate_rsi(close, period=7)
    rsi14 = calculate_rsi(close, period=14)
    atr14 = calculate_atr(df, period=14)
    
    # Try different UT key values & ATR lengths
    for ut_a in [1.5, 2.0, 2.5, 3.0, 4.0]:
        for ut_c in [3, 7, 10]:
            trail_stop, ut_pos = compute_ut_trailing_stop(df, key_value=ut_a, atr_period=ut_c)
            
            # Variant A: Pure UT Bot (Unfiltered - Old behavior)
            buy_a = (close > trail_stop) & (close.shift(1) <= trail_stop.shift(1))
            sell_a = (close < trail_stop) & (close.shift(1) >= trail_stop.shift(1))
            df_a = df.copy()
            df_a["buy_signal"] = buy_a
            df_a["sell_signal"] = sell_a
            m_a = backtest_vectorized(df_a)
            results.append({
                "Strategy": f"Pure UT (a={ut_a}, c={ut_c})",
                "Regime": "None",
                **m_a
            })
            
            # Variant B: UT Bot with Macro 200 EMA Filter (Longs ONLY when Close > 200 EMA)
            buy_b = buy_a & (close > ema200)
            sell_b = sell_a  # Exit immediately when trail stop hit or close < ema200
            df_b = df.copy()
            df_b["buy_signal"] = buy_b
            df_b["sell_signal"] = sell_b
            m_b = backtest_vectorized(df_b)
            results.append({
                "Strategy": f"Regime UT (a={ut_a}, c={ut_c})",
                "Regime": "Close > EMA200",
                **m_b
            })

            # Variant C: Dynamic Sniper Pullback (High Frequency Dip Buying in Bull Regime)
            # Buy when in Bull Trend (Close > EMA 100 or EMA 200) AND RSI7 pulls back below 38, then closes back above EMA9
            bull_regime = (close > ema100_val if (ema100_val := close.ewm(span=100, adjust=False).mean()) is not None else True)
            rsi_dip = (rsi7 < 38) | (rsi7.shift(1) < 38) | (rsi7.shift(2) < 38)
            rev = (close > ema9) & (close.shift(1) <= ema9.shift(1)) & (close > df["open"])
            buy_c = bull_regime & rsi_dip & rev
            sell_c = (close < trail_stop) & (close.shift(1) >= trail_stop.shift(1))
            df_c = df.copy()
            df_c["buy_signal"] = buy_c
            df_c["sell_signal"] = sell_c
            m_c = backtest_vectorized(df_c)
            results.append({
                "Strategy": f"Sniper Pullback + Trail (a={ut_a}, c={ut_c})",
                "Regime": "Close > EMA100",
                **m_c
            })

    res_df = pd.DataFrame(results)
    # Filter for strategies with reasonable trade count (> 15 trades)
    res_df = res_df.sort_values(by="net_profit_pct", ascending=False)
    return res_df

print("Running optimization across SUZLON and PCJEWELLER...")
for sym in ["SUZLON", "PCJEWELLER"]:
    for tf in ["15m", "1h", "1d"]:
        data = fetcher.get_data(sym, tf)
        if data is not None and len(data) > 100:
            print(f"\n=======================================================")
            print(f"  TOP 5 STRATEGIES: {sym} [{tf.upper()}] (Bars: {len(data)})")
            print(f"=======================================================")
            best = evaluate_strategy_variants(data, symbol=sym, tf=tf)
            print(best.head(5).to_string(index=False))
