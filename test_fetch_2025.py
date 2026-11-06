import time
import pandas as pd
from pathlib import Path

# Test TvDatafeed with larger n_bars
try:
    from tvDatafeed import TvDatafeed, Interval
    tv = TvDatafeed()
    print("Testing TvDatafeed for PCJEWELLER (n_bars=10000)...")
    df_tv = tv.get_hist(symbol="PCJEWELLER", exchange="NSE", interval=Interval.in_15_minute, n_bars=10000)
    if df_tv is not None:
        print(f"TvDatafeed PCJEWELLER: Start = {df_tv.index[0]} | End = {df_tv.index[-1]} | Total = {len(df_tv)}")
    else:
        print("TvDatafeed returned None")
except Exception as e:
    print("TvDatafeed error:", e)

# Test yfinance
try:
    import yfinance as yf
    print("\nTesting yfinance for PCJEWELLER.NS...")
    # yfinance allows 60d for 15m, let's verify what it returns
    df_yf = yf.download("PCJEWELLER.NS", interval="15m", period="60d")
    print(f"yfinance 60d: {len(df_yf)} bars")
except Exception as e:
    print("yfinance error:", e)
