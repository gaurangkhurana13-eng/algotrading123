from tvDatafeed import TvDatafeed, Interval
import pandas as pd

candidates = [
    "HINDCOPPER", "HFCL", "NCC", "ENGINERSIN", "MANAPPURAM",
    "DELTACORP", "SAMMAANCAP", "MARKSANS", "BALRAMCHIN", "FSL"
]

tv = TvDatafeed()
print("Checking availability and prices of 10 High-Beta Low-Price Candidates...")
for sym in candidates:
    try:
        df = tv.get_hist(sym, "NSE", interval=Interval.in_15_minute, n_bars=100)
        if df is not None and len(df) > 0:
            last_p = df['close'].iloc[-1]
            print(f"AVAILABLE: {sym:<12} | Bars: {len(df):<3} | Last Price: Rs {last_p:.2f}")
        else:
            print(f"UNAVAILABLE: {sym}")
    except Exception as e:
        print(f"ERROR: {sym} -> {e}")
