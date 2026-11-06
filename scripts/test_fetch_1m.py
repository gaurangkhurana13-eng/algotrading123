import sys
from tvDatafeed import TvDatafeed, Interval
import pandas as pd

def test():
    tv = TvDatafeed()
    print("Testing tvDatafeed 1-minute fetch for PCJEWELLER...")
    df = tv.get_hist("PCJEWELLER", "NSE", interval=Interval.in_1_minute, n_bars=10000)
    if df is not None:
        print(f"Fetched {len(df)} bars.")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        print(df.tail(3))
    else:
        print("Failed to fetch.")

if __name__ == "__main__":
    test()
