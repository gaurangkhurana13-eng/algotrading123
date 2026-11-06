from tvDatafeed import TvDatafeed, Interval
import pandas as pd

tv = TvDatafeed()

test_symbols = ['ELECTCAST', 'IFCI', 'IRCON', 'JKBANK', 'RPOWER', 'PCJEWELLER']

for sym in test_symbols:
    try:
        df = tv.get_hist(symbol=sym, exchange='NSE', interval=Interval.in_15_minute, n_bars=10000)
        if df is not None and not df.empty:
            print(f"{sym}: {len(df)} bars. Min date: {df.index.min()}, Max date: {df.index.max()}")
        else:
            print(f"{sym}: No data returned")
    except Exception as e:
        print(f"{sym}: Error {e}")
