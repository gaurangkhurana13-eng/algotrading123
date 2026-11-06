import pickle
from pathlib import Path
from tvDatafeed import TvDatafeed, Interval

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
H1_PKL = ROOT / "data" / "2016_H1_UNSEEN" / "universe_30_15m.pkl"

with open(H1_PKL, "rb") as f:
    d = pickle.load(f)

symbols = sorted(list(set(k.replace(".NS", "") for k in d.keys())))
print(f"Total 30-Stock Universe Symbols ({len(symbols)}):")
print(symbols)

OUT_DIR = ROOT / "data" / "2026_1M_30_STOCKS"
OUT_DIR.mkdir(parents=True, exist_ok=True)

tv = TvDatafeed()
pool = {}

for sym in symbols:
    print(f"Fetching 1m data for {sym}...", end=" ", flush=True)
    try:
        df = tv.get_hist(sym, "NSE", interval=Interval.in_1_minute, n_bars=10000)
        if df is None or len(df) == 0:
            print("Retrying BSE...", end=" ", flush=True)
            df = tv.get_hist(sym, "BSE", interval=Interval.in_1_minute, n_bars=10000)
            
        if df is not None and len(df) > 500:
            df = df.copy()
            df.columns = [c.lower() for c in df.columns]
            df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
            df["volume"] = df["volume"].fillna(0)
            pool[sym] = df
            pool[sym + ".NS"] = df
            print(f"OK ({len(df)} bars: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')})")
        else:
            bars_cnt = len(df) if df is not None else 0
            print(f"FAILED (only {bars_cnt} bars)")
    except Exception as e:
        print(f"ERROR: {e}")

out_file = OUT_DIR / "universe_30_1m.pkl"
with open(out_file, "wb") as f:
    pickle.dump(pool, f)

print(f"\nSuccessfully stored {len(pool)//2} stocks into {out_file}")
