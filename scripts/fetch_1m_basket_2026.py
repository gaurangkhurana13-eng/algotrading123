import os, pickle
from pathlib import Path
from tvDatafeed import TvDatafeed, Interval
import pandas as pd

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
OUT_DIR = ROOT / "data" / "2026_1M_BASKET"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = [
    "PCJEWELLER", "SOUTHBANK", "SUZLON", "IDEA", "MTNL",
    "DELTACORP", "IFCI", "PTC", "RITES", "TEXRAIL",
    "UTKARSHBNK", "ZEEL", "TATASTEEL", "PNB", "RPOWER"
]

def main():
    tv = TvDatafeed()
    pool = {}
    print(f"Fetching 1-minute data for {len(SYMBOLS)} symbols from NSE...")
    
    for sym in SYMBOLS:
        try:
            print(f"Fetching {sym}...", end=" ", flush=True)
            df = tv.get_hist(sym, "NSE", interval=Interval.in_1_minute, n_bars=10000)
            if df is None or len(df) == 0:
                print(f"Retrying {sym} from BSE...", end=" ", flush=True)
                df = tv.get_hist(sym, "BSE", interval=Interval.in_1_minute, n_bars=10000)
            
            if df is not None and len(df) > 500:
                df = df.copy()
                df.columns = [c.lower() for c in df.columns]
                df = df[["open", "high", "low", "close", "volume"]].dropna(subset=["close"])
                df["volume"] = df["volume"].fillna(0)
                pool[sym] = df
                pool[sym + ".NS"] = df
                print(f"OK ({len(df)} bars, {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')})")
            else:
                print("FAILED / insufficient bars")
        except Exception as e:
            print(f"ERROR: {e}")
            
    out_file = OUT_DIR / "universe_1m_basket.pkl"
    with open(out_file, "wb") as f:
        pickle.dump(pool, f)
    print(f"\nSuccessfully saved {len(pool)//2} symbols to {out_file}")

if __name__ == "__main__":
    main()
