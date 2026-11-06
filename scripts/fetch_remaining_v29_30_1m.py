import os, pickle
from pathlib import Path
from tvDatafeed import TvDatafeed, Interval

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
BASKET_PKL = ROOT / "data" / "2026_1M_BASKET" / "universe_1m_basket.pkl"

with open(BASKET_PKL, "rb") as f:
    pool = pickle.load(f)

V29_30_UNIVERSE = [
    "J&KBANK", "UTKARSHBNK", "MANALIPETC", "IDEA", "PCJEWELLER",
    "NIACL", "MTNL", "IFCI", "TEXRAIL", "SWSOLAR",
    "TTML", "RENUKA", "RCF", "NBCC", "JYOTHYLAB",
    "RBLBANK", "ELECTCAST", "PNB", "GSFC", "TATASTEEL",
    "SUZLON", "CENTRALBK", "IOB", "IRCON", "NMDC",
    "RITES", "NCC", "ZEEL", "DELTACORP", "RPOWER"
]

remaining = [s for s in V29_30_UNIVERSE if s not in pool]
print(f"Already have: {len(pool)//2} stocks. Need to fetch {len(remaining)} remaining stocks:")
print(remaining)

tv = TvDatafeed()
for sym in remaining:
    print(f"Fetching 1m for {sym}...", end=" ", flush=True)
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
            print(f"OK ({len(df)} bars)")
        else:
            bars_cnt = len(df) if df is not None else 0
            print(f"FAILED ({bars_cnt} bars)")
    except Exception as e:
        print(f"ERROR: {e}")

out_file = ROOT / "data" / "2026_1M_V29_30" / "universe_v29_30_1m.pkl"
out_file.parent.mkdir(parents=True, exist_ok=True)
with open(out_file, "wb") as f:
    pickle.dump(pool, f)

print(f"\nSuccessfully saved COMPLETE {len(pool)//2} stocks of V29_30_UNIVERSE to {out_file}!")
