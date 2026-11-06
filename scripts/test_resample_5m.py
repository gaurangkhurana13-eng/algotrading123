import pickle
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
pkl_1m = ROOT / "data" / "2026_1M_V29_30" / "universe_v29_30_1m.pkl"
with open(pkl_1m, "rb") as f:
    pool_1m = pickle.load(f)

print(f"Loaded 1m pool: {len(pool_1m)//2} stocks")
pcj = pool_1m["PCJEWELLER"]
print(f"PCJEWELLER 1m bars: {len(pcj)}")

# Resample to 5m
pcj_5m = pcj.resample("5min").agg({
    "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
}).dropna()
print(f"PCJEWELLER 5m bars: {len(pcj_5m)}")
print(pcj_5m.head(3))
