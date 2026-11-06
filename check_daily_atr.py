import pickle
import numpy as np
import pandas as pd
from pathlib import Path

p = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\data\2025-26\universe_14_15m.pkl")
with open(p, "rb") as f:
    d = pickle.load(f)

for k, df in d.items():
    daily = df.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    tr = daily["high"] - daily["low"]
    atr_pct = (tr / daily["close"]) * 100.0
    print(f"{k:15s}: Median Daily Range% = {atr_pct.median():.2f}%, 10th percentile = {atr_pct.quantile(0.10):.2f}%")
