import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calculate_atr

with open(ROOT_DIR / "data" / "2025-26" / "universe_14_15m.pkl", "rb") as f:
    univ_data = pickle.load(f)

all_atrs = []
for k, df in univ_data.items():
    if k.endswith(".NS"):
        continue
    c = df["close"]
    atr = calculate_atr(df, 14)
    atr_pct = (atr / c) * 100.0
    all_atrs.extend(atr_pct.dropna().tolist())

s = pd.Series(all_atrs)
print("=== 15-MINUTE ATR% REALITY ===")
print(s.describe(percentiles=[0.10, 0.25, 0.50, 0.75, 0.90]))
