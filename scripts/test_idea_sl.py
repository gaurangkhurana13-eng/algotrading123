import pickle
import pandas as pd
import sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    univ = pickle.load(f)

df = univ["IDEA.NS"]
from framework import calculate_atr
atr = calculate_atr(df, 14)
print("ATR at 09:45:", atr.loc["2026-08-14 09:45:00"])
active_sl = 13.78 - atr.loc["2026-08-14 09:45:00"]
print("active_sl:", active_sl)
print("round(active_sl, 2):", round(active_sl, 2))
print("13.71 <= active_sl:", 13.71 <= active_sl)
