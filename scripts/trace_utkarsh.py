import pickle
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    univ = pickle.load(f)

df = univ["UTKARSHBNK.NS"].loc["2026-04-06 10:30":"2026-04-06 14:00"]
print(df[["open", "high", "low", "close", "volume"]])
