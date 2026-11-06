import pickle
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    univ = pickle.load(f)

for ticker in ["PCJEWELLER.NS", "RPOWER.NS", "ZEEL.NS", "DELTACORP.NS"]:
    df = univ[ticker].copy()
    df["month"] = df.index.to_period("M")
    m_agg = df.groupby("month").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        vol=("volume", "sum")
    )
    print(f"=== {ticker} MONTHLY PRICE ACTION ===")
    print(m_agg)
    print()
