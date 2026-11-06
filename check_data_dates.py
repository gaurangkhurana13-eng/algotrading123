import pickle
from pathlib import Path

root = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
with open(root / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    d = pickle.load(f)

for sym in ["PCJEWELLER.NS", "RPOWER.NS", "DELTACORP.NS"]:
    if sym in d:
        df = d[sym]
        print(f"{sym}: Start = {df.index[0]} | End = {df.index[-1]} | Total Candles = {len(df)}")
