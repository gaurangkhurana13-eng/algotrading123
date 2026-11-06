import pickle
from pathlib import Path

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\data")
with open(ROOT / "unseen_batch3_10_15m_1y.pkl", "rb") as f:
    d = pickle.load(f)

print("Tickers in unseen_batch3_10_15m_1y.pkl:")
print(list(d.keys()))
