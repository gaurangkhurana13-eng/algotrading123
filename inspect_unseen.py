import pickle
from pathlib import Path

root = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
with open(root / "data" / "unseen_high_beta_10_15m_1y.pkl", "rb") as f:
    d = pickle.load(f)
print("Unseen High Beta Tickers:", list(d.keys()))

with open(root / "data" / "unseen_10_15m_1y.pkl", "rb") as f:
    d2 = pickle.load(f)
print("Unseen 10 Tickers:", list(d2.keys()))

with open(root / "data" / "unseen_batch3_10_15m_1y.pkl", "rb") as f:
    d3 = pickle.load(f)
print("Unseen Batch 3 Tickers:", list(d3.keys()))
