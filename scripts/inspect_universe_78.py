import pickle
from pathlib import Path

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
with open(ROOT / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    d = pickle.load(f)
print("Keys in universe_78_15m_1y:", len(d))
first_k = list(d.keys())[0]
print(first_k, d[first_k].index.min(), d[first_k].index.max(), len(d[first_k]))
