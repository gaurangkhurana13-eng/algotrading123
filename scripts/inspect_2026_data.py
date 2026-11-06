import pickle
from pathlib import Path

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
with open(ROOT / "data" / "2025-26" / "universe_14_15m.pkl", "rb") as f:
    d = pickle.load(f)
print("Keys in 2025-26 universe:", list(d.keys()))
for k, df in d.items():
    print(k, df.index.min(), df.index.max(), len(df))
