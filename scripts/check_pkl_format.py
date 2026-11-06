import pickle
from pathlib import Path

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
p = ROOT / "data" / "2024-25" / "suzlon_15m_1y.pkl"
with open(p, "rb") as f:
    df = pickle.load(f)

print("Type:", type(df))
if isinstance(df, dict):
    print("Dict keys:", list(df.keys())[:5])
else:
    print("Columns:", df.columns)
    print("Index:", df.index[:5])
    print("Index type:", type(df.index))
    print("Index min:", df.index.min(), "max:", df.index.max())
