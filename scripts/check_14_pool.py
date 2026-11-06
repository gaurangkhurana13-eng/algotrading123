import pickle
from pathlib import Path

pkl_path = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\data\2024-25\universe_14_15m.pkl")
with open(pkl_path, "rb") as f:
    pool = pickle.load(f)

print("14-stock pool keys:", sorted(list(pool.keys())))
