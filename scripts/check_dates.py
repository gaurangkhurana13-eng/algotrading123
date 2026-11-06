import pickle
from pathlib import Path

BASE = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
paths = [
    BASE / "data" / "universe_78_1y" / "centralbk_15m_1y.pkl",
    BASE / "data" / "2024-25" / "centralbk_15m_1y.pkl",
    BASE / "data" / "2024-25" / "universe_30_15m.pkl"
]

for p in paths:
    try:
        with open(p, "rb") as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                print(f"{p.name}: dict with {len(data)} keys: {list(data.keys())[:5]}")
                first_k = list(data.keys())[0]
                df = data[first_k]
                print(f"  Sample {first_k} Start: {df.index[0]} End: {df.index[-1]} Len: {len(df)}")
            else:
                print(f"{p.name}: DataFrame Start: {data.index[0]} End: {data.index[-1]} Len: {len(data)}")
    except Exception as e:
        print(f"{p.name} Error: {e}")
