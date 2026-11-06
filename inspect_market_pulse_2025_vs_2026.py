import pickle
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")

# 2026 pool
with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    u1 = pickle.load(f)

# 2025 pool
with open(ROOT_DIR / "data" / "2025-26" / "universe_14_15m.pkl", "rb") as f:
    u2025 = pickle.load(f)

def get_market_breadth(pool, label):
    daily_rets = []
    for sym, df in pool.items():
        if len(df) < 200:
            continue
        c = df["close"]
        df["_date"] = df.index.date
        d_open = df.groupby("_date")["open"].transform("first")
        d_ret = (c / d_open) - 1.0
        daily_rets.append(d_ret)
    b_series = pd.concat(daily_rets, axis=1).mean(axis=1)
    daily_b = b_series.groupby(b_series.index.date).last()
    print(f"\n=== {label} ===")
    print(f"Total Trading Days: {len(daily_b)}")
    print(f"Positive Days: {(daily_b > 0).sum()} ({(daily_b > 0).mean()*100:.1f}%)")
    print(f"Mean Daily Return: {daily_b.mean()*100:.3f}%")
    print(f"20-day rolling mean range: {daily_b.rolling(20).mean().min()*100:.3f}% to {daily_b.rolling(20).mean().max()*100:.3f}%")

get_market_breadth(u1, "2026 POOL")
get_market_breadth(u2025, "2025 CHOP POOL")
