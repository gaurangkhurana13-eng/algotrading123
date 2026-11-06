import pickle
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
data_dir = ROOT_DIR / "data"

with open(data_dir / "unseen_batch3_10_15m_1y.pkl", "rb") as f:
    d = pickle.load(f)

print(f"Loaded {len(d)} unseen stocks from batch 3:")
for k, df in d.items():
    sym = k.replace(".NS", "").lower()
    p = data_dir / f"{sym}_15m_1y.pkl"
    with open(p, "wb") as f_out:
        pickle.dump(df, f_out)
    print(f"  Saved {p.name:<24} | {len(df)} candles | {df.index[0].date()} to {df.index[-1].date()} | Price: Rs {df['close'].iloc[-1]:.2f}")
