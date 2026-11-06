import pickle
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
f1_path = ROOT_DIR / "data" / "unseen_high_beta_10_15m_1y.pkl"
f2_path = ROOT_DIR / "data" / "unseen_10_15m_1y.pkl"

if f1_path.exists():
    with open(f1_path, "rb") as f:
        d1 = pickle.load(f)
    print("unseen_high_beta_10:", list(d1.keys()))

if f2_path.exists():
    with open(f2_path, "rb") as f:
        d2 = pickle.load(f)
    print("unseen_10:", list(d2.keys()))
