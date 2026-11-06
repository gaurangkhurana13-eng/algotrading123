import pickle
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
data_dir = ROOT_DIR / "data"

with open(data_dir / "unseen_high_beta_10_15m_1y.pkl", "rb") as f:
    d_beta = pickle.load(f)

with open(data_dir / "unseen_10_15m_1y.pkl", "rb") as f:
    d_unseen = pickle.load(f)

# Save individual pickles for Gate V2 grounding
extra_symbols = {
    "DELTACORP": d_beta["DELTACORP.NS"],
    "RITES": d_beta["RITES.NS"],
    "NCC": d_beta["NCC.NS"],
    "ZEEL": d_unseen["ZEEL.NS"]
}

for sym, df in extra_symbols.items():
    p = data_dir / f"{sym.lower()}_15m_1y.pkl"
    with open(p, "wb") as f:
        pickle.dump(df, f)
    print(f"Saved {p.name}: {len(df)} candles")
