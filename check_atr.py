import pickle
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    univ = pickle.load(f)

core = [
    "PCJEWELLER.NS", "RPOWER.NS", "DELTACORP.NS", "RITES.NS",
    "IDEA.NS", "MTNL.NS", "ZEEL.NS", "UTKARSHBNK.NS", "IFCI.NS", "NCC.NS"
]

print(f"{'Symbol':<15} | {'Median Close':<12} | {'Median ATR14':<12} | {'0.05 / Median ATR':<18}")
print("-" * 65)

for sym in core:
    if sym not in univ:
        continue
    df = univ[sym]
    c = df["close"]
    tr = df["high"] - df["low"]
    atr = tr.rolling(14).mean()
    med_c = c.median()
    med_atr = atr.median()
    ratio = 0.05 / med_atr if med_atr > 0 else 0
    print(f"{sym:<15} | Rs {med_c:<10.2f} | Rs {med_atr:<10.3f} | {ratio:<18.3f}")
