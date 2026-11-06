import pickle
from pathlib import Path

V29_30_UNIVERSE = [
    "CENTRALBK", "DELTACORP", "ELECTCAST", "GSFC", "IDEA", "IFCI", "IOB", "IRCON",
    "J&KBANK", "JYOTHYLAB", "MANALIPETC", "MTNL", "NBCC", "NCC", "NIACL", "NMDC",
    "PCJEWELLER", "PNB", "RBLBANK", "RCF", "RENUKA", "RITES", "RPOWER", "SUZLON",
    "SWSOLAR", "TATASTEEL", "TEXRAIL", "TTML", "UTKARSHBNK", "ZEEL"
]

pkl_path = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\data\2024-25\universe_30_15m.pkl")
with open(pkl_path, "rb") as f:
    pool = pickle.load(f)

print(f"Total symbols in pool: {len(pool)}")
missing = [s for s in V29_30_UNIVERSE if s not in pool]
print(f"Missing symbols: {missing}")
for s in V29_30_UNIVERSE[:5]:
    df = pool[s]
    print(f"  {s}: {len(df)} bars, {df.index.min()} to {df.index.max()}")
