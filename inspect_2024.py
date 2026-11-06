import pickle
from pathlib import Path
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
with open(ROOT / "data" / "2024-25" / "universe_14_15m.pkl", "rb") as f:
    p2024 = pickle.load(f)

print(f"Total keys in 2024-25: {len(p2024)}")
print("Sample keys:", list(p2024.keys())[:10])

# check date range of 2024-25
fk = list(p2024.keys())[0]
df = p2024[fk]
print("Date range:", df.index.min(), "to", df.index.max())
print("Bars in 2024 H2 (2024-07-01 to 2024-12-31):", len(df.loc['2024-07-01':'2024-12-31']))
