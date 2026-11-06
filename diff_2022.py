import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies")
df36 = pd.read_csv(ROOT / "V36" / "fold_2022" / "trades_intraday.csv")
df37 = pd.read_csv(ROOT / "V37" / "fold_2022" / "trades_intraday.csv")

diff = df36.compare(df37)
print(f"Differences in 2022 trades CSV (V36 vs V37): {len(diff)}")
if len(diff) > 0:
    print(diff)
else:
    print("Zero differences! Every single row, timestamp, share, and fill is 100% identical.")
