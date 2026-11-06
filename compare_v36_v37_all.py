import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies")
v36_dir = ROOT / "V36"
v37_dir = ROOT / "V37"

folds = ["fold_2022", "fold_2025_last5m", "fold_2026", "unseen_2023_mid4m", "unseen_2025_q4", "unseen_2026", "fold_2024_h1"]

print(f"{'Fold':<20} | {'V36 Trades':<10} | {'V37 Trades':<10} | {'Identical?':<10} | {'V36 Net':<10} | {'V37 Net':<10}")
print("-" * 75)

for f in folds:
    df36 = pd.read_csv(v36_dir / f / "trades_intraday.csv")
    df37 = pd.read_csv(v37_dir / f / "trades_intraday.csv")
    identical = df36.equals(df37)
    print(f"{f:<20} | {len(df36):<10} | {len(df37):<10} | {str(identical):<10} | {df36['net_pnl'].sum():<10.2f} | {df37['net_pnl'].sum():<10.2f}")
