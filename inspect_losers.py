import pandas as pd
from pathlib import Path

p34 = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V34\fold_2026\trades_intraday.csv")
p33 = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V33\fold_2026\trades_intraday.csv")

df34 = pd.read_csv(p34)
df33 = pd.read_csv(p33)

merged = pd.merge(df34, df33, on=["symbol", "entry_time"], suffixes=("_v34", "_v33"), how="left")
only_v34 = merged[merged["net_pnl_v33"].isna()].copy()

print("Worst 10 losing extra trades:")
print(only_v34.sort_values("net_pnl_v34")[["symbol", "entry_time", "net_pnl_v34", "exit_reason_v34"]].head(10))

print("\nBest 10 winning extra trades:")
print(only_v34.sort_values("net_pnl_v34", ascending=False)[["symbol", "entry_time", "net_pnl_v34", "exit_reason_v34"]].head(10))
