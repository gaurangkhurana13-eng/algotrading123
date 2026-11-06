import pandas as pd
from pathlib import Path

p33 = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V33\fold_2026\trades_intraday.csv")
p34 = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V34\fold_2026\trades_intraday.csv")

df33 = pd.read_csv(p33)
df34 = pd.read_csv(p34)

print("V33: count =", len(df33))
print("  Gross Profit =", df33["gross_pnl"].sum())
print("  Charges      =", df33["charges"].sum())
print("  Net Profit   =", df33["net_pnl"].sum())
print("  Ending wallet =", 500 + df33["net_pnl"].sum())

print("\nV34: count =", len(df34))
print("  Gross Profit =", df34["gross_pnl"].sum())
print("  Charges      =", df34["charges"].sum())
print("  Net Profit   =", df34["net_pnl"].sum())
print("  Ending wallet =", 500 + df34["net_pnl"].sum())

w33 = df33[df33["net_pnl"] > 0]
l33 = df33[df33["net_pnl"] < 0]
w34 = df34[df34["net_pnl"] > 0]
l34 = df34[df34["net_pnl"] < 0]

print(f"\nV33: Wins={len(w33)} (total win Rs {w33['net_pnl'].sum():.2f}, avg Rs {w33['net_pnl'].mean():.2f})")
print(f"V33: Losses={len(l33)} (total loss Rs {l33['net_pnl'].sum():.2f}, avg Rs {l33['net_pnl'].mean():.2f})")

print(f"\nV34: Wins={len(w34)} (total win Rs {w34['net_pnl'].sum():.2f}, avg Rs {w34['net_pnl'].mean():.2f})")
print(f"V34: Losses={len(l34)} (total loss Rs {l34['net_pnl'].sum():.2f}, avg Rs {l34['net_pnl'].mean():.2f})")

# Let's inspect the biggest winners in V33 vs V34
print("\nTop 5 Winners in V33:")
print(df33.sort_values("net_pnl", ascending=False)[["symbol", "entry_time", "exit_time", "net_pnl", "exit_reason"]].head(5))

print("\nTop 5 Winners in V34:")
print(df34.sort_values("net_pnl", ascending=False)[["symbol", "entry_time", "exit_time", "net_pnl", "exit_reason"]].head(5))

# Let's see exit reasons
print("\nV33 exit reasons:")
print(df33["exit_reason"].value_counts())
print("\nV34 exit reasons:")
print(df34["exit_reason"].value_counts())
