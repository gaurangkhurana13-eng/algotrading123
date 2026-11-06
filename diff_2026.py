import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies")
df36 = pd.read_csv(ROOT / "V36" / "fold_2026" / "trades_intraday.csv")
df37 = pd.read_csv(ROOT / "V37" / "fold_2026" / "trades_intraday.csv")

diff_in_36 = df36[~df36['entry_time'].isin(df37['entry_time'])]
diff_in_37 = df37[~df37['entry_time'].isin(df36['entry_time'])]

print(f"Trades in V36 not in V37 ({len(diff_in_36)}):")
print(diff_in_36[['symbol', 'entry_time', 'net_pnl', 'exit_reason']])

print(f"\nTrades in V37 not in V36 ({len(diff_in_37)}):")
print(diff_in_37[['symbol', 'entry_time', 'net_pnl', 'exit_reason']])

# Let's also check MTNL trade in V37
print("\nMTNL trades in V37:")
print(df37[df37['symbol'] == 'MTNL'][['entry_time', 'shares', 'gross_pnl', 'charges', 'net_pnl']])

print("\nMTNL trades in V36:")
print(df36[df36['symbol'] == 'MTNL'][['entry_time', 'shares', 'gross_pnl', 'charges', 'net_pnl']])
