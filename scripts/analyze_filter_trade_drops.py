import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
df34 = pd.read_csv(ROOT / "strategies/V34.3/trades_intraday.csv")
df42 = pd.read_csv(ROOT / "strategies/V42_FOLD_1/trades_intraday.csv")

# Match by (symbol, entry_time)
df34['key'] = df34['symbol'] + "_" + df34['entry_time']
df42['key'] = df42['symbol'] + "_" + df42['entry_time']

common = df34[df34['key'].isin(df42['key'])]
v34_only = df34[~df34['key'].isin(df42['key'])]
v42_only = df42[~df42['key'].isin(df34['key'])]

print(f"Total V34.3 trades: {len(df34)}")
print(f"Total V42 trades:   {len(df42)}")
print(f"Common trades:      {len(common)}")
print(f"V34.3 only trades:  {len(v34_only)}")
print(f"V42 only trades:    {len(v42_only)}")

print("\n--- PERFORMANCE OF V34.3-ONLY TRADES (TRADES FILTERED OUT BY V42) ---")
w = (v34_only['net_pnl'] > 0).sum()
l = (v34_only['net_pnl'] < 0).sum()
net = v34_only['net_pnl'].sum()
print(f"V34.3-only count: {len(v34_only)} | Wins: {w} | Losses: {l} | Net PnL: Rs {net:+.2f}")
print("Top 10 winners among V34.3-only trades:")
print(v34_only.sort_values('net_pnl', ascending=False)[['symbol', 'entry_time', 'exit_time', 'shares', 'gross_pnl', 'net_pnl']].head(10).to_string(index=False))

print("\n--- PERFORMANCE OF COMMON TRADES ---")
print(f"Common trades in V34.3: Net PnL = Rs {common['net_pnl'].sum():.2f}")
c42 = df42[df42['key'].isin(df34['key'])]
print(f"Common trades in V42:   Net PnL = Rs {c42['net_pnl'].sum():.2f}")
