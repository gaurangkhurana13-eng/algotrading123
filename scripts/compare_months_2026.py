import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
df34 = pd.read_csv(ROOT / "strategies/V34.3/trades_intraday.csv")
df42 = pd.read_csv(ROOT / "strategies/V42_FOLD_1/trades_intraday.csv")

df34['dt'] = pd.to_datetime(df34['entry_time'])
df42['dt'] = pd.to_datetime(df42['entry_time'])

df34['month'] = df34['dt'].dt.to_period('M')
df42['month'] = df42['dt'].dt.to_period('M')

print("MONTHLY BREAKDOWN IN 2026:")
months = sorted(list(set(df34['month']).union(set(df42['month']))))

for m in months:
    sub34 = df34[df34['month'] == m]
    sub42 = df42[df42['month'] == m]
    net34 = sub34['net_pnl'].sum() if not sub34.empty else 0.0
    net42 = sub42['net_pnl'].sum() if not sub42.empty else 0.0
    tr34 = len(sub34)
    tr42 = len(sub42)
    wr34 = (sub34['net_pnl'] > 0).mean() * 100.0 if not sub34.empty else 0.0
    wr42 = (sub42['net_pnl'] > 0).mean() * 100.0 if not sub42.empty else 0.0
    print(f"Month: {m} | V34.3: Trades={tr34:2d}, WR={wr34:4.1f}%, Net=Rs {net34:+7.2f} | V42: Trades={tr42:2d}, WR={wr42:4.1f}%, Net=Rs {net42:+7.2f}")
