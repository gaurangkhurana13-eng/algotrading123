import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
df34 = pd.read_csv(ROOT / "strategies/V34.3/trades_intraday.csv")
df42 = pd.read_csv(ROOT / "strategies/V42_FOLD_1/trades_intraday.csv")

df34['dt'] = pd.to_datetime(df34['entry_time'])
df42['dt'] = pd.to_datetime(df42['entry_time'])

june34 = df34[(df34['dt'] >= '2026-06-01') & (df34['dt'] <= '2026-06-30 23:59:59')]
june42 = df42[(df42['dt'] >= '2026-06-01') & (df42['dt'] <= '2026-06-30 23:59:59')]

print("JUNE 2026 TRADES IN V34.3:")
print(june34[['symbol', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'shares', 'gross_pnl', 'charges', 'net_pnl', 'exit_reason']].to_string(index=False))

print("\nJUNE 2026 TRADES IN V42:")
print(june42[['symbol', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'shares', 'gross_pnl', 'charges', 'net_pnl', 'exit_reason']].to_string(index=False))
