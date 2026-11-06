import sys, pickle
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")

df34 = pd.read_csv(ROOT / "strategies/V34.3/trades_intraday.csv")
df42 = pd.read_csv(ROOT / "strategies/V42_FOLD_1/trades_intraday.csv")

print("="*95)
print("DEEP FORENSIC COMPARISON: V34.3 VS V42 IN 2026 BULL RUN")
print("="*95)
print(f"V34.3: Trades={len(df34)} | Wins={(df34['net_pnl']>0).sum()} | Net PnL=Rs {df34['net_pnl'].sum():.2f} | Gross=Rs {df34['gross_pnl'].sum():.2f} | Charges=Rs {df34['charges'].sum():.2f}")
print(f"V42:   Trades={len(df42)} | Wins={(df42['net_pnl']>0).sum()} | Net PnL=Rs {df42['net_pnl'].sum():.2f} | Gross=Rs {df42['gross_pnl'].sum():.2f} | Charges=Rs {df42['charges'].sum():.2f}")

# Top 10 winners
print("\n--- TOP 10 WINNERS V34.3 ---")
print(df34.sort_values('net_pnl', ascending=False)[['symbol', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'shares', 'net_pnl', 'exit_reason']].head(10).to_string(index=False))

print("\n--- TOP 10 WINNERS V42 ---")
print(df42.sort_values('net_pnl', ascending=False)[['symbol', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'shares', 'net_pnl', 'exit_reason']].head(10).to_string(index=False))

# Average win/loss comparison
w34 = df34[df34['net_pnl']>0]['net_pnl']
l34 = df34[df34['net_pnl']<0]['net_pnl']
w42 = df42[df42['net_pnl']>0]['net_pnl']
l42 = df42[df42['net_pnl']<0]['net_pnl']

print(f"\nV34.3 Avg Win: Rs {w34.mean():.2f} | Median Win: Rs {w34.median():.2f} | Max Win: Rs {w34.max():.2f}")
print(f"V42   Avg Win: Rs {w42.mean():.2f} | Median Win: Rs {w42.median():.2f} | Max Win: Rs {w42.max():.2f}")

print(f"V34.3 Avg Loss: Rs {l34.mean():.2f} | Median Loss: Rs {l34.median():.2f} | Max Loss: Rs {l34.min():.2f}")
print(f"V42   Avg Loss: Rs {l42.mean():.2f} | Median Loss: Rs {l42.median():.2f} | Max Loss: Rs {l42.min():.2f}")
