import pickle, os, sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

data = pickle.load(open(ROOT / 'scripts/v42_19_folds_master_results.pkl', 'rb'))
trades = data['all_trades']
df_trades = pd.DataFrame(trades)

print("=" * 80)
print(f"ANALYSIS OF EXISTING {len(df_trades)} BHAVISHYA V42 TRADES")
print("=" * 80)

wins = df_trades[df_trades['net_pnl'] > 0]
losses = df_trades[df_trades['net_pnl'] < 0]
be = df_trades[df_trades['net_pnl'] == 0]

print(f"Total Trades: {len(df_trades)}")
print(f"Wins: {len(wins)} ({len(wins)/len(df_trades)*100:.1f}%) | Losses: {len(losses)} ({len(losses)/len(df_trades)*100:.1f}%) | Breakeven: {len(be)}")
print(f"Total Net PnL: Rs {df_trades['net_pnl'].sum():.2f}")
print(f"Gross PnL: Rs {df_trades['gross_pnl'].sum():.2f} | Total Charges: Rs {df_trades['charges'].sum():.2f}")
print(f"Profit Factor: {wins['net_pnl'].sum() / abs(losses['net_pnl'].sum()):.2f}")
print(f"Average Win: Rs {wins['net_pnl'].mean():.2f}")
print(f"Median Win:  Rs {wins['net_pnl'].median():.2f}")
print(f"Average Loss: Rs {losses['net_pnl'].mean():.2f}")
print(f"Win/Loss Ratio: {abs(wins['net_pnl'].mean() / losses['net_pnl'].mean()):.2f}")
print(f"Max Win: Rs {wins['net_pnl'].max():.2f}")
print(f"Exit Reasons Breakdown:")
print(df_trades['exit_reason'].value_counts())

# Inspect the win distribution
print("\nWin Distribution (Net PnL quantiles for winning trades):")
print(wins['net_pnl'].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))

# Check R multiples realized
# Risk = entry_price - sl
df_trades['risk'] = df_trades['entry_price'] - df_trades['sl']
df_trades['realized_r'] = (df_trades['exit_price'] - df_trades['entry_price']) / (df_trades['risk'] + 1e-6)

print("\nRealized R-Multiple across all winning trades:")
win_r = df_trades[df_trades['net_pnl'] > 0]['realized_r']
print(f"Average Realized R for Wins: {win_r.mean():.2f}R")
print(f"Median Realized R for Wins:  {win_r.median():.2f}R")
print(f"Percent of Wins with Realized R < 1.0R: {(win_r < 1.0).mean() * 100:.1f}%")
print(f"Percent of Wins with Realized R < 0.6R: {(win_r < 0.6).mean() * 100:.1f}%")
print(f"Percent of Wins with Realized R >= 2.0R: {(win_r >= 2.0).mean() * 100:.1f}%")
