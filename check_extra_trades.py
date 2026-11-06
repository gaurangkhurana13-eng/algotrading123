import pandas as pd

v34 = pd.read_csv(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V34.3\fold_2026\trades_intraday.csv")
v36 = pd.read_csv(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V36\fold_2026\trades_intraday.csv")

# Find trades in v36 that don't match v34 by (symbol, entry_time)
common = pd.merge(v36, v34, on=['symbol', 'entry_time'], suffixes=('_v36', '_v34'))
extra_v36 = v36[~v36.set_index(['symbol', 'entry_time']).index.isin(v34.set_index(['symbol', 'entry_time']).index)]
missed_v34 = v34[~v34.set_index(['symbol', 'entry_time']).index.isin(v36.set_index(['symbol', 'entry_time']).index)]

print(f"Common trades between both: {len(common)}")
print(f"Extra trades taken ONLY by V36: {len(extra_v36)}")
print(f"Trades taken ONLY by V34.3: {len(missed_v34)}")

print(f"\nNet PnL of Common trades in V34.3: {common['net_pnl_v34'].sum():.2f}")
print(f"Net PnL of Common trades in V36:   {common['net_pnl_v36'].sum():.2f}")
print(f"Common difference: {common['net_pnl_v36'].sum() - common['net_pnl_v34'].sum():+.2f}")

print(f"\nNet PnL of Extra trades taken ONLY by V36: {extra_v36['net_pnl'].sum():.2f} (WR: {(extra_v36['net_pnl']>0).mean()*100:.1f}%, Wins={(extra_v36['net_pnl']>0).sum()}, Losses={(extra_v36['net_pnl']<0).sum()})")
print(f"Net PnL of Trades taken ONLY by V34.3:    {missed_v34['net_pnl'].sum():.2f}")
