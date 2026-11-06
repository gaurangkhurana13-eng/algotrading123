import pandas as pd

v34 = pd.read_csv(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V34.3\fold_2026\trades_intraday.csv")
v36 = pd.read_csv(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V36\fold_2026\trades_intraday.csv")

v34_apr = v34[v34['entry_time'].str.startswith('2026-04')]
v36_apr = v36[v36['entry_time'].str.startswith('2026-04')]

print(f"V34.3 April: Trades={len(v34_apr)}, Net={v34_apr['net_pnl'].sum():.2f}")
print(f"V36 April:   Trades={len(v36_apr)}, Net={v36_apr['net_pnl'].sum():.2f}")

diff_trades = v36_apr[~v36_apr['entry_time'].isin(v34_apr['entry_time'])]
print(f"\nExtra trades in V36 April ({len(diff_trades)} trades):")
print(diff_trades[['symbol', 'entry_time', 'net_pnl', 'exit_reason']])
print(f"Extra trades net PnL: {diff_trades['net_pnl'].sum():.2f}")
