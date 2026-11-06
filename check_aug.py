import pandas as pd

v34 = pd.read_csv(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V34.3\fold_2026\trades_intraday.csv")
v36 = pd.read_csv(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V36\fold_2026\trades_intraday.csv")

v34_aug = v34[v34['entry_time'].str.startswith('2026-08')]
v36_aug = v36[v36['entry_time'].str.startswith('2026-08')]

print("V34.3 August Trades:")
print(v34_aug[['symbol', 'entry_time', 'shares', 'net_pnl', 'exit_reason']])

print("\nV36 August Trades:")
print(v36_aug[['symbol', 'entry_time', 'shares', 'net_pnl', 'exit_reason']])
