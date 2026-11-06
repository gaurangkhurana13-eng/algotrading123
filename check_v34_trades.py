import pandas as pd

v34 = pd.read_csv(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V34.3\fold_2026\trades_intraday.csv")

print("Trades 88 to 94 in V34.3:")
print(v34.loc[88:94][['symbol', 'entry_time', 'exit_time', 'shares', 'net_pnl', 'exit_reason']])
