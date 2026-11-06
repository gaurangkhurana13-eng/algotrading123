import pandas as pd

v36 = pd.read_csv(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V36\fold_2026\trades_intraday.csv")

# Let's inspect trades leading up to trade 137
print("Trades 130 to 137 in V36:")
print(v36.loc[130:137][['symbol', 'entry_time', 'exit_time', 'shares', 'net_pnl', 'exit_reason']])
