import pandas as pd

v34 = pd.read_csv(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V34.3\fold_2026\trades_intraday.csv")
v36 = pd.read_csv(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V36\fold_2026\trades_intraday.csv")

cum34 = 500.0 + v34['net_pnl'].cumsum()
hwm34 = cum34.cummax()
dd34 = (hwm34 - cum34) / hwm34

cum36 = 500.0 + v36['net_pnl'].cumsum()
hwm36 = cum36.cummax()
dd36 = (hwm36 - cum36) / hwm36

print(f"V34.3 Max DD: {dd34.max()*100:.2f}% (Peak: {hwm34.max():.2f})")
print(f"V36   Max DD: {dd36.max()*100:.2f}% (Peak: {hwm36.max():.2f})")

print("\nV34.3 Drawdown > 15% occurrences:")
print(v34[dd34 > 0.15][['entry_time', 'symbol', 'net_pnl']])

print("\nV36 Drawdown > 15% occurrences:")
print(v36[dd36 > 0.15][['entry_time', 'symbol', 'net_pnl']])
