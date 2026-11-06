import pandas as pd

v34 = pd.read_csv(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V34.3\fold_2026\trades_intraday.csv")
v36 = pd.read_csv(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V36\fold_2026\trades_intraday.csv")

cum_34 = 500.0 + v34['net_pnl'].cumsum()
cum_36 = 500.0 + v36['net_pnl'].cumsum()

print(f"V34.3 Equity right before trade 94: {cum_34.iloc[93]:.2f}")
print(f"V36 Equity right before trade 137:   {cum_36.iloc[136]:.2f}")
