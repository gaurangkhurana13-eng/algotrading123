import pandas as pd

v36 = pd.read_csv(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V36\fold_2026\trades_intraday.csv")
cum = 500.0 + v36['net_pnl'].cumsum()
hwm = cum.cummax()
dd = (hwm - cum) / hwm

print(f"Trade 136 exit equity: {cum.iloc[136]:.2f}")
print(f"HWM at that point:     {hwm.iloc[136]:.2f}")
print(f"Drawdown percentage:   {dd.iloc[136]*100:.2f}%")
