import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
df34 = pd.read_csv(ROOT / "strategies/V34.3/trades_intraday.csv")
df42 = pd.read_csv(ROOT / "strategies/V42_FOLD_1/trades_intraday.csv")

df34['cum_pnl'] = 500.0 + df34['net_pnl'].cumsum().shift(1).fillna(0.0)
df42['cum_pnl'] = 500.0 + df42['net_pnl'].cumsum().shift(1).fillna(0.0)

t34 = df34[df34['shares']==492].iloc[0]
entry_t = t34['entry_time']
t42 = df42[df42['entry_time']==entry_t].iloc[0]

print(f"Trade at {entry_t}:")
print(f"  V34.3: Equity before trade: Rs {t34['cum_pnl']:.2f} | Shares: {t34['shares']} | Net PnL: Rs {t34['net_pnl']:.2f}")
print(f"  V42:   Equity before trade: Rs {t42['cum_pnl']:.2f} | Shares: {t42['shares']} | Net PnL: Rs {t42['net_pnl']:.2f}")
