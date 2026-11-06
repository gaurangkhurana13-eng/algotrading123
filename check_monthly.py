import pandas as pd

v36 = pd.read_csv(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V36\fold_2026\trades_intraday.csv")

# Let's inspect win rate and PnL by month or by trigger
v36['month'] = pd.to_datetime(v36['entry_time']).dt.to_period('M')
summary = v36.groupby('month').agg(
    trades=('net_pnl', 'count'),
    wins=('net_pnl', lambda x: (x > 0).sum()),
    net_pnl=('net_pnl', 'sum'),
    charges=('charges', 'sum')
)
summary['wr'] = (summary['wins'] / summary['trades']) * 100
print("V36 Monthly Breakdown in 2026 Bull:")
print(summary)

v34 = pd.read_csv(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V34.3\fold_2026\trades_intraday.csv")
v34['month'] = pd.to_datetime(v34['entry_time']).dt.to_period('M')
summary34 = v34.groupby('month').agg(
    trades=('net_pnl', 'count'),
    wins=('net_pnl', lambda x: (x > 0).sum()),
    net_pnl=('net_pnl', 'sum'),
    charges=('charges', 'sum')
)
summary34['wr'] = (summary34['wins'] / summary34['trades']) * 100
print("\nV34.3 Monthly Breakdown in 2026 Bull:")
print(summary34)
