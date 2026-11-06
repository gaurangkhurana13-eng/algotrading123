import pandas as pd
from pathlib import Path

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
df34 = pd.read_csv(ROOT / "strategies/V34.3/trades_intraday.csv")
df42 = pd.read_csv(ROOT / "strategies/V42_FOLD_1/trades_intraday.csv")

df34['key'] = df34['symbol'] + "_" + df34['entry_time']
df42['key'] = df42['symbol'] + "_" + df42['entry_time']

merged = pd.merge(df34, df42, on='key', suffixes=('_34', '_42'))
print(f"Total merged trades: {len(merged)}")

diff_shares = (merged['shares_34'] != merged['shares_42']).sum()
diff_exit_t = (merged['exit_time_34'] != merged['exit_time_42']).sum()
diff_exit_p = (merged['exit_price_34'] != merged['exit_price_42']).sum()

print(f"Trades with different shares: {diff_shares} / {len(merged)}")
print(f"Trades with different exit time: {diff_exit_t} / {len(merged)}")
print(f"Trades with different exit price: {diff_exit_p} / {len(merged)}")

print("\nSample comparison of top 10 divergent trades by PnL diff:")
merged['pnl_diff'] = merged['net_pnl_34'] - merged['net_pnl_42']
sample = merged.sort_values('pnl_diff', ascending=False)[[
    'symbol_34', 'entry_time_34', 'entry_price_34',
    'shares_34', 'shares_42',
    'exit_time_34', 'exit_time_42',
    'exit_price_34', 'exit_price_42',
    'net_pnl_34', 'net_pnl_42', 'pnl_diff'
]].head(10)

print(sample.to_string(index=False))
