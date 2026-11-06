import pandas as pd

v34 = pd.read_csv(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V34.3\fold_2026\trades_intraday.csv")
v34_sep = v34[v34['entry_time'].str.startswith('2026-09')]

print("V34.3 September 2026 Trades:")
for idx, r in v34_sep.iterrows():
    print(f"[{idx}] {r['symbol']} | {r['entry_time']} -> {r['exit_time']} | Shares: {r['shares']} | Net: {r['net_pnl']} | Exit: {r['exit_reason']}")
