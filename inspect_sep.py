import pickle
import pandas as pd
import numpy as np

v36 = pd.read_csv(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V36\fold_2026\trades_intraday.csv")
v36_sep = v36[v36['entry_time'].str.startswith('2026-09')]

print("V36 September 2026 Trades:")
for idx, r in v36_sep.iterrows():
    print(f"[{idx}] {r['symbol']} | {r['entry_time']} -> {r['exit_time']} | Shares: {r['shares']} | Net: {r['net_pnl']} | Exit: {r['exit_reason']}")
