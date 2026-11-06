import pickle
import pandas as pd
import numpy as np

# Let us check which trigger fired for trades 132, 134, 135
v36 = pd.read_csv(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V36\fold_2026\trades_intraday.csv")

# Let's inspect the code of run_v36.py to see how triggers are recorded or can be evaluated
for idx in [132, 134, 135]:
    r = v36.loc[idx]
    print(f"Trade {idx}: {r['symbol']} at {r['entry_time']}, exit {r['exit_time']} at {r['exit_price']}, pnl {r['net_pnl']}")
