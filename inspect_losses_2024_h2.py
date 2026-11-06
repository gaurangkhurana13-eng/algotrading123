import pandas as pd
from pathlib import Path
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
import pickle
import sys
sys.path.insert(0, str(ROOT / "strategies" / "V37"))
from run_v37 import run_simulation, V29_30_UNIVERSE

with open(ROOT / "data" / "2024-25" / "universe_14_15m.pkl", "rb") as f:
    p2024 = pickle.load(f)

cap, tr = run_simulation(p2024, "2024-07-01", "2024-12-31", "2024 H2", universe_filter=V29_30_UNIVERSE)
df = pd.DataFrame(tr)

print("Exit Reasons in 2024 H2:")
print(df['exit_reason'].value_counts())
print("\nPnL by Exit Reason:")
print(df.groupby('exit_reason')['net_pnl'].agg(['count', 'sum', 'mean']))

print("\nWorst 5 Trades in 2024 H2:")
print(df.nsmallest(5, 'net_pnl')[['symbol', 'entry_time', 'exit_time', 'net_pnl', 'exit_reason']])
