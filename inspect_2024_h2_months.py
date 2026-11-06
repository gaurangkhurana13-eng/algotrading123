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
df['month'] = pd.to_datetime(df['entry_time']).dt.to_period('M')

print("\n--- 2024 H2 Monthly Breakdown (V37) ---")
summary = df.groupby('month').agg(
    trades=('net_pnl', 'count'),
    wins=('net_pnl', lambda x: (x > 0).sum()),
    losses=('net_pnl', lambda x: (x < 0).sum()),
    net_pnl=('net_pnl', 'sum'),
    charges=('charges', 'sum')
)
summary['wr'] = (summary['wins'] / summary['trades']) * 100
print(summary)
