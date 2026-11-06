import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

# Let's import run_v36 functions
sys.path.insert(0, str(ROOT_DIR / "strategies" / "V36"))
from run_v36 import run_simulation, V29_30_UNIVERSE

with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    u1 = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_high_beta_10_15m_1y.pkl", "rb") as f:
    u2 = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_10_15m_1y.pkl", "rb") as f:
    u3 = pickle.load(f)
p2026 = {}
p2026.update(u1); p2026.update(u2); p2026.update(u3)

cap, trades = run_simulation(p2026, "2026-01-01", "2026-12-31", "2026 Bull", universe_filter=V29_30_UNIVERSE)
tdf = pd.DataFrame(trades)

# Check max drawdown in the run
cum = 500.0 + tdf['net_pnl'].cumsum()
hwm = cum.cummax()
dd = (hwm - cum) / hwm
print(f"Base V36 2026 Bull: Net={tdf['net_pnl'].sum():.2f}, Trades={len(tdf)}, Max DD={dd.max()*100:.2f}%, End={cap:.2f}")
