import os, sys, pickle
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    u1 = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_high_beta_10_15m_1y.pkl", "rb") as f:
    u2 = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_10_15m_1y.pkl", "rb") as f:
    u3 = pickle.load(f)
p2026 = {}
p2026.update(u1); p2026.update(u2); p2026.update(u3)

from strategies.V34.run_v34 import run_simulation, V29_30_UNIVERSE

# Let's inspect the trades_intraday.csv of V34 vs V33
df33 = pd.read_csv(ROOT_DIR / "strategies" / "V33" / "fold_2026" / "trades_intraday.csv")
df34 = pd.read_csv(ROOT_DIR / "strategies" / "V34" / "fold_2026" / "trades_intraday.csv")

# Match by symbol and entry_time
merged = pd.merge(df34, df33, on=["symbol", "entry_time"], suffixes=("_v34", "_v33"), how="left")
common = merged[~merged["net_pnl_v33"].isna()]
only_v34 = merged[merged["net_pnl_v33"].isna()]

print(f"Total V34 trades: {len(df34)}")
print(f"Common trades (also in V33): {len(common)}")
print(f"  V34 net on common trades: Rs {common['net_pnl_v34'].sum():.2f}")
print(f"  V33 net on common trades: Rs {common['net_pnl_v33'].sum():.2f}")
print(f"  Diff on common trades: Rs {common['net_pnl_v34'].sum() - common['net_pnl_v33'].sum():.2f}")

print(f"\nNew extra trades ONLY in V34: {len(only_v34)}")
print(f"  Net on new extra trades: Rs {only_v34['net_pnl_v34'].sum():.2f}")
print(f"  Wins in new trades: {(only_v34['net_pnl_v34'] > 0).sum()} / {len(only_v34)}")
print(f"  Gross win on new trades: Rs {only_v34[only_v34['net_pnl_v34'] > 0]['net_pnl_v34'].sum():.2f}")
print(f"  Gross loss on new trades: Rs {only_v34[only_v34['net_pnl_v34'] < 0]['net_pnl_v34'].sum():.2f}")
print(f"  Charges on new trades: Rs {only_v34['charges_v34'].sum():.2f}")

print("\nWhere did the diff on COMMON trades come from?")
print(common[abs(common['net_pnl_v34'] - common['net_pnl_v33']) > 10][["symbol", "entry_time", "net_pnl_v33", "net_pnl_v34", "exit_price_v33", "exit_price_v34", "shares_v33", "shares_v34"]].head(10))
