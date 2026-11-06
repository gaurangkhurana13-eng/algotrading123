import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

with open(ROOT_DIR / "data" / "unseen_batch3_10_15m_1y.pkl", "rb") as f:
    unseen_batch3 = pickle.load(f)

from test_v18_on_unseen_batch3 import df_res
print("=== BATCH 3 INDIVIDUAL TICKER PnL BREAKDOWN ===")
if len(df_res) > 0:
    for t, grp in df_res.groupby("ticker"):
        n = len(grp)
        w = (grp["net_pnl"] > 0).sum()
        net = grp["net_pnl"].sum()
        wr = (w / n * 100.0)
        print(f"{t:>15}: {n:>2} trades | WR {wr:>5.1f}% | Net Rs {net:>7.2f}")
