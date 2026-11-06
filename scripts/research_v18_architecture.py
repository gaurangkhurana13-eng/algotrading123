import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr
import strategy_evaluate_gate_v2 as gate

# Load all candidate data pools
with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    univ = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_high_beta_10_15m_1y.pkl", "rb") as f:
    d_beta = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_10_15m_1y.pkl", "rb") as f:
    d_unseen = pickle.load(f)

# Let's inspect which tickers are available in univ, d_beta, d_unseen
print("Univ tickers count:", len(univ))
print("Beta tickers count:", len(d_beta))
print("Unseen tickers count:", len(d_unseen))

base_10_keys = [
    "PCJEWELLER.NS", "RPOWER.NS", "DELTACORP.NS", "RITES.NS",
    "IDEA.NS", "MTNL.NS", "ZEEL.NS", "UTKARSHBNK.NS", "IFCI.NS", "NCC.NS"
]

candidates = {}
for k, df in univ.items():
    if k not in base_10_keys:
        candidates[k] = df
for k, df in d_beta.items():
    if k not in base_10_keys and k not in candidates:
        candidates[k] = df
for k, df in d_unseen.items():
    if k not in base_10_keys and k not in candidates:
        candidates[k] = df

print("Additional candidates to test:", len(candidates))
