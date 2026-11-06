import sys, pickle, importlib.util
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT / "strategies"))
from framework import calc_dhan_charges, calculate_atr

# Test folds
test_folds = [
    ('2020_LAST4M', ROOT / 'data/2020_LAST4M/universe_30_15m.pkl', '2020-09-01', '2020-12-31'),
    ('2020_FULL', ROOT / 'data/2020_FULL/universe_30_15m.pkl', '2020-01-01', '2020-12-31'),
    ('2024_FULL', ROOT / 'data/2024-25/universe_30_15m.pkl', '2024-01-01', '2024-12-31'),
    ('2023_FULL', ROOT / 'data/2023-24/universe_30_15m.pkl', '2023-01-02', '2023-12-29'),
    ('2021_FULL', ROOT / 'data/2021-22/universe_30_15m.pkl', '2021-01-01', '2021-12-31')
]

core_27 = {
    'CANBK', 'DELTACORP', 'FEDERALBNK', 'GMRINFRA', 'GSFC', 'IDFCFIRSTB', 'IOC',
    'JYOTHYLAB', 'MANAPPURAM', 'MOTHERSON', 'NBCC', 'NCC', 'NATIONALUM', 'NMDC',
    'POONAWALLA', 'PNB', 'RBLBANK', 'RCF', 'SAIL', 'SUZLON', 'TATACHEM',
    'TATASTEEL', 'TATAPOWER', 'TRIDENT', 'UNIONBANK', 'YESBANK', 'ZEEL'
}

# Load V39 code directly and build run_simulation_v39_1
with open(ROOT / 'strategies/V39/run_v39.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Modify code to add prev_day_close in data prep
# In V39:
# day_open = df_full.groupby("_date")["open"].transform("first")
# We add:
# daily_close = df_full.groupby("_date")["close"].last()
# prev_close_map = daily_close.shift(1).to_dict()
# df_full["prev_close"] = df_full["_date"].map(prev_close_map)
# df_full["open_gap_pct"] = (day_open - df_full["prev_close"]) / (df_full["prev_close"] + 1e-9) * 100.0

# And in candidate loop:
# if max_gap_up is not None and t_data["open_gap_pct"].iloc[idx] > max_gap_up:
#     continue

print("Code check ready. Writing temporary test engine...")
