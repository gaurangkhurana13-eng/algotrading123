import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

# Import base components
from strategies.V36.run_v36 import (
    V29_30_UNIVERSE, SECTOR_MAP, SYM_TO_SECTOR,
    compute_vwap_shifted, compute_er10, compute_adx_di
)

# Load 2026 data
with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    u1 = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_high_beta_10_15m_1y.pkl", "rb") as f:
    u2 = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_10_15m_1y.pkl", "rb") as f:
    u3 = pickle.load(f)
p2026 = {}
p2026.update(u1); p2026.update(u2); p2026.update(u3)

# Load 2024 H1 data
with open(ROOT_DIR / "data" / "2024-25" / "universe_14_15m.pkl", "rb") as f:
    p2024 = pickle.load(f)

# Load 2023 Mid 4M data
with open(ROOT_DIR / "data" / "2023-24" / "universe_30_15m.pkl", "rb") as f:
    p2023 = pickle.load(f)

# Load 2022 data
with open(ROOT_DIR / "data" / "2022-23" / "universe_30_15m.pkl", "rb") as f:
    p2022 = pickle.load(f)

print("Data loaded successfully.")
