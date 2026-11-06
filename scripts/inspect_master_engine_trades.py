import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

# Import the simulate_engine from test_v18_master_engine
from test_v18_master_engine import simulate_engine

res = simulate_engine(alpha_win=0.082, single_risk_cap=130.0, hard_sl_mult=0.85)
trades = res.get("trades", [])
# Let's inspect trades if available or re-run to print
