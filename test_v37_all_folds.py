import pickle
import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

from tune_v37 import run_sim_candidate, V29_30_UNIVERSE

# Load all datasets
with open(ROOT_DIR / "data" / "2022-23" / "universe_30_15m.pkl", "rb") as f:
    p2022 = pickle.load(f)
with open(ROOT_DIR / "data" / "2025-26" / "universe_14_15m.pkl", "rb") as f:
    p2025 = pickle.load(f)
with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    u1 = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_high_beta_10_15m_1y.pkl", "rb") as f:
    u2 = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_10_15m_1y.pkl", "rb") as f:
    u3 = pickle.load(f)
p2026 = {}
p2026.update(u1); p2026.update(u2); p2026.update(u3)

with open(ROOT_DIR / "data" / "2023-24" / "universe_30_15m.pkl", "rb") as f:
    p2023 = pickle.load(f)
with open(ROOT_DIR / "data" / "2024-25" / "universe_14_15m.pkl", "rb") as f:
    p2024 = pickle.load(f)

clean_78 = {k.replace(".NS", "") for k in u1.keys()}
unseen_stocks_2026 = clean_78 - V29_30_UNIVERSE

print("=" * 80)
print("TESTING CANDIDATE: T1B RVOL >= 1.50 ACROSS ALL 7 FOLDS")
print("=" * 80)

print("\n[FOLD 1: 2022 Bear]")
run_sim_candidate(p2022, "2022-01-01", "2022-12-31", "Fold 1: 2022 Bear", dd_guard_thresh=0.0, symbol_loss_cooldown_days=0, t1b_rvol_min=1.50)

print("\n[FOLD 2: 2025 Chop]")
run_sim_candidate(p2025, "2025-08-01", "2025-12-31", "Fold 2: 2025 Chop", universe_filter=V29_30_UNIVERSE, dd_guard_thresh=0.0, symbol_loss_cooldown_days=0, t1b_rvol_min=1.50)

print("\n[FOLD 3: 2026 Bull]")
run_sim_candidate(p2026, "2026-01-01", "2026-12-31", "Fold 3: 2026 Bull", universe_filter=V29_30_UNIVERSE, dd_guard_thresh=0.0, symbol_loss_cooldown_days=0, t1b_rvol_min=1.50)

print("\n[FOLD 4: 2023 Mid 4M]")
run_sim_candidate(p2023, "2023-05-01", "2023-08-31", "Fold 4: 2023 Mid 4M", universe_filter=V29_30_UNIVERSE, dd_guard_thresh=0.0, symbol_loss_cooldown_days=0, t1b_rvol_min=1.50)

print("\n[FOLD 5: 2025 Q4]")
run_sim_candidate(p2025, "2025-10-01", "2025-12-31", "Fold 5: 2025 Q4", dd_guard_thresh=0.0, symbol_loss_cooldown_days=0, t1b_rvol_min=1.50)

print("\n[FOLD 6: 2026 Unseen 52 Midcaps]")
run_sim_candidate(u1, "2026-01-01", "2026-12-31", "Fold 6: 2026 Unseen", universe_filter=unseen_stocks_2026, dd_guard_thresh=0.0, symbol_loss_cooldown_days=0, t1b_rvol_min=1.50)

print("\n[FOLD 7: 2024 H1 Election Year]")
run_sim_candidate(p2024, "2024-01-01", "2024-06-30", "Fold 7: 2024 H1", universe_filter=V29_30_UNIVERSE, dd_guard_thresh=0.0, symbol_loss_cooldown_days=0, t1b_rvol_min=1.50)

print("\n" + "=" * 80)
