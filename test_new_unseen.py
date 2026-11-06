import pickle
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT / "strategies"))
from tune_v37 import run_sim_candidate, V29_30_UNIVERSE

with open(ROOT / "data" / "2024-25" / "universe_14_15m.pkl", "rb") as f:
    p2024 = pickle.load(f)

with open(ROOT / "data" / "2025-26" / "universe_14_15m.pkl", "rb") as f:
    p2025 = pickle.load(f)

with open(ROOT / "data" / "2023-24" / "universe_30_15m.pkl", "rb") as f:
    p2023 = pickle.load(f)

print("=" * 80)
print("TESTING V36 vs V37 ON BRAND NEW UNSEEN FOLDS")
print("=" * 80)

print("\n--- UNSEEN FOLD 1: 2024 H2 (Jul 1 - Dec 31, 2024 Post-Election) ---")
print("V36:")
run_sim_candidate(p2024, "2024-07-01", "2024-12-31", "V36: 2024 H2", universe_filter=V29_30_UNIVERSE, t1b_rvol_min=1.30)
print("V37:")
run_sim_candidate(p2024, "2024-07-01", "2024-12-31", "V37: 2024 H2", universe_filter=V29_30_UNIVERSE, t1b_rvol_min=1.50)

print("\n--- UNSEEN FOLD 2: 2025 H1 (Jan 1 - Jul 31, 2025 First 7 Months) ---")
print("V36:")
run_sim_candidate(p2025, "2025-01-01", "2025-07-31", "V36: 2025 H1", universe_filter=V29_30_UNIVERSE, t1b_rvol_min=1.30)
print("V37:")
run_sim_candidate(p2025, "2025-01-01", "2025-07-31", "V37: 2025 H1", universe_filter=V29_30_UNIVERSE, t1b_rvol_min=1.50)

print("\n--- UNSEEN FOLD 3: 2023 Q1 (Jan 1 - Apr 30, 2023 Hindenburg Crash) ---")
print("V36:")
run_sim_candidate(p2023, "2023-01-01", "2023-04-30", "V36: 2023 Q1", universe_filter=V29_30_UNIVERSE, t1b_rvol_min=1.30)
print("V37:")
run_sim_candidate(p2023, "2023-01-01", "2023-04-30", "V37: 2023 Q1", universe_filter=V29_30_UNIVERSE, t1b_rvol_min=1.50)

print("=" * 80)
