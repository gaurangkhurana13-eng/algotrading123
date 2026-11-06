import pickle
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT / "strategies"))

from strategies.V36.run_v36 import run_simulation as run_sim_v36, V29_30_UNIVERSE
from strategies.V37.run_v37 import run_simulation as run_sim_v37

with open(ROOT / "data" / "2024-25" / "universe_14_15m.pkl", "rb") as f:
    p2024 = pickle.load(f)

print("=== RUNNING V36 on 2024 H2 (2024-07-01 to 2024-12-31) ===")
cap36, tr36 = run_sim_v36(p2024, "2024-07-01", "2024-12-31", "V36: 2024 H2", universe_filter=V29_30_UNIVERSE)

print("\n=== RUNNING V37 on 2024 H2 (2024-07-01 to 2024-12-31) ===")
cap37, tr37 = run_sim_v37(p2024, "2024-07-01", "2024-12-31", "V37: 2024 H2", universe_filter=V29_30_UNIVERSE)
