import pickle
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT / "strategies"))

from strategies.V34.run_v34_3 import run_simulation as run_sim_v34, V29_30_UNIVERSE

with open(ROOT / "data" / "2024-25" / "universe_14_15m.pkl", "rb") as f:
    p2024 = pickle.load(f)

print("=== RUNNING V34.3 on 2024 H2 (2024-07-01 to 2024-12-31) ===")
cap34, tr34 = run_sim_v34(p2024, "2024-07-01", "2024-12-31", "V34.3: 2024 H2", universe_filter=V29_30_UNIVERSE)
