import sys
from pathlib import Path
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT / "strategies" / "V34.3"))
from run_v34_3 import run_simulation, V29_30_UNIVERSE
import pickle

with open(ROOT / "data" / "2024-25" / "universe_14_15m.pkl", "rb") as f:
    p2024 = pickle.load(f)

print("=== RUNNING V34.3 on 2024 H2 (2024-07-01 to 2024-12-31) ===")
cap34, tr34 = run_simulation(p2024, "2024-07-01", "2024-12-31", "V34.3: 2024 H2", universe_filter=V29_30_UNIVERSE)
