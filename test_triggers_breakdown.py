import os, sys, pickle
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies" / "V34"))

# Let's inspect the triggers in run_v34 by logging which trigger fired for each trade
from run_v34 import run_simulation, V29_30_UNIVERSE

with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    u1 = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_high_beta_10_15m_1y.pkl", "rb") as f:
    u2 = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_10_15m_1y.pkl", "rb") as f:
    u3 = pickle.load(f)
p2026 = {}
p2026.update(u1); p2026.update(u2); p2026.update(u3)

# We can run with each trigger individually or check
# Let's test only T1 in V34 (disable T2 and T3)
# and let's test only T1+T2
# and let's test T1+T2+T3
print("Comparing triggers impact in 2026...")
