import sys, os, pickle, shutil, importlib.util
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR))

# Import V34
import strategies.V34.run_v34 as v34_mod

# Import V34.2 via importlib (due to dot in name)
spec = importlib.util.spec_from_file_location("v34_2_mod", ROOT_DIR / "strategies" / "V34.2" / "run_v34_2.py")
v34_2_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v34_2_mod)

print("=" * 80)
print("BHAVISHYA V34 VS V34.2: 1000% UNSEEN OUT-OF-SAMPLE AUDIT RUN")
print("=" * 80)

# 1. Prepare 2026 Unseen Universe (52 Midcaps from universe_78_15m_1y.pkl outside V29_30)
with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    u78 = pickle.load(f)

clean_78 = {k.replace(".NS", "") for k in u78.keys()}
unseen_stocks_2026 = clean_78 - v34_mod.V29_30_UNIVERSE
print(f"Loaded 2026 universe_78: {len(clean_78)} symbols.")
print(f"Filtered out 30 in-sample stocks. Unseen pool size: {len(unseen_stocks_2026)} stocks.")
print(f"Unseen sample: {sorted(list(unseen_stocks_2026))[:15]} ...\n")

# 2. Prepare 2025 Q4 Universe (Last 3 months of 2025: Oct 1 - Dec 31, 2025)
with open(ROOT_DIR / "data" / "2025-26" / "universe_14_15m.pkl", "rb") as f:
    p2025 = pickle.load(f)
clean_2025 = {k.replace(".NS", "") for k in p2025.keys()}
print(f"Loaded 2025 Q4 pool: {len(clean_2025)} symbols ({sorted(list(clean_2025))}).\n")

out_v34_2026 = ROOT_DIR / "strategies" / "V34" / "unseen_2026"
out_v34_2025_q4 = ROOT_DIR / "strategies" / "V34" / "unseen_2025_q4"

out_v34_2_2026 = ROOT_DIR / "strategies" / "V34.2" / "unseen_2026"
out_v34_2_2025_q4 = ROOT_DIR / "strategies" / "V34.2" / "unseen_2025_q4"

# ==========================================
# RUN V34
# ==========================================
print(">>> RUNNING BHAVISHYA V34...")
# A. 2026 Unseen 52 stocks
print("  [V34] 2026 Whole Horizon on 52 Unseen Stocks...")
cap_v34_2026, tr_v34_2026 = v34_mod.run_simulation(
    u78, "2026-01-01", "2026-12-31", "V34 Unseen 2026", universe_filter=unseen_stocks_2026
)
v34_mod.save_gate_v2_artifacts(out_v34_2026, "V34 2026 Unseen 52 Stocks", tr_v34_2026)

# B. 2025 Last 3 Months (2025-10-01 to 2025-12-31)
print("  [V34] 2025 Q4 (Oct-Dec 2025)...")
cap_v34_2025, tr_v34_2025 = v34_mod.run_simulation(
    p2025, "2025-10-01", "2025-12-31", "V34 2025 Q4"
)
v34_mod.save_gate_v2_artifacts(out_v34_2025_q4, "V34 2025 Q4 (Last 3M)", tr_v34_2025)

# ==========================================
# RUN V34.2
# ==========================================
print("\n>>> RUNNING BHAVISHYA V34.2...")
# A. 2026 Unseen 52 stocks
print("  [V34.2] 2026 Whole Horizon on 52 Unseen Stocks...")
cap_v34_2_2026, tr_v34_2_2026 = v34_2_mod.run_simulation(
    u78, "2026-01-01", "2026-12-31", "V34.2 Unseen 2026", universe_filter=unseen_stocks_2026
)
v34_2_mod.save_gate_v2_artifacts(out_v34_2_2026, "V34.2 2026 Unseen 52 Stocks", tr_v34_2_2026)

# B. 2025 Last 3 Months (2025-10-01 to 2025-12-31)
print("  [V34.2] 2025 Q4 (Oct-Dec 2025)...")
cap_v34_2_2025, tr_v34_2_2025 = v34_2_mod.run_simulation(
    p2025, "2025-10-01", "2025-12-31", "V34.2 2025 Q4"
)
v34_2_mod.save_gate_v2_artifacts(out_v34_2_2025_q4, "V34.2 2025 Q4 (Last 3M)", tr_v34_2_2025)

print("\n" + "=" * 80)
print("ALL SIMULATIONS COMPLETED SUCCESSFULLY! ARTIFACTS SAVED.")
print("=" * 80)
