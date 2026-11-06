import os, sys, pickle
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR))
from tune_v34_2 import test_engine, p2022, p2025, p2026, V29_30_UNIVERSE

print("=" * 75)
print("TESTING CONFIG 1: T1 + T2 SHARPENED ACROSS ALL 3 FOLDS")
print("=" * 75)
test_engine(p2022, "2022-01-01", "2022-12-31", "Fold 1: 2022 Full Year (T1+T2)", t2_enabled=True, t3_enabled=False, t2_min_rvol=1.20, t2_time_window=("09:45", "13:15"))
test_engine(p2025, "2025-08-01", "2025-12-31", "Fold 2: 2025 Last 5M (T1+T2)", t2_enabled=True, t3_enabled=False, t2_min_rvol=1.20, t2_time_window=("09:45", "13:15"), universe_filter=V29_30_UNIVERSE)
test_engine(p2026, "2026-01-01", "2026-12-31", "Fold 3: 2026 Bull (T1+T2)", t2_enabled=True, t3_enabled=False, t2_min_rvol=1.20, t2_time_window=("09:45", "13:15"), universe_filter=V29_30_UNIVERSE)

print("\n" + "=" * 75)
print("TESTING CONFIG 2: T1 ONLY (PURE UNCHOKED CORE) ACROSS ALL 3 FOLDS")
print("=" * 75)
test_engine(p2022, "2022-01-01", "2022-12-31", "Fold 1: 2022 Full Year (T1 Only)", t2_enabled=False, t3_enabled=False)
test_engine(p2025, "2025-08-01", "2025-12-31", "Fold 2: 2025 Last 5M (T1 Only)", t2_enabled=False, t3_enabled=False, universe_filter=V29_30_UNIVERSE)
test_engine(p2026, "2026-01-01", "2026-12-31", "Fold 3: 2026 Bull (T1 Only)", t2_enabled=False, t3_enabled=False, universe_filter=V29_30_UNIVERSE)
