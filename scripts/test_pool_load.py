import pickle, sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

from framework import calc_dhan_charges, calculate_atr

# Load all 19 folds and their trades
data = pickle.load(open(ROOT / 'scripts/v42_19_folds_master_results.pkl', 'rb'))
trades = data['all_trades']

# Let's verify which pool contains each trade
folds_def = [
    (0, "2021", ROOT / 'data/2021-22/universe_30_15m.pkl', "2021-01-01", "2021-12-31"),
    (1, "2026", 'COMPOSITE_1', "2026-01-01", "2026-12-31"),
    (2, "2022", ROOT / 'data/2022-23/universe_30_15m.pkl', "2022-01-01", "2022-12-31"),
    (3, "2023", ROOT / 'data/2023-24/universe_30_15m.pkl', "2023-01-02", "2023-12-29"),
    (4, "2023_H1", ROOT / 'data/2023-24/universe_30_15m.pkl', "2023-01-01", "2023-04-30"),
    (5, "2023_MID", ROOT / 'data/2023-24/universe_30_15m.pkl', "2023-05-01", "2023-08-31"),
    (6, "2023_H2", ROOT / 'data/2023-24/universe_30_15m.pkl', "2023-09-01", "2023-12-31"),
    (7, "2024_H1", ROOT / 'data/2024-25/universe_14_15m.pkl', "2024-01-01", "2024-06-30"),
    (8, "2024_H2", ROOT / 'data/2024-25/universe_14_15m.pkl', "2024-07-01", "2024-12-31"),
    (9, "2025_H1", ROOT / 'data/2025-26/universe_14_15m.pkl', "2025-01-01", "2025-07-31"),
    (10, "2025_Q4", ROOT / 'data/2025-26/universe_14_15m.pkl', "2025-10-01", "2025-12-31"),
    (11, "2025_L5M", ROOT / 'data/2025-26/universe_14_15m.pkl', "2025-08-01", "2025-12-31"),
    (12, "2026_VIRG", 'COMPOSITE_12', "2026-01-01", "2026-10-31"),
    (13, "2026_UNSEEN", 'COMPOSITE_13', "2026-01-01", "2026-12-31"),
    (14, "2020", ROOT / 'data/2020_FULL/universe_30_15m.pkl', "2020-01-01", "2020-12-31"),
    (15, "2019", ROOT / 'data/2019_UNSEEN/universe_30_15m.pkl', "2019-01-01", "2019-12-31"),
    (16, "2018", ROOT / 'data/2018_UNSEEN/universe_30_15m.pkl', "2018-01-01", "2018-12-31"),
    (17, "2017", ROOT / 'data/2017_UNSEEN/universe_30_15m.pkl', "2017-01-01", "2017-12-31"),
    (18, "2016_H2", ROOT / 'data/2016_H2_UNSEEN/universe_30_15m.pkl', "2016-07-01", "2016-12-31"),
]

def load_pool_by_spec(spec):
    if spec == 'COMPOSITE_1':
        u1 = pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))
        u2 = pickle.load(open(ROOT / 'data/unseen_high_beta_10_15m_1y.pkl', 'rb'))
        u3 = pickle.load(open(ROOT / 'data/unseen_10_15m_1y.pkl', 'rb'))
        return {**u1, **u2, **u3}
    elif spec == 'COMPOSITE_12':
        b1 = pickle.load(open(ROOT / 'data/unseen_10_15m_1y.pkl', 'rb'))
        b3 = pickle.load(open(ROOT / 'data/unseen_batch3_10_15m_1y.pkl', 'rb'))
        return {**b1, **b3}
    elif spec == 'COMPOSITE_13':
        return pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))
    else:
        return pickle.load(open(spec, 'rb'))

print("Loading pool index mapping...")
# We will cache the pools as needed
pool_cache = {}
for fid, lbl, spec, s_dt, e_dt in folds_def:
    pool_cache[fid] = load_pool_by_spec(spec)
print("All pools loaded successfully.")
