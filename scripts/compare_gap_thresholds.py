import os, sys, pickle, importlib.util
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT / "strategies"))

# Read V39 source
with open(ROOT / 'strategies/V39/run_v39.py', 'r', encoding='utf-8') as f:
    v39_code = f.read()

# Build V39.1 function generator
def build_v39_1_mod(max_gap_up: float):
    # 1. Update data prep to include prev_close and open_gap_pct
    old_prep = '        day_open = df_full.groupby("_date")["open"].transform("first")\n        df_full["day_ret"] = (df_full["close"] / day_open) - 1.0'
    new_prep = '''        day_open = df_full.groupby("_date")["open"].transform("first")
        daily_close = df_full.groupby("_date")["close"].last()
        prev_close_map = daily_close.shift(1).to_dict()
        df_full["prev_close"] = df_full["_date"].map(prev_close_map)
        df_full["open_gap_pct"] = (day_open - df_full["prev_close"]) / (df_full["prev_close"] + 1e-9) * 100.0
        df_full["day_ret"] = (df_full["close"] / day_open) - 1.0'''
    
    # 2. Add open_gap_pct to data_map dict
    old_dict = '            "c_300": c_300, "day_open": day_open,'
    new_dict = '            "c_300": c_300, "day_open": day_open, "open_gap_pct": df_full["open_gap_pct"],'
    
    # 3. Add gap check in candidate loop
    old_cand = '            # Day Trader Skill: Time Machine Odds Check\n            if min_tm_odds > 0:\n                odds = t_data["tm_odds"].iloc[idx]\n                if not np.isnan(odds) and odds < min_tm_odds:\n                    continue'
    new_cand = f'''            # Day Trader Skill: Time Machine Odds Check
            if min_tm_odds > 0:
                odds = t_data["tm_odds"].iloc[idx]
                if not np.isnan(odds) and odds < min_tm_odds:
                    continue

            # V39.1: Morning Opening Gap-Up Exhaustion Filter
            gap_pct = t_data["open_gap_pct"].iloc[idx]
            if not np.isnan(gap_pct) and gap_pct > {max_gap_up}:
                continue'''
                
    modified = v39_code.replace(old_prep, new_prep).replace(old_dict, new_dict).replace(old_cand, new_cand)
    
    # create temporary module
    mod = {}
    exec(modified, mod)
    return mod

test_folds = [
    ('2020_LAST4M', ROOT / 'data/2020_LAST4M/universe_30_15m.pkl', '2020-09-01', '2020-12-31'),
    ('2020_FULL', ROOT / 'data/2020_FULL/universe_30_15m.pkl', '2020-01-01', '2020-12-31'),
    ('2024_FULL', ROOT / 'data/2024-25/universe_30_15m.pkl', '2024-01-01', '2024-12-31'),
    ('2023_FULL', ROOT / 'data/2023-24/universe_30_15m.pkl', '2023-01-02', '2023-12-29')
]

core_27 = {
    'CANBK', 'DELTACORP', 'FEDERALBNK', 'GMRINFRA', 'GSFC', 'IDFCFIRSTB', 'IOC',
    'JYOTHYLAB', 'MANAPPURAM', 'MOTHERSON', 'NBCC', 'NCC', 'NATIONALUM', 'NMDC',
    'POONAWALLA', 'PNB', 'RBLBANK', 'RCF', 'SAIL', 'SUZLON', 'TATACHEM',
    'TATASTEEL', 'TATAPOWER', 'TRIDENT', 'UNIONBANK', 'YESBANK', 'ZEEL'
}

print("=== COMPARING GAP THRESHOLDS: Baseline vs 2.0% vs 1.5% vs 1.2% ===")

# Baseline V39
spec = importlib.util.spec_from_file_location('v39_base', str(ROOT / 'strategies/V39/run_v39.py'))
v39_base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v39_base)

mod_20 = build_v39_1_mod(2.0)
mod_15 = build_v39_1_mod(1.5)
mod_12 = build_v39_1_mod(1.2)

for fname, pkl_path, s_dt, e_dt in test_folds:
    with open(pkl_path, 'rb') as f:
        u = pickle.load(f)
    print(f"\n--- FOLD: {fname} ---")
    
    for label, m in [('V39 Baseline', v39_base.run_simulation), ('V39.1 (gap<=2.0%)', mod_20['run_simulation']), ('V39.1 (gap<=1.5%)', mod_15['run_simulation']), ('V39.1 (gap<=1.2%)', mod_12['run_simulation'])]:
        cap, trades = m(u, s_dt, e_dt, f'{label}_{fname}', universe_filter=core_27)
        df_t = pd.DataFrame(trades)
        n = len(df_t)
        net = df_t['net_pnl'].sum() if n > 0 else 0.0
        wr = (len(df_t[df_t['net_pnl'] > 0]) / n * 100) if n > 0 else 0.0
        print(f"  {label:20s}: {n:3d} trades | WR: {wr:5.1f}% | Net: Rs {net:+7.2f} | End Wallet: Rs {cap:7.2f}")
