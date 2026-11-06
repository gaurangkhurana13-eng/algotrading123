import sys, pickle
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
from scripts.test_v41_titan_prototype import folds_def, load_pool_by_spec, V29_30_UNIVERSE

print("="*80)
print("INSPECTING REGIME INDICATORS ACROSS ALL 15 FOLDS")
print("="*80)

for fid, fname, pdesc, pool_spec, s_dt, e_dt, u_filt in folds_def:
    pool = load_pool_by_spec(pool_spec)
    if u_filt == 'COMPOSITE_13_UNIV':
        clean_78 = {k.replace('.NS', '') for k in pool.keys()}
        filt = clean_78 - V29_30_UNIVERSE
    elif u_filt is None:
        filt = set(k.replace('.NS', '') for k in pool.keys())
    else:
        filt = u_filt

    # Gather closes
    closes = {}
    c300s = {}
    day_rets = {}
    for k, df in pool.items():
        clean = k.replace('.NS', '')
        if clean not in filt: continue
        sub = df.loc[s_dt:e_dt]
        if len(sub) < 150: continue
        c = sub['close']
        c300 = c.shift(min(300, len(c)-1))
        df_sub = sub.copy()
        df_sub['_date'] = df_sub.index.date
        d_open = df_sub.groupby('_date')['open'].transform('first')
        d_ret = (c / d_open) - 1.0
        closes[clean] = c
        c300s[clean] = (c > c300).astype(float)
        day_rets[clean] = d_ret

    if not closes: continue
    df_c300 = pd.DataFrame(c300s)
    df_day = pd.DataFrame(day_rets)
    mean_breadth = df_c300.mean(axis=1).mean()
    p10_breadth = df_c300.mean(axis=1).quantile(0.10)
    p90_breadth = df_c300.mean(axis=1).quantile(0.90)
    mean_day_ret = df_day.mean(axis=1).mean()

    print(f"[{fid:2d}] {fname:30s} | Breadth Mean: {mean_breadth:.2f} (10%: {p10_breadth:.2f}, 90%: {p90_breadth:.2f}) | DayRet Mean: {mean_day_ret*100:+.2f}%")
