import os, sys, pickle, importlib.util
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

# Import our test engine from test_v42_prototype
from scripts.test_v42_prototype import simulate_engine

V29_30_UNIVERSE = {
    "J&KBANK", "UTKARSHBNK", "MANALIPETC", "IDEA", "PCJEWELLER",
    "NIACL", "MTNL", "IFCI", "TEXRAIL", "SWSOLAR",
    "TTML", "RENUKA", "RCF", "NBCC", "JYOTHYLAB",
    "RBLBANK", "ELECTCAST", "PNB", "GSFC", "TATASTEEL",
    "SUZLON", "CENTRALBK", "IOB", "IRCON", "NMDC",
    "RITES", "NCC", "ZEEL", "DELTACORP", "RPOWER"
}

core_27 = {
    'CANBK', 'DELTACORP', 'FEDERALBNK', 'GMRINFRA', 'GSFC', 'IDFCFIRSTB', 'IOC',
    'JYOTHYLAB', 'MANAPPURAM', 'MOTHERSON', 'NBCC', 'NCC', 'NATIONALUM', 'NMDC',
    'POONAWALLA', 'PNB', 'RBLBANK', 'RCF', 'SAIL', 'SUZLON', 'TATACHEM',
    'TATASTEEL', 'TATAPOWER', 'TRIDENT', 'UNIONBANK', 'YESBANK', 'ZEEL'
}

stocks_30_legacy = {
    'CANBK', 'PNB', 'BANKBARODA', 'SBIN', 'IOC',
    'BPCL', 'GAIL', 'COALINDIA', 'TATASTEEL', 'TATAPOWER',
    'TATAMOTORS', 'VEDL', 'HINDALCO', 'JINDALSTEL', 'JSWSTEEL',
    'BEL', 'ADANIPOWER', 'ADANIENT', 'RECLTD', 'PFC',
    'DLF', 'MOTHERSON', 'ITC', 'ICICIBANK', 'AXISBANK',
    'RELIANCE', 'INFY', 'SUNPHARMA', 'TECHM', 'CIPLA'
}

folds_def = [
    # Pre-2020 Crucial Regimes:
    ("2016_H2", ROOT / 'data/2016_H2_UNSEEN/universe_30_15m.pkl', "2016-07-01", "2016-12-31", stocks_30_legacy),
    ("2017_FULL", ROOT / 'data/2017_UNSEEN/universe_30_15m.pkl', "2017-01-01", "2017-12-31", stocks_30_legacy),
    ("2018_FULL", ROOT / 'data/2018_UNSEEN/universe_30_15m.pkl', "2018-01-01", "2018-12-31", stocks_30_legacy),
    ("2019_FULL", ROOT / 'data/2019_UNSEEN/universe_30_15m.pkl', "2019-01-01", "2019-12-31", None),
    ("2020_FULL", ROOT / 'data/2020_FULL/universe_30_15m.pkl', "2020-01-01", "2020-12-31", core_27),
    # Post-2020 Regimes:
    ("2021_FULL", ROOT / 'data/2021-22/universe_30_15m.pkl', "2021-01-01", "2021-12-31", {'CENTRALBK', 'DELTACORP', 'GSFC', 'IDEA', 'IOB', 'JYOTHYLAB', 'NBCC', 'NCC', 'NIACL', 'NMDC', 'PNB', 'RBLBANK', 'RCF', 'RENUKA', 'RITES', 'SUZLON', 'SWSOLAR', 'TATASTEEL', 'TTML', 'ZEEL'}),
    ("2022_BEAR", ROOT / 'data/2022-23/universe_30_15m.pkl', "2022-01-01", "2022-12-31", V29_30_UNIVERSE),
    ("2023_FULL", ROOT / 'data/2023-24/universe_30_15m.pkl', "2023-01-02", "2023-12-29", V29_30_UNIVERSE),
    ("2024_H1", ROOT / 'data/2024-25/universe_14_15m.pkl', "2024-01-01", "2024-06-30", V29_30_UNIVERSE),
    ("2024_H2", ROOT / 'data/2024-25/universe_14_15m.pkl', "2024-07-01", "2024-12-31", V29_30_UNIVERSE),
    ("2025_H1", ROOT / 'data/2025-26/universe_14_15m.pkl', "2025-01-01", "2025-07-31", V29_30_UNIVERSE),
    ("2025_LAST5M", ROOT / 'data/2025-26/universe_14_15m.pkl', "2025-08-01", "2025-12-31", V29_30_UNIVERSE),
    ("2026_BULL", 'COMPOSITE_1', "2026-01-01", "2026-12-31", V29_30_UNIVERSE),
]

def load_pool_by_spec(spec):
    if spec == 'COMPOSITE_1':
        u1 = pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))
        u2 = pickle.load(open(ROOT / 'data/unseen_high_beta_10_15m_1y.pkl', 'rb'))
        u3 = pickle.load(open(ROOT / 'data/unseen_10_15m_1y.pkl', 'rb'))
        return {**u1, **u2, **u3}
    else:
        return pickle.load(open(spec, 'rb'))

print("=" * 95)
print("COMPARING V41 BASELINE vs V42 (PROTO4: BE @ 0.45 ATR + 2-STRIKE HALT + RVOL>=1.30)")
print("=" * 95)

results = []
for label, pool_spec, st, ed, u_filt in folds_def:
    pool = load_pool_by_spec(pool_spec)
    
    # Run V41 Baseline
    res_v41 = simulate_engine(pool, st, ed, u_filt, be_mfe_atr=0.0, be_lock_offset=0.0, daily_loss_limit=0, rvol_thresh=1.20)
    
    # Run V42 Proto4
    res_v42 = simulate_engine(pool, st, ed, u_filt, be_mfe_atr=0.45, be_lock_offset=0.05, daily_loss_limit=2, rvol_thresh=1.30)
    
    diff_net = res_v42["net_pnl"] - res_v41["net_pnl"]
    diff_dd = res_v42["max_dd"] - res_v41["max_dd"]
    
    print(f"{label:12s} | V41: Net Rs {res_v41['net_pnl']:+8.2f} (DD {res_v41['max_dd']:4.1f}%, Tr {res_v41['trades']:2d}) -> V42: Net Rs {res_v42['net_pnl']:+8.2f} (DD {res_v42['max_dd']:4.1f}%, Tr {res_v42['trades']:2d}) | Diff Net: {diff_net:+8.2f} | Diff DD: {diff_dd:+5.1f}%")
    results.append((label, res_v41, res_v42))

total_v41 = sum(r[1]["net_pnl"] for r in results)
total_v42 = sum(r[2]["net_pnl"] for r in results)
print("=" * 95)
print(f"TOTAL MULTI-ERA NET PnL: V41 = Rs {total_v41:+8.2f} | V42 = Rs {total_v42:+8.2f} | NET ENHANCEMENT = Rs {total_v42 - total_v41:+8.2f}")
print("=" * 95)
