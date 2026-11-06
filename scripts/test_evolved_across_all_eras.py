import sys, pickle
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

from scripts.test_v42_evolution_hypotheses import simulate_v42_evolved

stocks_30_legacy = {
    'CANBK', 'PNB', 'BANKBARODA', 'SBIN', 'IOC',
    'BPCL', 'GAIL', 'COALINDIA', 'TATASTEEL', 'TATAPOWER',
    'TATAMOTORS', 'VEDL', 'HINDALCO', 'JINDALSTEL', 'JSWSTEEL',
    'BEL', 'ADANIPOWER', 'ADANIENT', 'RECLTD', 'PFC',
    'DLF', 'MOTHERSON', 'ITC', 'ICICIBANK', 'AXISBANK',
    'RELIANCE', 'INFY', 'SUNPHARMA', 'TECHM', 'CIPLA'
}
V29_30_UNIVERSE = {
    "J&KBANK", "UTKARSHBNK", "MANALIPETC", "IDEA", "PCJEWELLER",
    "NIACL", "MTNL", "IFCI", "TEXRAIL", "SWSOLAR",
    "TTML", "RENUKA", "RCF", "NBCC", "JYOTHYLAB",
    "RBLBANK", "ELECTCAST", "PNB", "GSFC", "TATASTEEL",
    "SUZLON", "CENTRALBK", "IOB", "IRCON", "NMDC",
    "RITES", "NCC", "ZEEL", "DELTACORP", "RPOWER"
}

p_2016_h1 = pickle.load(open(ROOT / 'data/2016_H1_UNSEEN/universe_30_15m.pkl', 'rb'))
p_2016_h2 = pickle.load(open(ROOT / 'data/2016_H2_UNSEEN/universe_30_15m.pkl', 'rb'))
p_2017    = pickle.load(open(ROOT / 'data/2017_UNSEEN/universe_30_15m.pkl', 'rb'))
p_2018    = pickle.load(open(ROOT / 'data/2018_UNSEEN/universe_30_15m.pkl', 'rb'))
p_2019    = pickle.load(open(ROOT / 'data/2019_UNSEEN/universe_30_15m.pkl', 'rb'))
p_2020    = pickle.load(open(ROOT / 'data/2020_FULL/universe_30_15m.pkl', 'rb'))
p_2021    = pickle.load(open(ROOT / 'data/2021-22/universe_30_15m.pkl', 'rb'))
p_2022    = pickle.load(open(ROOT / 'data/2022-23/universe_30_15m.pkl', 'rb'))
p_2023    = pickle.load(open(ROOT / 'data/2023-24/universe_30_15m.pkl', 'rb'))

p_2026 = pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))
p_hb = pickle.load(open(ROOT / 'data/unseen_high_beta_10_15m_1y.pkl', 'rb'))
p_u10 = pickle.load(open(ROOT / 'data/unseen_10_15m_1y.pkl', 'rb'))
pool_2026 = {**p_2026, **p_hb, **p_u10}

core_27 = {
    'CANBK', 'DELTACORP', 'FEDERALBNK', 'GMRINFRA', 'GSFC', 'IDFCFIRSTB', 'IOC',
    'JYOTHYLAB', 'MANAPPURAM', 'MOTHERSON', 'NBCC', 'NCC', 'NATIONALUM', 'NMDC',
    'POONAWALLA', 'PNB', 'RBLBANK', 'RCF', 'SAIL', 'SUZLON', 'TATACHEM',
    'TATASTEEL', 'TATAPOWER', 'TRIDENT', 'UNIONBANK', 'YESBANK', 'ZEEL'
}
u_2021 = {'CENTRALBK', 'DELTACORP', 'GSFC', 'IDEA', 'IOB', 'JYOTHYLAB', 'NBCC', 'NCC', 'NIACL', 'NMDC', 'PNB', 'RBLBANK', 'RCF', 'RENUKA', 'RITES', 'SUZLON', 'SWSOLAR', 'TATASTEEL', 'TTML', 'ZEEL'}

eras = [
    ("2016_H1", p_2016_h1, "2016-01-01", "2016-06-30", stocks_30_legacy),
    ("2016_H2", p_2016_h2, "2016-07-01", "2016-12-31", stocks_30_legacy),
    ("2017",    p_2017,    "2017-01-01", "2017-12-31", stocks_30_legacy),
    ("2018",    p_2018,    "2018-01-01", "2018-12-31", stocks_30_legacy),
    ("2019",    p_2019,    "2019-01-01", "2019-12-31", None),
    ("2020",    p_2020,    "2020-01-01", "2020-12-31", core_27),
    ("2021",    p_2021,    "2021-01-01", "2021-12-31", u_2021),
    ("2022",    p_2022,    "2022-01-01", "2022-12-31", V29_30_UNIVERSE),
    ("2023",    p_2023,    "2023-01-02", "2023-12-29", V29_30_UNIVERSE),
    ("2026",    pool_2026, "2026-01-01", "2026-12-31", V29_30_UNIVERSE),
]

print("="*115)
print(f"{'Era':10s} | {'V42 Baseline Net (DD)':25s} | {'V42 Evolved Base 14 (DD)':27s} | {'V42 Evolved Base 25 (DD)':27s}")
print("="*115)

tot_base = 0.0
tot_ev14 = 0.0
tot_ev25 = 0.0

for name, pool, s_dt, e_dt, u_filt in eras:
    # 1. Baseline V42
    _, net_b, wr_b, dd_b, tr_b = simulate_v42_evolved(pool, s_dt, e_dt, u_filt, base_risk=14.0, adaptive_be=False, smart_strike=False)
    # 2. Evolved Base 14
    _, net_14, wr_14, dd_14, tr_14 = simulate_v42_evolved(pool, s_dt, e_dt, u_filt, base_risk=14.0, adaptive_be=True, smart_strike=True)
    # 3. Evolved Base 25
    _, net_25, wr_25, dd_25, tr_25 = simulate_v42_evolved(pool, s_dt, e_dt, u_filt, base_risk=25.0, adaptive_be=True, smart_strike=True)

    tot_base += net_b
    tot_ev14 += net_14
    tot_ev25 += net_25

    str_b = f"Rs {net_b:+8.2f} (DD {dd_b:4.1f}%, {tr_b:2d}t)"
    str_14 = f"Rs {net_14:+8.2f} (DD {dd_14:4.1f}%, {tr_14:2d}t)"
    str_25 = f"Rs {net_25:+8.2f} (DD {dd_25:4.1f}%, {tr_25:2d}t)"

    print(f"{name:10s} | {str_b:25s} | {str_14:27s} | {str_25:27s}")

print("="*115)
print(f"{'TOTAL':10s} | Rs {tot_base:+8.2f}               | Rs {tot_ev14:+8.2f}                 | Rs {tot_ev25:+8.2f}")
print("="*115)
