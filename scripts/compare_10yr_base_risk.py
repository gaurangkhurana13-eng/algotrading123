import sys, pickle
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

from scripts.test_base_risk_sensitivity import simulate_v42_risk_tuning

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

p_2016 = pickle.load(open(ROOT / 'data/2016_H2_UNSEEN/universe_30_15m.pkl', 'rb'))
p_2017 = pickle.load(open(ROOT / 'data/2017_UNSEEN/universe_30_15m.pkl', 'rb'))
p_2018 = pickle.load(open(ROOT / 'data/2018_UNSEEN/universe_30_15m.pkl', 'rb'))
p_2019 = pickle.load(open(ROOT / 'data/2019_UNSEEN/universe_30_15m.pkl', 'rb'))
p_2020 = pickle.load(open(ROOT / 'data/2020_FULL/universe_30_15m.pkl', 'rb'))
p_2021 = pickle.load(open(ROOT / 'data/2021-22/universe_30_15m.pkl', 'rb'))
p_2022 = pickle.load(open(ROOT / 'data/2022-23/universe_30_15m.pkl', 'rb'))
p_2023 = pickle.load(open(ROOT / 'data/2023-24/universe_30_15m.pkl', 'rb'))
p_2024 = pickle.load(open(ROOT / 'data/2024-25/universe_14_15m.pkl', 'rb'))
p_2025 = pickle.load(open(ROOT / 'data/2025-26/universe_14_15m.pkl', 'rb'))
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

timeline = [
    ("2016_H2", p_2016, "2016-07-01", "2016-12-31", stocks_30_legacy),
    ("2017", p_2017, "2017-01-01", "2017-12-31", stocks_30_legacy),
    ("2018", p_2018, "2018-01-01", "2018-12-31", stocks_30_legacy),
    ("2019", p_2019, "2019-01-01", "2019-12-31", None),
    ("2020", p_2020, "2020-01-01", "2020-12-31", core_27),
    ("2021", p_2021, "2021-01-01", "2021-12-31", u_2021),
    ("2022", p_2022, "2022-01-01", "2022-12-31", V29_30_UNIVERSE),
    ("2023", p_2023, "2023-01-02", "2023-12-29", V29_30_UNIVERSE),
    ("2024", p_2024, "2024-01-01", "2024-12-31", V29_30_UNIVERSE),
    ("2025", p_2025, "2025-01-01", "2025-12-31", V29_30_UNIVERSE),
    ("2026", pool_2026, "2026-01-01", "2026-12-31", V29_30_UNIVERSE),
]

print("=" * 95)
print("10-YEAR ANNUAL RESULTS WITH BASE RISK = ₹25.0 (5% RISK ON ₹500 CAPITAL)")
print("=" * 95)

total_net_14 = 0.0
total_net_25 = 0.0

for era_name, pool, s_dt, e_dt, u_filt in timeline:
    _, net14, wr14, dd14, tr14 = simulate_v42_risk_tuning(pool, s_dt, e_dt, u_filt, base_risk=14.0)
    _, net25, wr25, dd25, tr25 = simulate_v42_risk_tuning(pool, s_dt, e_dt, u_filt, base_risk=25.0)
    total_net_14 += net14
    total_net_25 += net25
    diff = net25 - net14
    print(f"{era_name:10s} | Base ₹14: Net ₹{net14:+8.2f} (DD {dd14:4.1f}%) -> Base ₹25: Net ₹{net25:+8.2f} (DD {dd25:4.1f}%) | Diff: ₹{diff:+8.2f}")

print("=" * 95)
print(f"TOTAL 10-YEAR NET PnL: Base ₹14 = ₹{total_net_14:+8.2f} | Base ₹25 = ₹{total_net_25:+8.2f} | EXTRA PROFIT = ₹{total_net_25 - total_net_14:+8.2f}")
print("=" * 95)
