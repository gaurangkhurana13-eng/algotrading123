import sys, pickle
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

from scripts.test_v42_prototype import simulate_engine

# Let's inspect the 2026 fold 1 pool
p_2026 = pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))
p_highbeta = pickle.load(open(ROOT / 'data/unseen_high_beta_10_15m_1y.pkl', 'rb'))
p_unseen10 = pickle.load(open(ROOT / 'data/unseen_10_15m_1y.pkl', 'rb'))
pool_2026 = {**p_2026, **p_highbeta, **p_unseen10}

V29_30_UNIVERSE = {
    "J&KBANK", "UTKARSHBNK", "MANALIPETC", "IDEA", "PCJEWELLER",
    "NIACL", "MTNL", "IFCI", "TEXRAIL", "SWSOLAR",
    "TTML", "RENUKA", "RCF", "NBCC", "JYOTHYLAB",
    "RBLBANK", "ELECTCAST", "PNB", "GSFC", "TATASTEEL",
    "SUZLON", "CENTRALBK", "IOB", "IRCON", "NMDC",
    "RITES", "NCC", "ZEEL", "DELTACORP", "RPOWER"
}

# Baseline V42: be_mfe_atr=0.45, be_lock_offset=0.05, daily_loss_limit=2, rvol_thresh=1.30
res_baseline = simulate_engine(pool_2026, "2026-01-01", "2026-12-31", V29_30_UNIVERSE, be_mfe_atr=0.45, be_lock_offset=0.05, daily_loss_limit=2, rvol_thresh=1.30)
print(f"2026 Baseline V42 (BE@0.45): Net=₹{res_baseline['net_pnl']:+8.2f}, Tr={res_baseline['trades']}, WR={res_baseline['win_rate']}%, DD={res_baseline['max_dd']}%")

# If BE is at 0.60 ATR instead of 0.45:
res_be60 = simulate_engine(pool_2026, "2026-01-01", "2026-12-31", V29_30_UNIVERSE, be_mfe_atr=0.60, be_lock_offset=0.05, daily_loss_limit=2, rvol_thresh=1.30)
print(f"2026 With BE@0.60:           Net=₹{res_be60['net_pnl']:+8.2f}, Tr={res_be60['trades']}, WR={res_be60['win_rate']}%, DD={res_be60['max_dd']}%")

# If BE is at 0.70 ATR:
res_be70 = simulate_engine(pool_2026, "2026-01-01", "2026-12-31", V29_30_UNIVERSE, be_mfe_atr=0.70, be_lock_offset=0.05, daily_loss_limit=2, rvol_thresh=1.30)
print(f"2026 With BE@0.70:           Net=₹{res_be70['net_pnl']:+8.2f}, Tr={res_be70['trades']}, WR={res_be70['win_rate']}%, DD={res_be70['max_dd']}%")

# But let's check what BE@0.60 or BE@0.70 does to 2017 & 2018!
p_2017 = pickle.load(open(ROOT / 'data/2017_UNSEEN/universe_30_15m.pkl', 'rb'))
p_2018 = pickle.load(open(ROOT / 'data/2018_UNSEEN/universe_30_15m.pkl', 'rb'))
stocks_30_legacy = {
    'CANBK', 'PNB', 'BANKBARODA', 'SBIN', 'IOC',
    'BPCL', 'GAIL', 'COALINDIA', 'TATASTEEL', 'TATAPOWER',
    'TATAMOTORS', 'VEDL', 'HINDALCO', 'JINDALSTEL', 'JSWSTEEL',
    'BEL', 'ADANIPOWER', 'ADANIENT', 'RECLTD', 'PFC',
    'DLF', 'MOTHERSON', 'ITC', 'ICICIBANK', 'AXISBANK',
    'RELIANCE', 'INFY', 'SUNPHARMA', 'TECHM', 'CIPLA'
}
res_17_base = simulate_engine(p_2017, "2017-01-01", "2017-12-31", stocks_30_legacy, be_mfe_atr=0.45, be_lock_offset=0.05, daily_loss_limit=2, rvol_thresh=1.30)
res_17_be60 = simulate_engine(p_2017, "2017-01-01", "2017-12-31", stocks_30_legacy, be_mfe_atr=0.60, be_lock_offset=0.05, daily_loss_limit=2, rvol_thresh=1.30)
res_18_base = simulate_engine(p_2018, "2018-01-01", "2018-12-31", stocks_30_legacy, be_mfe_atr=0.45, be_lock_offset=0.05, daily_loss_limit=2, rvol_thresh=1.30)
res_18_be60 = simulate_engine(p_2018, "2018-01-01", "2018-12-31", stocks_30_legacy, be_mfe_atr=0.60, be_lock_offset=0.05, daily_loss_limit=2, rvol_thresh=1.30)

print(f"\n2017 Base (BE@0.45): Net=₹{res_17_base['net_pnl']:+8.2f}, DD={res_17_base['max_dd']}% | BE@0.60: Net=₹{res_17_be60['net_pnl']:+8.2f}, DD={res_17_be60['max_dd']}%")
print(f"2018 Base (BE@0.45): Net=₹{res_18_base['net_pnl']:+8.2f}, DD={res_18_base['max_dd']}% | BE@0.60: Net=₹{res_18_be60['net_pnl']:+8.2f}, DD={res_18_be60['max_dd']}%")
