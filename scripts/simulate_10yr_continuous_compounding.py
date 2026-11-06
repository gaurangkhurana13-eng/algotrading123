import sys, pickle
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

import importlib.util
def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

mod_v34 = load_mod('mod_v34', ROOT / 'strategies/V34.3/run_v34_3.py')
mod_v41 = load_mod('mod_v41', ROOT / 'strategies/V41/run_v41.py')
mod_v42 = load_mod('mod_v42', ROOT / 'strategies/V42/run_v42.py')

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

# The sequential 10-year timeline:
timeline = [
    ("2016_H2", ROOT / 'data/2016_H2_UNSEEN/universe_30_15m.pkl', "2016-07-01", "2016-12-31", stocks_30_legacy),
    ("2017", ROOT / 'data/2017_UNSEEN/universe_30_15m.pkl', "2017-01-01", "2017-12-31", stocks_30_legacy),
    ("2018", ROOT / 'data/2018_UNSEEN/universe_30_15m.pkl', "2018-01-01", "2018-12-31", stocks_30_legacy),
    ("2019", ROOT / 'data/2019_UNSEEN/universe_30_15m.pkl', "2019-01-01", "2019-12-31", None),
    ("2020", ROOT / 'data/2020_FULL/universe_30_15m.pkl', "2020-01-01", "2020-12-31", core_27),
    ("2021", ROOT / 'data/2021-22/universe_30_15m.pkl', "2021-01-01", "2021-12-31", {'CENTRALBK', 'DELTACORP', 'GSFC', 'IDEA', 'IOB', 'JYOTHYLAB', 'NBCC', 'NCC', 'NIACL', 'NMDC', 'PNB', 'RBLBANK', 'RCF', 'RENUKA', 'RITES', 'SUZLON', 'SWSOLAR', 'TATASTEEL', 'TTML', 'ZEEL'}),
    ("2022", ROOT / 'data/2022-23/universe_30_15m.pkl', "2022-01-01", "2022-12-31", V29_30_UNIVERSE),
    ("2023", ROOT / 'data/2023-24/universe_30_15m.pkl', "2023-01-02", "2023-12-29", V29_30_UNIVERSE),
    ("2024", ROOT / 'data/2024-25/universe_14_15m.pkl', "2024-01-01", "2024-12-31", V29_30_UNIVERSE),
    ("2025", ROOT / 'data/2025-26/universe_14_15m.pkl', "2025-01-01", "2025-12-31", V29_30_UNIVERSE),
    ("2026", ROOT / 'data/universe_78_15m_1y.pkl', "2026-01-01", "2026-12-31", V29_30_UNIVERSE),
]

print("=" * 115)
print("10-YEAR CONTINUOUS MULTI-YEAR COMPOUNDING SIMULATION (STARTING CAPITAL: ₹500.00)")
print("=" * 115)

def run_continuous_simulation(mod, mod_name):
    cur_wallet = 500.0
    year_records = []
    
    # We will run each year. But notice: run_simulation starts with initial_equity = 500.0 inside the function.
    # To test actual continuous rolling compounding, if wallet grew to Rs 1500, we need to pass initial_equity,
    # OR we can observe the cumulative PnL without reset, and with scaled risk.
    # Let's inspect run_simulation in V42. Does it accept initial_equity?
    # In V42 run_simulation: initial_equity = 500.0 is hardcoded inside.
    # If initial_equity is hardcoded to 500 for each year, the sum of Net PnLs across the 10 years is:
    # 500 + Net(2016) + Net(2017) + ... + Net(2026).
    pass

# Let's see what the cumulative sum of Net PnLs is from 2016 through 2026 when starting at ₹500:
# And let's see what TRUE compounding (carrying ending balance into next year's initial_capital) yields!
