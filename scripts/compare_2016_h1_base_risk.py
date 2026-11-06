import sys, pickle
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

from scripts.test_base_risk_sensitivity import simulate_v42_risk_tuning

stocks_30_2016 = {
    'CANBK', 'PNB', 'BANKBARODA', 'SBIN', 'IOC',
    'BPCL', 'GAIL', 'COALINDIA', 'TATASTEEL', 'TATAPOWER',
    'TATAMOTORS', 'VEDL', 'HINDALCO', 'JINDALSTEL', 'JSWSTEEL',
    'BEL', 'ADANIPOWER', 'ADANIENT', 'RECLTD', 'PFC',
    'DLF', 'MOTHERSON', 'ITC', 'ICICIBANK', 'AXISBANK',
    'RELIANCE', 'INFY', 'SUNPHARMA', 'TECHM', 'CIPLA'
}

DATA_PKL = ROOT / "data/2016_H1_UNSEEN/universe_30_15m.pkl"
with open(DATA_PKL, 'rb') as f:
    universe = pickle.load(f)

cap14, net14, wr14, dd14, tr14 = simulate_v42_risk_tuning(universe, '2016-01-01', '2016-06-30', stocks_30_2016, base_risk=14.0)
cap25, net25, wr25, dd25, tr25 = simulate_v42_risk_tuning(universe, '2016-01-01', '2016-06-30', stocks_30_2016, base_risk=25.0)

print(f"Base 14: Trades={tr14}, WR={wr14}%, Net=Rs {net14:.2f}, Cap=Rs {cap14:.2f}, DD={dd14}%")
print(f"Base 25: Trades={tr25}, WR={wr25}%, Net=Rs {net25:.2f}, Cap=Rs {cap25:.2f}, DD={dd25}%")
