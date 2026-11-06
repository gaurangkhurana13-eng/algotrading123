
import sys, pickle
from pathlib import Path
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))
from strategies.V42_3.run_v42_3 import run_simulation_v42_3

stocks_30_legacy = ('CANBK', 'PNB', 'BANKBARODA', 'SBIN', 'IOC', 'BPCL', 'GAIL', 'COALINDIA', 'TATASTEEL', 'TATAPOWER', 'TATAMOTORS', 'VEDL', 'HINDALCO', 'JINDALSTEL', 'JSWSTEEL', 'BEL', 'ADANIPOWER', 'ADANIENT', 'RECLTD', 'PFC', 'DLF', 'MOTHERSON', 'ITC', 'ICICIBANK', 'AXISBANK', 'RELIANCE', 'INFY', 'SUNPHARMA', 'TECHM', 'CIPLA')

with open(Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\data\2016_H1_UNSEEN") / "universe_30_15m.pkl", "rb") as f:
    pool = pickle.load(f)

cap, trades, wr, dd, pf = run_simulation_v42_3(
    pool, "2016-01-01", "2016-06-30", "2016_H1", universe_filter=stocks_30_legacy, base_risk=25.0
)
