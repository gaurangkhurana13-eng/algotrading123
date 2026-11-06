import pickle, pandas as pd, sys
from pathlib import Path

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

from scripts.test_v43_synthesis import simulate_v43_era
from scripts.multi_slot_sim_engine import V29_30_UNIVERSE

pool = pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))
res = simulate_v43_era(pool, "2026-01-01", "2026-10-29", universe_filter=V29_30_UNIVERSE, initial_equity=500.0, include_ma20_reclaim=True)
cols = ['symbol', 'direction', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'shares', 'sl', 'gross_pnl', 'charges', 'net_pnl', 'exit_reason']
df = pd.DataFrame(res['trades'])[cols]
df.to_csv(Path(__file__).parent / 'trades_intraday.csv', index=False)
