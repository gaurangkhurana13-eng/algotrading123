import pickle, importlib.util, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
spec = importlib.util.spec_from_file_location('mod_V37_2018', str(ROOT / 'strategies/V37/run_v37.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DATA_PKL = ROOT / 'data/2018_UNSEEN/universe_30_15m.pkl'
with open(DATA_PKL, 'rb') as f:
    universe = pickle.load(f)

stocks_30_2018 = {
    'CANBK', 'PNB', 'BANKBARODA', 'SBIN', 'IOC',
    'BPCL', 'GAIL', 'COALINDIA', 'TATASTEEL', 'TATAPOWER',
    'TATAMOTORS', 'VEDL', 'HINDALCO', 'JINDALSTEL', 'JSWSTEEL',
    'BEL', 'ADANIPOWER', 'ADANIENT', 'RECLTD', 'PFC',
    'DLF', 'MOTHERSON', 'ITC', 'ICICIBANK', 'AXISBANK',
    'RELIANCE', 'INFY', 'SUNPHARMA', 'TECHM', 'CIPLA'
}

cap, trades = mod.run_simulation(universe, '2018-01-01', '2018-12-31', 'V37_2018', universe_filter=stocks_30_2018)
cols = [
    'symbol', 'direction', 'entry_time', 'exit_time', 'entry_price',
    'exit_price', 'shares', 'sl', 'gross_pnl', 'charges', 'net_pnl', 'exit_reason'
]
df = pd.DataFrame(trades)
if not df.empty:
    df = df[cols]
else:
    df = pd.DataFrame(columns=cols)
df.to_csv(Path(__file__).parent / 'trades_intraday.csv', index=False)
