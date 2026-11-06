import pickle, importlib.util, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
spec = importlib.util.spec_from_file_location('mod_V39_2020_FIRST4M', str(ROOT / 'strategies/V39/run_v39.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

DATA_PKL = ROOT / 'data/2020_FIRST4M/universe_30_15m.pkl'
with open(DATA_PKL, 'rb') as f:
    universe = pickle.load(f)

stocks_27 = {
    'CANBK', 'DELTACORP', 'FEDERALBNK', 'GMRINFRA', 'GSFC', 'IDFCFIRSTB', 'IOC',
    'JYOTHYLAB', 'MANAPPURAM', 'MOTHERSON', 'NBCC', 'NCC', 'NATIONALUM', 'NMDC',
    'POONAWALLA', 'PNB', 'RBLBANK', 'RCF', 'SAIL', 'SUZLON', 'TATACHEM',
    'TATASTEEL', 'TATAPOWER', 'TRIDENT', 'UNIONBANK', 'YESBANK', 'ZEEL'
}

cap, trades = mod.run_simulation(universe, '2020-01-01', '2020-04-30', 'V39_2020_FIRST4M', universe_filter=stocks_27)
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
