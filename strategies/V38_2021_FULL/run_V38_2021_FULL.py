import pickle, importlib.util, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
spec = importlib.util.spec_from_file_location('mod_v38_2021_full', str(ROOT / 'strategies/V38/run_v38.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

p2021 = pickle.load(open(ROOT / 'data/2021-22/universe_30_15m.pkl', 'rb'))
UNIV = {
    'CENTRALBK', 'DELTACORP', 'GSFC', 'IDEA', 'IOB', 'JYOTHYLAB', 'NBCC', 'NCC',
    'NIACL', 'NMDC', 'PNB', 'RBLBANK', 'RCF', 'RENUKA', 'RITES', 'SUZLON',
    'SWSOLAR', 'TATASTEEL', 'TTML', 'ZEEL'
}
cap, trades = mod.run_simulation(p2021, '2021-01-01', '2021-12-31', 'V38_2021_FULL', universe_filter=UNIV)
cols = [
    'symbol', 'direction', 'entry_time', 'exit_time', 'entry_price',
    'exit_price', 'shares', 'sl', 'gross_pnl', 'charges', 'net_pnl', 'exit_reason'
]
pd.DataFrame(trades)[cols].to_csv(Path(__file__).parent / 'trades_intraday.csv', index=False)
