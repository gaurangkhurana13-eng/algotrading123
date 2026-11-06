import pickle, importlib.util, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
spec = importlib.util.spec_from_file_location('mod_v34_3', str(ROOT / 'strategies/V34.3/run_v34_3.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

p2021 = pickle.load(open(ROOT / 'data/2021-22/universe_30_15m.pkl', 'rb'))
UNIV = {
    'CENTRALBK', 'DELTACORP', 'GSFC', 'IDEA', 'IOB', 'JYOTHYLAB', 'NBCC', 'NCC',
    'NIACL', 'NMDC', 'PNB', 'RBLBANK', 'RCF', 'RENUKA', 'RITES', 'SUZLON',
    'SWSOLAR', 'TATASTEEL', 'TTML', 'ZEEL'
}
cap, trades = mod.run_simulation(p2021, '2021-01-01', '2021-12-31', 'V34_3 2021 FULL', universe_filter=UNIV)
pd.DataFrame(trades).to_csv(Path(__file__).parent / 'trades_intraday.csv', index=False)
