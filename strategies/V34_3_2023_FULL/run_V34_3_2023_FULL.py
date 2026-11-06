import pickle, importlib.util, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
spec = importlib.util.spec_from_file_location('mod_v34_3', str(ROOT / 'strategies/V34.3/run_v34_3.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

p2023 = pickle.load(open(ROOT / 'data/2023-24/universe_30_15m.pkl', 'rb'))
UNIV = {
    'DELTACORP', 'ELECTCAST', 'GSFC', 'IDEA', 'IFCI', 'IOB', 'IRCON', 'J&KBANK',
    'JYOTHYLAB', 'MANALIPETC', 'MTNL', 'NBCC', 'NCC', 'NIACL', 'NMDC', 'PCJEWELLER',
    'PNB', 'RBLBANK', 'RCF', 'RENUKA', 'RITES', 'RPOWER', 'SUZLON', 'SWSOLAR',
    'TATASTEEL', 'TEXRAIL', 'TTML', 'UTKARSHBNK', 'ZEEL', 'CENTRALBK'
}
cap, trades = mod.run_simulation(p2023, '2023-01-02', '2023-12-29', 'V34_3 2023 FULL', universe_filter=UNIV)
pd.DataFrame(trades).to_csv(Path(__file__).parent / 'trades_intraday.csv', index=False)
