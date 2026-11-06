import pickle, importlib.util, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
spec = importlib.util.spec_from_file_location('mod_v34_3', str(ROOT / 'strategies/V34.3/run_v34_3.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

b1 = pickle.load(open(ROOT / 'data/unseen_10_15m_1y.pkl', 'rb'))
b3 = pickle.load(open(ROOT / 'data/unseen_batch3_10_15m_1y.pkl', 'rb'))
combined = {**b1, **b3}
syms = set(k.replace('.NS', '') for k in combined.keys())
first_df = list(combined.values())[0]
s = first_df.index.min().strftime('%Y-%m-%d')
e = first_df.index.max().strftime('%Y-%m-%d')

cap, trades = mod.run_simulation(combined, s, e, 'V34_3 VIRGIN 20', universe_filter=syms)
pd.DataFrame(trades).to_csv(Path(__file__).parent / 'trades_intraday.csv', index=False)
