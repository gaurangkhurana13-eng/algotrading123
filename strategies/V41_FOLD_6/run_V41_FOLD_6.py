import pickle, importlib.util, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
spec = importlib.util.spec_from_file_location('v41', str(ROOT / 'strategies/V41/run_v41.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# load pool
pool_spec = r'C:/Users/admin/.gemini/antigravity/scratch/new_trade/data/2023-24/universe_30_15m.pkl'
if pool_spec == 'COMPOSITE_1':
    u1 = pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))
    u2 = pickle.load(open(ROOT / 'data/unseen_high_beta_10_15m_1y.pkl', 'rb'))
    u3 = pickle.load(open(ROOT / 'data/unseen_10_15m_1y.pkl', 'rb'))
    pool = {**u1, **u2, **u3}
elif pool_spec == 'COMPOSITE_12':
    b1 = pickle.load(open(ROOT / 'data/unseen_10_15m_1y.pkl', 'rb'))
    b3 = pickle.load(open(ROOT / 'data/unseen_batch3_10_15m_1y.pkl', 'rb'))
    pool = {**b1, **b3}
elif pool_spec == 'COMPOSITE_13':
    pool = pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))
else:
    pool = pickle.load(open(pool_spec, 'rb'))

filt = {'CENTRALBK', 'RBLBANK', 'IDEA', 'RITES', 'IOB', 'SWSOLAR', 'MTNL', 'IRCON', 'SUZLON', 'ELECTCAST', 'RENUKA', 'NCC', 'TATASTEEL', 'PNB', 'IFCI', 'RCF', 'NMDC', 'PCJEWELLER', 'GSFC', 'NIACL', 'UTKARSHBNK', 'J&KBANK', 'DELTACORP', 'RPOWER', 'MANALIPETC', 'JYOTHYLAB', 'ZEEL', 'NBCC', 'TEXRAIL', 'TTML'}
cap, trades = mod.run_simulation(pool, '2023-09-01', '2023-12-31', 'V41_FOLD_6', universe_filter=filt)
cols = ['symbol', 'direction', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'shares', 'sl', 'gross_pnl', 'charges', 'net_pnl', 'exit_reason']
df = pd.DataFrame(trades)
if not df.empty:
    df = df[cols]
else:
    df = pd.DataFrame(columns=cols)
df.to_csv(Path(__file__).parent / 'trades_intraday.csv', index=False)
