import sys, pickle
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

import importlib.util

spec = importlib.util.spec_from_file_location("mod_v42", str(ROOT / "strategies/V42/run_v42.py"))
mod_v42 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod_v42)

DATA_PKL = ROOT / "data/2016_H1_UNSEEN/universe_30_15m.pkl"
with open(DATA_PKL, 'rb') as f:
    universe = pickle.load(f)

stocks_30_2016 = {
    'CANBK', 'PNB', 'BANKBARODA', 'SBIN', 'IOC',
    'BPCL', 'GAIL', 'COALINDIA', 'TATASTEEL', 'TATAPOWER',
    'TATAMOTORS', 'VEDL', 'HINDALCO', 'JINDALSTEL', 'JSWSTEEL',
    'BEL', 'ADANIPOWER', 'ADANIENT', 'RECLTD', 'PFC',
    'DLF', 'MOTHERSON', 'ITC', 'ICICIBANK', 'AXISBANK',
    'RELIANCE', 'INFY', 'SUNPHARMA', 'TECHM', 'CIPLA'
}

def eval_trades(trades, initial_cap=500.0):
    df_t = pd.DataFrame(trades)
    n = len(df_t)
    if n > 0:
        win_trades = df_t[df_t['net_pnl'] > 0]
        loss_trades = df_t[df_t['net_pnl'] < 0]
        wr = (len(win_trades) / n) * 100.0
        net = df_t['net_pnl'].sum()
        gw = win_trades['net_pnl'].sum()
        gl = abs(loss_trades['net_pnl'].sum())
        pf = (gw / gl) if gl > 0 else 999.0
        avg_w = win_trades['net_pnl'].mean() if len(win_trades) > 0 else 0.0
        avg_l = abs(loss_trades['net_pnl'].mean()) if len(loss_trades) > 0 else 0.0
        gross = df_t['gross_pnl'].sum()
        charges = df_t['charges'].sum()
        
        eq = [initial_cap]
        cur = initial_cap
        for p in df_t['net_pnl']:
            cur += p
            eq.append(cur)
        pk = eq[0]
        mdd = 0.0
        for e in eq:
            if e > pk: pk = e
            dd = (pk - e) / pk * 100.0
            if dd > mdd: mdd = dd
    else:
        wr = net = pf = mdd = avg_w = avg_l = gross = charges = 0.0
    return {
        'trades': n, 'wr': round(wr, 1), 'gross': round(gross, 2),
        'charges': round(charges, 2), 'net': round(net, 2), 'ending_cap': round(500 + net, 2),
        'pf': round(pf, 2), 'max_dd': round(mdd, 1), 'avg_win': round(avg_w, 2), 'avg_loss': round(avg_l, 2)
    }

# 1. Official V42 Standard (base_risk = 14.0)
cap_std, trades_std = mod_v42.run_simulation(universe, '2016-01-01', '2016-06-30', 'V42_Standard', universe_filter=stocks_30_2016)
m_std = eval_trades(trades_std)

# 2. V42 Turbo simulation (running the exact same code with base_risk = 25.0)
# Let's inspect how base_risk changes the trade sizing in run_simulation
v42_code = open(ROOT / "strategies/V42/run_v42.py", "r", encoding="utf-8").read()
v42_turbo_code = v42_code.replace("raw_r = 14.0 +", "raw_r = 25.0 +")

# Execute turbo dynamically
scope = {}
exec(v42_turbo_code, scope)
cap_turbo, trades_turbo = scope['run_simulation'](universe, '2016-01-01', '2016-06-30', 'V42_Turbo', universe_filter=stocks_30_2016)
m_turbo = eval_trades(trades_turbo)

print("="*100)
print(f"{'Metric':20s} | {'V42 Standard (Base 14.0)':25s} | {'V42 Turbo (Base 25.0)':25s}")
print("="*100)
for k in m_std.keys():
    print(f"{k:20s} | {str(m_std[k]):25s} | {str(m_turbo[k]):25s}")
print("="*100)
