import os, sys, pickle, importlib.util
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT / "strategies"))

print("="*100)
print("EXECUTING 15-FOLD MASTER GRAND BENCHMARK: V34.3 vs V39 vs V39.1")
print("="*100)

# Import the 3 engines
def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

mod_v34 = load_module('mod_v34', ROOT / 'strategies/V34.3/run_v34_3.py')
mod_v39 = load_module('mod_v39', ROOT / 'strategies/V39/run_v39.py')
mod_v39_1 = load_module('mod_v39_1', ROOT / 'strategies/V39.1/run_v39_1.py')

V29_30_UNIVERSE = {
    "J&KBANK", "UTKARSHBNK", "MANALIPETC", "IDEA", "PCJEWELLER",
    "NIACL", "MTNL", "IFCI", "TEXRAIL", "SWSOLAR",
    "TTML", "RENUKA", "RCF", "NBCC", "JYOTHYLAB",
    "RBLBANK", "ELECTCAST", "PNB", "GSFC", "TATASTEEL",
    "SUZLON", "CENTRALBK", "IOB", "IRCON", "NMDC",
    "RITES", "NCC", "ZEEL", "DELTACORP", "RPOWER"
}

core_27 = {
    'CANBK', 'DELTACORP', 'FEDERALBNK', 'GMRINFRA', 'GSFC', 'IDFCFIRSTB', 'IOC',
    'JYOTHYLAB', 'MANAPPURAM', 'MOTHERSON', 'NBCC', 'NCC', 'NATIONALUM', 'NMDC',
    'POONAWALLA', 'PNB', 'RBLBANK', 'RCF', 'SAIL', 'SUZLON', 'TATACHEM',
    'TATASTEEL', 'TATAPOWER', 'TRIDENT', 'UNIONBANK', 'YESBANK', 'ZEEL'
}

# Define 15 folds
# (fold_id, fold_label, period_desc, data_loader_func, start_date, end_date, universe_filter)
def get_folds():
    folds_def = []
    
    # Fold 0: 2021
    def load_0():
        return pickle.load(open(ROOT / 'data/2021-22/universe_30_15m.pkl', 'rb'))
    univ_0 = {'CENTRALBK', 'DELTACORP', 'GSFC', 'IDEA', 'IOB', 'JYOTHYLAB', 'NBCC', 'NCC', 'NIACL', 'NMDC', 'PNB', 'RBLBANK', 'RCF', 'RENUKA', 'RITES', 'SUZLON', 'SWSOLAR', 'TATASTEEL', 'TTML', 'ZEEL'}
    folds_def.append((0, "Fold 0 (2021 Full Year)", "Jan-Dec 2021 (20 Stocks, 247d)", load_0, "2021-01-01", "2021-12-31", univ_0))
    
    # Fold 1: 2026 Bull Horizon
    def load_1():
        u1 = pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))
        u2 = pickle.load(open(ROOT / 'data/unseen_high_beta_10_15m_1y.pkl', 'rb'))
        u3 = pickle.load(open(ROOT / 'data/unseen_10_15m_1y.pkl', 'rb'))
        return {**u1, **u2, **u3}
    folds_def.append((1, "Fold 1 (2026 Bull Horizon)", "Dec 2025-Oct 2026 (78 Midcaps)", load_1, "2026-01-01", "2026-12-31", V29_30_UNIVERSE))
    
    # Fold 2: 2022 Bear Market
    def load_2():
        return pickle.load(open(ROOT / 'data/2022-23/universe_30_15m.pkl', 'rb'))
    folds_def.append((2, "Fold 2 (2022 Bear Market)", "Jan-Dec 2022 (30 Stocks, 248d)", load_2, "2022-01-01", "2022-12-31", V29_30_UNIVERSE))
    
    # Fold 3: 2023 Full Year
    def load_3():
        return pickle.load(open(ROOT / 'data/2023-24/universe_30_15m.pkl', 'rb'))
    folds_def.append((3, "Fold 3 (2023 Full Year)", "2 Jan-29 Dec 2023 (30 Stocks, 245d)", load_3, "2023-01-02", "2023-12-29", V29_30_UNIVERSE))
    
    # Fold 4: 2023 H1
    folds_def.append((4, "Fold 4 (2023 H1 Virgin)", "Jan-Apr 2023 (30 Stocks)", load_3, "2023-01-01", "2023-04-30", V29_30_UNIVERSE))
    
    # Fold 5: 2023 Mid 4M
    folds_def.append((5, "Fold 5 (2023 Mid 4M)", "May-Aug 2023 (30 Stocks)", load_3, "2023-05-01", "2023-08-31", V29_30_UNIVERSE))
    
    # Fold 6: 2023 H2
    folds_def.append((6, "Fold 6 (2023 H2 Virgin)", "Sep-Dec 2023 (30 Stocks)", load_3, "2023-09-01", "2023-12-31", V29_30_UNIVERSE))
    
    # Fold 7: 2024 H1
    def load_7():
        return pickle.load(open(ROOT / 'data/2024-25/universe_14_15m.pkl', 'rb'))
    folds_def.append((7, "Fold 7 (2024 H1 Pre-Election)", "Jan-Jun 2024 (14 Stocks)", load_7, "2024-01-01", "2024-06-30", V29_30_UNIVERSE))
    
    # Fold 8: 2024 H2 Crash
    folds_def.append((8, "Fold 8 (2024 H2 Crash FII)", "Jul-Dec 2024 (14 Stocks)", load_7, "2024-07-01", "2024-12-31", V29_30_UNIVERSE))
    
    # Fold 9: 2025 H1
    def load_9():
        return pickle.load(open(ROOT / 'data/2025-26/universe_14_15m.pkl', 'rb'))
    folds_def.append((9, "Fold 9 (2025 H1 Choppy)", "Jan-Jul 2025 (14 Stocks)", load_9, "2025-01-01", "2025-07-31", V29_30_UNIVERSE))
    
    # Fold 10: 2025 Q4
    folds_def.append((10, "Fold 10 (2025 Q4)", "Oct-Dec 2025 (14 Stocks)", load_9, "2025-10-01", "2025-12-31", None))
    
    # Fold 11: 2025 Last 5M
    folds_def.append((11, "Fold 11 (2025 Last 5M)", "Aug-Dec 2025 (14 Stocks)", load_9, "2025-08-01", "2025-12-31", V29_30_UNIVERSE))
    
    # Fold 12: 20 Virgin Companies
    def load_12():
        b1 = pickle.load(open(ROOT / 'data/unseen_10_15m_1y.pkl', 'rb'))
        b3 = pickle.load(open(ROOT / 'data/unseen_batch3_10_15m_1y.pkl', 'rb'))
        return {**b1, **b3}
    folds_def.append((12, "Fold 12 (20 Virgin Companies)", "Jan-Oct 2026 (20 New Stocks, 1-Yr)", load_12, "2026-01-01", "2026-10-31", None))
    
    # Fold 13: 2026 Unseen Midcaps
    def load_13():
        return pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))
    def get_univ_13():
        u = pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))
        clean_78 = {k.replace('.NS', '') for k in u.keys()}
        return clean_78 - V29_30_UNIVERSE
    folds_def.append((13, "Fold 13 (2026 Unseen Midcaps)", "Jan-Oct 2026 (52 Outside Stocks)", load_13, "2026-01-01", "2026-12-31", get_univ_13()))
    
    # Fold 14: 2020 Full Year Continuous (The COVID Crucible)
    def load_14():
        return pickle.load(open(ROOT / 'data/2020_FULL/universe_30_15m.pkl', 'rb'))
    folds_def.append((14, "Fold 14 (2020 Full COVID Year)", "Jan-Dec 2020 (27 Core Stocks, 250d)", load_14, "2020-01-01", "2020-12-31", core_27))
    
    return folds_def

folds = get_folds()
results = []

def run_engine_eval(mod, pool, s_dt, e_dt, label, univ_filt):
    cap, trades = mod.run_simulation(pool, s_dt, e_dt, label, universe_filter=univ_filt)
    df_t = pd.DataFrame(trades)
    n = len(df_t)
    if n > 0:
        win_trades = df_t[df_t['net_pnl'] > 0]
        loss_trades = df_t[df_t['net_pnl'] < 0]
        wr = (len(win_trades) / n) * 100.0
        net = df_t['net_pnl'].sum()
        gwins = win_trades['net_pnl'].sum()
        gloss = abs(loss_trades['net_pnl'].sum())
        pf = (gwins / gloss) if gloss > 0 else 999.0
        
        # Max DD
        eq = [500.0]
        cur = 500.0
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
        wr = net = pf = mdd = 0.0
    return {
        'trades': n,
        'win_rate': round(wr, 1),
        'net_pnl': round(net, 2),
        'pf': round(pf, 2),
        'max_dd': round(mdd, 1),
        'end_cap': round(cap, 2)
    }

for fid, fname, pdesc, loader, s_dt, e_dt, u_filt in folds:
    print(f"\n---> Evaluating {fname} ({pdesc})...")
    pool = loader()
    if u_filt is None:
        filt = set(k.replace('.NS', '') for k in pool.keys())
    else:
        filt = u_filt
        
    res_v34 = run_engine_eval(mod_v34, pool, s_dt, e_dt, f"V34_{fid}", filt)
    res_v39 = run_engine_eval(mod_v39, pool, s_dt, e_dt, f"V39_{fid}", filt)
    res_v39_1 = run_engine_eval(mod_v39_1, pool, s_dt, e_dt, f"V39_1_{fid}", filt)
    
    print(f"  V34.3 : {res_v34['trades']:3d}t | WR: {res_v34['win_rate']:4.1f}% | Net: Rs {res_v34['net_pnl']:+8.2f} | PF: {res_v34['pf']:4.2f} | DD: {res_v34['max_dd']:4.1f}%")
    print(f"  V39   : {res_v39['trades']:3d}t | WR: {res_v39['win_rate']:4.1f}% | Net: Rs {res_v39['net_pnl']:+8.2f} | PF: {res_v39['pf']:4.2f} | DD: {res_v39['max_dd']:4.1f}%")
    print(f"  V39.1 : {res_v39_1['trades']:3d}t | WR: {res_v39_1['win_rate']:4.1f}% | Net: Rs {res_v39_1['net_pnl']:+8.2f} | PF: {res_v39_1['pf']:4.2f} | DD: {res_v39_1['max_dd']:4.1f}%")
    
    results.append({
        'fold_id': fid,
        'fold_name': fname,
        'period_desc': pdesc,
        'v34_trades': res_v34['trades'], 'v34_wr': res_v34['win_rate'], 'v34_net': res_v34['net_pnl'], 'v34_dd': res_v34['max_dd'],
        'v39_trades': res_v39['trades'], 'v39_wr': res_v39['win_rate'], 'v39_net': res_v39['net_pnl'], 'v39_dd': res_v39['max_dd'],
        'v39_1_trades': res_v39_1['trades'], 'v39_1_wr': res_v39_1['win_rate'], 'v39_1_net': res_v39_1['net_pnl'], 'v39_1_dd': res_v39_1['max_dd']
    })

print("\n" + "="*100)
print("15-FOLD COMPARISON MATRIX: V34.3 vs V39 vs V39.1")
print("="*100)
df_res = pd.DataFrame(results)
print(df_res.to_string(index=False))

# Calculate cumulative totals
v34_tot_net = sum(r['v34_net'] for r in results)
v34_tot_trd = sum(r['v34_trades'] for r in results)
v39_tot_net = sum(r['v39_net'] for r in results)
v39_tot_trd = sum(r['v39_trades'] for r in results)
v39_1_tot_net = sum(r['v39_1_net'] for r in results)
v39_1_tot_trd = sum(r['v39_1_trades'] for r in results)

print("\n" + "="*80)
print(f"CUMULATIVE 15-FOLD GRAND TOTALS:")
print(f"  V34.3 : {v34_tot_trd:4d} Trades | Total Net: Rs {v34_tot_net:+9.2f}")
print(f"  V39   : {v39_tot_trd:4d} Trades | Total Net: Rs {v39_tot_net:+9.2f}")
print(f"  V39.1 : {v39_1_tot_trd:4d} Trades | Total Net: Rs {v39_1_tot_net:+9.2f}")
print("="*80)
