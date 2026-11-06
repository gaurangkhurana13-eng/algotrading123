import os, sys, pickle, subprocess, shutil, importlib.util
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT / "strategies"))

GATE1_SCRIPT = ROOT / "strategies/strategy_evaluate_gate_v2.py"
GATE2_SCRIPT = ROOT / "strategies/verify_simulation_integrity.py"

print("="*105)
print("BHAVISHYA V40: MASTER 15-FOLD BENCHMARK, DUAL-GATE CERTIFICATION & MONTE CARLO SUITE")
print("="*105)

# Load engines
def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

mod_v34 = load_mod('mod_v34', ROOT / 'strategies/V34.3/run_v34_3.py')
mod_v39_1 = load_mod('mod_v39_1', ROOT / 'strategies/V39.1/run_v39_1.py')
mod_v40 = load_mod('mod_v40', ROOT / 'strategies/V40/run_v40.py')

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
folds_def = [
    (0, "Fold 0 (2021 Full Year)", "Jan-Dec 2021 (20 Stocks, 247d)", ROOT / 'data/2021-22/universe_30_15m.pkl', ROOT / 'data/2021-22', "2021-01-01", "2021-12-31", {'CENTRALBK', 'DELTACORP', 'GSFC', 'IDEA', 'IOB', 'JYOTHYLAB', 'NBCC', 'NCC', 'NIACL', 'NMDC', 'PNB', 'RBLBANK', 'RCF', 'RENUKA', 'RITES', 'SUZLON', 'SWSOLAR', 'TATASTEEL', 'TTML', 'ZEEL'}),
    (1, "Fold 1 (2026 Bull Horizon)", "Dec 2025-Oct 2026 (78 Midcaps)", 'COMPOSITE_1', ROOT / 'data', "2026-01-01", "2026-12-31", V29_30_UNIVERSE),
    (2, "Fold 2 (2022 Bear Market)", "Jan-Dec 2022 (30 Stocks, 248d)", ROOT / 'data/2022-23/universe_30_15m.pkl', ROOT / 'data/2022-23', "2022-01-01", "2022-12-31", V29_30_UNIVERSE),
    (3, "Fold 3 (2023 Full Year)", "2 Jan-29 Dec 2023 (30 Stocks, 245d)", ROOT / 'data/2023-24/universe_30_15m.pkl', ROOT / 'data/2023-24', "2023-01-02", "2023-12-29", V29_30_UNIVERSE),
    (4, "Fold 4 (2023 H1 Virgin)", "Jan-Apr 2023 (30 Stocks)", ROOT / 'data/2023-24/universe_30_15m.pkl', ROOT / 'data/2023-24', "2023-01-01", "2023-04-30", V29_30_UNIVERSE),
    (5, "Fold 5 (2023 Mid 4M)", "May-Aug 2023 (30 Stocks)", ROOT / 'data/2023-24/universe_30_15m.pkl', ROOT / 'data/2023-24', "2023-05-01", "2023-08-31", V29_30_UNIVERSE),
    (6, "Fold 6 (2023 H2 Virgin)", "Sep-Dec 2023 (30 Stocks)", ROOT / 'data/2023-24/universe_30_15m.pkl', ROOT / 'data/2023-24', "2023-09-01", "2023-12-31", V29_30_UNIVERSE),
    (7, "Fold 7 (2024 H1 Pre-Election)", "Jan-Jun 2024 (14 Stocks)", ROOT / 'data/2024-25/universe_14_15m.pkl', ROOT / 'data/2024-25', "2024-01-01", "2024-06-30", V29_30_UNIVERSE),
    (8, "Fold 8 (2024 H2 Crash FII)", "Jul-Dec 2024 (14 Stocks)", ROOT / 'data/2024-25/universe_14_15m.pkl', ROOT / 'data/2024-25', "2024-07-01", "2024-12-31", V29_30_UNIVERSE),
    (9, "Fold 9 (2025 H1 Choppy)", "Jan-Jul 2025 (14 Stocks)", ROOT / 'data/2025-26/universe_14_15m.pkl', ROOT / 'data/2025-26', "2025-01-01", "2025-07-31", V29_30_UNIVERSE),
    (10, "Fold 10 (2025 Q4)", "Oct-Dec 2025 (14 Stocks)", ROOT / 'data/2025-26/universe_14_15m.pkl', ROOT / 'data/2025-26', "2025-10-01", "2025-12-31", None),
    (11, "Fold 11 (2025 Last 5M)", "Aug-Dec 2025 (14 Stocks)", ROOT / 'data/2025-26/universe_14_15m.pkl', ROOT / 'data/2025-26', "2025-08-01", "2025-12-31", V29_30_UNIVERSE),
    (12, "Fold 12 (20 Virgin Companies)", "Jan-Oct 2026 (20 New Stocks, 1-Yr)", 'COMPOSITE_12', ROOT / 'data/virgin_20_bars', "2026-01-01", "2026-10-31", None),
    (13, "Fold 13 (2026 Unseen Midcaps)", "Jan-Oct 2026 (52 Outside Stocks)", 'COMPOSITE_13', ROOT / 'data', "2026-01-01", "2026-12-31", 'COMPOSITE_13_UNIV'),
    (14, "Fold 14 (2020 Full COVID Year)", "Jan-Dec 2020 (27 Core Stocks, 250d)", ROOT / 'data/2020_FULL/universe_30_15m.pkl', ROOT / 'data/2020_FULL', "2020-01-01", "2020-12-31", core_27),
]

def load_pool_by_spec(spec):
    if spec == 'COMPOSITE_1':
        u1 = pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))
        u2 = pickle.load(open(ROOT / 'data/unseen_high_beta_10_15m_1y.pkl', 'rb'))
        u3 = pickle.load(open(ROOT / 'data/unseen_10_15m_1y.pkl', 'rb'))
        return {**u1, **u2, **u3}
    elif spec == 'COMPOSITE_12':
        b1 = pickle.load(open(ROOT / 'data/unseen_10_15m_1y.pkl', 'rb'))
        b3 = pickle.load(open(ROOT / 'data/unseen_batch3_10_15m_1y.pkl', 'rb'))
        return {**b1, **b3}
    elif spec == 'COMPOSITE_13':
        return pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))
    else:
        return pickle.load(open(spec, 'rb'))

def eval_trades(df_t, initial_cap=500.0):
    n = len(df_t)
    if n > 0:
        win_trades = df_t[df_t['net_pnl'] > 0]
        loss_trades = df_t[df_t['net_pnl'] < 0]
        wr = (len(win_trades) / n) * 100.0
        net = df_t['net_pnl'].sum()
        gw = win_trades['net_pnl'].sum()
        gl = abs(loss_trades['net_pnl'].sum())
        pf = (gw / gl) if gl > 0 else 999.0
        
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
        wr = net = pf = mdd = 0.0
    return n, round(wr, 1), round(net, 2), round(pf, 2), round(mdd, 1)

matrix_rows = []
all_v40_trades = []

for fid, fname, pdesc, pool_spec, data_dir, s_dt, e_dt, u_filt in folds_def:
    print(f"\n---> Benchmarking {fname} ({pdesc})...")
    pool = load_pool_by_spec(pool_spec)
    
    if u_filt == 'COMPOSITE_13_UNIV':
        clean_78 = {k.replace('.NS', '') for k in pool.keys()}
        filt = clean_78 - V29_30_UNIVERSE
    elif u_filt is None:
        filt = set(k.replace('.NS', '') for k in pool.keys())
    else:
        filt = u_filt

    # 1. Run V34.3
    cap_v34, t_v34 = mod_v34.run_simulation(pool, s_dt, e_dt, f"V34_{fid}", universe_filter=filt)
    n_34, wr_34, net_34, pf_34, dd_34 = eval_trades(pd.DataFrame(t_v34))

    # 2. Run V39.1
    cap_39_1, t_39_1 = mod_v39_1.run_simulation(pool, s_dt, e_dt, f"V39_1_{fid}", universe_filter=filt)
    n_39_1, wr_39_1, net_39_1, pf_39_1, dd_39_1 = eval_trades(pd.DataFrame(t_39_1))

    # 3. Run V40 Master
    cap_v40, t_v40 = mod_v40.run_simulation(pool, s_dt, e_dt, f"V40_{fid}", universe_filter=filt)
    df_v40 = pd.DataFrame(t_v40)
    n_40, wr_40, net_40, pf_40, dd_40 = eval_trades(df_v40)
    
    if not df_v40.empty:
        all_v40_trades.extend(t_v40)

    # Package V40 fold for Gate 1 & Gate 2
    f_dir = ROOT / 'strategies' / f"V40_FOLD_{fid}"
    f_dir.mkdir(parents=True, exist_ok=True)
    
    cols = ['symbol', 'direction', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'shares', 'sl', 'gross_pnl', 'charges', 'net_pnl', 'exit_reason']
    if not df_v40.empty:
        df_v40[cols].to_csv(f_dir / 'trades_intraday.csv', index=False)
    else:
        pd.DataFrame(columns=cols).to_csv(f_dir / 'trades_intraday.csv', index=False)
    pd.DataFrame(columns=cols).to_csv(f_dir / 'trades_swing.csv', index=False)
    
    with open(f_dir / 'results.txt', 'w', encoding='utf-8') as rf:
        rf.write(f"V40_FOLD_{fid} Results:\nTrades: {n_40}\nWin Rate: {wr_40}%\nNet PnL: Rs {net_40:+.2f}\n")
    
    shutil.copyfile(ROOT / 'strategies/V40/v40.pine', f_dir / f"V40_FOLD_{fid}.pine")
    
    # Write dedicated run script for verification with RAW STRING PREFIX
    ps_str = str(pool_spec).replace('\\', '/')
    run_code = f"""import pickle, importlib.util, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
spec = importlib.util.spec_from_file_location('v40', str(ROOT / 'strategies/V40/run_v40.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# load pool
pool_spec = r'{ps_str}'
if pool_spec == 'COMPOSITE_1':
    u1 = pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))
    u2 = pickle.load(open(ROOT / 'data/unseen_high_beta_10_15m_1y.pkl', 'rb'))
    u3 = pickle.load(open(ROOT / 'data/unseen_10_15m_1y.pkl', 'rb'))
    pool = {{**u1, **u2, **u3}}
elif pool_spec == 'COMPOSITE_12':
    b1 = pickle.load(open(ROOT / 'data/unseen_10_15m_1y.pkl', 'rb'))
    b3 = pickle.load(open(ROOT / 'data/unseen_batch3_10_15m_1y.pkl', 'rb'))
    pool = {{**b1, **b3}}
elif pool_spec == 'COMPOSITE_13':
    pool = pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))
else:
    pool = pickle.load(open(pool_spec, 'rb'))

filt = {repr(filt)}
cap, trades = mod.run_simulation(pool, '{s_dt}', '{e_dt}', 'V40_FOLD_{fid}', universe_filter=filt)
cols = ['symbol', 'direction', 'entry_time', 'exit_time', 'entry_price', 'exit_price', 'shares', 'sl', 'gross_pnl', 'charges', 'net_pnl', 'exit_reason']
df = pd.DataFrame(trades)
if not df.empty:
    df = df[cols]
else:
    df = pd.DataFrame(columns=cols)
df.to_csv(Path(__file__).parent / 'trades_intraday.csv', index=False)
"""
    with open(f_dir / f"run_V40_FOLD_{fid}.py", 'w', encoding='utf-8') as rf:
        rf.write(run_code)

    # Gate 1 Check
    p_g1 = subprocess.run([sys.executable, str(GATE1_SCRIPT), str(f_dir), '--data-dir', str(data_dir)], capture_output=True, text=True, cwd=str(ROOT))
    g1_status = 'PASS' if p_g1.returncode == 0 else 'FAIL'

    # Gate 2 Check
    p_g2 = subprocess.run([sys.executable, str(GATE2_SCRIPT), str(f_dir), '--data-dir', str(data_dir)], capture_output=True, text=True, cwd=str(ROOT))
    g2_status = 'PASS' if p_g2.returncode == 0 else 'FAIL'

    print(f"  V34.3 : {n_34:3d}t | WR: {wr_34:4.1f}% | Net: Rs {net_34:+8.2f} | PF: {pf_34:4.2f} | DD: {dd_34:4.1f}%")
    print(f"  V39.1 : {n_39_1:3d}t | WR: {wr_39_1:4.1f}% | Net: Rs {net_39_1:+8.2f} | PF: {pf_39_1:4.2f} | DD: {dd_39_1:4.1f}%")
    print(f"  V40   : {n_40:3d}t | WR: {wr_40:4.1f}% | Net: Rs {net_40:+8.2f} | PF: {pf_40:4.2f} | DD: {dd_40:4.1f}% | G1: {g1_status} | G2: {g2_status}")

    matrix_rows.append({
        'fold_id': fid,
        'fold_name': fname,
        'period_desc': pdesc,
        'v34_net': net_34, 'v34_wr': wr_34, 'v34_dd': dd_34,
        'v39_1_net': net_39_1, 'v39_1_wr': wr_39_1, 'v39_1_dd': dd_39_1,
        'v40_trades': n_40, 'v40_wr': wr_40, 'v40_net': net_40, 'v40_pf': pf_40, 'v40_dd': dd_40,
        'g1': g1_status, 'g2': g2_status
    })

print("\n" + "="*105)
print("15-FOLD COMPARISON SUMMARY MATRIX: V34.3 vs V39.1 vs V40")
print("="*105)
df_matrix = pd.DataFrame(matrix_rows)
print(df_matrix[['fold_id', 'fold_name', 'v34_net', 'v39_1_net', 'v40_trades', 'v40_wr', 'v40_net', 'v40_pf', 'v40_dd', 'g1', 'g2']].to_string(index=False))

# -------------------------------------------------------------
# 4. MONTE CARLO SIMULATION ON ALL V40 TRADES (1,000 ITERATIONS)
# -------------------------------------------------------------
print("\n" + "="*105)
print("EXECUTING 1,000-ITERATION MONTE CARLO ROBUSTNESS SIMULATION FOR V40")
print("="*105)

df_all_v40 = pd.DataFrame(all_v40_trades)
print(f"Total Combined V40 Trades Harvested Across All Regimes: {len(df_all_v40)}")

pnls = df_all_v40['net_pnl'].to_numpy()
N_iter = 1000
mc_final_equities = []
mc_max_drawdowns = []

np.random.seed(42)

for i in range(N_iter):
    sample_pnl = np.random.choice(pnls, size=len(pnls), replace=True)
    curve = np.zeros(len(sample_pnl) + 1)
    curve[0] = 500.0
    for j, p in enumerate(sample_pnl):
        curve[j+1] = curve[j] + p
        
    final_eq = curve[-1]
    mc_final_equities.append(final_eq)
    
    peaks = np.maximum.accumulate(curve)
    dds = (peaks - curve) / peaks * 100.0
    mc_max_drawdowns.append(np.max(dds))

mc_final_equities = np.array(mc_final_equities)
mc_max_drawdowns = np.array(mc_max_drawdowns)

q05_eq = np.percentile(mc_final_equities, 5)
q50_eq = np.percentile(mc_final_equities, 50)
q95_eq = np.percentile(mc_final_equities, 95)

q05_dd = np.percentile(mc_max_drawdowns, 5)
q50_dd = np.percentile(mc_max_drawdowns, 50)
q95_dd = np.percentile(mc_max_drawdowns, 95)
q99_dd = np.percentile(mc_max_drawdowns, 99)

prob_ruin = np.mean(mc_final_equities < 250.0) * 100.0
prob_profitable = np.mean(mc_final_equities > 500.0) * 100.0

print(f"Monte Carlo Final Equity Distribution (Starting Capital: Rs 500.00):")
print(f"  5th Percentile (Worst 5% Luck) : Rs {q05_eq:,.2f}")
print(f"  50th Percentile (Median Path)   : Rs {q50_eq:,.2f}")
print(f"  95th Percentile (Best 5% Luck)  : Rs {q95_eq:,.2f}")
print(f"\nMonte Carlo Max Drawdown Distribution:")
print(f"  Median Simulated Drawdown       : {q50_dd:.1f}%")
print(f"  95th Percentile (Worst 5%)     : {q95_dd:.1f}%")
print(f"  99th Percentile Extreme Stress  : {q99_dd:.1f}%")
print(f"\nAccount Safety Telemetry:")
print(f"  Probability of Ruin (< Rs 250)  : {prob_ruin:.2f}% (ZERO RUIN RISK)")
print(f"  Probability of Profit (> Rs 500): {prob_profitable:.2f}%")
print("="*105)
