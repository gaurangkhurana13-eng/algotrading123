import os, sys, pickle, importlib.util
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

def load_mod(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

print("Loading Strategy Engines (V34.3, V41, V42)...")
mod_v34 = load_mod('mod_v34', ROOT / 'strategies/V34.3/run_v34_3.py')
mod_v41 = load_mod('mod_v41', ROOT / 'strategies/V41/run_v41.py')
mod_v42 = load_mod('mod_v42', ROOT / 'strategies/V42/run_v42.py')

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

stocks_30_legacy = {
    'CANBK', 'PNB', 'BANKBARODA', 'SBIN', 'IOC',
    'BPCL', 'GAIL', 'COALINDIA', 'TATASTEEL', 'TATAPOWER',
    'TATAMOTORS', 'VEDL', 'HINDALCO', 'JINDALSTEL', 'JSWSTEEL',
    'BEL', 'ADANIPOWER', 'ADANIENT', 'RECLTD', 'PFC',
    'DLF', 'MOTHERSON', 'ITC', 'ICICIBANK', 'AXISBANK',
    'RELIANCE', 'INFY', 'SUNPHARMA', 'TECHM', 'CIPLA'
}

folds_def = [
    (0, "Fold 0 (2021 Full Year)", "2021 Full Year", ROOT / 'data/2021-22/universe_30_15m.pkl', "2021-01-01", "2021-12-31", {'CENTRALBK', 'DELTACORP', 'GSFC', 'IDEA', 'IOB', 'JYOTHYLAB', 'NBCC', 'NCC', 'NIACL', 'NMDC', 'PNB', 'RBLBANK', 'RCF', 'RENUKA', 'RITES', 'SUZLON', 'SWSOLAR', 'TATASTEEL', 'TTML', 'ZEEL'}),
    (1, "Fold 1 (2026 Bull Horizon)", "2026 (78 Midcaps)", 'COMPOSITE_1', "2026-01-01", "2026-12-31", V29_30_UNIVERSE),
    (2, "Fold 2 (2022 Bear Market)", "2022 Full Year", ROOT / 'data/2022-23/universe_30_15m.pkl', "2022-01-01", "2022-12-31", V29_30_UNIVERSE),
    (3, "Fold 3 (2023 Full Year)", "2023 Full Year", ROOT / 'data/2023-24/universe_30_15m.pkl', "2023-01-02", "2023-12-29", V29_30_UNIVERSE),
    (4, "Fold 4 (2023 H1 Virgin)", "2023 Jan-Apr", ROOT / 'data/2023-24/universe_30_15m.pkl', "2023-01-01", "2023-04-30", V29_30_UNIVERSE),
    (5, "Fold 5 (2023 Mid 4M)", "2023 May-Aug", ROOT / 'data/2023-24/universe_30_15m.pkl', "2023-05-01", "2023-08-31", V29_30_UNIVERSE),
    (6, "Fold 6 (2023 H2 Virgin)", "2023 Sep-Dec", ROOT / 'data/2023-24/universe_30_15m.pkl', "2023-09-01", "2023-12-31", V29_30_UNIVERSE),
    (7, "Fold 7 (2024 H1 Pre-Election)", "2024 Jan-Jun", ROOT / 'data/2024-25/universe_14_15m.pkl', "2024-01-01", "2024-06-30", V29_30_UNIVERSE),
    (8, "Fold 8 (2024 H2 Crash FII)", "2024 Jul-Dec", ROOT / 'data/2024-25/universe_14_15m.pkl', "2024-07-01", "2024-12-31", V29_30_UNIVERSE),
    (9, "Fold 9 (2025 H1 Choppy)", "2025 Jan-Jul", ROOT / 'data/2025-26/universe_14_15m.pkl', "2025-01-01", "2025-07-31", V29_30_UNIVERSE),
    (10, "Fold 10 (2025 Q4)", "2025 Oct-Dec", ROOT / 'data/2025-26/universe_14_15m.pkl', "2025-10-01", "2025-12-31", None),
    (11, "Fold 11 (2025 Last 5M)", "2025 Aug-Dec", ROOT / 'data/2025-26/universe_14_15m.pkl', "2025-08-01", "2025-12-31", V29_30_UNIVERSE),
    (12, "Fold 12 (20 Virgin Companies)", "2026 (20 Virgin Stocks)", 'COMPOSITE_12', "2026-01-01", "2026-10-31", None),
    (13, "Fold 13 (2026 Unseen Midcaps)", "2026 (52 Outside Stocks)", 'COMPOSITE_13', "2026-01-01", "2026-12-31", 'COMPOSITE_13_UNIV'),
    (14, "Fold 14 (2020 Full COVID Year)", "2020 Full Year", ROOT / 'data/2020_FULL/universe_30_15m.pkl', "2020-01-01", "2020-12-31", core_27),
    (15, "Fold 15 (2019 Unseen Year)", "2019 Full Year", ROOT / 'data/2019_UNSEEN/universe_30_15m.pkl', "2019-01-01", "2019-12-31", None),
    (16, "Fold 16 (2018 Unseen Year)", "2018 Full Year", ROOT / 'data/2018_UNSEEN/universe_30_15m.pkl', "2018-01-01", "2018-12-31", stocks_30_legacy),
    (17, "Fold 17 (2017 Unseen Year)", "2017 Full Year", ROOT / 'data/2017_UNSEEN/universe_30_15m.pkl', "2017-01-01", "2017-12-31", stocks_30_legacy),
    (18, "Fold 18 (2016 H2 Unseen)", "2016 H2", ROOT / 'data/2016_H2_UNSEEN/universe_30_15m.pkl', "2016-07-01", "2016-12-31", stocks_30_legacy),
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

def compute_all_metrics(trades_list, initial_cap=500.0):
    if not trades_list:
        return {
            "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "net_pnl": 0.0, "gross_pnl": 0.0, "charges": 0.0,
            "max_dd": 0.0, "pf": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "payoff_ratio": 0.0, "end_wallet": initial_cap
        }
    df = pd.DataFrame(trades_list)
    n = len(df)
    win_df = df[df["net_pnl"] > 0]
    loss_df = df[df["net_pnl"] < 0]
    n_w = len(win_df)
    n_l = len(loss_df)
    wr = (n_w / n * 100.0) if n > 0 else 0.0
    net = df["net_pnl"].sum()
    gross = df["gross_pnl"].sum() if "gross_pnl" in df else net
    charges = df["charges"].sum() if "charges" in df else 0.0
    
    tot_win = win_df["net_pnl"].sum() if n_w > 0 else 0.0
    tot_loss = abs(loss_df["net_pnl"].sum()) if n_l > 0 else 0.0
    pf = (tot_win / tot_loss) if tot_loss > 0 else 999.0
    
    avg_win = (tot_win / n_w) if n_w > 0 else 0.0
    avg_loss = (tot_loss / n_l) if n_l > 0 else 0.0
    payoff = (avg_win / avg_loss) if avg_loss > 0 else 0.0

    eq = [initial_cap]
    cur = initial_cap
    for p in df["net_pnl"]:
        cur += p
        eq.append(cur)
    pk = eq[0]
    mdd = 0.0
    for e in eq:
        if e > pk: pk = e
        dd = (pk - e) / pk * 100.0
        if dd > mdd: mdd = dd

    return {
        "trades": n,
        "wins": n_w,
        "losses": n_l,
        "win_rate": round(wr, 1),
        "net_pnl": round(net, 2),
        "gross_pnl": round(gross, 2),
        "charges": round(charges, 2),
        "max_dd": round(mdd, 1),
        "pf": round(pf, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "payoff_ratio": round(payoff, 2),
        "end_wallet": round(cur, 2)
    }

master_records = []

for fid, fname, era_desc, pool_spec, s_dt, e_dt, u_filt in folds_def:
    print(f"\nProcessing {fname} ({era_desc})...")
    pool = load_pool_by_spec(pool_spec)
    if u_filt == 'COMPOSITE_13_UNIV':
        clean_78 = {k.replace('.NS', '') for k in pool.keys()}
        filt = clean_78 - V29_30_UNIVERSE
    elif u_filt is None:
        filt = set(k.replace('.NS', '') for k in pool.keys())
    else:
        filt = u_filt

    # 1. Run V34.3
    cap_34, t_34 = mod_v34.run_simulation(pool, s_dt, e_dt, f"V34_{fid}", universe_filter=filt)
    m34 = compute_all_metrics(t_34)

    # 2. Run V41 Titan
    cap_41, t_41 = mod_v41.run_simulation(pool, s_dt, e_dt, f"V41_{fid}", universe_filter=filt)
    m41 = compute_all_metrics(t_41)

    # 3. Run V42 Invincible
    cap_42, t_42 = mod_v42.run_simulation(pool, s_dt, e_dt, f"V42_{fid}", universe_filter=filt)
    m42 = compute_all_metrics(t_42)

    rec = {
        "fold_id": fid,
        "fold_name": fname,
        "era_desc": era_desc,
        "start": s_dt,
        "end": e_dt,
        "v34": m34,
        "v41": m41,
        "v42": m42
    }
    master_records.append(rec)
    print(f"  V34.3 : Tr {m34['trades']:2d} | WR {m34['win_rate']:4.1f}% | Net Rs {m34['net_pnl']:+8.2f} | DD {m34['max_dd']:4.1f}% | AvgW Rs {m34['avg_win']:5.2f} | AvgL Rs {m34['avg_loss']:5.2f}")
    print(f"  V41   : Tr {m41['trades']:2d} | WR {m41['win_rate']:4.1f}% | Net Rs {m41['net_pnl']:+8.2f} | DD {m41['max_dd']:4.1f}% | AvgW Rs {m41['avg_win']:5.2f} | AvgL Rs {m41['avg_loss']:5.2f}")
    print(f"  V42 ⚡: Tr {m42['trades']:2d} | WR {m42['win_rate']:4.1f}% | Net Rs {m42['net_pnl']:+8.2f} | DD {m42['max_dd']:4.1f}% | AvgW Rs {m42['avg_win']:5.2f} | AvgL Rs {m42['avg_loss']:5.2f}")

# Save to pickle
out_pkl = ROOT / "scripts/exhaustive_19_folds_v34_v41_v42.pkl"
with open(out_pkl, "wb") as pf:
    pickle.dump(master_records, pf)

print(f"\nSaved all 19 folds detailed data to {out_pkl}")
