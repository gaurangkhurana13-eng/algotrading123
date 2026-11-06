import pickle
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
spec = importlib.util.spec_from_file_location("mod_v39", str(ROOT / "strategies/V39/run_v39.py"))
mod_v39 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod_v39)

V29_30_UNIVERSE = mod_v39.V29_30_UNIVERSE

# Pre-load datasets
print("--- PRE-LOADING ALL DATASETS ---")
p2021 = pickle.load(open(ROOT / "data/2021-22/universe_30_15m.pkl", "rb"))
p2022 = pickle.load(open(ROOT / "data/2022-23/universe_30_15m.pkl", "rb"))
p2023 = pickle.load(open(ROOT / "data/2023-24/universe_30_15m.pkl", "rb"))
p2024_14 = pickle.load(open(ROOT / "data/2024-25/universe_14_15m.pkl", "rb"))
p2025_14 = pickle.load(open(ROOT / "data/2025-26/universe_14_15m.pkl", "rb"))

# 2026 bull datasets
u1 = pickle.load(open(ROOT / "data/universe_78_15m_1y.pkl", "rb"))
u2 = pickle.load(open(ROOT / "data/unseen_high_beta_10_15m_1y.pkl", "rb"))
u3 = pickle.load(open(ROOT / "data/unseen_10_15m_1y.pkl", "rb"))
p2026 = {}
p2026.update(u1); p2026.update(u2); p2026.update(u3)

# Virgin 20 dataset
b1 = pickle.load(open(ROOT / "data/unseen_10_15m_1y.pkl", "rb"))
b3 = pickle.load(open(ROOT / "data/unseen_batch3_10_15m_1y.pkl", "rb"))
p_virgin20 = {**b1, **b3}
first_df_v20 = list(p_virgin20.values())[0]
s_v20 = first_df_v20.index.min().strftime('%Y-%m-%d')
e_v20 = first_df_v20.index.max().strftime('%Y-%m-%d')
v20_syms = set(k.replace('.NS', '') for k in p_virgin20.keys())

# 2026 Unseen 52 Midcaps
clean_78 = {k.replace(".NS", "") for k in u1.keys()}
unseen_52_syms = clean_78 - V29_30_UNIVERSE

# 2021 20 Stocks
UNIV_2021 = {
    'CENTRALBK', 'DELTACORP', 'GSFC', 'IDEA', 'IOB', 'JYOTHYLAB', 'NBCC', 'NCC',
    'NIACL', 'NMDC', 'PNB', 'RBLBANK', 'RCF', 'RENUKA', 'RITES', 'SUZLON',
    'SWSOLAR', 'TATASTEEL', 'TTML', 'ZEEL'
}

import sys
sys.stdout.reconfigure(encoding='utf-8')

folds = [
    (0, "2021 Full Year (Verified)", p2021, "2021-01-01", "2021-12-31", UNIV_2021, "Jan–Dec 2021 (20 Stocks, 247d)"),
    (1, "2026 Bull Horizon", p2026, "2026-01-01", "2026-12-31", V29_30_UNIVERSE, "Dec 25–Oct 26 (78 Midcaps)"),
    (2, "2022 Bear Market", p2022, "2022-01-01", "2022-12-31", V29_30_UNIVERSE, "Jan–Dec 2022 (30 Stocks)"),
    (3, "2023 Full Year Continuous", p2023, "2023-01-02", "2023-12-29", V29_30_UNIVERSE, "2 Jan–29 Dec 2023 (30 Stocks, 245d)"),
    (4, "2023 H1 Virgin", p2023, "2023-01-01", "2023-04-30", V29_30_UNIVERSE, "Jan–Apr 2023 (30 Stocks)"),
    (5, "2023 Mid 4M", p2023, "2023-05-01", "2023-08-31", V29_30_UNIVERSE, "May–Aug 2023 (30 Stocks)"),
    (6, "2023 H2 Virgin", p2023, "2023-09-01", "2023-12-31", V29_30_UNIVERSE, "Sep–Dec 2023 (30 Stocks)"),
    (7, "2024 H1 Pre-Election", p2024_14, "2024-01-01", "2024-06-30", V29_30_UNIVERSE, "Jan–Jun 2024 (14 Stocks)"),
    (8, "2024 H2 Crash (FII)", p2024_14, "2024-07-01", "2024-12-31", V29_30_UNIVERSE, "Jul–Dec 2024 (14 Stocks)"),
    (9, "2025 H1 Choppy", p2025_14, "2025-01-01", "2025-07-31", V29_30_UNIVERSE, "Jan–Jul 2025 (14 Stocks)"),
    (10, "2025 Q4", p2025_14, "2025-10-01", "2025-12-31", None, "Oct–Dec 2025 (14 Stocks)"),
    (11, "2025 Last 5M", p2025_14, "2025-08-01", "2025-12-31", V29_30_UNIVERSE, "Aug–Dec 2025 (14 Stocks)"),
    (12, "20 Virgin Companies", p_virgin20, s_v20, e_v20, v20_syms, "Jan–Oct 2026 (20 New Stocks, 1-Yr)"),
    (13, "2026 Unseen Midcaps", u1, "2026-01-01", "2026-12-31", unseen_52_syms, "Jan–Oct 2026 (52 Outside Stocks)"),
]

print(f"Configured {len(folds)} folds. Starting simulation across all 14 folds...")

results = []

for f_id, name, pool, start, end, univ, period_desc in folds:
    cap, trades = mod_v39.run_simulation(
        pool,
        start,
        end,
        f"Fold {f_id}: {name}",
        universe_filter=univ,
        min_tm_odds=0.40,
        skip_midday_deadzone=True
    )
    tdf = pd.DataFrame(trades)
    n_trd = len(tdf)
    if n_trd > 0:
        wins = tdf[tdf["net_pnl"] > 0]
        losses = tdf[tdf["net_pnl"] < 0]
        wr = len(wins) / n_trd * 100.0
        net = tdf["net_pnl"].sum()
        cum = 500.0 + tdf["net_pnl"].cumsum()
        hwm = cum.cummax()
        dd = (hwm - cum) / hwm * 100.0
        max_dd = dd.max() if len(dd) > 0 else 0.0
        pf = (wins["net_pnl"].sum() / abs(losses["net_pnl"].sum())) if not losses.empty and abs(losses["net_pnl"].sum()) > 0 else 999.0
    else:
        wr = 0.0
        net = 0.0
        max_dd = 0.0
        pf = 0.0

    status = "🟢" if net >= 0 else "🔴"
    results.append({
        "Fold": f_id,
        "Regime": name,
        "Period": period_desc,
        "Trades": n_trd,
        "WR%": wr,
        "Net_PnL": net,
        "PF": pf,
        "Max_DD%": max_dd,
        "Final_Cap": cap,
        "Status": status,
        "Summary": f"{n_trd}t | {wr:.1f}% | {net:+.2f} {status}"
    })
    print(f"Fold {f_id:2d} | {name:30s} | {n_trd:3d}t | WR: {wr:5.1f}% | Net: Rs {net:+8.2f} | End: Rs {cap:7.2f} | {status}")

res_df = pd.DataFrame(results)
print("\n" + "=" * 95)
print("BHAVISHYA V39: ALL 14 FOLDS SCORECARD")
print("=" * 95)
print(res_df[["Fold", "Regime", "Trades", "WR%", "Net_PnL", "PF", "Max_DD%", "Status"]].to_string(index=False))

total_trades = res_df["Trades"].sum()
total_net = res_df["Net_PnL"].sum()
green_folds = len(res_df[res_df["Net_PnL"] >= 0])
red_folds = len(res_df[res_df["Net_PnL"] < 0])

print("\n" + "=" * 95)
print(f"CUMULATIVE V39 TOTAL: {total_trades} trades across all 14 folds | Total Net PnL: Rs {total_net:+.2f}")
print(f"SCORECARD: {green_folds} GREEN FOLDS | {red_folds} RED FOLDS (Win Rate: {green_folds/len(res_df)*100:.1f}%)")
print("=" * 95)

# Save results for subsequent reporting
with open(ROOT / "scripts" / "v39_14_folds_results.pkl", "wb") as f:
    pickle.dump(results, f)
