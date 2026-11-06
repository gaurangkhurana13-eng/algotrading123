import sys, pickle
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")

with open(ROOT / 'scripts/exhaustive_19_folds_v34_v41_v42.pkl', 'rb') as f:
    recs = pickle.load(f)

# Let's map folds to distinct calendar years:
# 2016: Fold 18 (2016 H2)
# 2017: Fold 17 (2017 Full Year)
# 2018: Fold 16 (2018 Full Year)
# 2019: Fold 15 (2019 Full Year)
# 2020: Fold 14 (2020 Full Year)
# 2021: Fold 0  (2021 Full Year)
# 2022: Fold 2  (2022 Full Year)
# 2023: Fold 3  (2023 Full Year)
# 2024: Fold 7 (H1) + Fold 8 (H2)
# 2025: Fold 9 (H1) + Fold 10 (Q4)
# 2026: Fold 1 (78 Midcaps) + Fold 12 (Virgin 20) + Fold 13 (52 Outside)

rec_map = {r['fold_id']: r for r in recs}

print("=" * 140)
print("ANNUAL CALENDAR YEAR COMPARISON: V34.3 vs V41 vs V42 INVINCIBLE (2016 - 2026)")
print("=" * 140)

calendar_groups = [
    ("2016 (H2 Shock)", [18]),
    ("2017 (Bull Drift)", [17]),
    ("2018 (Deep Bear)", [16]),
    ("2019 (NBFC Stress)", [15]),
    ("2020 (COVID Crucible)", [14]),
    ("2021 (Bull Market)", [0]),
    ("2022 (War / Bear)", [2]),
    ("2023 (Momentum Run)", [3]),
    ("2024 (Election + Crash)", [7, 8]),
    ("2025 (Chop + Breakout)", [9, 10]),
    ("2026 (Modern Expansion)", [1, 12, 13]),
]

annual_summary = []

for year_label, fold_ids in calendar_groups:
    for v_key in ['v34', 'v41', 'v42']:
        tot_trades = sum(rec_map[fid][v_key]['trades'] for fid in fold_ids)
        tot_wins = sum(rec_map[fid][v_key]['wins'] for fid in fold_ids)
        tot_losses = sum(rec_map[fid][v_key]['losses'] for fid in fold_ids)
        tot_net = sum(rec_map[fid][v_key]['net_pnl'] for fid in fold_ids)
        max_dd = max(rec_map[fid][v_key]['max_dd'] for fid in fold_ids)
        wr = (tot_wins / tot_trades * 100.0) if tot_trades > 0 else 0.0
        
        # Weighted average wins and losses
        sum_win_val = sum(rec_map[fid][v_key]['avg_win'] * rec_map[fid][v_key]['wins'] for fid in fold_ids)
        sum_loss_val = sum(rec_map[fid][v_key]['avg_loss'] * rec_map[fid][v_key]['losses'] for fid in fold_ids)
        avg_w = (sum_win_val / tot_wins) if tot_wins > 0 else 0.0
        avg_l = (sum_loss_val / tot_losses) if tot_losses > 0 else 0.0
        pf = (sum_win_val / sum_loss_val) if sum_loss_val > 0 else 999.0
        payoff = (avg_w / avg_l) if avg_l > 0 else 0.0
        
        annual_summary.append({
            "Year": year_label,
            "Version": v_key.upper(),
            "Trades": tot_trades,
            "WR": f"{wr:.1f}%",
            "Net PnL": f"₹{tot_net:+8.2f}",
            "Max DD": f"{max_dd:.1f}%",
            "Avg Win": f"₹{avg_w:.2f}",
            "Avg Loss": f"₹{avg_l:.2f}",
            "Payoff": f"{payoff:.2f}",
            "PF": f"{pf:.2f}"
        })

df_ann = pd.DataFrame(annual_summary)
for y, g in df_ann.groupby("Year", sort=False):
    print(f"\n--- CALENDAR ERA: {y} ---")
    print(g[['Version', 'Trades', 'WR', 'Net PnL', 'Max DD', 'Avg Win', 'Avg Loss', 'Payoff', 'PF']].to_string(index=False))
