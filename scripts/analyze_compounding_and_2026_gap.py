import sys, pickle
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")

with open(ROOT / 'scripts/exhaustive_19_folds_v34_v41_v42.pkl', 'rb') as f:
    recs = pickle.load(f)

rec_map = {r['fold_id']: r for r in recs}

print("=" * 115)
print("1. CONTINUOUS 10-YEAR ACCOUNT PROGRESSION (STARTING CAPITAL: ₹500 IN 2016 H2)")
print("=" * 115)

# Annual distinct timeline:
annual_folds = [
    ("2016 (H2 Shock)", 18),
    ("2017 (Bull Drift)", 17),
    ("2018 (Deep Bear)", 16),
    ("2019 (NBFC Stress)", 15),
    ("2020 (COVID Crucible)", 14),
    ("2021 (Bull Market)", 0),
    ("2022 (War / Bear)", 2),
    ("2023 (Momentum Run)", 3),
    ("2024 (Election + Crash)", 7), # Fold 7 + 8
    ("2025 (Chop + Breakout)", 9), # Fold 9 + 10
    ("2026 (78 Midcap Basket)", 1),
]

w_34 = 500.0
w_41 = 500.0
w_42 = 500.0

table_rows = []

for era_name, fid in annual_folds:
    if era_name == "2024 (Election + Crash)":
        pnl_34 = rec_map[7]['v34']['net_pnl'] + rec_map[8]['v34']['net_pnl']
        pnl_41 = rec_map[7]['v41']['net_pnl'] + rec_map[8]['v41']['net_pnl']
        pnl_42 = rec_map[7]['v42']['net_pnl'] + rec_map[8]['v42']['net_pnl']
    elif era_name == "2025 (Chop + Breakout)":
        pnl_34 = rec_map[9]['v34']['net_pnl'] + rec_map[10]['v34']['net_pnl']
        pnl_41 = rec_map[9]['v41']['net_pnl'] + rec_map[10]['v41']['net_pnl']
        pnl_42 = rec_map[9]['v42']['net_pnl'] + rec_map[10]['v42']['net_pnl']
    else:
        pnl_34 = rec_map[fid]['v34']['net_pnl']
        pnl_41 = rec_map[fid]['v41']['net_pnl']
        pnl_42 = rec_map[fid]['v42']['net_pnl']

    # Update wallets
    if w_34 > 0:
        w_34 += pnl_34
        if w_34 <= 0:
            w_34_str = "₹0.00 (BLOWN UP!)"
        else:
            w_34_str = f"₹{w_34:,.2f}"
    else:
        w_34_str = "₹0.00 (DEAD)"

    w_41 += pnl_41
    w_42 += pnl_42

    table_rows.append({
        "Era": era_name,
        "V34 PnL": f"₹{pnl_34:+8.2f}",
        "V34 Wallet": w_34_str,
        "V41 PnL": f"₹{pnl_41:+8.2f}",
        "V41 Wallet": f"₹{w_41:,.2f}",
        "V42 PnL": f"₹{pnl_42:+8.2f}",
        "V42 Wallet": f"₹{w_42:,.2f}"
    })

df_progress = pd.DataFrame(table_rows)
print(df_progress.to_string(index=False))

print("\n" + "=" * 115)
print("2. DEEP DIVE: WHY DID V42 MAKE ₹1,620 IN 2026 INSTEAD OF V34's ₹4,363?")
print("=" * 115)

# Inspect 2026 Fold 1 trades
m1_34 = rec_map[1]['v34']
m1_41 = rec_map[1]['v41']
m1_42 = rec_map[1]['v42']

print(f"V34.3 in 2026: Trades={m1_34['trades']}, WR={m1_34['win_rate']}%, AvgWin=₹{m1_34['avg_win']}, AvgLoss=₹{m1_34['avg_loss']}, MaxDD={m1_34['max_dd']}%")
print(f"V41   in 2026: Trades={m1_41['trades']}, WR={m1_41['win_rate']}%, AvgWin=₹{m1_41['avg_win']}, AvgLoss=₹{m1_41['avg_loss']}, MaxDD={m1_41['max_dd']}%")
print(f"V42   in 2026: Trades={m1_42['trades']}, WR={m1_42['win_rate']}%, AvgWin=₹{m1_42['avg_win']}, AvgLoss=₹{m1_42['avg_loss']}, MaxDD={m1_42['max_dd']}%")
