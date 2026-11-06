import sys
import pickle
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
with open(ROOT / 'scripts/exhaustive_19_folds_v34_v41_v42.pkl', 'rb') as f:
    recs = pickle.load(f)

print("=" * 135)
print("EXHAUSTIVE 19-FOLD COMPARISON MATRIX: V34.3 vs V41 vs V42 INVINCIBLE")
print("=" * 135)

cols = [
    "Fold", "Period / Name",
    "V34 Tr", "V34 WR", "V34 Net", "V34 DD", "V34 AW", "V34 AL", "V34 PF",
    "V41 Tr", "V41 WR", "V41 Net", "V41 DD", "V41 AW", "V41 AL", "V41 PF",
    "V42 Tr", "V42 WR", "V42 Net", "V42 DD", "V42 AW", "V42 AL", "V42 PF"
]

rows = []
for r in recs:
    fid = r['fold_id']
    fname = r['fold_name']
    m34 = r['v34']
    m41 = r['v41']
    m42 = r['v42']
    rows.append({
        "Fold": fid,
        "Period / Name": fname,
        "V34 Tr": m34["trades"],
        "V34 WR": f"{m34['win_rate']:.1f}%",
        "V34 Net": f"₹{m34['net_pnl']:+8.2f}",
        "V34 DD": f"{m34['max_dd']:.1f}%",
        "V34 AW": f"₹{m34['avg_win']:.2f}",
        "V34 AL": f"₹{m34['avg_loss']:.2f}",
        "V34 PF": f"{m34['pf']:.2f}",

        "V41 Tr": m41["trades"],
        "V41 WR": f"{m41['win_rate']:.1f}%",
        "V41 Net": f"₹{m41['net_pnl']:+8.2f}",
        "V41 DD": f"{m41['max_dd']:.1f}%",
        "V41 AW": f"₹{m41['avg_win']:.2f}",
        "V41 AL": f"₹{m41['avg_loss']:.2f}",
        "V41 PF": f"{m41['pf']:.2f}",

        "V42 Tr": m42["trades"],
        "V42 WR": f"{m42['win_rate']:.1f}%",
        "V42 Net": f"₹{m42['net_pnl']:+8.2f}",
        "V42 DD": f"{m42['max_dd']:.1f}%",
        "V42 AW": f"₹{m42['avg_win']:.2f}",
        "V42 AL": f"₹{m42['avg_loss']:.2f}",
        "V42 PF": f"{m42['pf']:.2f}",
    })

df = pd.DataFrame(rows)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(df.to_string(index=False))

# Totals across all 19 folds
tot_v34_net = sum(r['v34']['net_pnl'] for r in recs)
tot_v41_net = sum(r['v41']['net_pnl'] for r in recs)
tot_v42_net = sum(r['v42']['net_pnl'] for r in recs)

tot_v34_tr = sum(r['v34']['trades'] for r in recs)
tot_v41_tr = sum(r['v41']['trades'] for r in recs)
tot_v42_tr = sum(r['v42']['trades'] for r in recs)

print("=" * 135)
print(f"TOTAL TRADES ACROSS ALL 19 FOLDS: V34.3 = {tot_v34_tr} | V41 = {tot_v41_tr} | V42 = {tot_v42_tr}")
print(f"TOTAL NET PnL ACROSS ALL 19 FOLDS: V34.3 = ₹{tot_v34_net:+,.2f} | V41 = ₹{tot_v41_net:+,.2f} | V42 = ₹{tot_v42_net:+,.2f}")
print("=" * 135)
