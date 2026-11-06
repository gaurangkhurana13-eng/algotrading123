import os
import shutil
import pickle
import importlib.util
from pathlib import Path
import pandas as pd

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
OUT_DIR = ROOT / "strategies" / "V34_3_2021_FULL"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Write run_V34_3_2021_FULL.py
run_script_content = '''import pickle, importlib.util, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
spec = importlib.util.spec_from_file_location('mod_v34_3', str(ROOT / 'strategies/V34.3/run_v34_3.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

p2021 = pickle.load(open(ROOT / 'data/2021-22/universe_30_15m.pkl', 'rb'))
UNIV = {
    'CENTRALBK', 'DELTACORP', 'GSFC', 'IDEA', 'IOB', 'JYOTHYLAB', 'NBCC', 'NCC',
    'NIACL', 'NMDC', 'PNB', 'RBLBANK', 'RCF', 'RENUKA', 'RITES', 'SUZLON',
    'SWSOLAR', 'TATASTEEL', 'TTML', 'ZEEL'
}
cap, trades = mod.run_simulation(p2021, '2021-01-01', '2021-12-31', 'V34_3 2021 FULL', universe_filter=UNIV)
pd.DataFrame(trades).to_csv(Path(__file__).parent / 'trades_intraday.csv', index=False)
'''

with open(OUT_DIR / "run_V34_3_2021_FULL.py", "w", encoding="utf-8") as f:
    f.write(run_script_content)
print("Saved run_V34_3_2021_FULL.py")

# 2. Copy Pine Script
src_pine = ROOT / "strategies" / "V34_3_2023_FULL" / "V34_3_2023_FULL.pine"
dst_pine = OUT_DIR / "V34_3_2021_FULL.pine"
shutil.copy2(src_pine, dst_pine)
print("Saved V34_3_2021_FULL.pine")

# 3. Copy official trades_swing.csv schema
src_swing = ROOT / "strategies" / "V34_3_2023_FULL" / "trades_swing.csv"
dst_swing = OUT_DIR / "trades_swing.csv"
shutil.copy2(src_swing, dst_swing)
print("Saved trades_swing.csv")

# 4. Execute simulation to generate trades_intraday.csv
spec = importlib.util.spec_from_file_location('mod_v34_3', str(ROOT / 'strategies/V34.3/run_v34_3.py'))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

p2021 = pickle.load(open(ROOT / 'data/2021-22/universe_30_15m.pkl', 'rb'))
UNIV = {
    'CENTRALBK', 'DELTACORP', 'GSFC', 'IDEA', 'IOB', 'JYOTHYLAB', 'NBCC', 'NCC',
    'NIACL', 'NMDC', 'PNB', 'RBLBANK', 'RCF', 'RENUKA', 'RITES', 'SUZLON',
    'SWSOLAR', 'TATASTEEL', 'TTML', 'ZEEL'
}
cap, trades = mod.run_simulation(p2021, '2021-01-01', '2021-12-31', 'V34_3 2021 FULL', universe_filter=UNIV)

tdf = pd.DataFrame(trades)
csv_path = OUT_DIR / 'trades_intraday.csv'
tdf.to_csv(csv_path, index=False)
print(f"Saved trades_intraday.csv with {len(tdf)} trades.")

# 5. Compute metrics for results.txt
wins = tdf[tdf["net_pnl"] > 0]
losses = tdf[tdf["net_pnl"] < 0]
wr = len(wins) / len(tdf) * 100.0 if len(tdf) > 0 else 0.0
net = tdf["net_pnl"].sum()
net_win = wins["net_pnl"].sum() if not wins.empty else 0.0
net_loss = abs(losses["net_pnl"].sum()) if not losses.empty else 0.0
pf = round(net_win / net_loss, 2) if net_loss > 0 else 999.0
avg_win = round(wins["net_pnl"].mean(), 2) if not wins.empty else 0.0
avg_loss = round(losses["net_pnl"].mean(), 2) if not losses.empty else 0.0
final_cap = 500.0 + net

results_content = f"""Trades: {len(tdf)}
Wins: {len(wins)}
Losses: {len(losses)}
Win Rate: {wr:.2f}%
Net PnL: Rs {net:.2f}
Profit Factor: {pf:.2f}
Avg Win: Rs {avg_win:.2f}
Avg Loss: Rs {avg_loss:.2f}
Final Wallet: Rs {final_cap:.2f}
"""

with open(OUT_DIR / "results.txt", "w", encoding="utf-8") as f:
    f.write(results_content)
print("Saved results.txt:")
print(results_content)
