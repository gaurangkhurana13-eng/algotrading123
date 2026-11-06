import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
import strategy_evaluate_gate_v2 as gate
from test_v17_bifurcated import df_v17, m, path, peak, mdd

V17_DIR = ROOT_DIR / "strategies" / "V17"
V17_DIR.mkdir(parents=True, exist_ok=True)

# 1. Save trades_intraday.csv
df_v17.to_csv(V17_DIR / "trades_intraday.csv", index=False)

# 2. Save empty trades_swing.csv
df_swing = pd.DataFrame(columns=[
    "symbol", "direction", "entry_time", "exit_time",
    "entry_price", "exit_price", "shares", "sl", "gross_pnl", "charges", "net_pnl", "exit_reason"
])
df_swing.to_csv(V17_DIR / "trades_swing.csv", index=False)

# 3. Save results.txt
results_txt = f"""----------------------------------------
INTRADAY
----------------------------------------
Total Trades: {m['n']}
Wins: {m['wins']}
Losses: {m['losses']}
Win Rate: {m['win_rate']:.2f}%
Gross Profit: Rs {m['gross_profit']:.2f}
Gross Loss: Rs {m['gross_loss']:.2f}
Net Profit: Rs {m['net']:.2f}
Profit Factor: {m['pf']:.2f}
Avg Win: Rs {m['avg_win']:.2f}
Avg Loss: Rs {m['avg_loss']:.2f}
Max Drawdown: {mdd:.2f}%

----------------------------------------
SWING
----------------------------------------
Total Trades: 0
Wins: 0
Losses: 0
Win Rate: 0.00%
Gross Profit: Rs 0.00
Gross Loss: Rs 0.00
Net Profit: Rs 0.00
Profit Factor: 0.00
Avg Win: Rs 0.00
Avg Loss: Rs 0.00
Max Drawdown: 0.00%
"""
with open(V17_DIR / "results.txt", "w") as f:
    f.write(results_txt)

# 4. Save v17.pine
with open(ROOT_DIR / "strategies" / "V16" / "v16.pine", "r") as f:
    pine = f.read()
with open(V17_DIR / "v17.pine", "w") as f:
    f.write(pine.replace("V16", "V17"))

# 5. Save run_v17.py
with open(ROOT_DIR / "scripts" / "test_v17_bifurcated.py", "r") as f:
    run_s = f.read()
with open(V17_DIR / "run_v17.py", "w") as f:
    f.write(run_s)

print("V17 Artefacts packaged successfully in strategies/V17!")
