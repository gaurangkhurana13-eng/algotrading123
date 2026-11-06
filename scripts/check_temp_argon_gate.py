import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
import strategy_evaluate_gate_v2 as gate

# Run gate directly on the DataFrame
from test_argon_exact_reconciled import df_res

# Create temporary folder to audit
temp_dir = ROOT_DIR / "strategies" / "TEMP_ARGON"
temp_dir.mkdir(parents=True, exist_ok=True)
df_res.to_csv(temp_dir / "trades_intraday.csv", index=False)
pd.DataFrame(columns=df_res.columns).to_csv(temp_dir / "trades_swing.csv", index=False)

# Results.txt
m = gate.metrics(df_res)
path = 500.0 + np.cumsum(df_res["net_pnl"].to_numpy(float))
peak = np.maximum.accumulate(np.maximum(path, 500.0))
mdd = float(((peak - path) / peak).max()) * 100.0 if len(path) else 0.0

with open(temp_dir / "results.txt", "w") as f:
    f.write(f"""----------------------------------------
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
""")

with open(ROOT_DIR / "strategies" / "V16" / "v16.pine", "r") as f:
    pine = f.read()
with open(temp_dir / "temp_argon.pine", "w") as f:
    f.write(pine)

with open(ROOT_DIR / "scripts" / "test_argon_exact_reconciled.py", "r") as f:
    run_s = f.read()
with open(temp_dir / "run_temp_argon.py", "w") as f:
    f.write(run_s)

print("Temporary strategy folder created for Gate V2 audit.")
