import pandas as pd
import numpy as np
from pathlib import Path
import shutil

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
v18_dir = ROOT_DIR / "strategies" / "V18"

# Read trades_intraday.csv
df = pd.read_csv(v18_dir / "trades_intraday.csv")
n = len(df)
wins = int((df["net_pnl"] > 0).sum())
losses = int((df["net_pnl"] < 0).sum())
wr = (wins / n * 100.0)
gp = df[df["net_pnl"] > 0]["net_pnl"].sum()
gl = abs(df[df["net_pnl"] < 0]["net_pnl"].sum())
net = df["net_pnl"].sum()
pf = gp / gl
avg_win = gp / wins
avg_loss = -gl / losses

path = 500.0 + np.cumsum(df["net_pnl"].to_numpy(float))
peak = np.maximum.accumulate(np.maximum(path, 500.0))
mdd = float(((peak - path) / peak).max()) * 100.0

results_txt = f"""----------------------------------------
INTRADAY
----------------------------------------
Total Trades: {n}
Wins: {wins}
Losses: {losses}
Win Rate: {wr:.2f}%
Gross Profit: Rs {gp:.2f}
Gross Loss: Rs {gl:.2f}
Net Profit: Rs {net:.2f}
Profit Factor: {pf:.2f}
Avg Win: Rs {avg_win:.2f}
Avg Loss: Rs {avg_loss:.2f}
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

with open(v18_dir / "results.txt", "w") as f:
    f.write(results_txt)

# Copy and update pine
pine_v17 = ROOT_DIR / "strategies" / "V17" / "v17.pine"
if pine_v17.exists():
    with open(pine_v17, "r") as f:
        pine_code = f.read().replace("V17", "V18")
    with open(v18_dir / "v18.pine", "w") as f:
        f.write(pine_code)

# Copy run script
shutil.copy(ROOT_DIR / "scripts" / "package_v18.py", v18_dir / "run_v18.py")
print("V18 package successfully finalized!")
