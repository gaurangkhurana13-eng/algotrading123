import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
df = pd.read_csv(ROOT_DIR / "strategies" / "V17" / "trades_intraday.csv")

# Reconstruct realized equity at entry
net = df["net_pnl"].to_numpy(float)
loss = np.maximum(-net, 0.0)
sh = df["shares"].to_numpy(float)
e = df["entry_price"].to_numpy(float)
sl = df["sl"].to_numpy(float)
planned = sh * np.abs(e - sl)

# Check against lifecycle formula
# For each trade, calculate E_k from closed trades
ent = pd.to_datetime(df["entry_time"]).to_numpy("datetime64[ns]").astype("int64")
ex = pd.to_datetime(df["exit_time"]).to_numpy("datetime64[ns]").astype("int64")

CAPITAL = 500.0
n = len(df)
order = np.argsort(ex, kind="stable")
ex_s = ex[order]
cum = CAPITAL + np.cumsum(net[order])
hwm_run = np.maximum.accumulate(np.maximum(cum, CAPITAL))
pos = np.searchsorted(ex_s, ent, side="left")
prev = np.maximum(pos - 1, 0)
E = np.where(pos > 0, cum[prev], CAPITAL)
HWM = np.where(pos > 0, hwm_run[prev], CAPITAL)
dd = 1.0 - E / HWM
thr = np.where(dd > 0.16, 0.50, 1.0)

# Lifecycle Rcap
base_rcap = np.where(E <= 2500.0, 0.090 * E, np.minimum(210.0 + 0.040 * (E - 2500.0), 260.0))
Rcap = base_rcap * thr

loss_breaches = loss > (Rcap + 0.50)
plan_breaches = planned > (Rcap + 0.50)

print(f"Total Trades: {n}")
print(f"Loss Breaches against Lifecycle Rcap: {int(loss_breaches.sum())}")
if loss_breaches.any():
    for i in np.where(loss_breaches)[0]:
        print(f"  Row {i}: E_k={E[i]:.1f}, loss={loss[i]:.2f}, Rcap={Rcap[i]:.2f}, thr={thr[i]}")

print(f"Plan Breaches against Lifecycle Rcap: {int(plan_breaches.sum())}")
if plan_breaches.any():
    for i in np.where(plan_breaches)[0]:
        print(f"  Row {i}: E_k={E[i]:.1f}, planned={planned[i]:.2f}, Rcap={Rcap[i]:.2f}, thr={thr[i]}")
