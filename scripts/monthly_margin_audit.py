import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
df_trades = pd.read_csv(ROOT_DIR / "strategies" / "V19" / "trades_intraday.csv")

df_trades["entry_time"] = pd.to_datetime(df_trades["entry_time"])
df_trades["exit_time"] = pd.to_datetime(df_trades["exit_time"])
df_trades["month_year"] = df_trades["entry_time"].dt.to_period("M")

# Calculate running equity before each trade
df_trades["equity_before"] = 500.0 + df_trades["net_pnl"].shift(1).fillna(0).cumsum()
df_trades["notional_deployed"] = df_trades["entry_price"] * df_trades["shares"]
df_trades["margin_deployed"] = df_trades["notional_deployed"] / 5.0
df_trades["max_avail_margin"] = df_trades["equity_before"] * 0.95
df_trades["idle_margin"] = (df_trades["max_avail_margin"] - df_trades["margin_deployed"]).clip(lower=0)
df_trades["utilization_pct"] = (df_trades["margin_deployed"] / (df_trades["max_avail_margin"] + 1e-9)) * 100.0
df_trades["idle_pct"] = 100.0 - df_trades["utilization_pct"]

# Group by month
months = []
for m, grp in df_trades.groupby("month_year"):
    start_eq = grp["equity_before"].iloc[0]
    end_eq = grp["equity_before"].iloc[-1] + grp["net_pnl"].iloc[-1]
    net_m = grp["net_pnl"].sum()
    n_trades = len(grp)
    wins = (grp["net_pnl"] > 0).sum()
    wr = (wins / n_trades) * 100.0
    avg_eq = grp["equity_before"].mean()
    avg_margin_used = grp["margin_deployed"].mean()
    avg_idle_margin = grp["idle_margin"].mean()
    avg_util = grp["utilization_pct"].mean()
    avg_idle_pct = grp["idle_pct"].mean()
    peak_notional = grp["notional_deployed"].max()
    peak_margin_used = grp["margin_deployed"].max()
    
    months.append({
        "Month": str(m),
        "Trades": n_trades,
        "Start Equity": round(start_eq, 2),
        "End Equity": round(end_eq, 2),
        "Net Profit": round(net_m, 2),
        "Win Rate (%)": round(wr, 1),
        "Avg Margin Used (Rs)": round(avg_margin_used, 2),
        "Avg Idle Cash (Rs)": round(avg_idle_margin, 2),
        "Avg Buying Power (Rs)": round(avg_eq * 5.0, 2),
        "Avg Utilization (%)": round(avg_util, 1),
        "Avg Idle Margin (%)": round(avg_idle_pct, 1),
        "Peak Notional (Rs)": round(peak_notional, 2)
    })

df_monthly = pd.DataFrame(months)
print(df_monthly.to_string(index=False))
