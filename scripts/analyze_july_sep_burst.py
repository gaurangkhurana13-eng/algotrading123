import pandas as pd
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
df_trades = pd.read_csv(ROOT_DIR / "strategies" / "V19" / "trades_intraday.csv")
df_trades["entry_time"] = pd.to_datetime(df_trades["entry_time"])
df_trades["month_year"] = df_trades["entry_time"].dt.to_period("M")

# Filter for July, August, September
burst_months = ["2026-07", "2026-08", "2026-09"]
df_burst = df_trades[df_trades["month_year"].astype(str).isin(burst_months)]

print("=== TOTAL BURST PERIOD (JULY - SEPT) SUMMARY ===")
print(f"Total Trades: {len(df_burst)}")
print(f"Total Net PnL: Rs {df_burst['net_pnl'].sum():.2f}")
print(f"Win Rate: {(df_burst['net_pnl'] > 0).mean()*100:.2f}%")
print()

print("=== STOCK-BY-STOCK BREAKDOWN IN BURST PERIOD ===")
stock_summary = df_burst.groupby("symbol").agg(
    trades=("net_pnl", "count"),
    wins=("net_pnl", lambda x: (x > 0).sum()),
    win_rate=("net_pnl", lambda x: (x > 0).mean() * 100),
    net_pnl=("net_pnl", "sum"),
    avg_win=("net_pnl", lambda x: x[x > 0].mean() if (x > 0).any() else 0),
    avg_loss=("net_pnl", lambda x: x[x < 0].mean() if (x < 0).any() else 0),
    max_win=("net_pnl", "max"),
    max_loss=("net_pnl", "min")
).reset_index().sort_values("net_pnl", ascending=False)

print(stock_summary.to_string(index=False))
print()

# Month by Month per Stock
print("=== MONTH BY MONTH NET PNL PER STOCK ===")
pivot = df_burst.pivot_table(index="symbol", columns="month_year", values="net_pnl", aggfunc="sum", fill_value=0)
pivot["Total"] = pivot.sum(axis=1)
print(pivot.sort_values("Total", ascending=False).to_string())

# Also check date range of the underlying data
print("\n=== DATASET EXACT TIMESTAMPS ===")
print("First trade:", df_trades['entry_time'].min())
print("Last trade:", df_trades['entry_time'].max())
