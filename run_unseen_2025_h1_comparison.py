import os
import sys
import pickle
import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")

# Load V34.3
p_v34 = ROOT / "strategies" / "V34.3" / "run_v34_3.py"
spec34 = importlib.util.spec_from_file_location("v34_3", p_v34)
mod_v34 = importlib.util.module_from_spec(spec34)
spec34.loader.exec_module(mod_v34)

# Load V36
sys.path.insert(0, str(ROOT / "strategies" / "V36"))
import run_v36 as mod_v36

# Load V37
sys.path.insert(0, str(ROOT / "strategies" / "V37"))
import run_v37 as mod_v37

# Load Data
DATA_PATH = ROOT / "data" / "2025-26" / "universe_14_15m.pkl"
with open(DATA_PATH, "rb") as f:
    p2025 = pickle.load(f)

START_DATE = "2025-01-01"
END_DATE = "2025-07-31"
FOLD_LABEL = "2025 H1 Unseen"
UNIV = mod_v37.V29_30_UNIVERSE

def evaluate_version(name, run_fn):
    print(f"\n=======================================================")
    print(f"=== RUNNING {name} on {FOLD_LABEL} ({START_DATE} to {END_DATE}) ===")
    print(f"=======================================================")
    end_cap, trades = run_fn(p2025, START_DATE, END_DATE, f"{name}: {FOLD_LABEL}", universe_filter=UNIV)
    df = pd.DataFrame(trades)
    
    if len(df) == 0:
        print(f"No trades executed for {name}")
        return None
        
    wins = df[df["net_pnl"] > 0]
    losses = df[df["net_pnl"] < 0]
    win_rate = (len(wins) / len(df)) * 100.0 if len(df) > 0 else 0.0
    gross_pnl = df["gross_pnl"].sum()
    charges = df["charges"].sum()
    net_pnl = df["net_pnl"].sum()
    gross_win = wins["net_pnl"].sum()
    gross_loss = abs(losses["net_pnl"].sum())
    pf = (gross_win / gross_loss) if gross_loss > 0 else (999.0 if gross_win > 0 else 0.0)
    
    # Running Equity and Max Drawdown
    equity = [500.0]
    for p in df["net_pnl"]:
        equity.append(equity[-1] + p)
    eq_series = pd.Series(equity)
    hwm = eq_series.cummax()
    dd_pts = hwm - eq_series
    dd_pct = (dd_pts / hwm) * 100.0
    max_dd_pts = dd_pts.max()
    max_dd_pct = dd_pct.max()

    print(f"\n[{name} METRICS SUMMARY]")
    print(f"  Total Trades:     {len(df)}")
    print(f"  Win Rate:         {win_rate:.1f}% ({len(wins)} Wins / {len(losses)} Losses)")
    print(f"  Gross PnL:        Rs {gross_pnl:+.2f}")
    print(f"  Dhan Charges:     Rs {charges:.2f}")
    print(f"  Net PnL:          Rs {net_pnl:+.2f}")
    print(f"  Profit Factor:    {pf:.2f}")
    print(f"  Max Drawdown:     Rs {max_dd_pts:.2f} ({max_dd_pct:.1f}%)")
    print(f"  Start Wallet:     Rs 500.00")
    print(f"  End Wallet:       Rs {end_cap:.2f}")
    
    print("\n[Exit Reason Breakdown]")
    exit_summary = df.groupby("exit_reason").agg(
        count=("net_pnl", "count"),
        net_pnl=("net_pnl", "sum"),
        avg_pnl=("net_pnl", "mean")
    )
    print(exit_summary)
    
    df["month"] = pd.to_datetime(df["entry_time"]).dt.to_period("M")
    print("\n[Monthly Performance Breakdown]")
    monthly_summary = df.groupby("month").agg(
        trades=("net_pnl", "count"),
        wins=("net_pnl", lambda x: (x > 0).sum()),
        losses=("net_pnl", lambda x: (x < 0).sum()),
        net_pnl=("net_pnl", "sum"),
        charges=("charges", "sum")
    )
    monthly_summary["wr%"] = (monthly_summary["wins"] / monthly_summary["trades"]) * 100.0
    print(monthly_summary)
    
    print("\n[Top/Bottom Symbols by Net PnL]")
    sym_summary = df.groupby("symbol").agg(
        trades=("net_pnl", "count"),
        net_pnl=("net_pnl", "sum"),
        wins=("net_pnl", lambda x: (x > 0).sum())
    ).sort_values("net_pnl", ascending=False)
    print(sym_summary)
    
    return {
        "name": name,
        "trades": len(df),
        "wr": win_rate,
        "gross_pnl": gross_pnl,
        "charges": charges,
        "net_pnl": net_pnl,
        "pf": pf,
        "max_dd_pct": max_dd_pct,
        "end_wallet": end_cap,
        "df": df
    }

res_v34 = evaluate_version("V34.3", mod_v34.run_simulation)
res_v36 = evaluate_version("V36", mod_v36.run_simulation)
res_v37 = evaluate_version("V37", mod_v37.run_simulation)

print("\n" + "="*80)
print(f"=== HEAD-TO-HEAD SUMMARY ON UNSEEN B (2025 H1: Jan 1 - Jul 31, 2025) ===")
print("="*80)
summary_table = pd.DataFrame([
    {
        "Version": r["name"],
        "Trades": r["trades"],
        "Win Rate": f"{r['wr']:.1f}%",
        "Gross PnL": f"Rs {r['gross_pnl']:+.2f}",
        "Charges": f"Rs {r['charges']:.2f}",
        "Net PnL": f"Rs {r['net_pnl']:+.2f}",
        "Profit Factor": f"{r['pf']:.2f}",
        "Max DD %": f"{r['max_dd_pct']:.1f}%",
        "End Wallet": f"Rs {r['end_wallet']:.2f}"
    }
    for r in [res_v34, res_v36, res_v37] if r is not None
])
print(summary_table.to_string(index=False))
