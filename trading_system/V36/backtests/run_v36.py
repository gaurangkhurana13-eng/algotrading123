"""
Execution pipeline for Bhavishya V36:
Claude Sonnet High-Conviction Breakout & Microstructure Architecture.
Strictly lookahead-free, real Dhan MIS transaction costs, 100% causal bar-by-bar execution.
"""

import os
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root in sys.path
current_dir = Path(__file__).resolve().parent
version_dir = current_dir.parent
project_root = version_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from trading_system.V36.src.strategy import build_dataset, run_walk_forward_pipeline
from trading_system.V36.src.backtester import execute_backtest, run_global_monte_carlo

def main():
    print("=" * 80)
    print("BHAVISHYA V36 EXECUTION PIPELINE — CLAUDE SONNET HIGH-CONVICTION ARCHITECTURE")
    print("=" * 80)

    config_path = version_dir / "config" / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    out_dirs = {
        "trades": str(version_dir / "trades"),
        "monte_carlo": str(version_dir / "monte_carlo"),
        "reports": str(version_dir / "reports"),
        "logs": str(version_dir / "logs")
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    cached_data = project_root / "data" / "universe_78_15m.pkl"
    if not cached_data.exists():
        raise FileNotFoundError(f"Universe data not found at {cached_data}")

    print(f"\nStep 1: Processing Master Feature Matrix with Nifty 50 Index Gate from {cached_data.name}...")
    df_pool = build_dataset(str(cached_data), config)
    print(f"Master Matrix Assembled: {len(df_pool):,} samples across stocks.")

    print("\nStep 2: Running 5-Fold Chronological Purged Walk-Forward with 7-Factor Conviction Signals...")
    oos_df = run_walk_forward_pipeline(df_pool, config)
    print(f"Walk-Forward Complete: {len(oos_df):,} calibrated out-of-sample bars ready.")

    print("\nStep 3: Executing Backtest with Risk-Utilization Gate, Chandelier Ratchet & Partial TP...")
    results = execute_backtest(oos_df, df_pool, config, out_dirs)

    print("\nStep 4: Running 10,000-Iteration Global Portfolio Monte Carlo Simulation...")
    df_trades = results["df_trades"]
    mc_res = run_global_monte_carlo(df_trades, config, out_dirs)

    print(f"\n--- V36 PERFORMANCE SUMMARY ---")
    print(f"Total Candidates Evaluated:            {results['n_candidates']}")
    print(f"Approved Trades:                       {results['n_approved']}")
    print(f"Rejected Candidates:                   {results['n_rejected']}")
    print(f"Total Completed Trades:                {results['n_trades']}")
    print(f"Wins / Losses:                         {results['wins']} Wins / {results['losses']} Losses")
    print(f"Win Rate:                              {results['win_rate']}%")
    print(f"Average Win:                           Rs {results['avg_win']}")
    print(f"Average Loss:                          Rs {results['avg_loss']}")
    print(f"Payoff Ratio (Win/Loss):               {results['payoff_ratio']}:1")
    print(f"Profit Factor:                         {results['profit_factor']}")
    print(f"Total Dhan Brokerage & Taxes:          Rs {results['total_charges']}")
    print(f"Net Profit / Loss:                     Rs {results['net_pnl']}")
    print(f"Final Capital:                         Rs {results['final_capital']} (ROI: {results['roi']}%)")
    print(f"Sharpe Ratio:                          {results['sharpe']}")
    print(f"Sortino Ratio:                         {results['sortino']}")
    print(f"Monte Carlo 50% Account Ruin Risk:     {mc_res['ruin_prob']}%")
    print(f"Monte Carlo Expected Drawdown:         {mc_res['expected_dd']}%")
    print(f"Monte Carlo Median Capital:            Rs {mc_res['median_capital']}")

    if len(df_trades) > 0 and "exit_reason" in df_trades.columns:
        print("\n--- EXIT REASON BREAKDOWN ---")
        for reason, r_df in df_trades.groupby("exit_reason"):
            r_wins = int(r_df["winner"].sum())
            r_total = len(r_df)
            r_wr = round((r_wins / r_total) * 100, 2)
            r_pnl = round(float(r_df["net_pnl"].sum()), 2)
            print(f"  * {reason}: {r_total} trades | {r_wins}W / {r_total - r_wins}L ({r_wr}% WR) | Net PnL: Rs {r_pnl}")

    print("\nStep 5: Generating Performance Visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if len(df_trades) > 0:
        eq = np.array([500.0] + df_trades["capital"].tolist())
        axes[0, 0].plot(eq, marker="o", color="#00ff88", linewidth=2, markersize=4)
        axes[0, 0].axhline(500.0, color="gray", linestyle="--", alpha=0.6)
        axes[0, 0].set_title(f"V36 Equity Curve (Capital: Rs 500 -> Rs {results['final_capital']})", fontsize=12, fontweight="bold")
        axes[0, 0].set_xlabel("Trade Number")
        axes[0, 0].set_ylabel("Account Balance (Rs)")
        axes[0, 0].grid(True, alpha=0.3)

        colors = ["#00ff88" if p > 0 else "#ff4444" for p in df_trades["net_pnl"]]
        axes[0, 1].bar(range(1, len(df_trades) + 1), df_trades["net_pnl"], color=colors, alpha=0.85)
        axes[0, 1].axhline(0, color="black", linewidth=1)
        axes[0, 1].set_title(f"Net PnL per Trade (Avg Win: Rs {results['avg_win']}, Avg Loss: Rs {results['avg_loss']})", fontsize=12, fontweight="bold")
        axes[0, 1].set_xlabel("Trade Number")
        axes[0, 1].set_ylabel("Net PnL (Rs)")
        axes[0, 1].grid(True, alpha=0.3)

        reasons_summary = df_trades["exit_reason"].value_counts()
        axes[1, 0].barh(reasons_summary.index, reasons_summary.values, color="#4da6ff")
        axes[1, 0].set_title("Trades by Exit Mechanism", fontsize=12, fontweight="bold")
        axes[1, 0].set_xlabel("Trade Count")
        axes[1, 0].grid(True, alpha=0.3)

        cum_charges = np.cumsum(df_trades["charges"].values)
        axes[1, 1].plot(cum_charges, marker="s", color="#ffaa00", linewidth=2)
        axes[1, 1].set_title(f"Cumulative Dhan MIS Frictions (Total: Rs {results['total_charges']})", fontsize=12, fontweight="bold")
        axes[1, 1].set_xlabel("Trade Number")
        axes[1, 1].set_ylabel("Charges (Rs)")
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, "No Trades Executed", ha="center")

    plt.tight_layout()
    chart_path = os.path.join(out_dirs["reports"], "v36_performance.png")
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"Chart saved to {chart_path}")

    # Write Markdown Final Report
    report_path = os.path.join(out_dirs["reports"], "V36_FINAL_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# BHAVISHYA V36 MASTER PERFORMANCE REPORT\n\n")
        f.write(f"- **System Name**: {config['system_name']}\n")
        f.write(f"- **Universe**: 78 NSE Stocks (15m bars, 60 trading days)\n")
        f.write(f"- **Initial Capital**: Rs {config['capital']:.2f} (5x MIS Leverage, Rs 2,450 Notional Cap)\n\n")
        f.write("## 1. Executive Summary & Core Metrics\n\n")
        f.write("| Metric | Value |\n")
        f.write("| :--- | :--- |\n")
        f.write(f"| **Total Candidates Evaluated** | {results['n_candidates']} |\n")
        f.write(f"| **Total Approved Trades** | {results['n_trades']} |\n")
        f.write(f"| **Winning Trades** | {results['wins']} |\n")
        f.write(f"| **Losing Trades** | {results['losses']} |\n")
        f.write(f"| **Win Rate** | **{results['win_rate']}%** |\n")
        f.write(f"| **Average Win** | **Rs {results['avg_win']}** |\n")
        f.write(f"| **Average Loss** | **Rs {results['avg_loss']}** |\n")
        f.write(f"| **Payoff Ratio (Win/Loss)** | **{results['payoff_ratio']}:1** |\n")
        f.write(f"| **Profit Factor** | **{results['profit_factor']}** |\n")
        f.write(f"| **Total Dhan Charges** | Rs {results['total_charges']} |\n")
        f.write(f"| **Net PnL** | **Rs {results['net_pnl']}** |\n")
        f.write(f"| **Final Capital** | **Rs {results['final_capital']}** |\n")
        f.write(f"| **Net Return on Capital (ROI)** | **+{results['roi']}%** |\n")
        f.write(f"| **Sharpe Ratio (Annualized)** | {results['sharpe']} |\n")
        f.write(f"| **Sortino Ratio (Annualized)** | {results['sortino']} |\n")
        f.write(f"| **Monte Carlo Ruin Probability** | {mc_res['ruin_prob']}% |\n")
        f.write(f"| **Monte Carlo Expected Drawdown** | {mc_res['expected_dd']}% |\n\n")

        f.write("## 2. Exit Mechanism Breakdown\n\n")
        f.write("| Exit Reason | Trades | Wins | Losses | Win Rate | Net PnL |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
        if len(df_trades) > 0 and "exit_reason" in df_trades.columns:
            for reason, r_df in df_trades.groupby("exit_reason"):
                r_wins = int(r_df["winner"].sum())
                r_tot = len(r_df)
                r_wr = round((r_wins / r_tot) * 100, 2)
                r_pnl = round(float(r_df["net_pnl"].sum()), 2)
                f.write(f"| `{reason}` | {r_tot} | {r_wins} | {r_tot - r_wins} | {r_wr}% | Rs {r_pnl} |\n")
        f.write("\n")

    print(f"Report written to {report_path}")
    print("\nV36 Pipeline Execution Complete.")

if __name__ == "__main__":
    main()
