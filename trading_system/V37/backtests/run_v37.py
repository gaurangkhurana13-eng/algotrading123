"""
Execution pipeline for Bhavishya V37:
High-Conviction ML Stacking Ensemble with Noise-Immune Tiered Breathing Stops & Chandelier Runner Ratchet.
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

from trading_system.V37.src.strategy import build_dataset, run_walk_forward_pipeline
from trading_system.V37.src.backtester import execute_backtest, run_global_monte_carlo

def generate_performance_chart(df_trades: pd.DataFrame, initial_capital: float, out_path: str):
    if len(df_trades) == 0:
        return

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor("#0f172a")

    for row in axes:
        for ax in row:
            ax.set_facecolor("#1e293b")
            ax.tick_params(colors="#cbd5e1", labelsize=9)
            ax.xaxis.label.set_color("#cbd5e1")
            ax.yaxis.label.set_color("#cbd5e1")
            ax.title.set_color("#f8fafc")
            for spine in ax.spines.values():
                spine.set_color("#334155")

    # Panel 1: Cumulative Equity & Drawdown
    ax1 = axes[0, 0]
    equity = [initial_capital] + list(df_trades["capital"].values)
    trade_nums = list(range(len(equity)))
    ax1.plot(trade_nums, equity, color="#38bdf8", linewidth=2.5, label="Portfolio Equity (Rs)")
    peak = np.maximum.accumulate(equity)
    ax1.plot(trade_nums, peak, color="#94a3b8", linestyle="--", alpha=0.6, label="High Watermark")
    ax1.fill_between(trade_nums, equity, peak, color="#f43f5e", alpha=0.18, label="Drawdown Area")
    ax1.set_title("Bhavishya V37 — Cumulative Equity & Drawdown (Rs 500 Base)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Trade Number")
    ax1.set_ylabel("Capital (Rs)")
    ax1.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")

    # Panel 2: Trade-by-Trade Net PnL Bar Chart
    ax2 = axes[0, 1]
    pnls = df_trades["net_pnl"].values
    colors = ["#10b981" if p > 0 else "#f43f5e" for p in pnls]
    ax2.bar(range(1, len(pnls) + 1), pnls, color=colors, alpha=0.85, width=0.7)
    ax2.axhline(0, color="#94a3b8", linewidth=1.0)
    ax2.axhline(-15.50, color="#fbbf24", linestyle=":", linewidth=1.5, label="Max Loss Cap (-Rs 15.50)")
    ax2.set_title("Trade-by-Trade Net PnL Distribution (After Dhan MIS Charges)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Trade Index")
    ax2.set_ylabel("Net PnL (Rs)")
    ax2.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")

    # Panel 3: Tier-Wise Performance Breakdown
    ax3 = axes[1, 0]
    if "tier" in df_trades.columns:
        tier_pnl = df_trades.groupby("tier")["net_pnl"].sum()
        tier_names = [t.split(" ")[0] for t in tier_pnl.index]
        bars = ax3.bar(tier_names, tier_pnl.values, color=["#38bdf8", "#818cf8", "#c084fc"], width=0.55)
        for b in bars:
            val = b.get_height()
            ax3.text(b.get_x() + b.get_width() / 2, val + (1.5 if val >= 0 else -3.5), f"Rs {val:.1f}",
                     ha="center", va="bottom" if val >= 0 else "top", color="#f8fafc", fontweight="bold", fontsize=9)
        ax3.set_title("Net PnL Contribution by Volatility Tier", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Total Net PnL (Rs)")

    # Panel 4: Exit Reason Breakdown
    ax4 = axes[1, 1]
    if "exit_reason" in df_trades.columns:
        exit_counts = df_trades["exit_reason"].value_counts().head(6)
        short_names = [r.replace("STAGNATION_", "STAG_").replace("_HIT", "").replace("_FLOOR", "") for r in exit_counts.index]
        y_pos = np.arange(len(short_names))
        ax4.barh(y_pos, exit_counts.values, color="#6366f1", alpha=0.85, height=0.6)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(short_names, fontsize=8)
        ax4.invert_yaxis()
        ax4.set_title("Top Exit Mechanism Frequencies", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Number of Exits")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()

def main():
    print("=" * 80)
    print("BHAVISHYA V37 EXECUTION PIPELINE — HIGH-CONVICTION CHANDELIER RUNNER ENGINE")
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

    print(f"\nStep 1: Processing Master Feature & Label Matrix from {cached_data.name}...")
    df_pool = build_dataset(str(cached_data), config)
    print(f"Master Matrix Assembled: {len(df_pool):,} samples across stocks.")

    print("\nStep 2: Running 5-Fold Chronological Purged Walk-Forward Pipeline...")
    oos_df = run_walk_forward_pipeline(df_pool, config)
    print(f"Walk-Forward Complete: {len(oos_df):,} calibrated out-of-sample bars ready.")

    print("\nStep 3: Executing Backtest with Noise-Immune Tiered Stops & Chandelier Ratchet...")
    results = execute_backtest(oos_df, df_pool, config, out_dirs)

    print("\nStep 4: Running 10,000-Iteration Global Portfolio Monte Carlo Simulation...")
    df_trades = results["df_trades"]
    mc_res = run_global_monte_carlo(df_trades, config, out_dirs)

    # Generate Chart
    chart_path = os.path.join(out_dirs["reports"], "v37_performance.png")
    generate_performance_chart(df_trades, config.get("capital", 500.0), chart_path)
    print(f"Performance chart generated at {chart_path}")

    # Generate Markdown Report
    report_path = os.path.join(out_dirs["reports"], "V37_FINAL_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# BHAVISHYA V37 MASTER PERFORMANCE REPORT\n\n")
        f.write(f"- **System Name**: {config.get('system_name', 'Bhavishya V37')}\n")
        f.write(f"- **Universe**: 78 NSE Stocks (15m bars, 60 trading days, 4 OOS Walk-Forward Folds)\n")
        f.write(f"- **Initial Capital**: Rs {config.get('capital', 500.0):.2f} (5x MIS Leverage, Rs 2,450 Notional Cap)\n\n")
        f.write("## 1. Executive Summary & Core Metrics\n\n")
        f.write("| Metric | Value |\n| :--- | :--- |\n")
        f.write(f"| **Total Candidates Evaluated** | {results['n_candidates']} |\n")
        f.write(f"| **Total Approved Trades** | {results['n_approved']} |\n")
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
        f.write(f"| **Monte Carlo Expected Drawdown** | {mc_res['expected_dd']}% |\n")
        f.write(f"| **Monte Carlo Median Capital** | Rs {mc_res['median_capital']} |\n\n")

        if len(df_trades) > 0 and "tier" in df_trades.columns:
            f.write("## 2. Volatility Tier Performance Breakdown\n\n")
            f.write("| Tier | Trades | Wins | Losses | Win Rate | Net PnL | Avg Win | Avg Loss |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for tier, t_df in df_trades.groupby("tier"):
                t_w = int(t_df["winner"].sum())
                t_tot = len(t_df)
                t_l = t_tot - t_w
                t_wr = round(t_w / t_tot * 100, 1)
                t_pnl = round(float(t_df["net_pnl"].sum()), 2)
                t_aw = round(float(t_df[t_df["winner"]]["net_pnl"].mean()), 2) if t_w > 0 else 0
                t_al = round(float(t_df[~t_df["winner"]]["net_pnl"].mean()), 2) if t_l > 0 else 0
                f.write(f"| `{tier}` | {t_tot} | {t_w} | {t_l} | {t_wr}% | Rs {t_pnl} | Rs {t_aw} | Rs {t_al} |\n")
            f.write("\n")

        if len(df_trades) > 0 and "exit_reason" in df_trades.columns:
            f.write("## 3. Exit Mechanism Breakdown\n\n")
            f.write("| Exit Reason | Trades | Wins | Losses | Win Rate | Net PnL |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for reason, r_df in df_trades.groupby("exit_reason"):
                r_w = int(r_df["winner"].sum())
                r_tot = len(r_df)
                r_wr = round(r_w / r_tot * 100, 1)
                r_pnl = round(float(r_df["net_pnl"].sum()), 2)
                f.write(f"| `{reason}` | {r_tot} | {r_w} | {r_tot - r_w} | {r_wr}% | Rs {r_pnl} |\n")

    print(f"\n--- V37 PERFORMANCE SUMMARY ---")
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

    if len(df_trades) > 0 and "tier" in df_trades.columns:
        print("\n--- TIER-WISE PERFORMANCE BREAKDOWN ---")
        for tier, t_df in df_trades.groupby("tier"):
            t_wins = int(t_df["winner"].sum())
            t_total = len(t_df)
            t_wr = round((t_wins / t_total) * 100, 2)
            t_pnl = round(float(t_df["net_pnl"].sum()), 2)
            t_avg_w = round(float(t_df[t_df["winner"]]["net_pnl"].mean()), 2) if t_wins > 0 else 0
            t_avg_l = round(float(t_df[~t_df["winner"]]["net_pnl"].mean()), 2) if t_total - t_wins > 0 else 0
            print(f"  * {tier}: {t_total} trades | {t_wins}W / {t_total - t_wins}L ({t_wr}% WR) | Net PnL: Rs {t_pnl} | AvgWin: Rs {t_avg_w} | AvgLoss: Rs {t_avg_l}")

    if len(df_trades) > 0 and "exit_reason" in df_trades.columns:
        print("\n--- EXIT REASON BREAKDOWN ---")
        for reason, r_df in df_trades.groupby("exit_reason"):
            r_wins = int(r_df["winner"].sum())
            r_total = len(r_df)
            r_wr = round((r_wins / r_total) * 100, 2)
            r_pnl = round(float(r_df["net_pnl"].sum()), 2)
            print(f"  * {reason}: {r_total} trades | {r_wins}W / {r_total - r_wins}L ({r_wr}% WR) | Net PnL: Rs {r_pnl}")

if __name__ == "__main__":
    main()
