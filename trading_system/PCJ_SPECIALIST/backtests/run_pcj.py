"""
Execution Pipeline for Bhavishya PC Jeweller Specialist Intraday CatBoost System.
Evaluates 100% untouched out-of-sample data across expanding walk-forward folds.
Real Dhan MIS charges and slippage included.
"""

import os
import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

current_dir = Path(__file__).resolve().parent
version_dir = current_dir.parent
project_root = version_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from trading_system.PCJ_SPECIALIST.src.model import prepare_pcj_dataset, run_pcj_walk_forward
from trading_system.PCJ_SPECIALIST.src.backtester import execute_pcj_backtest, run_pcj_monte_carlo

def generate_pcj_chart(df_trades: pd.DataFrame, initial_capital: float, out_path: str):
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

    # Panel 1: Cumulative Equity Curve
    ax1 = axes[0, 0]
    equity = [initial_capital] + list(df_trades["capital"].values)
    trade_nums = list(range(len(equity)))
    ax1.plot(trade_nums, equity, color="#38bdf8", linewidth=2.5, label="Portfolio Equity (Rs)")
    peak = np.maximum.accumulate(equity)
    ax1.plot(trade_nums, peak, color="#94a3b8", linestyle="--", alpha=0.6, label="High Watermark")
    ax1.fill_between(trade_nums, equity, peak, color="#f43f5e", alpha=0.18, label="Drawdown Area")
    ax1.set_title("PC Jeweller Specialist — Cumulative Equity (Rs 500 Base, 5x MIS)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Trade Number")
    ax1.set_ylabel("Capital (Rs)")
    ax1.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")

    # Panel 2: Trade-by-Trade Net PnL
    ax2 = axes[0, 1]
    pnls = df_trades["net_pnl"].values
    colors = ["#10b981" if p > 0 else "#f43f5e" for p in pnls]
    ax2.bar(range(1, len(pnls) + 1), pnls, color=colors, alpha=0.85, width=0.7)
    ax2.axhline(0, color="#94a3b8", linewidth=1.0)
    ax2.axhline(-15.50, color="#fbbf24", linestyle=":", linewidth=1.5, label="Max Loss Cap (-Rs 15.50)")
    ax2.set_title("Trade-by-Trade Net PnL (Dhan MIS Charges Deducted)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Trade Index")
    ax2.set_ylabel("Net PnL (Rs)")
    ax2.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")

    # Panel 3: Fold-Wise Performance
    ax3 = axes[1, 0]
    if "fold" in df_trades.columns:
        fold_pnl = df_trades.groupby("fold")["net_pnl"].sum()
        fold_names = [f"Fold {f}" for f in fold_pnl.index]
        bars = ax3.bar(fold_names, fold_pnl.values, color="#818cf8", width=0.55)
        for b in bars:
            val = b.get_height()
            ax3.text(b.get_x() + b.get_width() / 2, val + (1.0 if val >= 0 else -2.5), f"Rs {val:.1f}",
                     ha="center", va="bottom" if val >= 0 else "top", color="#f8fafc", fontweight="bold", fontsize=9)
        ax3.set_title("Net PnL by Chronological Walk-Forward Fold", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Total Net PnL (Rs)")

    # Panel 4: Exit Reason Distribution
    ax4 = axes[1, 1]
    if "exit_reason" in df_trades.columns:
        exit_counts = df_trades["exit_reason"].value_counts().head(6)
        short_names = [r.replace("STAGNATION_", "STAG_").replace("_HIT", "").replace("_FLOOR", "") for r in exit_counts.index]
        y_pos = np.arange(len(short_names))
        ax4.barh(y_pos, exit_counts.values, color="#34d399", alpha=0.85, height=0.6)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(short_names, fontsize=8)
        ax4.invert_yaxis()
        ax4.set_title("Top Exit Mechanisms", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Trade Count")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()

def main():
    print("=" * 80)
    print("BHAVISHYA PC JEWELLER SPECIALIST INTRADAY CATBOOST SYSTEM")
    print("=" * 80)

    config_path = version_dir / "config" / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    out_dirs = {
        "trades": str(version_dir / "trades"),
        "monte_carlo": str(version_dir / "monte_carlo"),
        "reports": str(version_dir / "reports")
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    data_path = project_root / "data" / "pcjeweller_15m_1y.pkl"
    if not data_path.exists():
        raise FileNotFoundError(f"PC Jeweller data not found at {data_path}")

    print(f"\nStep 1: Preparing Master Dataset from {data_path.name}...")
    df_prepared = prepare_pcj_dataset(str(data_path), config)
    print(f"Dataset Prepared: {len(df_prepared):,} samples.")

    print("\nStep 2: Executing Expanding Walk-Forward CatBoost Training (5 Folds)...")
    oos_df = run_pcj_walk_forward(df_prepared, config)
    print(f"Walk-Forward Complete: {len(oos_df):,} out-of-sample bars evaluated.")

    print("\nStep 3: Running Intraday Causal Backtest with Dual-Constraint Sizing & Ratchets...")
    results = execute_pcj_backtest(oos_df, df_prepared, config, out_dirs)

    print("\nStep 4: Running 10,000-Iteration Global Monte Carlo Simulation...")
    df_trades = results["df_trades"]
    mc_res = run_pcj_monte_carlo(df_trades, config, out_dirs)

    chart_path = os.path.join(out_dirs["reports"], "pcj_performance.png")
    generate_pcj_chart(df_trades, config.get("capital", 500.0), chart_path)
    print(f"Performance chart saved to {chart_path}")

    # Generate Markdown Report
    report_path = os.path.join(out_dirs["reports"], "PCJ_FINAL_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# BHAVISHYA PC JEWELLER SPECIALIST SYSTEM — MASTER PERFORMANCE REPORT\n\n")
        f.write(f"- **Asset**: PCJEWELLER (NSE Equity Cash Segment)\n")
        f.write(f"- **Timeframe**: 15-Minute Intraday Bars\n")
        f.write(f"- **Total Dataset Horizon**: 203 Trading Days (~10 Months, Jan 2026 to Oct 2026)\n")
        f.write(f"- **Initial Capital**: Rs {config.get('capital', 500.0):.2f} (5x MIS Leverage, Rs 2,450 Margin Cap)\n")
        f.write(f"- **Model**: Dedicated CatBoostClassifier with Fold-Relative Z-Score Hurdle\n\n")

        f.write("## 1. Executive Summary & Core Metrics\n\n")
        f.write("| Metric | Value |\n| :--- | :--- |\n")
        f.write(f"| **Out-Of-Sample Test Bars** | {len(oos_df):,} bars ({len(set(oos_df.index.date))} trading days) |\n")
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
        f.write(f"| **Monte Carlo 50% Ruin Risk** | {mc_res['ruin_prob']}% |\n")
        f.write(f"| **Monte Carlo Expected Drawdown** | {mc_res['expected_dd']}% |\n")
        f.write(f"| **Monte Carlo Median Capital** | Rs {mc_res['median_capital']} |\n\n")

        if len(df_trades) > 0 and "fold" in df_trades.columns:
            f.write("## 2. Walk-Forward Fold Performance Breakdown\n\n")
            f.write("| Fold | Trades | Wins | Losses | Win Rate | Net PnL | Avg Win | Avg Loss |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for fold_id, f_df in df_trades.groupby("fold"):
                f_w = int(f_df["winner"].sum())
                f_tot = len(f_df)
                f_l = f_tot - f_w
                f_wr = round(f_w / f_tot * 100, 1)
                f_pnl = round(float(f_df["net_pnl"].sum()), 2)
                f_aw = round(float(f_df[f_df["winner"]]["net_pnl"].mean()), 2) if f_w > 0 else 0
                f_al = round(float(f_df[~f_df["winner"]]["net_pnl"].mean()), 2) if f_l > 0 else 0
                f.write(f"| `Fold {fold_id}` | {f_tot} | {f_w} | {f_l} | {f_wr}% | Rs {f_pnl} | Rs {f_aw} | Rs {f_al} |\n")
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

    print("\n--- PERFORMANCE SUMMARY ---")
    print(f"Total Approved Trades:                 {results['n_approved']}")
    print(f"Wins / Losses:                         {results['wins']} Wins / {results['losses']} Losses")
    print(f"Win Rate:                              {results['win_rate']}%")
    print(f"Average Win:                           Rs {results['avg_win']}")
    print(f"Average Loss:                          Rs {results['avg_loss']}")
    print(f"Payoff Ratio (Win/Loss):               {results['payoff_ratio']}:1")
    print(f"Profit Factor:                         {results['profit_factor']}")
    print(f"Total Dhan Charges:                    Rs {results['total_charges']}")
    print(f"Net Profit / Loss:                     Rs {results['net_pnl']}")
    print(f"Final Capital:                         Rs {results['final_capital']} (ROI: {results['roi']}%)")
    print(f"Monte Carlo Ruin Probability:          {mc_res['ruin_prob']}%")
    print(f"Monte Carlo Expected Drawdown:         {mc_res['expected_dd']}%")

    if len(df_trades) > 0 and "fold" in df_trades.columns:
        print("\n--- FOLD-WISE PERFORMANCE BREAKDOWN ---")
        for fold_id, f_df in df_trades.groupby("fold"):
            f_w = int(f_df["winner"].sum())
            f_tot = len(f_df)
            f_wr = round(f_w / f_tot * 100, 1)
            f_pnl = round(float(f_df["net_pnl"].sum()), 2)
            print(f"  * Fold {fold_id}: {f_tot} trades | {f_w}W / {f_tot - f_w}L ({f_wr}% WR) | Net PnL: Rs {f_pnl}")

if __name__ == "__main__":
    main()
