"""
Visualizer & Charting Module
----------------------------
Generates high-resolution performance plots matching TradingView equity curves:
1. Multi-timeframe Equity Comparison Plot
2. Single-timeframe Deep Dive with Underwater Drawdown & Daily Midpoint Overlay
"""

import os
from typing import Dict, Any
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def set_dark_theme():
    plt.style.use("dark_background")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.edgecolor"] = "#334155"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#1e293b"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.5


def plot_equity_curve(eq_series: pd.Series, symbol: str, timeframe: str, metrics: Dict[str, Any], output_path: str):
    """Plots individual timeframe equity curve with underwater drawdown."""
    set_dark_theme()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    # 1. Equity Curve
    initial = metrics["initial_capital"]
    color = "#10b981" if eq_series.iloc[-1] >= initial else "#ef4444"
    ax1.plot(eq_series.index, eq_series.values, color=color, linewidth=1.8, label=f"Equity ({metrics['net_profit_pct']:+.2f}%)")
    ax1.axhline(initial, color="#64748b", linestyle=":", linewidth=1, label=f"Initial: INR {initial:,.0f}")
    ax1.set_title(
        f"{symbol} [{timeframe.upper()}] - 50% Daily Midpoint Strategy\n"
        f"Net Profit: INR {metrics['net_profit']:+,.2f} ({metrics['net_profit_pct']:+.2f}%) | "
        f"Win Rate: {metrics['win_rate']}% | Profit Factor: {metrics['profit_factor']} | Max DD: -{metrics['max_drawdown_pct']}%",
        fontsize=12, fontweight="bold", pad=12, color="#f8fafc"
    )
    ax1.set_ylabel("Portfolio Value (INR)", color="#cbd5e1")
    ax1.grid(True)
    ax1.legend(loc="upper left", framealpha=0.3)

    # 2. Underwater Drawdown
    running_max = eq_series.cummax()
    dd_series = (eq_series - running_max) / running_max * 100.0
    ax2.fill_between(dd_series.index, dd_series.values, 0, color="#ef4444", alpha=0.4)
    ax2.plot(dd_series.index, dd_series.values, color="#ef4444", linewidth=1)
    ax2.set_ylabel("Drawdown (%)", color="#cbd5e1")
    ax2.set_xlabel("Date", color="#cbd5e1")
    ax2.grid(True)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_multi_timeframe_comparison(results_dict: Dict[str, Dict[str, Any]], symbol: str, output_path: str):
    """
    Plots a multi-panel comparison of all timeframes side by side,
    mirroring the 4 screenshots tested on TradingView!
    """
    set_dark_theme()
    n = len(results_dict)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.2 * n), sharex=False)
    if n == 1:
        axes = [axes]

    colors = {
        "1d": "#38bdf8",
        "1h": "#10b981",
        "30m": "#2dd4bf",
        "15m": "#f59e0b",
        "5m": "#ef4444"
    }

    for i, (tf, res) in enumerate(results_dict.items()):
        ax = axes[i]
        eq = res["equity_curve"]
        m = res["metrics"]
        tf_color = colors.get(tf.lower(), "#94a3b8")
        is_profit = m["net_profit"] >= 0
        fill_color = "#10b981" if is_profit else "#ef4444"

        ax.plot(eq.index, eq.values, color=tf_color, linewidth=1.5, label=f"{tf.upper()} Equity")
        ax.axhline(m["initial_capital"], color="#475569", linestyle=":", linewidth=1)
        ax.fill_between(eq.index, eq.values, m["initial_capital"], color=fill_color, alpha=0.15)

        title_str = (
            f"Timeframe: {tf.upper()} | Net Profit: INR {m['net_profit']:+,.2f} ({m['net_profit_pct']:+.2f}%) | "
            f"Win Rate: {m['win_rate']}% | Trades: {m['total_trades']} | Max DD: -{m['max_drawdown_pct']}% | PF: {m['profit_factor']}"
        )
        ax.set_title(title_str, fontsize=11, fontweight="bold", color="#f8fafc", loc="left")
        ax.set_ylabel("Equity (INR)", color="#cbd5e1", fontsize=9)
        ax.grid(True)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %y"))

    fig.suptitle(f"Multi-Timeframe Comparative Performance Analysis: {symbol}\nPrevious Day 50% Midpoint Strategy", fontsize=14, fontweight="bold", y=0.995, color="#f1f5f9")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
