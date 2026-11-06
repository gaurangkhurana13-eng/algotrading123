"""
Master Quantitative Pipeline Runner
-----------------------------------
Executes end-to-end backtesting for any equity across multiple timeframes:
1. Downloads and validates data from yfinance and TradingView
2. Applies 50% Daily Midpoint Strategy (with 0.03% commissions & ATR TP/SL)
3. Outputs precision metrics and trade logs
4. Generates visual equity curves in reports/
"""

import os
import sys
import argparse
from typing import Dict, Any

# Ensure UTF-8 output encoding for Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
from tabulate import tabulate

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher import DataFetcher
from src.strategy import DailyMidpointStrategy, AIConfluenceStrategy
from src.backtester import BacktestEngine
from src.visualizer import plot_equity_curve, plot_multi_timeframe_comparison


def run_pipeline(
    symbol: str = "PCJEWELLER",
    timeframes = ("1d", "1h", "30m", "15m", "5m"),
    initial_capital: float = 100000.0,
    position_pct: float = 20.0,
    commission_pct: float = 0.03,
    sl_mult: float = 1.5,
    tp_mult: float = 3.0,
    force_download: bool = False
):
    print("=" * 80)
    print(f"  QUANTITATIVE BACKTESTING PIPELINE: {symbol.upper()}")
    print(f"  Initial Capital: INR {initial_capital:,.2f} | Size: {position_pct}% | Commission: {commission_pct}%")
    print(f"  Timeframes: {', '.join(timeframes)}")
    print("=" * 80)

    fetcher = DataFetcher(data_dir="data")
    engine = BacktestEngine(initial_capital=initial_capital, position_pct=position_pct, commission_pct=commission_pct)
    strategy = DailyMidpointStrategy(sl_multiplier=sl_mult, tp_multiplier=tp_mult)

    # 1. Fetch Daily Data first (used as anchor for higher-timeframe levels)
    print("\n[Step 1/4] Fetching Daily Reference Data...")
    try:
        daily_df = fetcher.get_data(symbol, "1d", force_download=force_download)
    except Exception as e:
        print(f"Error fetching daily anchor data: {e}")
        daily_df = None

    # 2. Iterate through each timeframe
    results_dict: Dict[str, Dict[str, Any]] = {}
    summary_rows = []

    print("\n[Step 2/4] Processing Timeframes & Executing Strategies...")
    for tf in timeframes:
        print(f"\n---> Analyzing [{tf.upper()}] ...")
        try:
            prefer_tv = tf in ["5m", "15m", "30m"]
            intraday_df = fetcher.get_data(symbol, tf, force_download=force_download, prefer_tv=prefer_tv)
            if intraday_df is None or intraday_df.empty:
                print(f"Skipping {tf}: No data available.")
                continue

            # Prepare data with 50% Daily Midpoint mapping
            prepared_df = strategy.prepare_data(intraday_df, daily_df=daily_df)
            signals_df = strategy.generate_signals(prepared_df)

            # Execute backtest
            backtest_res = engine.run(signals_df, use_tp_sl=True, trade_direction="Both")
            results_dict[tf] = backtest_res

            # Save detailed trades
            trades_df = backtest_res["trades_df"]
            trades_path = os.path.join("reports", f"{symbol}_{tf}_trades.csv")
            if not trades_df.empty:
                trades_df.to_csv(trades_path, index=False)

            # Plot individual chart
            chart_path = os.path.join("reports", f"equity_{symbol}_{tf}.png")
            plot_equity_curve(backtest_res["equity_curve"], symbol, tf, backtest_res["metrics"], chart_path)

            m = backtest_res["metrics"]
            date_range = f"{intraday_df.index.min().strftime('%d-%b-%y')} to {intraday_df.index.max().strftime('%d-%b-%y')}"
            summary_rows.append([
                tf.upper(),
                len(intraday_df),
                date_range,
                f"INR {m['net_profit']:+,.2f}",
                f"{m['net_profit_pct']:+.2f}%",
                f"{m['win_rate']}%",
                m['total_trades'],
                f"{m['winning_trades']}/{m['losing_trades']}",
                m['profit_factor'],
                f"{m['max_drawdown_pct']}%",
                f"INR {m['total_commissions']:,.2f}"
            ])
            print(f"[{tf.upper()}] Done! Net Profit: INR {m['net_profit']:+,.2f} ({m['net_profit_pct']:+.2f}%) | Trades: {m['total_trades']} | Win Rate: {m['win_rate']}%")

        except Exception as e:
            print(f"Error processing {tf}: {e}")
            import traceback
            traceback.print_exc()

    # 3. Print Consolidated Summary Table
    print("\n" + "=" * 100)
    print(f"  PERFORMANCE SUMMARY TABLE: {symbol.upper()}")
    print("=" * 100)
    headers = [
        "Timeframe", "Candles", "Date Range", "Net Profit (INR)", 
        "Return (%)", "Win Rate", "Trades", "W / L", "Profit Factor", "Max Drawdown", "Commissions"
    ]
    print(tabulate(summary_rows, headers=headers, tablefmt="fancy_grid"))

    # Save summary table to CSV
    summary_df = pd.DataFrame(summary_rows, columns=headers)
    summary_df.to_csv(os.path.join("reports", f"summary_{symbol}.csv"), index=False)

    # 4. Generate Multi-Timeframe Comparative Plot
    if results_dict:
        print("\n[Step 3/4] Generating Consolidated Multi-Timeframe Chart...")
        comp_path = os.path.join("reports", f"multi_timeframe_comparison_{symbol}.png")
        plot_multi_timeframe_comparison(results_dict, symbol, comp_path)
        print(f"Consolidated chart saved to: {comp_path}")

    print("\n[Step 4/4] Pipeline Complete! All datasets in 'data/' and reports in 'reports/'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Professional Quant Backtesting Pipeline")
    parser.add_argument("--symbol", type=str, default="PCJEWELLER", help="Stock ticker (e.g. PCJEWELLER, RELIANCE, TCS)")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial Capital in INR")
    parser.add_argument("--force-download", action="store_true", help="Force fresh download from sources")
    args = parser.parse_args()

    run_pipeline(
        symbol=args.symbol,
        timeframes=("1d", "1h", "30m", "15m", "5m"),
        initial_capital=args.capital,
        force_download=args.force_download
    )
