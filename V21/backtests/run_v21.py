"""
Bhavishya Strategy V21 Execution & Forensic Backtest Suite
Asset Class: Indian Equities (NSE Cash 15m, Dhan 5x MIS)
Audit Engine: Gate V2 Compliance
"""

import pickle
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "V21" / "src"))
from engine import BhavishyaV21Engine, compute_v21_features, DhanFrictionEngine

# 1. Load Data Pools
print("Loading historical data pools...")
with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    univ = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_high_beta_10_15m_1y.pkl", "rb") as f:
    d_beta = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_10_15m_1y.pkl", "rb") as f:
    d_unseen = pickle.load(f)

all_pools = {}
all_pools.update(univ)
all_pools.update(d_beta)
all_pools.update(d_unseen)

def get_symbol_clean(ticker: str) -> str:
    return ticker.replace(".NS", "")

def prepare_basket(tickers: list) -> dict:
    basket_dfs = {}
    for t in tickers:
        key = t if t in all_pools else (f"{t}.NS" if f"{t}.NS" in all_pools else None)
        if key and key in all_pools:
            clean_name = get_symbol_clean(key)
            df = all_pools[key].copy()
            if "timestamp" not in df.columns:
                df["timestamp"] = df.index
            basket_dfs[clean_name] = df
    return basket_dfs

# Core Tickers
core_10 = [
    "PCJEWELLER", "RPOWER", "DELTACORP", "RITES", "IDEA",
    "MTNL", "ZEEL", "UTKARSHBNK", "IFCI", "NCC"
]

v19_14 = [
    "PCJEWELLER", "RPOWER", "DELTACORP", "RITES", "IDEA",
    "MTNL", "ZEEL", "UTKARSHBNK", "IFCI", "NCC",
    "TEXRAIL", "JPPOWER", "PTC", "NIACL"
]

expanded_18 = [
    "PCJEWELLER", "RPOWER", "DELTACORP", "RITES", "IDEA",
    "MTNL", "ZEEL", "UTKARSHBNK", "IFCI", "NCC",
    "TEXRAIL", "JPPOWER", "PTC", "NIACL",
    "HUDCO", "IRCON", "NBCC", "SUZLON"
]

def run_test(name: str, tickers: list):
    print(f"\n{'='*70}")
    print(f"RUNNING V21 BACKTEST: {name} ({len(tickers)} tickers)")
    print(f"{'='*70}")
    dfs = prepare_basket(tickers)
    print(f"Loaded {len(dfs)} valid symbols.")
    
    engine = BhavishyaV21Engine(
        initial_capital=500.0,
        leverage=5.0,
        hard_unit_risk_cap=102.0,
        physical_stop_atr=0.84,
        pyramid_scale_ratio=0.70,
        max_slots_cap=3
    )
    
    trades_df, summary = engine.run_backtest(dfs)
    
    print(f"Total Trades: {summary.get('total_trades', 0)}")
    print(f"Win Rate: {summary.get('win_rate_pct', 0.0)}%")
    print(f"Net Profit: Rs {summary.get('net_profit_rs', 0.0)}")
    print(f"Final Wallet: Rs {summary.get('final_wallet_rs', 500.0)} ({round(summary.get('final_wallet_rs', 500.0)/500.0, 2)}x)")
    print(f"Profit Factor: {summary.get('net_profit_factor', 0.0)}")
    print(f"Avg Win: Rs {summary.get('avg_win_rs', 0.0)}")
    print(f"Avg Loss: Rs {summary.get('avg_loss_rs', 0.0)}")
    print(f"Worst Loss: Rs {summary.get('worst_single_loss_rs', 0.0)}")
    print(f"Max Drawdown: {summary.get('max_drawdown_pct', 0.0)}%")
    print(f"Dhan Taxes: Rs {summary.get('total_taxes_rs', 0.0)}")
    
    if not trades_df.empty:
        print("\nExit Reasons Distribution:")
        print(trades_df["exit_reason"].value_counts())
        
        pyramids = trades_df[trades_df["qty_pyramid"] > 0]
        print(f"\nPyramided Trades Count: {len(pyramids)}")
        if not pyramids.empty:
            print(f"Pyramid Net Profit: Rs {round(pyramids['net_pnl'].sum(), 2)}")
            print(f"Pyramid Avg Net PnL: Rs {round(pyramids['net_pnl'].mean(), 2)}")
            
    return trades_df, summary

if __name__ == "__main__":
    t_core, s_core = run_test("V21 Core 10 Basket", core_10)
    t_14, s_14 = run_test("V21 V19-Baseline 14 Basket", v19_14)
    t_18, s_18 = run_test("V21 Expanded 18 Basket", expanded_18)
    
    # Save the 14-Basket or 18-Basket for Gate V2
    chosen_df = t_14 if not t_14.empty else t_core
    chosen_summary = s_14 if not s_14.empty else s_core
    
    # Format for Gate V2
    if not chosen_df.empty:
        # Create output trades_intraday.csv
        gate_df = pd.DataFrame()
        gate_df["symbol"] = chosen_df["symbol"]
        gate_df["direction"] = "long"
        gate_df["entry_time"] = chosen_df["entry_time"]
        gate_df["exit_time"] = chosen_df["exit_time"]
        gate_df["entry_price"] = chosen_df["entry_price"]
        gate_df["exit_price"] = chosen_df["exit_price"]
        gate_df["shares"] = chosen_df["qty_base"] + chosen_df["qty_pyramid"]
        gate_df["sl"] = round(chosen_df["entry_price"] - 0.84 * 1.0, 2) # approx ref
        gate_df["gross_pnl"] = chosen_df["gross_pnl"]
        gate_df["charges"] = chosen_df["dhan_taxes"]
        gate_df["net_pnl"] = chosen_df["net_pnl"]
        gate_df["exit_reason"] = chosen_df["exit_reason"]
        
        # Save CSVs
        chosen_df.to_csv(ROOT_DIR / "V21" / "trades" / "v21_trades.csv", index=False)
        gate_df.to_csv(ROOT_DIR / "strategies" / "V21" / "trades_intraday.csv", index=False)
        
        # Empty swing csv
        pd.DataFrame(columns=gate_df.columns).to_csv(
            ROOT_DIR / "strategies" / "V21" / "trades_swing.csv", index=False
        )
        
        # Save results.txt
        wins = chosen_df[chosen_df["net_pnl"] > 0]
        losses = chosen_df[chosen_df["net_pnl"] <= 0]
        gross_profit = round(chosen_df[chosen_df["gross_pnl"] > 0]["gross_pnl"].sum(), 2)
        gross_loss = round(abs(chosen_df[chosen_df["gross_pnl"] < 0]["gross_pnl"].sum()), 2)
        
        res_text = f"""----------------------------------------
INTRADAY
----------------------------------------
Total Trades: {len(chosen_df)}
Wins: {len(wins)}
Losses: {len(losses)}
Win Rate: {chosen_summary['win_rate_pct']:.2f}%
Gross Profit: Rs {gross_profit:.2f}
Gross Loss: Rs {gross_loss:.2f}
Net Profit: Rs {chosen_summary['net_profit_rs']:.2f}
Profit Factor: {chosen_summary['net_profit_factor']:.2f}
Avg Win: Rs {chosen_summary['avg_win_rs']:.2f}
Avg Loss: Rs {chosen_summary['avg_loss_rs']:.2f}
Max Drawdown: {chosen_summary['max_drawdown_pct']:.2f}%

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
        with open(ROOT_DIR / "strategies" / "V21" / "results.txt", "w") as f:
            f.write(res_text)
        print("\nSaved trades_intraday.csv and results.txt for Gate V2 audit.")
