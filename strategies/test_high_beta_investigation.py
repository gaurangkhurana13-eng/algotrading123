import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory
sys.path.insert(0, str(Path(__file__).parent))
from V4.run_v4 import backtest_v4
from framework import compute_metrics, calculate_atr

df_raw = pd.read_pickle("C:/Users/admin/.gemini/antigravity/scratch/new_trade/data/universe_78_15m.pkl")
tickers_all = sorted(list(set([c[0] for c in df_raw.columns if c[0] != "^NSEI"])))

# 1. Backtest every ticker and store detailed trade logs
ticker_data = {}
all_trades_dict = {}

for t in tickers_all:
    sub = df_raw[t].dropna()
    if len(sub) < 150:
        continue
    df_t = pd.DataFrame({
        "open": sub["Open"], "high": sub["High"], "low": sub["Low"],
        "close": sub["Close"], "volume": sub["Volume"]
    }, index=sub.index)
    atr14 = calculate_atr(df_t, 14)
    atr_pct = float((atr14 / df_t["close"]).mean())
    avg_vol = float(df_t["volume"].mean())
    med_price = float(df_t["close"].median())
    
    trades = backtest_v4(df_t, mode="intraday")
    if len(trades) > 0:
        trades["ticker"] = t
        all_trades_dict[t] = trades
        m = compute_metrics(trades, 500.0)
    else:
        all_trades_dict[t] = pd.DataFrame()
        m = {
            "total_trades": 0, "win_rate": 0.0, "net_pnl": 0.0, "profit_factor": 0.0,
            "avg_win": 0.0, "avg_loss": 0.0, "avg_charge": 0.0, "max_drawdown": 0.0,
            "final_capital": 500.0, "roi_pct": 0.0, "wins": 0, "losses": 0
        }
        
    ticker_data[t] = {
        "ticker": t,
        "atr_pct": atr_pct,
        "avg_vol": avg_vol,
        "med_price": med_price,
        "trades": m["total_trades"],
        "wins": m.get("wins", 0),
        "losses": m.get("losses", 0),
        "win_rate": m["win_rate"],
        "net_pnl": m["net_pnl"],
        "profit_factor": m["profit_factor"],
        "avg_win": m["avg_win"],
        "avg_loss": m["avg_loss"],
        "avg_charge": m["avg_charge"]
    }

df_summary = pd.DataFrame.from_dict(ticker_data, orient="index")

print("=" * 90)
print("SECTION 1: SPECIFIC RETAIL HIGH-BETA STOCKS (FROM USER PROMPT)")
print("=" * 90)
prompt_retail = ["PCJEWELLER.NS", "IFCI.NS", "SUZLON.NS", "RPOWER.NS", "MMTC.NS", "MANALIPETC.NS"]
for t in prompt_retail:
    if t in df_summary.index:
        row = df_summary.loc[t]
        print(f"Ticker: {t:<15} | ATR%: {row['atr_pct']*100:5.2f}% | AvgVol: {int(row['avg_vol']):>10,} | Trades: {row['trades']:2} | WR: {row['win_rate']:5.1f}% | Net: Rs {row['net_pnl']:6.2f} | PF: {row['profit_factor']:4.2f} | AvgWin: Rs {row['avg_win']:5.2f} | AvgLoss: Rs {row['avg_loss']:6.2f} | AvgChg: Rs {row['avg_charge']:4.2f}")

trades_prompt = [all_trades_dict[t] for t in prompt_retail if t in all_trades_dict and len(all_trades_dict[t]) > 0]
if trades_prompt:
    df_p = pd.concat(trades_prompt, ignore_index=True)
    mp = compute_metrics(df_p, 500.0)
    print("-" * 90)
    print(f"COMBINED PROMPT RETAIL BASKET (PCJEWELLER, IFCI, SUZLON, RPOWER, MMTC, MANALIPETC):")
    print(f"Total Trades: {mp['total_trades']} | WR: {mp['win_rate']}% ({mp['wins']}W/{mp['losses']}L) | Net PnL: Rs {mp['net_pnl']} | PF: {mp['profit_factor']} | AvgWin: Rs {mp['avg_win']} | AvgLoss: Rs {mp['avg_loss']} | AvgChg: Rs {mp['avg_charge']}")

print("\n" + "=" * 90)
print("SECTION 2: MANDATORY ATR% >= 0.80% STOCKS (THE STRICT HIGH-BETA THRESHOLD)")
print("=" * 90)
atr80_tickers = df_summary[df_summary["atr_pct"] >= 0.0080].index.tolist()
for t in atr80_tickers:
    row = df_summary.loc[t]
    print(f"Ticker: {t:<15} | ATR%: {row['atr_pct']*100:5.2f}% | AvgVol: {int(row['avg_vol']):>10,} | Trades: {row['trades']:2} | WR: {row['win_rate']:5.1f}% | Net: Rs {row['net_pnl']:6.2f} | PF: {row['profit_factor']:4.2f} | AvgWin: Rs {row['avg_win']:5.2f} | AvgLoss: Rs {row['avg_loss']:6.2f} | AvgChg: Rs {row['avg_charge']:4.2f}")

trades_atr80 = [all_trades_dict[t] for t in atr80_tickers if t in all_trades_dict and len(all_trades_dict[t]) > 0]
if trades_atr80:
    df_80 = pd.concat(trades_atr80, ignore_index=True)
    m80 = compute_metrics(df_80, 500.0)
    print("-" * 90)
    print(f"COMBINED STRICT ATR >= 0.80% BASKET ({len(atr80_tickers)} Tickers):")
    print(f"Total Trades: {m80['total_trades']} | WR: {m80['win_rate']}% ({m80['wins']}W/{m80['losses']}L) | Net PnL: Rs {m80['net_pnl']} | PF: {m80['profit_factor']} | AvgWin: Rs {m80['avg_win']} | AvgLoss: Rs {m80['avg_loss']} | AvgChg: Rs {m80['avg_charge']}")

print("\n" + "=" * 90)
print("SECTION 3: DISCOVERING THE OPTIMAL HIGH-BETA BASKET")
print("=" * 90)

# What are the best baskets?
# Basket A: PCJEWELLER alone (Highest Win Rate & PF in the ATR >= 0.80% set)
# Basket B: Liquid High-Beta Momentum (e.g. ATR% >= 0.50% & Avg Volume >= 500k & Net PnL > 0)
# Basket C: Top Performers in High-Beta / Retail Space (PCJEWELLER, RPOWER, HCC, IDEA, UJJIVANSFB, MTNL)

baskets = {
    "Basket 1: Single Top High-Beta Stock (PCJEWELLER)": ["PCJEWELLER.NS"],
    "Basket 2: Strict ATR >= 0.80% Universe (PCJEWELLER, SUBEXLTD, MANALIPETC, MOREPENLAB)": atr80_tickers,
    "Basket 3: High-Beta Retail Momentum Leaders (PCJEWELLER, RPOWER, HCC, IDEA)": ["PCJEWELLER.NS", "RPOWER.NS", "HCC.NS", "IDEA.NS"],
    "Basket 4: Top Profit High-Beta Basket (PCJEWELLER, RPOWER, IDEA, MTNL, HCC, UJJIVANSFB)": ["PCJEWELLER.NS", "RPOWER.NS", "IDEA.NS", "MTNL.NS", "HCC.NS", "UJJIVANSFB.NS"],
    "Basket 5: High-Beta + High-Liquidity Basket (ATR% >= 0.50% & Avg Volume >= 800,000)": df_summary[(df_summary["atr_pct"] >= 0.0050) & (df_summary["avg_vol"] >= 800000)].index.tolist(),
    "Basket 6: Prompt Retail Winners (PCJEWELLER, RPOWER, MMTC)": ["PCJEWELLER.NS", "RPOWER.NS", "MMTC.NS"]
}

for b_name, b_tickers in baskets.items():
    b_trades = [all_trades_dict[t] for t in b_tickers if t in all_trades_dict and len(all_trades_dict[t]) > 0]
    if b_trades:
        b_df = pd.concat(b_trades, ignore_index=True)
        mb = compute_metrics(b_df, 500.0)
        print(f"\n>>> {b_name}")
        print(f"    Tickers ({len(b_tickers)}): {b_tickers}")
        print(f"    Trades       : {mb['total_trades']}")
        print(f"    Win Rate     : {mb['win_rate']}% ({mb['wins']} Wins / {mb['losses']} Losses)")
        print(f"    Net PnL      : Rs {mb['net_pnl']}")
        print(f"    Profit Factor: {mb['profit_factor']}")
        print(f"    Avg Win      : Rs {mb['avg_win']}")
        print(f"    Avg Loss     : Rs {mb['avg_loss']}")
        print(f"    Avg Charges  : Rs {mb['avg_charge']}")
