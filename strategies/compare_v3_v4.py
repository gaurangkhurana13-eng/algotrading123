import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from V3.run_v3 import backtest_v3
from V4.run_v4 import backtest_v4
from framework import compute_metrics

df_raw = pd.read_pickle("C:/Users/admin/.gemini/antigravity/scratch/new_trade/data/universe_78_15m.pkl")

tickers_baskets = {
    "ATR >= 0.80% (4 stocks)": ["PCJEWELLER.NS", "MOREPENLAB.NS", "SUBEXLTD.NS", "MANALIPETC.NS"],
    "Retail Momentum Core 4 (PCJ, RPOWER, HCC, IDEA)": ["PCJEWELLER.NS", "RPOWER.NS", "HCC.NS", "IDEA.NS"],
    "Retail Top 6 (PCJ, RPOWER, IDEA, MTNL, HCC, UJJIVAN)": ["PCJEWELLER.NS", "RPOWER.NS", "IDEA.NS", "MTNL.NS", "HCC.NS", "UJJIVANSFB.NS"]
}

for b_name, b_tickers in tickers_baskets.items():
    t3_list = []
    t4_list = []
    for t in b_tickers:
        sub = df_raw[t].dropna()
        df_t = pd.DataFrame({"open": sub["Open"], "high": sub["High"], "low": sub["Low"], "close": sub["Close"], "volume": sub["Volume"]}, index=sub.index)
        t3 = backtest_v3(df_t, mode="intraday")
        t4 = backtest_v4(df_t, mode="intraday")
        if len(t3) > 0:
            t3["ticker"] = t
            t3_list.append(t3)
        if len(t4) > 0:
            t4["ticker"] = t
            t4_list.append(t4)
            
    df3 = pd.concat(t3_list, ignore_index=True) if t3_list else pd.DataFrame()
    df4 = pd.concat(t4_list, ignore_index=True) if t4_list else pd.DataFrame()
    
    m3 = compute_metrics(df3, 500.0) if len(df3) > 0 else {}
    m4 = compute_metrics(df4, 500.0) if len(df4) > 0 else {}
    
    print("=" * 80)
    print(f"BASKET: {b_name}")
    print(f"  V3 (Fixed TP 2.2x ATR): Trades={m3.get('total_trades',0)}, WR={m3.get('win_rate',0)}%, Net=Rs {m3.get('net_pnl',0)}, PF={m3.get('profit_factor',0)}, AvgWin=Rs {m3.get('avg_win',0)}, AvgLoss=Rs {m3.get('avg_loss',0)}")
    print(f"  V4 (Moonshot Trailing): Trades={m4.get('total_trades',0)}, WR={m4.get('win_rate',0)}%, Net=Rs {m4.get('net_pnl',0)}, PF={m4.get('profit_factor',0)}, AvgWin=Rs {m4.get('avg_win',0)}, AvgLoss=Rs {m4.get('avg_loss',0)}")
