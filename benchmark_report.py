import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher import DataFetcher
from src.strategy import calculate_atr, calculate_rsi
from fast_engine import compute_ut_trailing_stop

fetcher = DataFetcher(data_dir="data")

def run_quant_sniper_benchmark(df, mode="Dual", preset="High-Frequency", commission_pct=0.03):
    # Resolve preset
    if preset == "High-Frequency":
        a, c = 2.0, 7
    elif preset == "Balanced":
        a, c = 2.5, 10
    else:
        a, c = 4.0, 10
        
    close = df["close"]
    ema200 = close.ewm(span=200, adjust=False).mean()
    trail_stop, _ = compute_ut_trailing_stop(df, key_value=a, atr_period=c)
    
    raw_buy = (close > trail_stop) & (close.shift(1) <= trail_stop.shift(1))
    raw_sell = (close < trail_stop) & (close.shift(1) >= trail_stop.shift(1))
    
    if mode == "Long Only":
        buy_signal = raw_buy & (close > ema200)
    else:
        buy_signal = raw_buy
    sell_signal = raw_sell
    
    capital = 100000.0
    pos = 0 # 1, -1, 0
    entry_price = 0.0
    entry_idx = 0
    trades = []
    
    for i in range(1, len(df)):
        cur_c = close.iloc[i]
        cur_buy = buy_signal.iloc[i]
        cur_sell = sell_signal.iloc[i]
        
        if pos == 0:
            if cur_buy:
                pos = 1
                entry_price = cur_c
                entry_idx = i
            elif cur_sell and mode == "Dual":
                pos = -1
                entry_price = cur_c
                entry_idx = i
        elif pos == 1:
            if cur_sell:
                pnl_pct = ((cur_c - entry_price) / entry_price) * 100.0 - (2 * commission_pct)
                capital += capital * (pnl_pct / 100.0)
                trades.append({"dir": "LONG", "pnl_pct": pnl_pct, "capital": capital})
                
                if mode == "Dual":
                    pos = -1
                    entry_price = cur_c
                    entry_idx = i
                else:
                    pos = 0
        elif pos == -1:
            if cur_buy:
                pnl_pct = ((entry_price - cur_c) / entry_price) * 100.0 - (2 * commission_pct)
                capital += capital * (pnl_pct / 100.0)
                trades.append({"dir": "SHORT", "pnl_pct": pnl_pct, "capital": capital})
                
                pos = 1
                entry_price = cur_c
                entry_idx = i
                
    if not trades:
        return {"trades": 0, "win_rate": 0.0, "net_profit": 0.0, "pf": 0.0, "max_dd": 0.0}
        
    tdf = pd.DataFrame(trades)
    wins = tdf[tdf["pnl_pct"] > 0]
    losses = tdf[tdf["pnl_pct"] <= 0]
    total_trades = len(tdf)
    win_rate = (len(wins) / total_trades) * 100.0
    net_profit = ((capital - 100000.0) / 100000.0) * 100.0
    
    gp = wins["pnl_pct"].sum()
    gl = abs(losses["pnl_pct"].sum())
    pf = (gp / gl) if gl > 0 else (99.0 if gp > 0 else 0.0)
    
    eq = tdf["capital"]
    peak = eq.cummax()
    dd = (eq - peak) / peak * 100.0
    max_dd = abs(dd.min()) if not dd.empty else 0.0
    
    return {
        "trades": total_trades,
        "win_rate": round(win_rate, 1),
        "net_profit": round(net_profit, 2),
        "pf": round(pf, 2),
        "max_dd": round(max_dd, 2)
    }

print("=" * 85)
print("  FINAL QUANT SNIPER PRO [v6] SYSTEM BENCHMARK REPORT")
print("=" * 85)
summary_rows = []
for sym in ["PCJEWELLER", "SUZLON"]:
    for tf in ["15m", "1h", "1d"]:
        data = fetcher.get_data(sym, tf)
        if data is not None and len(data) > 100:
            # Dual High-Frequency (Intraday / F&O)
            m_dual_hf = run_quant_sniper_benchmark(data, mode="Dual", preset="High-Frequency")
            summary_rows.append({
                "Asset": sym, "TF": tf.upper(), "Mode": "Dual (Intraday/F&O)", "Preset": "High-Freq",
                **m_dual_hf
            })
            # Dual Balanced Swing
            m_dual_bal = run_quant_sniper_benchmark(data, mode="Dual", preset="Balanced")
            summary_rows.append({
                "Asset": sym, "TF": tf.upper(), "Mode": "Dual (Intraday/F&O)", "Preset": "Balanced",
                **m_dual_bal
            })
            # Long-Only Macro Gated (Delivery / Cash)
            m_long = run_quant_sniper_benchmark(data, mode="Long Only", preset="Balanced")
            summary_rows.append({
                "Asset": sym, "TF": tf.upper(), "Mode": "Long-Only (Delivery)", "Preset": "Macro-Gated",
                **m_long
            })

report_df = pd.DataFrame(summary_rows)
print(report_df.to_string(index=False))
