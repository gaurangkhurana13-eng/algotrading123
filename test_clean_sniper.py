import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher import DataFetcher
from src.strategy import calculate_atr

fetcher = DataFetcher(data_dir="data")

def test_clean_intraday_sniper(df, a=2.0, c=1, hma_len=31, commission_pct=0.05):
    """Replicates PC Jeweller 5M Clean Sniper [STRATEGY]"""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    
    # 1. Hull MA 31
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    def wma(s, length):
        weights = np.arange(1, length + 1)
        return s.rolling(length).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
        
    half_len = int(round(hma_len / 2))
    sqrt_len = int(round(np.sqrt(hma_len)))
    raw_hma = 2 * wma(close, half_len) - wma(close, hma_len)
    hma = wma(raw_hma, sqrt_len)
    hma_bull = hma > hma.shift(1)
    
    # 2. UT Bot Trailing Stop
    atr = calculate_atr(df, period=c)
    n_loss = a * atr
    
    trail_stop = np.zeros(len(df))
    prev_stop = 0.0
    for i in range(1, len(df)):
        cur_c = close.iloc[i]
        prev_c = close.iloc[i-1]
        loss = n_loss.iloc[i]
        if cur_c > prev_stop and prev_c > prev_stop:
            curr = max(prev_stop, cur_c - loss)
        elif cur_c < prev_stop and prev_c < prev_stop:
            curr = min(prev_stop, cur_c + loss)
        elif cur_c > prev_stop:
            curr = cur_c - loss
        else:
            curr = cur_c + loss
        trail_stop[i] = curr
        prev_stop = curr
        
    ts_series = pd.Series(trail_stop, index=df.index)
    
    # Triggers
    buy_cross = (close > ts_series) & (close.shift(1) <= ts_series.shift(1))
    exit_cross = (close < ts_series) & (close.shift(1) >= ts_series.shift(1))
    
    # Simulate trade execution
    capital = 1000.0
    pos = 0
    entry_p = 0.0
    trades = []
    
    for i in range(1, len(df)):
        cur_c = close.iloc[i]
        h_bull = hma_bull.iloc[i] if not pd.isna(hma_bull.iloc[i]) else False
        
        # Intraday hour check (Groww square-off at 15:10)
        t = df.index[i]
        is_sqoff = (t.hour == 15 and t.minute >= 10)
        in_window = (t.hour >= 9 and (t.hour < 15 or (t.hour == 15 and t.minute < 10)))
        
        if pos == 0:
            if buy_cross.iloc[i] and h_bull and in_window:
                pos = 1
                entry_p = cur_c
                entry_idx = i
        elif pos == 1:
            if exit_cross.iloc[i] or is_sqoff:
                pnl_pct = ((cur_c - entry_p) / entry_p) * 100.0 - (2 * commission_pct)
                pnl_abs = capital * (pnl_pct / 100.0)
                capital += pnl_abs
                trades.append({"pnl_pct": pnl_pct, "capital": capital})
                pos = 0
                
    if not trades:
        return {"trades": 0, "win_rate": 0.0, "net_profit": 0.0, "pf": 0.0, "max_dd": 0.0}
        
    tdf = pd.DataFrame(trades)
    wins = tdf[tdf["pnl_pct"] > 0]
    losses = tdf[tdf["pnl_pct"] <= 0]
    total_trades = len(tdf)
    win_rate = (len(wins) / total_trades) * 100.0
    net_profit = ((capital - 1000.0) / 1000.0) * 100.0
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

print("Running test on Clean Intraday Sniper:")
for sym in ["PCJEWELLER", "SUZLON"]:
    for tf in ["15m", "1h", "1d"]:
        data = fetcher.get_data(sym, tf)
        if data is not None and len(data) > 100:
            m = test_clean_intraday_sniper(data, a=2.0, c=1, hma_len=31)
            print(f"{sym} {tf}: Trades={m['trades']} | WinRate={m['win_rate']}% | Profit={m['net_profit']}% | PF={m['pf']} | MaxDD={m['max_dd']}%")
