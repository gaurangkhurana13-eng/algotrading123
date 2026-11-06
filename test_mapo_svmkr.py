import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher import DataFetcher
from src.strategy import calculate_atr

fetcher = DataFetcher(data_dir="data")

def test_mapo_svmkr(df, a=6.0, c=4, n=31, commission_pct=0.03):
    close = df["close"]
    
    # Hull MA
    def wma(s, length):
        weights = np.arange(1, length + 1)
        return s.rolling(length).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
    half_len = int(round(n / 2))
    sqrt_len = int(round(np.sqrt(n)))
    raw_hma = 2 * wma(close, half_len) - wma(close, n)
    hma = wma(raw_hma, sqrt_len)
    hma_bull = close > hma
    
    # UT Bot
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
    
    # MAPO approximation (percentage of moving averages where close > MA)
    ma_bull_counts = np.zeros(len(df))
    for m in range(10, 101, 5):
        ma_bull_counts += (close > close.rolling(m).mean()).astype(int)
    mapo_pct = (ma_bull_counts / 19.0) * 100.0
    mapo_bull = mapo_pct >= 50.0
    
    # UCS LRS (Slope of Linreg)
    # Linreg 50
    # slope = ta.ema(linreg - linreg[1], 5)
    # lrsBull = slope > 0
    
    buy_sig = (close > ts_series) & (close.shift(1) <= ts_series.shift(1)) & hma_bull & mapo_bull
    sell_sig = (close < ts_series) & (close.shift(1) >= ts_series.shift(1))
    
    capital = 1000.0
    pos = 0
    entry_p = 0.0
    trades = []
    
    for i in range(1, len(df)):
        cur_c = close.iloc[i]
        if pos == 0:
            if buy_sig.iloc[i]:
                pos = 1
                entry_p = cur_c
        elif pos == 1:
            if sell_sig.iloc[i]:
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

print("Testing MAPO + SVMKR on PCJEWELLER and SUZLON:")
for sym in ["PCJEWELLER", "SUZLON"]:
    for tf in ["15m", "1h", "1d"]:
        data = fetcher.get_data(sym, tf)
        if data is not None and len(data) > 100:
            # Swing parameters: a=6, c=4, n=31
            m_swing = test_mapo_svmkr(data, a=6.0, c=4, n=31)
            # Intraday parameters: a=3, c=4, n=21
            m_intra = test_mapo_svmkr(data, a=3.0, c=4, n=21)
            print(f"[{sym} {tf.upper()}] Swing (6,4): Trades={m_swing['trades']} | WinRate={m_swing['win_rate']}% | Profit={m_swing['net_profit']}% | PF={m_swing['pf']} | MaxDD={m_swing['max_dd']}%")
            print(f"[{sym} {tf.upper()}] Intra (3,4): Trades={m_intra['trades']} | WinRate={m_intra['win_rate']}% | Profit={m_intra['net_profit']}% | PF={m_intra['pf']} | MaxDD={m_intra['max_dd']}%")
