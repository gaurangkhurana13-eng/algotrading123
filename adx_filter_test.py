import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher import DataFetcher
from src.strategy import calculate_atr, calculate_rsi
from fast_engine import compute_ut_trailing_stop

fetcher = DataFetcher(data_dir="data")

def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    
    tr = calculate_atr(df, period=1)
    
    alpha = 1.0 / period
    tr_smooth = pd.Series(tr).ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean() / tr_smooth)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean() / tr_smooth)
    
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return adx.fillna(0)

def backtest_hybrid_sniper(df, ut_a=2.0, ut_c=7, adx_thresh=20, rsi_ob=70, rsi_os=30, commission_pct=0.03):
    trail_stop, _ = compute_ut_trailing_stop(df, key_value=ut_a, atr_period=ut_c)
    close = df["close"]
    adx = calculate_adx(df, period=14)
    rsi = calculate_rsi(close, period=14)
    ema200 = close.ewm(span=200, adjust=False).mean()
    
    # Base UT Crossovers
    base_buy = (close > trail_stop) & (close.shift(1) <= trail_stop.shift(1))
    base_sell = (close < trail_stop) & (close.shift(1) >= trail_stop.shift(1))
    
    # Filter: ADX > threshold (avoids dead chop)
    # Filter: RSI not overbought for Longs, not oversold for Shorts
    qual_buy = base_buy & (adx >= adx_thresh) & (rsi < rsi_ob)
    qual_sell = base_sell & (adx >= adx_thresh) & (rsi > rsi_os)
    
    capital = 100000.0
    pos = 0
    entry_price = 0.0
    entry_idx = 0
    trades = []
    
    for i in range(1, len(df)):
        c = close.iloc[i]
        curr_trail = trail_stop.iloc[i]
        
        if pos == 0:
            if qual_buy.iloc[i]:
                pos = 1
                entry_price = c
                entry_idx = i
            elif qual_sell.iloc[i]:
                pos = -1
                entry_price = c
                entry_idx = i
        elif pos == 1:
            # Exit long if sell signal OR trailing stop breached
            if base_sell.iloc[i] or c < curr_trail:
                pnl_pct = ((c - entry_price) / entry_price) * 100.0 - (2 * commission_pct)
                capital += capital * (pnl_pct / 100.0)
                trades.append({"dir": "LONG", "pnl_pct": pnl_pct, "capital": capital})
                # Check if qualifying short can be opened immediately
                if qual_sell.iloc[i]:
                    pos = -1
                    entry_price = c
                    entry_idx = i
                else:
                    pos = 0
        elif pos == -1:
            # Exit short if buy signal OR trailing stop breached
            if base_buy.iloc[i] or c > curr_trail:
                pnl_pct = ((entry_price - c) / entry_price) * 100.0 - (2 * commission_pct)
                capital += capital * (pnl_pct / 100.0)
                trades.append({"dir": "SHORT", "pnl_pct": pnl_pct, "capital": capital})
                if qual_buy.iloc[i]:
                    pos = 1
                    entry_price = c
                    entry_idx = i
                else:
                    pos = 0
                    
    if not trades:
        return {"total_trades": 0, "win_rate": 0.0, "net_profit_pct": 0.0, "profit_factor": 0.0, "max_drawdown_pct": 0.0}
        
    tdf = pd.DataFrame(trades)
    wins = tdf[tdf["pnl_pct"] > 0]
    losses = tdf[tdf["pnl_pct"] <= 0]
    total_trades = len(tdf)
    win_rate = (len(wins) / total_trades) * 100.0
    net_profit_pct = ((capital - 100000.0) / 100000.0) * 100.0
    
    gp = wins["pnl_pct"].sum()
    gl = abs(losses["pnl_pct"].sum())
    pf = (gp / gl) if gl > 0 else (99.0 if gp > 0 else 0.0)
    
    eq = tdf["capital"]
    peak = eq.cummax()
    dd = (eq - peak) / peak * 100.0
    max_dd = abs(dd.min()) if not dd.empty else 0.0
    
    return {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 1),
        "net_profit_pct": round(net_profit_pct, 2),
        "profit_factor": round(pf, 2),
        "max_drawdown_pct": round(max_dd, 2)
    }

print("Testing Trend Strength (ADX) & Momentum Exhaustion Filter:")
for sym in ["SUZLON", "PCJEWELLER"]:
    for tf in ["15m", "1h", "1d"]:
        data = fetcher.get_data(sym, tf)
        if data is not None and len(data) > 100:
            print(f"\n--- {sym} [{tf.upper()}] ---")
            for adx_t in [0, 15, 20, 25]:
                m = backtest_hybrid_sniper(data, ut_a=2.0, ut_c=7, adx_thresh=adx_t)
                print(f"  ADX >= {adx_t:<2} -> Trades: {m['total_trades']:<3} | WinRate: {m['win_rate']:>5.1f}% | Profit: {m['net_profit_pct']:>8.2f}% | PF: {m['profit_factor']:>5.2f} | MaxDD: {m['max_drawdown_pct']:>5.2f}%")
