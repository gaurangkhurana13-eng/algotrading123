"""
Backtest User's Exact Strategies (Quant Sniper Pro v6 & SVMKR + MAPO)
on 1-Year PC Jeweller Real 15-Minute Data with Full Dhan MIS Costs & Slippage.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Load PC Jeweller 1-Year Data
data_path = Path("C:/Users/admin/.gemini/antigravity/scratch/new_trade/data/pcjeweller_15m_1y.pkl")
df = pd.read_pickle(data_path)

def calc_dhan_charges(buy_p, sell_p, qty):
    buy_val = buy_p * qty
    sell_val = sell_p * qty
    turnover = buy_val + sell_val
    brokerage = min(20.0, buy_val * 0.0003) + min(20.0, sell_val * 0.0003)
    stt = sell_val * 0.00025
    etc = turnover * 0.0000325
    sebi = turnover * 0.000001
    stamp = buy_val * 0.00003
    gst = 0.18 * (brokerage + etc)
    return brokerage + stt + etc + gst + stamp + sebi

def calculate_atr(df, period):
    h = df['high']
    l = df['low']
    c = df['close']
    c_prev = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - c_prev).abs(), (l - c_prev).abs()))
    return tr.rolling(period).mean()

def run_quant_sniper_backtest(df, mode="intraday", long_only=True):
    close = df['close']
    high = df['high']
    low = df['low']
    open_p = df['open']
    
    # ATR & UT Bot
    c_len = 7
    a_mult = 2.0
    atr = calculate_atr(df, c_len)
    n_loss = a_mult * atr
    
    ema200 = close.ewm(span=200, adjust=False).mean()
    
    n = len(df)
    trail_stop = np.zeros(n)
    prev_stop = 0.0
    
    for i in range(1, n):
        cur_c = close.iloc[i]
        prev_c = close.iloc[i-1]
        loss = n_loss.iloc[i] if not np.isnan(n_loss.iloc[i]) else 0.5
        
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
    
    buy_cross = (close > ts_series) & (close.shift(1) <= ts_series.shift(1))
    sell_cross = (close < ts_series) & (close.shift(1) >= ts_series.shift(1))
    
    allow_long = close > ema200
    
    trades = []
    capital = 500.0
    pos = 0 # 1: Long, 0: Flat
    entry_price = 0.0
    entry_time = None
    entry_qty = 0
    slippage = 0.0005 # 0.05%
    
    for i in range(200, n):
        ts = df.index[i]
        t = ts.time()
        d = ts.date()
        c = close.iloc[i]
        
        # Intraday square-off check
        if mode == "intraday" and pos == 1:
            if t >= pd.Timestamp("15:10").time():
                # Force EOD exit
                exit_price = c * (1.0 - slippage)
                gross = (exit_price - entry_price) * entry_qty
                chg = calc_dhan_charges(entry_price, exit_price, entry_qty)
                net = gross - chg
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "qty": entry_qty,
                    "gross_pnl": round(gross, 2),
                    "charges": round(chg, 2),
                    "net_pnl": round(net, 2),
                    "exit_reason": "INTRADAY_15:10_SQUARE_OFF"
                })
                pos = 0
                continue
                
        # Check Entry
        if pos == 0:
            time_ok = (t >= pd.Timestamp("09:20").time()) and (t <= pd.Timestamp("13:30").time() if mode=="intraday" else True)
            if buy_cross.iloc[i] and allow_long.iloc[i] and time_ok:
                pos = 1
                entry_time = ts
                entry_price = c * (1.0 + slippage)
                # Max 5x leverage on 500 = 2450 notional
                entry_qty = max(1, int(2450.0 / entry_price))
        elif pos == 1:
            # Check Exit
            if sell_cross.iloc[i]:
                exit_price = c * (1.0 - slippage)
                gross = (exit_price - entry_price) * entry_qty
                chg = calc_dhan_charges(entry_price, exit_price, entry_qty)
                net = gross - chg
                trades.append({
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "qty": entry_qty,
                    "gross_pnl": round(gross, 2),
                    "charges": round(chg, 2),
                    "net_pnl": round(net, 2),
                    "exit_reason": "UT_TRAILING_STOP_CROSS"
                })
                pos = 0

    return pd.DataFrame(trades)

# Test Intraday & Swing
tdf_intra = run_quant_sniper_backtest(df, mode="intraday", long_only=True)
tdf_swing = run_quant_sniper_backtest(df, mode="swing", long_only=True)

def report_stats(tdf, name):
    print(f"\n{'='*50}\n  {name}\n{'='*50}")
    if len(tdf) == 0:
        print("No trades generated.")
        return
    n = len(tdf)
    wins = tdf[tdf['net_pnl'] > 0]
    losses = tdf[tdf['net_pnl'] < 0]
    wr = len(wins) / n * 100
    tot_pnl = tdf['net_pnl'].sum()
    tot_win = wins['net_pnl'].sum()
    tot_loss = abs(losses['net_pnl'].sum())
    pf = tot_win / tot_loss if tot_loss > 0 else 999.0
    avg_win = wins['net_pnl'].mean() if len(wins) > 0 else 0
    avg_loss = losses['net_pnl'].mean() if len(losses) > 0 else 0
    avg_chg = tdf['charges'].mean()
    
    print(f"Total Trades   : {n}")
    print(f"Win Rate       : {wr:.1f}% ({len(wins)} Wins / {len(losses)} Losses)")
    print(f"Total Net PnL  : Rs {tot_pnl:.2f}")
    print(f"Profit Factor  : {pf:.2f}")
    print(f"Avg Win        : Rs {avg_win:.2f}")
    print(f"Avg Loss       : Rs {avg_loss:.2f}")
    print(f"Avg Dhan Chg   : Rs {avg_chg:.2f}")

report_stats(tdf_intra, "Quant Sniper Pro v6 [INTRADAY MIS]")
report_stats(tdf_swing, "Quant Sniper Pro v6 [SWING / DELIVERY]")


def run_svmkr_mapo_backtest(df, mode="intraday"):
    close = df['close']
    high = df['high']
    low = df['low']
    open_p = df['open']
    
    # 1. UT Bot trailing stop with a=2.0, c=1
    c_len = 1
    a_mult = 2.0
    atr = calculate_atr(df, c_len)
    n_loss = a_mult * atr
    
    n = len(df)
    trail_stop = np.zeros(n)
    prev_stop = 0.0
    for i in range(1, n):
        cur_c = close.iloc[i]
        prev_c = close.iloc[i-1]
        loss = n_loss.iloc[i] if not np.isnan(n_loss.iloc[i]) else 0.5
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
    
    # 2. Hull MA 31
    def wma(s, length):
        weights = np.arange(1, length + 1)
        return s.rolling(length).apply(lambda p: np.dot(p, weights) / weights.sum(), raw=True)
    n_hma = 31
    half_len = int(round(n_hma / 2))
    sqrt_len = int(round(np.sqrt(n_hma)))
    raw_hma = 2 * wma(close, half_len) - wma(close, n_hma)
    hma = wma(raw_hma, sqrt_len)
    
    # 3. MAPO Consensus (> 50%)
    ma_bull_pct = np.zeros(n)
    for m in range(10, 101, 5):
        ma_bull_pct += (close > close.rolling(m).mean()).astype(int)
    ma_bull_pct = (ma_bull_pct / 19.0) * 100.0
    
    buy_cross = (close > ts_series) & (close.shift(1) <= ts_series.shift(1))
    sell_cross = (close < ts_series) & (close.shift(1) >= ts_series.shift(1))
    
    trades = []
    capital = 500.0
    pos = 0
    entry_price = 0.0
    entry_time = None
    entry_qty = 0
    slippage = 0.0005
    
    for i in range(105, n):
        ts = df.index[i]
        t = ts.time()
        c = close.iloc[i]
        
        if mode == "intraday" and pos == 1:
            if t >= pd.Timestamp("15:10").time():
                exit_price = c * (1.0 - slippage)
                gross = (exit_price - entry_price) * entry_qty
                chg = calc_dhan_charges(entry_price, exit_price, entry_qty)
                net = gross - chg
                trades.append({
                    "entry_time": entry_time, "exit_time": ts,
                    "entry_price": entry_price, "exit_price": exit_price,
                    "qty": entry_qty, "gross_pnl": round(gross, 2),
                    "charges": round(chg, 2), "net_pnl": round(net, 2),
                    "exit_reason": "INTRADAY_15:10_SQUARE_OFF"
                })
                pos = 0
                continue
                
        if pos == 0:
            time_ok = (t >= pd.Timestamp("09:20").time()) and (t <= pd.Timestamp("13:30").time() if mode=="intraday" else True)
            mapo_ok = ma_bull_pct.iloc[i] >= 50.0
            hma_ok = c > hma.iloc[i]
            if buy_cross.iloc[i] and mapo_ok and hma_ok and time_ok:
                pos = 1
                entry_time = ts
                entry_price = c * (1.0 + slippage)
                entry_qty = max(1, int(2450.0 / entry_price))
        elif pos == 1:
            if sell_cross.iloc[i]:
                exit_price = c * (1.0 - slippage)
                gross = (exit_price - entry_price) * entry_qty
                chg = calc_dhan_charges(entry_price, exit_price, entry_qty)
                net = gross - chg
                trades.append({
                    "entry_time": entry_time, "exit_time": ts,
                    "entry_price": entry_price, "exit_price": exit_price,
                    "qty": entry_qty, "gross_pnl": round(gross, 2),
                    "charges": round(chg, 2), "net_pnl": round(net, 2),
                    "exit_reason": "SVMKR_TRAILING_STOP_CROSS"
                })
                pos = 0

    return pd.DataFrame(trades)

tdf_svmkr_intra = run_svmkr_mapo_backtest(df, mode="intraday")
tdf_svmkr_swing = run_svmkr_mapo_backtest(df, mode="swing")

report_stats(tdf_svmkr_intra, "SVMKR + MAPO Unified [INTRADAY MIS]")
report_stats(tdf_svmkr_swing, "SVMKR + MAPO Unified [SWING / DELIVERY]")
