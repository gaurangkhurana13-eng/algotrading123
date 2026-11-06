"""
Test MACD + RSI + ADX Strategy across 78-Stock NSE Universe
"""

import pandas as pd
import numpy as np
from framework import calc_dhan_charges, calculate_atr

df_raw = pd.read_pickle("C:/Users/admin/.gemini/antigravity/scratch/new_trade/data/universe_78_15m.pkl")
tickers = list(set([c[0] for c in df_raw.columns]))

all_trades = []

for ticker in sorted(tickers):
    sub = df_raw[ticker].dropna()
    if len(sub) < 150:
        continue
    c = sub["Close"]
    h = sub["High"]
    l = sub["Low"]
    o = sub["Open"]
    v = sub["Volume"]
    
    atr14 = calculate_atr(pd.DataFrame({"high": h, "low": l, "close": c}), 14)
    ema50 = c.ewm(span=50).mean()
    
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(7).mean()
    loss = (-delta.clip(upper=0)).rolling(7).mean()
    rsi7 = 100 - (100 / (1 + gain / (loss + 1e-9)))
    
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    sig = macd.ewm(span=9).mean()
    
    # ADX
    tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
    atr10 = tr.rolling(10).mean()
    up_move = h - h.shift(1)
    down_move = l.shift(1) - l
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_di = 100 * (pd.Series(plus_dm, index=sub.index).rolling(10).mean() / (atr10 + 1e-9))
    minus_di = 100 * (pd.Series(minus_dm, index=sub.index).rolling(10).mean() / (atr10 + 1e-9))
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
    adx = dx.rolling(10).mean()
    
    entry_cond = (
        (macd > sig) & 
        (macd.shift(1) <= sig.shift(1)) & 
        (macd > 0) & 
        (c > ema50) & 
        (rsi7 > 52) & 
        (adx > 20) & 
        (plus_di > minus_di)
    )
    
    pos = 0
    entry_p = 0.0
    active_sl = 0.0
    active_tp = 0.0
    high_seen = 0.0
    phase = 0
    entry_atr = 0.0
    p1_trig = 0.90
    p1_lock = 0.30
    p2_trig = 2.00
    p2_lock = 1.20
    tp_mult = 2.50
    sl_mult = 1.10
    
    for i in range(50, len(sub)):
        ts = sub.index[i]
        t = ts.time()
        cur_c = c.iloc[i]
        cur_h = h.iloc[i]
        cur_l = l.iloc[i]
        
        # 15:10 Square-off
        if pos == 1 and t >= pd.Timestamp("15:10").time():
            exit_p = cur_c * 0.9995
            chg = calc_dhan_charges(entry_p, exit_p, qty)
            net = (exit_p - entry_p) * qty - chg
            all_trades.append({"ticker": ticker, "net_pnl": net, "charges": chg})
            pos = 0
            continue
            
        if pos == 1:
            high_seen = max(high_seen, cur_h)
            mfe_atr = (high_seen - entry_p) / entry_atr if entry_atr > 0 else 0
            if mfe_atr >= p2_trig and phase < 2:
                phase = 2
                active_sl = max(active_sl, entry_p + p2_lock * entry_atr)
            elif mfe_atr >= p1_trig and phase < 1:
                phase = 1
                active_sl = max(active_sl, entry_p + p1_lock * entry_atr)
                
            if cur_h >= active_tp:
                exit_p = active_tp * 0.9995
                chg = calc_dhan_charges(entry_p, exit_p, qty)
                net = (exit_p - entry_p) * qty - chg
                all_trades.append({"ticker": ticker, "net_pnl": net, "charges": chg})
                pos = 0
            elif cur_l <= active_sl:
                exit_p = active_sl * 0.9995
                chg = calc_dhan_charges(entry_p, exit_p, qty)
                net = (exit_p - entry_p) * qty - chg
                all_trades.append({"ticker": ticker, "net_pnl": net, "charges": chg})
                pos = 0
                
        if pos == 0:
            time_ok = (t >= pd.Timestamp("09:20").time()) and (t <= pd.Timestamp("13:30").time())
            if entry_cond.iloc[i] and time_ok:
                pos = 1
                entry_p = cur_c * 1.0005
                entry_atr = max(atr14.iloc[i], 0.05)
                active_sl = entry_p - sl_mult * entry_atr
                active_tp = entry_p + tp_mult * entry_atr
                high_seen = entry_p
                phase = 0
                
                # Sizing: Dual constraint (Rs 500 base, 5x MIS = Rs 2,450 notional, Rs 15.50 risk cap)
                stop_dist = sl_mult * entry_atr
                friction_share = 0.002067 * entry_p
                denom = stop_dist + friction_share
                qty_risk = int(15.50 / denom) if denom > 0 else 1
                qty_notional = int(2450.0 / entry_p)
                qty = max(1, min(qty_risk, qty_notional))

df_res = pd.DataFrame(all_trades)
print(f"Total Universe Trades: {len(df_res)}")
wins = df_res[df_res["net_pnl"] > 0]
losses = df_res[df_res["net_pnl"] <= 0]
wr = len(wins) / len(df_res) * 100
tot_pnl = df_res["net_pnl"].sum()
tot_win = wins["net_pnl"].sum()
tot_loss = abs(losses["net_pnl"].sum())
pf = tot_win / tot_loss if tot_loss > 0 else 999
print(f"Win Rate       : {wr:.1f}% ({len(wins)} Wins / {len(losses)} Losses)")
print(f"Total Net PnL  : Rs {tot_pnl:.2f}")
print(f"Profit Factor  : {pf:.2f}")
print(f"Avg Win        : Rs {wins['net_pnl'].mean():.2f}")
print(f"Avg Loss       : Rs {losses['net_pnl'].mean():.2f}")
print(f"Avg Dhan Chg   : Rs {df_res['charges'].mean():.2f}")
