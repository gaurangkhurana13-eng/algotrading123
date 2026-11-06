"""
Testing Combinations of MACD Zero Cross + RSI Bullish + Squeeze + Profit Ratchets
"""

import pandas as pd
import numpy as np
from framework import calc_dhan_charges, calculate_atr

df = pd.read_pickle("C:/Users/admin/.gemini/antigravity/scratch/new_trade/data/pcjeweller_15m_1y.pkl")
c = df["close"]
h = df["high"]
l = df["low"]
o = df["open"]
v = df["volume"]
atr14 = calculate_atr(df, 14)

# Indicators
ema9 = c.ewm(span=9).mean()
ema21 = c.ewm(span=21).mean()
ema50 = c.ewm(span=50).mean()
ema200 = c.ewm(span=200).mean()

delta = c.diff()
gain = delta.clip(lower=0).rolling(7).mean()
loss = (-delta.clip(upper=0)).rolling(7).mean()
rsi7 = 100 - (100 / (1 + gain / (loss + 1e-9)))

macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
sig = macd.ewm(span=9).mean()
macd_hist = macd - sig

# Squeeze
bb_mid = c.rolling(20).mean()
bb_std = c.rolling(20).std()
bb_up = bb_mid + 2.0 * bb_std
bb_dn = bb_mid - 2.0 * bb_std
kc_up = bb_mid + 1.5 * atr14
kc_dn = bb_mid - 1.5 * atr14
squeeze = (bb_up < kc_up) & (bb_dn > kc_dn)
squeeze_recent = squeeze.rolling(4).max() == 1

# ADX
tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
atr10 = tr.rolling(10).mean()
up_move = h - h.shift(1)
down_move = l.shift(1) - l
plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(10).mean() / (atr10 + 1e-9))
minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(10).mean() / (atr10 + 1e-9))
dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
adx = dx.rolling(10).mean()

def run_combo(entry_signal, p1_trig=1.5, p1_lock=0.6, p2_trig=2.5, p2_lock=1.5, tp_atr=3.5, sl_atr=1.3, name=""):
    trades = []
    pos = 0
    entry_p = 0.0
    active_sl = 0.0
    active_tp = 0.0
    high_seen = 0.0
    phase = 0
    entry_atr = 0.0
    
    for i in range(100, len(df)):
        ts = df.index[i]
        t = ts.time()
        cur_c = c.iloc[i]
        cur_h = h.iloc[i]
        cur_l = l.iloc[i]
        
        # Intraday MIS exit at 15:10
        if pos == 1 and t >= pd.Timestamp("15:10").time():
            exit_p = cur_c * 0.9995
            net = (exit_p - entry_p) * 100 - calc_dhan_charges(entry_p, exit_p, 100)
            trades.append(net)
            pos = 0
            continue
            
        if pos == 1:
            high_seen = max(high_seen, cur_h)
            mfe_atr = (high_seen - entry_p) / entry_atr if entry_atr > 0 else 0
            
            # Dynamic Asymmetric Ratchet
            if mfe_atr >= p2_trig and phase < 2:
                phase = 2
                active_sl = max(active_sl, entry_p + p2_lock * entry_atr)
            elif mfe_atr >= p1_trig and phase < 1:
                phase = 1
                active_sl = max(active_sl, entry_p + p1_lock * entry_atr)
                
            if cur_h >= active_tp:
                exit_p = active_tp * 0.9995
                net = (exit_p - entry_p) * 100 - calc_dhan_charges(entry_p, exit_p, 100)
                trades.append(net)
                pos = 0
            elif cur_l <= active_sl:
                exit_p = active_sl * 0.9995
                net = (exit_p - entry_p) * 100 - calc_dhan_charges(entry_p, exit_p, 100)
                trades.append(net)
                pos = 0
                
        if pos == 0:
            time_ok = (t >= pd.Timestamp("09:20").time()) and (t <= pd.Timestamp("13:30").time())
            if entry_signal.iloc[i] and time_ok:
                pos = 1
                entry_p = cur_c * 1.0005
                entry_atr = max(atr14.iloc[i], 0.05)
                active_sl = entry_p - sl_atr * entry_atr
                active_tp = entry_p + tp_atr * entry_atr
                high_seen = entry_p
                phase = 0
                
    if len(trades) < 5:
        return
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    wr = len(wins) / len(trades) * 100
    tot_pnl = sum(trades)
    pf = sum(wins) / abs(sum(losses)) if abs(sum(losses)) > 0 else 999
    print(f"{name:<50} | N={len(trades):<3} | WR={wr:5.1f}% | Net={tot_pnl:8.1f} | PF={pf:4.2f} | AvgW={np.mean(wins) if wins else 0:5.2f} | AvgL={np.mean(losses) if losses else 0:5.2f}")

print(f"{'Strategy Combo Name':<50} | {'N':<5} | {'WR %':<6} | {'Net (Rs)':<9} | {'PF':<5} | {'AvgWin':<6} | {'AvgLoss'}")
print("-" * 105)

# Test 1: Base MACD + RSI
c1 = (macd > sig) & (macd.shift(1) <= sig.shift(1)) & (macd > 0) & (c > ema50) & (rsi7 > 52)
run_combo(c1, p1_trig=1.2, p1_lock=0.4, tp_atr=3.0, sl_atr=1.2, name="MACD Bull + RSI > 52 (P1: 1.2R -> 0.4R)")
run_combo(c1, p1_trig=1.0, p1_lock=0.3, tp_atr=2.5, sl_atr=1.1, name="MACD Bull + RSI > 52 (Early Lock 1.0R)")

# Test 2: MACD + Squeeze Breakout Confluence
c2 = c1 & squeeze_recent
run_combo(c2, p1_trig=1.2, p1_lock=0.5, tp_atr=3.2, sl_atr=1.2, name="MACD + Post-Squeeze Expansion")

# Test 3: MACD + ADX Trend Confirmation (ADX > 20 and +DI > -DI)
c3 = c1 & (adx > 20) & (plus_di > minus_di)
run_combo(c3, p1_trig=1.1, p1_lock=0.4, tp_atr=2.8, sl_atr=1.2, name="MACD + RSI + ADX > 20 Trend")
run_combo(c3, p1_trig=0.9, p1_lock=0.3, tp_atr=2.2, sl_atr=1.0, name="MACD + RSI + ADX (Ultra-Fast Lock 0.9R)")

# Test 4: Dynamic Re-entry dip in trend (MACD bullish & price touches EMA 21 then closes green)
c4 = (macd > sig) & (c > ema50) & (l <= ema21 * 1.002) & (c > ema9) & (c > o)
run_combo(c4, p1_trig=1.2, p1_lock=0.5, tp_atr=2.5, sl_atr=1.1, name="Trend Dip to EMA 21 + MACD Bullish")
