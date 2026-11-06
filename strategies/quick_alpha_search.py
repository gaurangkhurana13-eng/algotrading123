"""
Rapid Quantitative Strategy Search across 10 distinct rule-based indicator setups
to find architectures that produce >= 65% Win Rate and strong Profit Factor.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from framework import calc_dhan_charges, calculate_atr

df = pd.read_pickle("C:/Users/admin/.gemini/antigravity/scratch/new_trade/data/pcjeweller_15m_1y.pkl")
c = df["close"]
h = df["high"]
l = df["low"]
o = df["open"]
v = df["volume"]
atr14 = calculate_atr(df, 14)

def run_sim(entry_signal, exit_signal=None, tp_atr=1.8, sl_atr=1.2, name=""):
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
            # Profit ratchet
            mfe_atr = (high_seen - entry_p) / entry_atr if entry_atr > 0 else 0
            if mfe_atr >= 1.2 and phase == 0:
                phase = 1
                active_sl = entry_p + 0.3 * entry_atr
            elif mfe_atr >= 2.0 and phase < 2:
                phase = 2
                active_sl = entry_p + 1.0 * entry_atr
                
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
            elif exit_signal is not None and exit_signal.iloc[i]:
                exit_p = cur_c * 0.9995
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
    print(f"{name:<45} | N={len(trades):<3} | WR={wr:5.1f}% | Net={tot_pnl:8.1f} | PF={pf:4.2f} | AvgW={np.mean(wins) if wins else 0:5.2f} | AvgL={np.mean(losses) if losses else 0:5.2f}")

print(f"{'Strategy Name':<45} | {'N':<5} | {'WR %':<6} | {'Net (Rs)':<9} | {'PF':<5} | {'AvgWin':<6} | {'AvgLoss'}")
print("-" * 95)

# Setup 1: EMA 9/21 Pullback
ema9 = c.ewm(span=9).mean()
ema21 = c.ewm(span=21).mean()
ema50 = c.ewm(span=50).mean()
rsi7 = pd.Series(index=df.index, dtype=float)
delta = c.diff()
gain = delta.clip(lower=0).rolling(7).mean()
loss = (-delta.clip(upper=0)).rolling(7).mean()
rs = gain / (loss + 1e-9)
rsi7 = 100 - (100 / (1 + rs))

pullback_entry = (ema9 > ema21) & (c > ema50) & (l <= ema21 * 1.003) & (c > ema9) & (c > o) & (rsi7 < 60)
run_sim(pullback_entry, tp_atr=1.8, sl_atr=1.2, name="EMA 9/21 Value Pullback")
run_sim(pullback_entry, tp_atr=2.2, sl_atr=1.0, name="EMA 9/21 Value Pullback (Tight SL)")
run_sim(pullback_entry, tp_atr=1.4, sl_atr=0.9, name="EMA 9/21 Quick Scalp")

# Setup 2: SuperTrend
def calc_supertrend(df, period=10, mult=3.0):
    atr = calculate_atr(df, period)
    hl2 = (df['high'] + df['low']) / 2
    upper = hl2 + mult * atr
    lower = hl2 - mult * atr
    st = np.zeros(len(df))
    trend = np.zeros(len(df))
    for i in range(1, len(df)):
        if df['close'].iloc[i-1] > st[i-1]:
            st[i] = max(lower.iloc[i], st[i-1]) if df['close'].iloc[i] > st[i-1] else upper.iloc[i]
            trend[i] = 1 if df['close'].iloc[i] > st[i-1] else -1
        else:
            st[i] = min(upper.iloc[i], st[i-1]) if df['close'].iloc[i] < st[i-1] else lower.iloc[i]
            trend[i] = -1 if df['close'].iloc[i] < st[i-1] else 1
    return pd.Series(trend, index=df.index)

st10_3 = calc_supertrend(df, 10, 3.0)
st7_2 = calc_supertrend(df, 7, 2.0)
st_entry = (st10_3 == 1) & (st10_3.shift(1) == -1) & (c > ema50)
run_sim(st_entry, tp_atr=2.5, sl_atr=1.5, name="SuperTrend (10, 3.0) Cross")

# Setup 3: Donchian 20 Channel Breakout
dc_high20 = h.rolling(20).max().shift(1)
dc_entry = (c > dc_high20) & (v > v.rolling(20).mean() * 1.3)
run_sim(dc_entry, tp_atr=2.0, sl_atr=1.2, name="Donchian 20 Breakout + Vol Surge")

# Setup 4: Keltner-Bollinger Squeeze Release
bb_mid = c.rolling(20).mean()
bb_std = c.rolling(20).std()
bb_up = bb_mid + 2.0 * bb_std
bb_dn = bb_mid - 2.0 * bb_std
kc_up = bb_mid + 1.5 * atr14
kc_dn = bb_mid - 1.5 * atr14
squeeze = (bb_up < kc_up) & (bb_dn > kc_dn)
squeeze_release = squeeze.shift(1) & ~squeeze & (c > bb_mid) & (c > o)
run_sim(squeeze_release, tp_atr=2.2, sl_atr=1.2, name="Keltner-Bollinger Squeeze Release")

# Setup 5: Morning Opening Range Breakout (09:15 - 09:45 ORB)
df["_time"] = df.index.time
df["_date"] = df.index.date
orb_mask = (df["_time"] >= pd.Timestamp("09:15").time()) & (df["_time"] <= pd.Timestamp("09:45").time())
orb_h = df[orb_mask].groupby("_date")["high"].max()
df["orb_high"] = df["_date"].map(orb_h)
orb_entry = (df["_time"] > pd.Timestamp("09:45").time()) & (df["_time"] <= pd.Timestamp("11:30").time()) & (c > df["orb_high"]) & (c.shift(1) <= df["orb_high"]) & (v > v.rolling(20).mean() * 1.2)
run_sim(orb_entry, tp_atr=2.0, sl_atr=1.2, name="Opening Range Breakout (ORB 09:45)")

# Setup 6: MACD Zero-Line Cross + RSI Bullish
macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
sig = macd.ewm(span=9).mean()
macd_entry = (macd > sig) & (macd.shift(1) <= sig.shift(1)) & (macd > 0) & (c > ema50) & (rsi7 > 50)
run_sim(macd_entry, tp_atr=2.0, sl_atr=1.2, name="MACD Zero-Line Cross + RSI Bullish")
