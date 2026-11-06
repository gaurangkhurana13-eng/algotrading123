"""
Testing Strategy V5 Candidate:
Squeeze Expansion + MACD Zero-Line + Fast RSI + Moonshot Ratchet
"""

import pandas as pd
import numpy as np
from framework import calc_dhan_charges, calculate_atr, compute_metrics

df = pd.read_pickle("C:/Users/admin/.gemini/antigravity/scratch/new_trade/data/pcjeweller_15m_1y.pkl")
c = df["close"]
h = df["high"]
l = df["low"]
o = df["open"]
v = df["volume"]
atr14 = calculate_atr(df, 14)

ema50 = c.ewm(span=50).mean()

# VWAP shifted
df_temp = df.copy()
df_temp["_date"] = df_temp.index.date
tp = (df_temp["high"] + df_temp["low"] + df_temp["close"]) / 3.0
df_temp["_tpv"] = tp * df_temp["volume"]
cum_tpv = df_temp.groupby("_date")["_tpv"].cumsum()
cum_vol = df_temp.groupby("_date")["volume"].cumsum()
vwap_lag = (cum_tpv / (cum_vol + 1e-9)).shift(1)

# RSI 7
delta = c.diff()
gain = delta.clip(lower=0).rolling(7).mean()
loss = (-delta.clip(upper=0)).rolling(7).mean()
rsi7 = 100 - (100 / (1 + gain / (loss + 1e-9)))

# MACD 12/26/9
macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
sig = macd.ewm(span=9).mean()

# ADX 10
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

# Bollinger & Keltner Squeeze
bb_mid = c.rolling(20).mean()
bb_std = c.rolling(20).std()
bb_up = bb_mid + 2.0 * bb_std
bb_dn = bb_mid - 2.0 * bb_std
kc_up = bb_mid + 1.5 * atr14
kc_dn = bb_mid - 1.5 * atr14
squeeze = (bb_up < kc_up) & (bb_dn > kc_dn)
squeeze_coiled = squeeze.rolling(5).max() == 1

# Relative volume
v_ma20 = v.rolling(20).mean() + 1e-6
rvol = v / v_ma20

# Entry: MACD Bullish Cross + RSI > 50 + ADX > 18 + Close > 50 EMA + Close > Shifted VWAP
entry_cond = (
    (macd > sig) & 
    (macd.shift(1) <= sig.shift(1)) & 
    (macd > 0) & 
    (c > ema50) & 
    (c > vwap_lag) & 
    (rsi7 > 50.0) & 
    (adx > 18.0) & 
    (plus_di > minus_di)
)

def run_v5_test(trail_dist=0.65, p1_lock=0.40, p2_trig=1.75):
    capital = 500.0
    notional_cap = 2450.0
    slippage = 0.0005
    sl_mult = 1.00
    p1_trig = 0.85
    trades = []
    pos = 0
    entry_p = 0.0
    active_sl = 0.0
    high_seen = 0.0
    phase = 0
    entry_atr = 0.0
    qty = 0
    
    for i in range(100, len(df)):
        ts = df.index[i]
        t = ts.time()
        cur_c = c.iloc[i]
        cur_h = h.iloc[i]
        cur_l = l.iloc[i]
        
        if pos == 1 and t >= pd.Timestamp("15:10").time():
            exit_p = cur_c * (1.0 - slippage)
            gross = (exit_p - entry_p) * qty
            chg = calc_dhan_charges(entry_p, exit_p, qty, is_intraday=True)
            net = round(gross - chg, 2)
            trades.append(net)
            pos = 0
            continue
            
        if pos == 1:
            high_seen = max(high_seen, cur_h)
            mfe_atr = (high_seen - entry_p) / entry_atr if entry_atr > 0 else 0
            
            if mfe_atr >= p2_trig:
                phase = 2
                active_sl = max(active_sl, high_seen - (trail_dist * entry_atr))
            elif mfe_atr >= p1_trig and phase < 1:
                phase = 1
                active_sl = max(active_sl, entry_p + (p1_lock * entry_atr))
                
            if cur_l <= active_sl:
                exit_p = active_sl * (1.0 - slippage)
                gross = (exit_p - entry_p) * qty
                chg = calc_dhan_charges(entry_p, exit_p, qty, is_intraday=True)
                net = round(gross - chg, 2)
                trades.append(net)
                pos = 0
                
        if pos == 0:
            time_ok = (t >= pd.Timestamp("09:20").time()) and (t <= pd.Timestamp("13:30").time())
            if entry_cond.iloc[i] and time_ok:
                pos = 1
                entry_p = cur_c * (1.0 + slippage)
                entry_atr = max(atr14.iloc[i], 0.05)
                active_sl = entry_p - sl_mult * entry_atr
                high_seen = entry_p
                phase = 0
                
                stop_dist = sl_mult * entry_atr
                friction_share = 0.002067 * entry_p
                denom = stop_dist + friction_share
                qty_risk = int(15.50 / denom) if denom > 0 else 1
                qty_notional = int(notional_cap / entry_p)
                qty = max(1, min(qty_risk, qty_notional))
                
    tdf = pd.Series(trades)
    wins = tdf[tdf > 0]
    losses = tdf[tdf <= 0]
    wr = len(wins) / len(tdf) * 100
    tot_pnl = tdf.sum()
    pf = wins.sum() / abs(losses.sum()) if abs(losses.sum()) > 0 else 999
    print(f"trail={trail_dist}, p1_lock={p1_lock}, p2_trig={p2_trig} | N={len(tdf)} | WR={wr:.1f}% | Net=Rs {tot_pnl:.2f} | PF={pf:.2f} | AvgW=Rs {wins.mean():.2f} | AvgL=Rs {losses.mean():.2f}")

run_v5_test(trail_dist=0.65, p1_lock=0.40, p2_trig=1.75)
run_v5_test(trail_dist=0.60, p1_lock=0.45, p2_trig=1.70)
run_v5_test(trail_dist=0.70, p1_lock=0.35, p2_trig=1.80)
