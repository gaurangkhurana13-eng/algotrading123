"""
Testing 5 Advanced Candidate Architectures to Beat V3
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

ema9 = c.ewm(span=9).mean()
ema21 = c.ewm(span=21).mean()
ema50 = c.ewm(span=50).mean()

delta = c.diff()
gain = delta.clip(lower=0).rolling(7).mean()
loss = (-delta.clip(upper=0)).rolling(7).mean()
rsi7 = 100 - (100 / (1 + gain / (loss + 1e-9)))

macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
sig = macd.ewm(span=9).mean()

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

v_ma20 = v.rolling(20).mean() + 1e-6
rvol = v / v_ma20

# VWAP shifted
df_temp = df.copy()
df_temp["_date"] = df_temp.index.date
tp = (df_temp["high"] + df_temp["low"] + df_temp["close"]) / 3.0
df_temp["_tpv"] = tp * df_temp["volume"]
cum_tpv = df_temp.groupby("_date")["_tpv"].cumsum()
cum_vol = df_temp.groupby("_date")["volume"].cumsum()
vwap_lag = (cum_tpv / (cum_vol + 1e-9)).shift(1)

# Squeeze
bb_mid = c.rolling(20).mean()
bb_std = c.rolling(20).std()
bb_up = bb_mid + 2.0 * bb_std
bb_dn = bb_mid - 2.0 * bb_std
kc_up = bb_mid + 1.5 * atr14
kc_dn = bb_mid - 1.5 * atr14
squeeze = (bb_up < kc_up) & (bb_dn > kc_dn)

def run_arch(entry_signal, name="", sl_mult=1.0, tp_mult=2.2, p1_trig=0.9, p1_lock=0.3, p2_trig=2.0, p2_lock=1.2, trail_after_p2=False):
    trades = []
    pos = 0
    entry_p = 0.0
    active_sl = 0.0
    active_tp = 0.0
    high_seen = 0.0
    phase = 0
    entry_atr = 0.0
    capital = 500.0
    notional_cap = 2450.0
    slippage = 0.0005
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
            
            if mfe_atr >= p2_trig and phase < 2:
                phase = 2
                active_sl = max(active_sl, entry_p + p2_lock * entry_atr)
            elif mfe_atr >= p1_trig and phase < 1:
                phase = 1
                active_sl = max(active_sl, entry_p + p1_lock * entry_atr)
                
            if trail_after_p2 and phase == 2:
                active_sl = max(active_sl, high_seen - 0.70 * entry_atr)
                
            if not trail_after_p2 and cur_h >= active_tp:
                exit_p = active_tp * (1.0 - slippage)
                gross = (exit_p - entry_p) * qty
                chg = calc_dhan_charges(entry_p, exit_p, qty, is_intraday=True)
                net = round(gross - chg, 2)
                trades.append(net)
                pos = 0
            elif cur_l <= active_sl:
                exit_p = active_sl * (1.0 - slippage)
                gross = (exit_p - entry_p) * qty
                chg = calc_dhan_charges(entry_p, exit_p, qty, is_intraday=True)
                net = round(gross - chg, 2)
                trades.append(net)
                pos = 0
                
        if pos == 0:
            time_ok = (t >= pd.Timestamp("09:20").time()) and (t <= pd.Timestamp("13:30").time())
            if entry_signal.iloc[i] and time_ok:
                pos = 1
                entry_p = cur_c * (1.0 + slippage)
                entry_atr = max(atr14.iloc[i], 0.05)
                active_sl = entry_p - sl_mult * entry_atr
                active_tp = entry_p + tp_mult * entry_atr
                high_seen = entry_p
                phase = 0
                
                stop_dist = sl_mult * entry_atr
                friction_share = 0.002067 * entry_p
                denom = stop_dist + friction_share
                qty_risk = int(15.50 / denom) if denom > 0 else 1
                qty_notional = int(notional_cap / entry_p)
                qty = max(1, min(qty_risk, qty_notional))
                
    if len(trades) < 5:
        return
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    wr = len(wins) / len(trades) * 100
    tot_pnl = sum(trades)
    pf = sum(wins) / abs(sum(losses)) if abs(sum(losses)) > 0 else 999
    print(f"{name:<55} | N={len(trades):<3} | WR={wr:5.1f}% | Net=Rs {tot_pnl:6.2f} | PF={pf:4.2f} | AvgWin=Rs {np.mean(wins) if wins else 0:5.2f} | AvgLoss=Rs {np.mean(losses) if losses else 0:5.2f}")

print(f"{'Strategy Architecture':<55} | {'N':<5} | {'WR %':<6} | {'Net (Rs)':<9} | {'PF':<5} | {'AvgWin':<6} | {'AvgLoss'}")
print("-" * 105)

# Arch 1: V3 Baseline
v3_sig = (macd > sig) & (macd.shift(1) <= sig.shift(1)) & (macd > 0) & (c > ema50) & (rsi7 > 52) & (adx > 20) & (plus_di > minus_di)
run_arch(v3_sig, name="V3 Benchmark (MACD+RSI+ADX)", sl_mult=1.0, tp_mult=2.2, p1_trig=0.9, p1_lock=0.3)

# Arch 2: V3 + VWAP Confirmation (Close > Shifted VWAP)
v3_vwap = v3_sig & (c > vwap_lag)
run_arch(v3_vwap, name="V3 + Above Shifted VWAP Anchor", sl_mult=1.0, tp_mult=2.2, p1_trig=0.9, p1_lock=0.3)

# Arch 3: V3 + Volume Acceleration (RVOL >= 1.25)
v3_vol = v3_sig & (rvol >= 1.25)
run_arch(v3_vol, name="V3 + Institutional RVOL >= 1.25", sl_mult=1.0, tp_mult=2.4, p1_trig=0.9, p1_lock=0.35)

# Arch 4: V3 with Dynamic Trailing Runner (trail after P2 instead of capping at TP)
run_arch(v3_sig, name="V3 + Dynamic Trailing Runner (Moonshot)", sl_mult=1.0, tp_mult=4.0, p1_trig=0.9, p1_lock=0.35, p2_trig=1.8, p2_lock=1.0, trail_after_p2=True)

# Arch 5: V3 + Dynamic Trailing Runner + VWAP Anchor
run_arch(v3_vwap, name="V3 + VWAP Anchor + Dynamic Trailing Runner", sl_mult=1.0, tp_mult=4.0, p1_trig=0.9, p1_lock=0.35, p2_trig=1.8, p2_lock=1.0, trail_after_p2=True)

# Arch 6: Opening Momentum Burst + V3
orb_mask = (df_temp["_time"] >= pd.Timestamp("09:15").time()) & (df_temp["_time"] <= pd.Timestamp("09:45").time()) if "_time" in df_temp.columns else pd.Series(False, index=df.index)
df_temp["_time"] = df_temp.index.time
orb_mask = (df_temp["_time"] >= pd.Timestamp("09:15").time()) & (df_temp["_time"] <= pd.Timestamp("09:45").time())
orb_h = df_temp[orb_mask].groupby("_date")["high"].max()
df_temp["orb_high"] = df_temp["_date"].map(orb_h)
orb_v3 = v3_sig & (c > df_temp["orb_high"])
run_arch(orb_v3, name="V3 + ORB High Breakout Confluence", sl_mult=1.0, tp_mult=2.4, p1_trig=0.9, p1_lock=0.35)
