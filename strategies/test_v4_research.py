"""
Research and Optimization for Strategy V4:
Adaptive Volatility Target, Dynamic Pullback Re-entries, and Tight Breakeven Ratchet.
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

# Order Flow CLV
bar_rng = h - l + 1e-6
clv = ((c - l) - (h - c)) / bar_rng

# Primary Breakout Entry
primary_entry = (
    (macd > sig) & 
    (macd.shift(1) <= sig.shift(1)) & 
    (macd > 0) & 
    (c > ema50) & 
    (rsi7 > 52.0) & 
    (adx > 18.0) & 
    (plus_di > minus_di)
)

# Secondary Pullback Re-entry: Trend is strong, price pulls back to EMA 21, then closes green above EMA 9 with positive CLV
pullback_reentry = (
    (macd > sig) & 
    (c > ema50) & 
    (ema9 > ema21) & 
    (l <= ema21 * 1.002) & 
    (c > ema9) & 
    (c > o) & 
    (clv >= 0.55) & 
    (adx > 18.0)
)

all_entries = primary_entry | pullback_reentry

def test_v4_sim(tp_base=2.6, sl_base=0.85, p1_trig=0.85, p1_lock=0.35, p2_trig=1.80, p2_lock=1.20):
    capital = 500.0
    notional_cap = 2450.0
    slippage = 0.0005
    trades = []
    pos = 0
    entry_p = 0.0
    active_sl = 0.0
    active_tp = 0.0
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
            trades.append({"net": net, "reason": "15:10", "chg": chg})
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
                exit_p = active_tp * (1.0 - slippage)
                gross = (exit_p - entry_p) * qty
                chg = calc_dhan_charges(entry_p, exit_p, qty, is_intraday=True)
                net = round(gross - chg, 2)
                trades.append({"net": net, "reason": "TP", "chg": chg})
                pos = 0
            elif cur_l <= active_sl:
                exit_p = active_sl * (1.0 - slippage)
                gross = (exit_p - entry_p) * qty
                chg = calc_dhan_charges(entry_p, exit_p, qty, is_intraday=True)
                net = round(gross - chg, 2)
                trades.append({"net": net, "reason": "SL_RATCHET", "chg": chg})
                pos = 0
                
        if pos == 0:
            time_ok = (t >= pd.Timestamp("09:20").time()) and (t <= pd.Timestamp("13:30").time())
            if all_entries.iloc[i] and time_ok:
                pos = 1
                entry_p = cur_c * (1.0 + slippage)
                entry_atr = max(atr14.iloc[i], 0.05)
                
                # Volatility-adaptive TP: If ATR expanding, expand target
                atr_expanding = entry_atr > atr14.iloc[i-1]
                eff_tp = tp_base + (0.4 if atr_expanding else 0.0)
                
                active_sl = entry_p - sl_base * entry_atr
                active_tp = entry_p + eff_tp * entry_atr
                high_seen = entry_p
                phase = 0
                
                stop_dist = sl_base * entry_atr
                friction_share = 0.002067 * entry_p
                denom = stop_dist + friction_share
                qty_risk = int(15.50 / denom) if denom > 0 else 1
                qty_notional = int(notional_cap / entry_p)
                qty = max(1, min(qty_risk, qty_notional))
                
    tdf = pd.DataFrame(trades)
    wins = tdf[tdf["net"] > 0]
    losses = tdf[tdf["net"] <= 0]
    wr = len(wins) / len(tdf) * 100
    tot_pnl = tdf["net"].sum()
    tot_win = wins["net"].sum()
    tot_loss = abs(losses["net"].sum())
    pf = tot_win / tot_loss if tot_loss > 0 else 999
    print(f"TP={tp_base}, SL={sl_base} | N={len(tdf):<3} | WR={wr:5.1f}% | Net=Rs {tot_pnl:6.2f} | PF={pf:4.2f} | AvgWin=Rs {wins['net'].mean():5.2f} | AvgLoss=Rs {losses['net'].mean():5.2f}")

print("Testing V4 Parameter Variants:")
test_v4_sim(tp_base=2.4, sl_base=0.85, p1_trig=0.85, p1_lock=0.35)
test_v4_sim(tp_base=2.6, sl_base=0.85, p1_trig=0.85, p1_lock=0.35)
test_v4_sim(tp_base=2.8, sl_base=0.90, p1_trig=0.90, p1_lock=0.40)
test_v4_sim(tp_base=2.2, sl_base=0.80, p1_trig=0.80, p1_lock=0.30)
