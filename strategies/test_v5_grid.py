"""
Grid Search and Alpha Exploration for Strategy V5:
SuperTrend + Keltner-Bollinger Squeeze Release + VWAP Anchor + Dynamic Moonshot Trailing Ratchet
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from framework import calc_dhan_charges, calculate_atr, compute_metrics

data_file = Path("C:/Users/admin/.gemini/antigravity/scratch/new_trade/data/pcjeweller_15m_1y.pkl")
df = pd.read_pickle(data_file)
c = df["close"]
h = df["high"]
l = df["low"]
o = df["open"]
v = df["volume"]

atr14 = calculate_atr(df, 14)
ema50 = c.ewm(span=50).mean()
ema20 = c.ewm(span=20).mean()
ema9 = c.ewm(span=9).mean()
ema21 = c.ewm(span=21).mean()

# Daily VWAP Shifted
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

# 1. SuperTrend Functions
def compute_supertrend(df, period=10, mult=2.5):
    atr = calculate_atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2.0
    basic_upper = hl2 + mult * atr
    basic_lower = hl2 - mult * atr
    final_upper = basic_upper.copy()
    final_lower = basic_lower.copy()
    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    curr_dir = 1
    for i in range(1, len(df)):
        if basic_lower.iloc[i] > final_lower.iloc[i-1] or df["close"].iloc[i-1] < final_lower.iloc[i-1]:
            final_lower.iloc[i] = basic_lower.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i-1]
            
        if basic_upper.iloc[i] < final_upper.iloc[i-1] or df["close"].iloc[i-1] > final_upper.iloc[i-1]:
            final_upper.iloc[i] = basic_upper.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i-1]
            
        if curr_dir == 1:
            if df["close"].iloc[i] < final_lower.iloc[i]:
                curr_dir = -1
                st.iloc[i] = final_upper.iloc[i]
            else:
                st.iloc[i] = final_lower.iloc[i]
        else:
            if df["close"].iloc[i] > final_upper.iloc[i]:
                curr_dir = 1
                st.iloc[i] = final_lower.iloc[i]
            else:
                st.iloc[i] = final_upper.iloc[i]
        direction.iloc[i] = curr_dir
    return st, direction

st_10_25, dir_10_25 = compute_supertrend(df, period=10, mult=2.5)
st_7_20, dir_7_20 = compute_supertrend(df, period=7, mult=2.0)

# 2. Keltner-Bollinger Squeeze
# Standard Carter Squeeze: BB (20, 2.0) vs KC (20, 1.5)
bb_mid = c.rolling(20).mean()
bb_std = c.rolling(20).std()
bb_up = bb_mid + 2.0 * bb_std
bb_dn = bb_mid - 2.0 * bb_std

# KC with 1.5 ATR (standard) and 1.0 or 2.0
kc_atr20 = calculate_atr(df, 20)
kc_up_15 = bb_mid + 1.5 * kc_atr20
kc_dn_15 = bb_mid - 1.5 * kc_atr20

squeeze_on_15 = (bb_up < kc_up_15) & (bb_dn > kc_dn_15)
# Squeeze release triggers
sq_rel_exact = squeeze_on_15.shift(1) & (~squeeze_on_15)
sq_rel_lookback2 = (squeeze_on_15.shift(1) | squeeze_on_15.shift(2)) & (~squeeze_on_15)
sq_rel_lookback3 = (squeeze_on_15.shift(1) | squeeze_on_15.shift(2) | squeeze_on_15.shift(3)) & (~squeeze_on_15)
sq_rel_lookback5 = (squeeze_on_15.shift(1) | squeeze_on_15.shift(2) | squeeze_on_15.shift(3) | squeeze_on_15.shift(4) | squeeze_on_15.shift(5)) & (~squeeze_on_15)

# KC with 1.0 ATR
kc_up_10 = bb_mid + 1.0 * kc_atr20
kc_dn_10 = bb_mid - 1.0 * kc_atr20
squeeze_on_10 = (bb_up < kc_up_10) & (bb_dn > kc_dn_10)
sq_rel_10_lookback3 = (squeeze_on_10.shift(1) | squeeze_on_10.shift(2) | squeeze_on_10.shift(3)) & (~squeeze_on_10)

# Backtest Engine
def backtest_strategy(entry_cond, sl_mult=1.0, p1_trig=0.85, p1_lock=0.35, p2_trig=1.80, trail_dist=0.65, tp_mult=4.0):
    capital = 500.0
    notional_cap = 2450.0
    r_cap = 15.50
    slippage = 0.0005
    trades = []
    pos = 0
    entry_p = 0.0
    qty = 0
    active_sl = 0.0
    active_tp = 0.0
    high_seen = 0.0
    phase = 0
    entry_atr = 0.0
    entry_t = None
    
    n = len(df)
    for i in range(100, n):
        ts = df.index[i]
        t = ts.time()
        cur_c = c.iloc[i]
        cur_h = h.iloc[i]
        cur_l = l.iloc[i]
        
        # 15:10 square off
        if pos == 1 and t >= pd.Timestamp("15:10").time():
            exit_p = cur_c * (1.0 - slippage)
            gross = (exit_p - entry_p) * qty
            chg = calc_dhan_charges(entry_p, exit_p, qty, is_intraday=True)
            net = round(gross - chg, 2)
            trades.append({"net_pnl": net, "gross_pnl": gross, "charges": chg, "reason": "15:10", "entry_p": entry_p, "exit_p": exit_p, "qty": qty})
            pos = 0
            continue
            
        if pos == 1:
            high_seen = max(high_seen, cur_h)
            mfe_atr = (high_seen - entry_p) / entry_atr if entry_atr > 0 else 0.0
            
            # Dynamic Moonshot Ratchet
            if mfe_atr >= p2_trig:
                phase = 2
                active_sl = max(active_sl, high_seen - (trail_dist * entry_atr))
            elif mfe_atr >= p1_trig and phase < 1:
                phase = 1
                active_sl = max(active_sl, entry_p + (p1_lock * entry_atr))
                
            if tp_mult > 0 and cur_h >= active_tp:
                exit_p = active_tp * (1.0 - slippage)
                gross = (exit_p - entry_p) * qty
                chg = calc_dhan_charges(entry_p, exit_p, qty, is_intraday=True)
                net = round(gross - chg, 2)
                trades.append({"net_pnl": net, "gross_pnl": gross, "charges": chg, "reason": "TP", "entry_p": entry_p, "exit_p": exit_p, "qty": qty})
                pos = 0
            elif cur_l <= active_sl:
                exit_p = active_sl * (1.0 - slippage)
                gross = (exit_p - entry_p) * qty
                chg = calc_dhan_charges(entry_p, exit_p, qty, is_intraday=True)
                net = round(gross - chg, 2)
                reason = "PHASE_2_TRAIL" if phase == 2 else ("PHASE_1_FLOOR" if phase == 1 else "INITIAL_SL")
                trades.append({"net_pnl": net, "gross_pnl": gross, "charges": chg, "reason": reason, "entry_p": entry_p, "exit_p": exit_p, "qty": qty})
                pos = 0
                
        if pos == 0:
            time_ok = (t >= pd.Timestamp("09:20").time()) and (t <= pd.Timestamp("13:30").time())
            if entry_cond.iloc[i] and time_ok:
                pos = 1
                entry_t = ts
                entry_p = cur_c * (1.0 + slippage)
                entry_atr = max(atr14.iloc[i], 0.05)
                
                stop_dist = sl_mult * entry_atr
                friction_share = 0.002067 * entry_p
                denom = stop_dist + friction_share
                
                qty_risk = int(r_cap / denom) if denom > 0 else 1
                qty_notional = int(notional_cap / entry_p)
                qty = max(1, min(qty_risk, qty_notional))
                
                active_sl = entry_p - stop_dist
                active_tp = entry_p + tp_mult * entry_atr if tp_mult > 0 else 999999.0
                high_seen = entry_p
                phase = 0
                
    tdf = pd.DataFrame(trades)
    return compute_metrics(tdf, capital=500.0)

print("Starting Candidate Exploration...")

candidates = []

# Candidate definitions:
# A. SuperTrend Cross + Squeeze in lookback + VWAP
for st_name, st_dir in [("ST(10,2.5)", dir_10_25), ("ST(7,2.0)", dir_7_20)]:
    st_cross = (st_dir == 1) & (st_dir.shift(1) == -1)
    st_cross_3 = st_cross | st_cross.shift(1) | st_cross.shift(2)
    st_bull = (st_dir == 1)
    
    for sq_name, sq_rel in [
        ("SqRel_exact", sq_rel_exact),
        ("SqRel_3bar", sq_rel_lookback3),
        ("SqRel_5bar", sq_rel_lookback5),
        ("SqRel_10_3bar", sq_rel_10_lookback3)
    ]:
        # Combo 1: ST Bullish AND SqRel fired AND c > vwap_lag
        c1 = st_bull & sq_rel & (c > vwap_lag)
        candidates.append((f"{st_name} Bull + {sq_name} + VWAP", c1))
        
        # Combo 2: ST Cross AND SqRel (or recent) AND c > vwap_lag
        c2 = st_cross_3 & sq_rel & (c > vwap_lag)
        candidates.append((f"{st_name} Cross3 + {sq_name} + VWAP", c2))
        
        # Combo 3: ST Bull + SqRel + VWAP + Macro EMA50
        c3 = st_bull & sq_rel & (c > vwap_lag) & (c > ema50)
        candidates.append((f"{st_name} Bull + {sq_name} + VWAP + EMA50", c3))
        
        # Combo 4: ST Bull + SqRel + VWAP + EMA50 + RSI7 > 50
        c4 = st_bull & sq_rel & (c > vwap_lag) & (c > ema50) & (rsi7 > 50)
        candidates.append((f"{st_name} Bull + {sq_name} + VWAP + EMA50 + RSI>50", c4))

        # Combo 5: ST Cross + SqRel (or recent squeeze) + VWAP + RSI>50
        c5 = st_cross & (sq_rel | squeeze_on_15.shift(1) | squeeze_on_15.shift(2)) & (c > vwap_lag)
        candidates.append((f"{st_name} Cross + SqCoiled + VWAP", c5))

        # Combo 6: V4 Base (MACD+ADX+RSI) + ST Bull + SqRel
        c6 = (macd > sig) & (macd.shift(1) <= sig.shift(1)) & (macd > 0) & (c > ema50) & (rsi7 > 52.0) & (adx > 18.0) & (plus_di > minus_di) & (c > vwap_lag) & st_bull & (sq_rel | squeeze_on_15.shift(1) | squeeze_on_15.shift(2) | squeeze_on_15.shift(3))
        candidates.append((f"V4 + {st_name} Bull + SqCoilOrRel", c6))

        # Combo 7: V4 Base + ST Bull
        c7 = (macd > sig) & (macd.shift(1) <= sig.shift(1)) & (macd > 0) & (c > ema50) & (rsi7 > 52.0) & (adx > 18.0) & (plus_di > minus_di) & (c > vwap_lag) & st_bull
        candidates.append((f"V4 + {st_name} Bull", c7))

        # Combo 8: ST Bull + SqRel + MACD > Sig + VWAP
        c8 = st_bull & sq_rel & (macd > sig) & (c > vwap_lag) & (c > ema50)
        candidates.append((f"{st_name} Bull + {sq_name} + MACD>Sig + VWAP", c8))

print(f"Total candidate setups: {len(candidates)}")

# Test on ratchet variations:
ratchet_configs = [
    # (sl_mult, p1_trig, p1_lock, p2_trig, trail_dist, tp_mult, label)
    (0.85, 0.85, 0.35, 1.80, 0.65, 4.0, "SL0.85 P1=0.85/0.35 P2=1.80/0.65 TP4.0"),
    (1.00, 0.85, 0.35, 1.80, 0.65, 4.0, "SL1.00 P1=0.85/0.35 P2=1.80/0.65 TP4.0"),
    (0.85, 0.85, 0.40, 1.60, 0.65, 4.0, "SL0.85 P1=0.85/0.40 P2=1.60/0.65 TP4.0"),
    (1.00, 0.85, 0.40, 1.60, 0.65, 4.0, "SL1.00 P1=0.85/0.40 P2=1.60/0.65 TP4.0"),
    (0.85, 0.85, 0.85, 1.80, 0.65, 4.0, "SL0.85 P1=0.85/0.85 P2=1.80/0.65 TP4.0"), # lock at +0.85
    (0.90, 0.85, 0.35, 1.80, 0.65, 4.0, "SL0.90 P1=0.85/0.35 P2=1.80/0.65 TP4.0"),
    (1.00, 0.85, 0.35, 1.70, 0.65, 4.0, "SL1.00 P1=0.85/0.35 P2=1.70/0.65 TP4.0"),
]

best_results = []
for name, cond in candidates:
    n_raw = cond.sum()
    if n_raw < 5:
        continue
    for sl_mult, p1_trig, p1_lock, p2_trig, trail_dist, tp_mult, r_label in ratchet_configs:
        m = backtest_strategy(cond, sl_mult=sl_mult, p1_trig=p1_trig, p1_lock=p1_lock, p2_trig=p2_trig, trail_dist=trail_dist, tp_mult=tp_mult)
        if m["total_trades"] >= 10:
            best_results.append((m["profit_factor"], m["win_rate"], m["net_pnl"], m["total_trades"], name, r_label, m))

# Sort by PF descending, then Win Rate
best_results.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)

print("\n--- TOP 25 CANDIDATES BY PROFIT FACTOR ---")
for pf, wr, pnl, n, name, r_label, m in best_results[:25]:
    print(f"PF={pf:5.2f} | WR={wr:5.1f}% | Net=Rs {pnl:7.2f} | N={n:3} | Win={m['avg_win']:5.2f} | Loss={m['avg_loss']:6.2f} | {name} | {r_label}")

print("\n--- TOP 25 CANDIDATES BY WIN RATE (N >= 12) ---")
best_by_wr = [r for r in best_results if r[3] >= 12]
best_by_wr.sort(key=lambda x: (x[1], x[0], x[2]), reverse=True)
for pf, wr, pnl, n, name, r_label, m in best_by_wr[:25]:
    print(f"WR={wr:5.1f}% | PF={pf:5.2f} | Net=Rs {pnl:7.2f} | N={n:3} | Win={m['avg_win']:5.2f} | Loss={m['avg_loss']:6.2f} | {name} | {r_label}")
