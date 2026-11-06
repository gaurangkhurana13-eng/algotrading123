"""
Inspect V4/V5 trade breakdown and find what separates the 15 wins from the 6 losses.
"""

import sys
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

# SuperTrend (10, 2.5) and (7, 2.0)
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

# Keltner-Bollinger Squeeze
bb_mid = c.rolling(20).mean()
bb_std = c.rolling(20).std()
bb_up = bb_mid + 2.0 * bb_std
bb_dn = bb_mid - 2.0 * bb_std
kc_atr20 = calculate_atr(df, 20)
kc_up = bb_mid + 1.5 * kc_atr20
kc_dn = bb_mid - 1.5 * kc_atr20

squeeze_on = (bb_up < kc_up) & (bb_dn > kc_dn)
squeeze_release = squeeze_on.shift(1) & (~squeeze_on)
squeeze_coiled_3 = squeeze_on.shift(1) | squeeze_on.shift(2) | squeeze_on.shift(3)
squeeze_released_recent = (squeeze_on.shift(1) | squeeze_on.shift(2) | squeeze_on.shift(3)) & (~squeeze_on)

# Squeeze momentum (Carter squeeze momentum histogram)
# Linear regression of close - average of (donchian mid and sma20)
# Or simplified: (close - bb_mid) / atr
sq_momentum = (c - bb_mid)

entry_v4 = (
    (macd > sig) & 
    (macd.shift(1) <= sig.shift(1)) & 
    (macd > 0) & 
    (c > ema50) & 
    (c > vwap_lag) & 
    (rsi7 > 52.0) & 
    (adx > 20.0) & 
    (plus_di > minus_di)
)

def run_detailed(entry_mask, sl_mult=0.90, p1_trig=0.85, p1_lock=0.35, p2_trig=1.80, trail_dist=0.65, tp_mult=4.0):
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
        
        if pos == 1 and t >= pd.Timestamp("15:10").time():
            exit_p = cur_c * (1.0 - slippage)
            gross = (exit_p - entry_p) * qty
            chg = calc_dhan_charges(entry_p, exit_p, qty, is_intraday=True)
            net = round(gross - chg, 2)
            trades.append({
                "entry_time": entry_t, "exit_time": ts,
                "entry_price": entry_p, "exit_price": exit_p, "qty": qty,
                "net_pnl": net, "charges": chg, "reason": "15:10",
                "mfe": (high_seen - entry_p) / entry_atr,
                "entry_idx": entry_idx
            })
            pos = 0
            continue
            
        if pos == 1:
            high_seen = max(high_seen, cur_h)
            mfe_atr = (high_seen - entry_p) / entry_atr if entry_atr > 0 else 0.0
            
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
                trades.append({
                    "entry_time": entry_t, "exit_time": ts,
                    "entry_price": entry_p, "exit_price": exit_p, "qty": qty,
                    "net_pnl": net, "charges": chg, "reason": "TP",
                    "mfe": (high_seen - entry_p) / entry_atr,
                    "entry_idx": entry_idx
                })
                pos = 0
            elif cur_l <= active_sl:
                exit_p = active_sl * (1.0 - slippage)
                gross = (exit_p - entry_p) * qty
                chg = calc_dhan_charges(entry_p, exit_p, qty, is_intraday=True)
                net = round(gross - chg, 2)
                reason = "PHASE_2_TRAIL" if phase == 2 else ("PHASE_1_FLOOR" if phase == 1 else "INITIAL_SL")
                trades.append({
                    "entry_time": entry_t, "exit_time": ts,
                    "entry_price": entry_p, "exit_price": exit_p, "qty": qty,
                    "net_pnl": net, "charges": chg, "reason": reason,
                    "mfe": (high_seen - entry_p) / entry_atr,
                    "entry_idx": entry_idx
                })
                pos = 0
                
        if pos == 0:
            time_ok = (t >= pd.Timestamp("09:20").time()) and (t <= pd.Timestamp("13:30").time())
            if entry_mask.iloc[i] and time_ok:
                pos = 1
                entry_t = ts
                entry_idx = i
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
    return tdf

tdf = run_detailed(entry_v4)
print("=== V4 21 TRADES DETAIL ===")
for idx, r in tdf.iterrows():
    e_i = r["entry_idx"]
    sq_on_val = squeeze_on.iloc[e_i]
    sq_coiled_val = squeeze_coiled_3.iloc[e_i]
    sq_rel_val = squeeze_released_recent.iloc[e_i]
    st10_val = dir_10_25.iloc[e_i]
    st7_val = dir_7_20.iloc[e_i]
    rvol_val = (v.iloc[e_i] / (v.rolling(20).mean().iloc[e_i] + 1e-9))
    print(f"Trade {idx+1:2d} | Entry: {str(r['entry_time']):19s} | PnL: Rs {r['net_pnl']:6.2f} | Reason: {r['reason']:13s} | MFE: {r['mfe']:4.2f}x | ST10={st10_val} ST7={st7_val} | SqCoiled={sq_coiled_val} SqRel={sq_rel_val} | RVOL={rvol_val:.2f}")

