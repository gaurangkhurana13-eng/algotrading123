import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    univ = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_high_beta_10_15m_1y.pkl", "rb") as f:
    d_beta = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_10_15m_1y.pkl", "rb") as f:
    d_unseen = pickle.load(f)

all_pools = {}
all_pools.update(univ)
all_pools.update(d_beta)
all_pools.update(d_unseen)

def compute_vwap_shifted(df: pd.DataFrame) -> pd.Series:
    df_temp = df.copy()
    df_temp["_date"] = df_temp.index.date
    tp = (df_temp["high"] + df_temp["low"] + df_temp["close"]) / 3.0
    df_temp["_tpv"] = tp * df_temp["volume"]
    cum_tpv = df_temp.groupby("_date")["_tpv"].cumsum()
    cum_vol = df_temp.groupby("_date")["volume"].cumsum()
    return (cum_tpv / (cum_vol + 1e-9)).shift(1)

def test_single_stock(ticker, df):
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]
    if len(df) < 500:
        return None
    atr14 = calculate_atr(df, 14)
    ema50 = c.ewm(span=50).mean()
    ema110 = c.ewm(span=110).mean()
    vwap_lag = compute_vwap_shifted(df)
    
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(7).mean()
    loss = (-delta.clip(upper=0)).rolling(7).mean()
    rsi7 = 100 - (100 / (1 + gain / (loss + 1e-9)))
    
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    sig = macd.ewm(span=9).mean()
    macd_cross_up = (macd > sig) & (macd.shift(1) <= sig.shift(1))
    
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
    
    entry_cond = (
        macd_cross_up & (macd > 0) &
        (c > ema50) & (c > ema110) &
        (c > vwap_lag) &
        (rsi7 > 50.0) & (adx > 18.0) & (plus_di > minus_di) &
        (c <= 450.0) & (c >= 10.0)
    )
    
    slippage = 0.0005
    trades = []
    in_pos = False
    entry_p = 0.0
    active_sl = 0.0
    high_seen = 0.0
    entry_atr = 0.0
    qty = 0
    bars_held = 0
    cur_date = None
    daily_traded = False
    
    for i in range(100, len(df)):
        ts = df.index[i]
        d = ts.date()
        if d != cur_date:
            cur_date = d
            daily_traded = False
            
        t_time = ts.time()
        cur_c = c.iloc[i]
        cur_h = h.iloc[i]
        cur_l = l.iloc[i]
        cur_atr = atr14.iloc[i]
        
        if in_pos:
            bars_held += 1
            high_seen = max(high_seen, cur_h)
            mfe_atr = (high_seen - entry_p) / entry_atr if entry_atr > 0 else 0.0
            
            # V17 standard harvester trail
            if mfe_atr >= 1.35:
                active_sl = max(active_sl, round(high_seen - (0.45 * entry_atr), 2))
            elif mfe_atr >= 1.70:
                active_sl = max(active_sl, round(high_seen - (0.65 * entry_atr), 2))
            elif mfe_atr >= 0.85:
                active_sl = max(active_sl, round(entry_p + (0.40 * entry_atr), 2))
                
            # Stop check
            if cur_l <= active_sl:
                raw_exit = active_sl * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                gross = round((exit_p - entry_p) * qty, 2)
                chg = round(calc_dhan_charges(entry_p, exit_p, qty, is_intraday=True), 2)
                trades.append(gross - chg)
                in_pos = False
                continue
                
            # Stagnation scratch
            if bars_held >= 3 and mfe_atr < 0.40 and cur_c < entry_p and (rsi7.iloc[i] < 46.0 or cur_c < vwap_lag.iloc[i]):
                raw_exit = cur_c * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                gross = round((exit_p - entry_p) * qty, 2)
                chg = round(calc_dhan_charges(entry_p, exit_p, qty, is_intraday=True), 2)
                trades.append(gross - chg)
                in_pos = False
                continue
                
            # 14:45 Square-off
            if t_time >= pd.Timestamp("14:45").time():
                raw_exit = cur_c * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                gross = round((exit_p - entry_p) * qty, 2)
                chg = round(calc_dhan_charges(entry_p, exit_p, qty, is_intraday=True), 2)
                trades.append(gross - chg)
                in_pos = False
                continue
                
        else:
            if daily_traded:
                continue
            if t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:15").time():
                continue
            if not entry_cond.iloc[i]:
                continue
                
            raw_entry = cur_c * (1.0 + slippage)
            entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
            entry_atr = max(cur_atr, 0.05)
            active_sl = round(entry_p - (1.00 * entry_atr), 2)
            
            # Unit risk of 20 Rs for comparison
            friction = 0.002067 * entry_p
            denom = entry_atr + friction
            qty = max(1, int(25.0 / denom))
            in_pos = True
            high_seen = entry_p
            bars_held = 0
            daily_traded = True

    if not trades:
        return None
    arr = np.array(trades)
    n = len(arr)
    wins = (arr > 0).sum()
    wr = wins / n * 100.0
    gp = arr[arr > 0].sum()
    gl = abs(arr[arr < 0].sum())
    pf = gp / gl if gl > 0 else 999.0
    net = arr.sum()
    return {"ticker": ticker, "n": n, "wr": wr, "pf": pf, "net": net}

results = []
for t, df in all_pools.items():
    res = test_single_stock(t, df)
    if res and res["n"] >= 10:
        results.append(res)

df_all = pd.DataFrame(results).sort_values("pf", ascending=False)
print("=== TOP PERFORMING STOCKS BY PROFIT FACTOR (MIN 10 TRADES) ===")
print(df_all.to_string(index=False))
