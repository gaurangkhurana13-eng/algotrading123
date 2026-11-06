import pickle
import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    univ = pickle.load(f)

# Test on PCJEWELLER, RPOWER, DELTACORP
core_3 = ["PCJEWELLER.NS", "RPOWER.NS", "DELTACORP.NS"]

for sym in core_3:
    if sym not in univ:
        continue
    df = univ[sym].copy()
    if "timestamp" not in df.columns:
        df["timestamp"] = df.index
    d = df.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(d["timestamp"])
    d["date"] = ts.dt.date
    d["time_hm"] = ts.dt.hour * 100 + ts.dt.minute
    o, h, l, c, v = d["open"], d["high"], d["low"], d["close"], d["volume"]
    
    # ATR
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    d["atr_14"] = tr.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    d["natr_14"] = d["atr_14"] / c.replace(0.0, np.nan)
    d["tick_to_atr"] = 0.05 / d["atr_14"].replace(0.0, np.nan)
    
    # VWAP
    typ = (h + l + c) / 3.0
    cum_pv = (typ * v).groupby(d["date"]).cumsum()
    cum_v = v.groupby(d["date"]).cumsum().replace(0.0, np.nan)
    vwap_curr = cum_pv / cum_v
    d["vwap_shifted"] = vwap_curr.groupby(d["date"]).shift(1).fillna(typ)
    
    # EMAs
    d["ema_50"] = c.ewm(span=50, adjust=False).mean()
    d["ema_110"] = c.ewm(span=110, adjust=False).mean()
    
    # MACD
    ema_12 = c.ewm(span=12, adjust=False).mean()
    ema_26 = c.ewm(span=26, adjust=False).mean()
    d["macd"] = ema_12 - ema_26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]
    
    # RSI 7
    delta = c.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / 7.0, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / 7.0, adjust=False).mean()
    d["rsi_7"] = 100.0 - (100.0 / (1.0 + (gain / loss.replace(0.0, 1e-9))))
    
    # ADX 10
    up_m = h.diff()
    dn_m = -l.diff()
    p_dm = np.where((up_m > dn_m) & (up_m > 0.0), up_m, 0.0)
    m_dm = np.where((dn_m > up_m) & (dn_m > 0.0), dn_m, 0.0)
    tr_10 = tr.ewm(alpha=1.0 / 10.0, adjust=False).mean().replace(0.0, 1e-9)
    d["plus_di"] = 100.0 * pd.Series(p_dm, index=d.index).ewm(alpha=1.0 / 10.0, adjust=False).mean() / tr_10
    d["minus_di"] = 100.0 * pd.Series(m_dm, index=d.index).ewm(alpha=1.0 / 10.0, adjust=False).mean() / tr_10
    dx = 100.0 * (d["plus_di"] - d["minus_di"]).abs() / (d["plus_di"] + d["minus_di"]).replace(0.0, 1e-9)
    d["adx_10"] = dx.ewm(alpha=1.0 / 10.0, adjust=False).mean()
    
    # ER10 & RVOL20
    disp_10 = (c - c.shift(10)).abs()
    path_10 = c.diff().abs().rolling(10).sum().replace(0.0, 1e-9)
    d["er_10"] = (disp_10 / path_10).clip(0.0, 1.0)
    vol_baseline = v.shift(1).rolling(20).mean().replace(0.0, np.nan)
    d["rvol_20"] = (v / vol_baseline).fillna(1.0)
    
    # Microstructure
    rng = (h - l).replace(0.0, 1e-9)
    body = (c - o).abs()
    wick = (rng - body).clip(lower=0.0)
    d["clv"] = (c - l) / rng
    d["bar_range_atr"] = (h - l) / d["atr_14"].replace(0.0, 1e-9)
    d["wbr_20"] = (wick.shift(1).rolling(20).sum() / rng.shift(1).rolling(20).sum().replace(0.0, 1e-9)).fillna(0.50)
    
    # Base V19 Entry Signal
    macd_cross_up = (d["macd"] > d["macd_signal"]) & (d["macd"].shift(1) <= d["macd_signal"].shift(1))
    v19_base = (
        (d["macd"] > 0) & (d["macd"] > d["macd_signal"]) &
        (c > d["ema_50"]) & (c > d["ema_110"]) &
        (c > d["vwap_shifted"]) &
        (d["rsi_7"] > 50.0) & (d["adx_10"] > 18.0) & (d["plus_di"] > d["minus_di"]) &
        (d["time_hm"] >= 930) & (d["time_hm"] <= 1430)
    )
    
    print(f"\n--- {sym} Signal Diagnostic ---")
    print(f"Base V19 Signals: {v19_base.sum()}")
    print(f"  + tick_to_atr <= 0.14: {(v19_base & (d['tick_to_atr'] <= 0.14)).sum()}")
    print(f"  + natr_14 >= 0.0056: {(v19_base & (d['natr_14'] >= 0.0056)).sum()}")
    print(f"  + wbr_20 <= 0.57: {(v19_base & (d['wbr_20'] <= 0.57)).sum()}")
    print(f"  + er_10 >= 0.26: {(v19_base & (d['er_10'] >= 0.26)).sum()}")
    print(f"  + rvol_20 >= 1.18: {(v19_base & (d['rvol_20'] >= 1.18)).sum()}")
    print(f"  + clv >= 0.66: {(v19_base & (d['clv'] >= 0.66)).sum()}")
    print(f"  + bar_range_atr in [0.45, 1.80]: {(v19_base & (d['bar_range_atr'] >= 0.45) & (d['bar_range_atr'] <= 1.80)).sum()}")
    print(f"  + macd_hist > hist.shift(1): {(v19_base & (d['macd_hist'] > d['macd_hist'].shift(1))).sum()}")
    
    # Combined all
    v21_all = (
        v19_base & 
        (d['macd_hist'] > d['macd_hist'].shift(1)) &
        (d['tick_to_atr'] <= 0.14) &
        (d['natr_14'] >= 0.0056) &
        (d['wbr_20'] <= 0.57) &
        (d['er_10'] >= 0.26) &
        (d['rvol_20'] >= 1.18) &
        (d['clv'] >= 0.66) &
        (d['bar_range_atr'] >= 0.45) & (d['bar_range_atr'] <= 1.80)
    )
    print(f"Total V21 Signals when ALL combined: {v21_all.sum()}")
