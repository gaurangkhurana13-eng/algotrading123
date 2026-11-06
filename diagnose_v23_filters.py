import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calculate_atr

with open(ROOT_DIR / "data" / "2025-26" / "universe_14_15m.pkl", "rb") as f:
    univ_data = pickle.load(f)

tickers_combined = [
    "PCJEWELLER.NS", "RPOWER.NS", "DELTACORP.NS", "RITES.NS",
    "IDEA.NS", "MTNL.NS", "ZEEL.NS", "UTKARSHBNK.NS", "IFCI.NS", "NCC.NS",
    "TEXRAIL.NS", "JPPOWER.NS", "PTC.NS", "NIACL.NS"
]

def compute_vwap_shifted(df: pd.DataFrame) -> pd.Series:
    df_temp = df.copy()
    df_temp["_date"] = df_temp.index.date
    tp = (df_temp["high"] + df_temp["low"] + df_temp["close"]) / 3.0
    df_temp["_tpv"] = tp * df_temp["volume"]
    cum_tpv = df_temp.groupby("_date")["_tpv"].cumsum()
    cum_vol = df_temp.groupby("_date")["volume"].cumsum()
    return (cum_tpv / (cum_vol + 1e-9)).shift(1)

def compute_er10(series_close: pd.Series) -> pd.Series:
    change = (series_close - series_close.shift(10)).abs()
    volatility = series_close.diff().abs().rolling(10).sum()
    return change / (volatility + 1e-9)

# Collect all signals across 14 stocks in June-Dec 2025
base_signals = 0
pass_adx20 = 0
pass_er40 = 0
pass_er35 = 0
pass_rvol14 = 0
pass_rvol12 = 0
pass_atrpct11 = 0
pass_mom20d = 0
pass_all = 0

for t in tickers_combined:
    raw_sym = t.replace(".NS", "")
    df = univ_data.get(t, univ_data.get(raw_sym))
    if df is None:
        continue
    df = df.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]
    atr14 = calculate_atr(df, 14)
    atr_pct = (atr14 / c) * 100.0
    ema50 = c.ewm(span=50).mean()
    ema110 = c.ewm(span=110).mean()
    vwap_lag = compute_vwap_shifted(df)
    v_sma20 = v.rolling(20).mean()
    rvol = v / (v_sma20 + 1e-9)
    er10 = compute_er10(c)
    
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
    
    # Base V22 condition
    base_v22 = (
        macd_cross_up & (macd > 0) &
        (c > ema50) & (c > ema110) &
        (c > vwap_lag) &
        (rsi7 > 50.0) & (adx > 18.0) & (plus_di > minus_di) &
        (c <= 450.0) & (c >= 10.0)
    )
    
    # Slice to June - Dec 2025
    mask = (df.index >= "2025-06-01 09:15:00") & (df.index <= "2025-12-31 15:30:00")
    b = base_v22[mask]
    base_signals += b.sum()
    
    pass_adx20 += (b & (adx[mask] >= 20.0)).sum()
    pass_er40 += (b & (er10[mask] >= 0.40)).sum()
    pass_er35 += (b & (er10[mask] >= 0.35)).sum()
    pass_rvol14 += (b & (rvol[mask] >= 1.4)).sum()
    pass_rvol12 += (b & (rvol[mask] >= 1.2)).sum()
    pass_atrpct11 += (b & (atr_pct[mask] >= 1.1)).sum()
    
    mom20d = (c > c.shift(500))
    pass_mom20d += (b & mom20d[mask]).sum()
    
    pass_all += (b & (adx[mask] >= 20.0) & (er10[mask] >= 0.40) & (rvol[mask] >= 1.4) & (atr_pct[mask] >= 1.1) & mom20d[mask]).sum()

print("=== FILTER RETENTION DIAGNOSIS (JUNE-DEC 2025) ===")
print(f"Base V22 Raw Signals:     {base_signals}")
print(f"Passing ADX >= 20:         {pass_adx20} ({pass_adx20/base_signals*100:.1f}%)")
print(f"Passing ER10 >= 0.40:      {pass_er40} ({pass_er40/base_signals*100:.1f}%)")
print(f"Passing ER10 >= 0.35:      {pass_er35} ({pass_er35/base_signals*100:.1f}%)")
print(f"Passing RVOL >= 1.4:       {pass_rvol14} ({pass_rvol14/base_signals*100:.1f}%)")
print(f"Passing RVOL >= 1.2:       {pass_rvol12} ({pass_rvol12/base_signals*100:.1f}%)")
print(f"Passing ATR% >= 1.1%:      {pass_atrpct11} ({pass_atrpct11/base_signals*100:.1f}%)")
print(f"Passing 20-Day Momentum:   {pass_mom20d} ({pass_mom20d/base_signals*100:.1f}%)")
print(f"Passing ALL 5 Simultaneously: {pass_all} ({pass_all/base_signals*100:.1f}%)")
