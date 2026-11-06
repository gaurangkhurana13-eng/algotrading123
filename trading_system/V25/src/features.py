"""
Feature engineering module for Bhavishya V25
Contains 35 stationary, lookahead-free indicators including:
  - Close Location Value (CLV): Intraday buying pressure proxy [-1, +1]
  - Upper Wick Ratio: Upper wick divided by ATR [0, 5]
  - 33 Standard Features shifted by 1 bar to strictly guarantee out-of-sample integrity.
"""
import numpy as np
import pandas as pd
import ta

FEATURE_COLS = [
    "ret_1","ret_3","ret_6",
    "atr_norm","atr_norm_z","atr_expanding",
    "rsi","rsi_z","macd_hist","macd_accel","macd_cross",
    "ema_dev_9","ema_dev_21","ema_dev_50","bullish_alignment",
    "adx","di_diff","adx_rising","bulls_dominate",
    "bb_pos","bb_width","squeeze_on","squeeze_release",
    "rvol","vol_zscore","vol_surge","obv_norm",
    "vwap_dev","above_vwap","near_close","morning",
    "squeeze_x_vol","ret_x_rvol",
    "clv", "upper_wick_ratio"
]

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    df["ret_1"] = c.pct_change(1)
    df["ret_3"] = c.pct_change(3)
    df["ret_6"] = c.pct_change(6)

    atr_raw = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
    atr_safe = atr_raw.bfill().fillna(0.01)
    df["atr_norm"]      = (atr_safe / c).fillna(0)
    df["atr_norm_z"]    = (df["atr_norm"] - df["atr_norm"].rolling(50).mean()) / (df["atr_norm"].rolling(50).std() + 1e-9)
    df["atr_expanding"] = (df["atr_norm"] > df["atr_norm"].shift(1)).astype(int)

    df["rsi"]   = ta.momentum.RSIIndicator(c, window=14).rsi()
    df["rsi_z"] = (df["rsi"] - df["rsi"].rolling(50).mean()) / (df["rsi"].rolling(50).std() + 1e-9)
    
    macd = ta.trend.MACD(c)
    df["macd_hist"]  = macd.macd_diff()
    df["macd_accel"] = df["macd_hist"].diff()
    df["macd_cross"] = (macd.macd() > macd.macd_signal()).astype(int)

    ema9  = c.ewm(span=9).mean()
    ema21 = c.ewm(span=21).mean()
    ema50 = c.ewm(span=50).mean()
    
    df["ema_dev_9"]  = (c - ema9) / (ema9 + 1e-9)
    df["ema_dev_21"] = (c - ema21) / (ema21 + 1e-9)
    df["ema_dev_50"] = (c - ema50) / (ema50 + 1e-9)
    df["bullish_alignment"] = ((ema9 > ema21) & (c > ema50)).astype(int)

    adx_ind = ta.trend.ADXIndicator(h, l, c, window=14)
    df["adx"] = adx_ind.adx()
    df["di_diff"] = adx_ind.adx_pos() - adx_ind.adx_neg()
    df["adx_rising"] = (df["adx"] > df["adx"].shift(1)).astype(int)
    df["bulls_dominate"] = (adx_ind.adx_pos() > adx_ind.adx_neg()).astype(int)

    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2.0)
    df["bb_pos"]   = (c - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-9)
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / (bb.bollinger_mavg() + 1e-9)

    kc_mavg = c.rolling(20).mean()
    kc_range = 1.5 * atr_safe
    kc_upper = kc_mavg + kc_range
    kc_lower = kc_mavg - kc_range
    df["squeeze_on"] = ((bb.bollinger_hband() < kc_upper) & (bb.bollinger_lband() > kc_lower)).astype(int)
    df["squeeze_release"] = ((df["squeeze_on"].shift(1) == 1) & (df["squeeze_on"] == 0)).astype(int)

    vol_ma20 = v.rolling(20).mean() + 1e-9
    df["rvol"] = v / vol_ma20
    df["vol_zscore"] = (v - vol_ma20) / (v.rolling(20).std() + 1e-9)
    df["vol_surge"] = (df["rvol"] > 1.5).astype(int)

    obv = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    df["obv_norm"] = (obv - obv.rolling(50).mean()) / (obv.rolling(50).std() + 1e-9)

    cum_vol = v.cumsum() + 1e-9
    cum_pv  = (c * v).cumsum()
    vwap    = cum_pv / cum_vol
    df["vwap_dev"]   = (c - vwap) / (vwap + 1e-9)
    df["above_vwap"] = (c > vwap).astype(int)

    bar_time = df.index.time
    df["near_close"] = [1 if t.hour >= 14 and t.minute >= 45 else 0 for t in bar_time]
    df["morning"]    = [1 if t.hour == 9 and t.minute <= 45 else 0 for t in bar_time]

    df["squeeze_x_vol"] = df["squeeze_on"] * df["rvol"]
    df["ret_x_rvol"]    = df["ret_1"] * df["rvol"]

    # Close Location Value (CLV): [ (C - L) - (H - C) ] / (H - L)
    bar_range = (h - l).replace(0, np.nan).bfill()
    df["clv"] = (((c - l) - (h - c)) / (bar_range + 1e-5)).clip(-1.0, 1.0).fillna(0)

    # Upper Wick Ratio to ATR
    df["upper_wick_ratio"] = ((h - c) / (atr_safe + 1e-5)).clip(0.0, 5.0).fillna(0)

    # Shift all features by 1 bar to strictly guarantee out-of-sample integrity
    df[FEATURE_COLS] = df[FEATURE_COLS].shift(1)
    return df
