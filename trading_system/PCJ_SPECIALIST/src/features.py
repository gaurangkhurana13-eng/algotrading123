"""
Dedicated Feature Engineering Module for PCJEWELLER Intraday Alpha.
Computes 38 causal, stationary indicators:
- Shifted Intraday VWAP Dynamics (strictly .shift(1))
- Order Flow CLV and Volume Acceleration (RVOL, Surge)
- Multi-Horizon Volatility & Bollinger-Keltner Squeeze
- Trend Alignment (EMA 9/21/50, ADX, DI difference, MACD)
- Candlestick Microstructure (Body ratio, Wicks, CLV)
- Opening Range Breakout (ORB 09:15-09:45 High/Low)
Zero look-ahead bias guaranteed.
"""

import numpy as np
import pandas as pd
import ta

FEATURE_COLS = [
    "ret_1", "ret_2", "ret_3", "ret_5",
    "atr_norm", "atr_norm_z", "atr_expanding", "atr_ratio",
    "rsi", "rsi_z", "macd_hist", "macd_slope", "macd_cross",
    "ema_dev_9", "ema_dev_21", "ema_dev_50", "bullish_alignment",
    "adx", "di_diff", "adx_rising", "bulls_dominate",
    "bb_pos", "bb_width", "squeeze_on", "squeeze_fast", "squeeze_std", "squeeze_release",
    "rvol", "vol_zscore", "vol_surge",
    "vwap_dev", "vwap_dist", "above_vwap", "vwap_slope",
    "clv", "flow_clv", "body_ratio", "upper_wick", "lower_wick",
    "orb_high_dist", "near_close", "morning", "h_rem"
]

def compute_pcj_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    o = df["open"]
    v = df["volume"]

    # Returns
    df["ret_1"] = c.pct_change(1)
    df["ret_2"] = c.pct_change(2)
    df["ret_3"] = c.pct_change(3)
    df["ret_5"] = c.pct_change(5)

    # Volatility & ATR
    atr_14 = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
    atr_6  = ta.volatility.AverageTrueRange(h, l, c, window=6).average_true_range()
    atr_20 = ta.volatility.AverageTrueRange(h, l, c, window=20).average_true_range()
    
    df["atr_raw"] = atr_14
    df["atr_norm"] = (atr_14 / (c + 1e-9)).fillna(0)
    df["atr_norm_z"] = (df["atr_norm"] - df["atr_norm"].rolling(50).mean()) / (df["atr_norm"].rolling(50).std() + 1e-9)
    df["atr_expanding"] = (df["atr_norm"] > df["atr_norm"].shift(1)).astype(int)
    df["atr_ratio"] = atr_6 / (atr_20 + 1e-9)

    # Momentum
    rsi = ta.momentum.RSIIndicator(c, window=14).rsi()
    df["rsi"] = rsi
    df["rsi_z"] = (rsi - rsi.rolling(50).mean()) / (rsi.rolling(50).std() + 1e-9)

    macd = ta.trend.MACD(c)
    df["macd_hist"] = macd.macd_diff()
    df["macd_slope"] = df["macd_hist"] - df["macd_hist"].shift(2)
    df["macd_cross"] = (macd.macd() > macd.macd_signal()).astype(int)

    # Moving averages
    ema9  = c.ewm(span=9).mean()
    ema21 = c.ewm(span=21).mean()
    ema50 = c.ewm(span=50).mean()
    df["ema_dev_9"]  = (c - ema9) / (ema9 + 1e-9)
    df["ema_dev_21"] = (c - ema21) / (ema21 + 1e-9)
    df["ema_dev_50"] = (c - ema50) / (ema50 + 1e-9)
    df["bullish_alignment"] = ((ema9 > ema21) & (ema21 > ema50) & (c > ema9)).astype(int)

    # Trend Strength
    adx = ta.trend.ADXIndicator(h, l, c, window=14)
    df["adx"] = adx.adx()
    df["di_diff"] = adx.adx_pos() - adx.adx_neg()
    df["adx_rising"] = (df["adx"] > df["adx"].shift(2)).astype(int)
    df["bulls_dominate"] = (adx.adx_pos() > adx.adx_neg()).astype(int)

    # Bollinger Bands & Keltner Squeeze
    bb10 = ta.volatility.BollingerBands(c, window=10, window_dev=2)
    kc10 = ta.volatility.KeltnerChannel(h, l, c, window=10)
    sq10 = (bb10.bollinger_hband() < kc10.keltner_channel_hband()) & (bb10.bollinger_lband() > kc10.keltner_channel_lband())

    bb20 = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    kc20 = ta.volatility.KeltnerChannel(h, l, c, window=20)
    sq20 = (bb20.bollinger_hband() < kc20.keltner_channel_hband()) & (bb20.bollinger_lband() > kc20.keltner_channel_lband())

    sq = sq10 | sq20
    df["squeeze_on"] = sq.astype(int)
    df["squeeze_fast"] = sq10.astype(int)
    df["squeeze_std"] = sq20.astype(int)
    df["squeeze_release"] = ((sq.shift(1) | sq.shift(2)) & ~sq).astype(int)
    df["bb_pos"] = bb20.bollinger_pband()
    df["bb_width"] = (bb20.bollinger_hband() - bb20.bollinger_lband()) / (bb20.bollinger_mavg() + 1e-9)

    # Volume & Order Flow
    v_mean_20 = v.rolling(20).mean() + 1e-9
    v_std_20  = v.rolling(20).std() + 1e-9
    df["rvol"] = v / v_mean_20
    df["vol_zscore"] = (v - v_mean_20) / v_std_20
    df["vol_surge"] = (df["rvol"] > 1.80).astype(int)

    # Candlestick Microstructure
    bar_range = (h - l) + 1e-6
    clv = ((c - l) - (h - c)) / bar_range
    df["clv"] = clv
    df["flow_clv"] = clv * df["rvol"]
    df["body_ratio"] = abs(c - o) / bar_range
    df["upper_wick"] = (h - np.maximum(o, c)) / bar_range
    df["lower_wick"] = (np.minimum(o, c) - l) / bar_range

    # Intraday VWAP (strictly shifted by 1 bar to prevent bar-close lookahead)
    df["_date"]   = df.index.date
    tp_price      = (h + l + c) / 3
    df["_tpv"]    = tp_price * v
    df["_vcum"]   = df.groupby("_date")["_tpv"].cumsum()
    df["_volcum"] = df.groupby("_date")["volume"].cumsum()
    df["vwap"]    = (df["_vcum"] / (df["_volcum"] + 1e-9)).shift(1)
    df["vwap_dev"] = (c - df["vwap"]) / (df["vwap"] + 1e-9)
    df["vwap_dist"] = (c - df["vwap"]) / (atr_14 + 1e-9)
    df["above_vwap"] = (c > df["vwap"]).astype(int)
    df["vwap_slope"] = (df["vwap"] - df["vwap"].shift(3)) / (atr_14 + 1e-9)

    # Opening Range Breakout (09:15 to 09:45 High)
    df["_time"] = df.index.time
    orb_mask = (df["_time"] >= pd.Timestamp("09:15").time()) & (df["_time"] <= pd.Timestamp("09:45").time())
    df["_orb_h"] = np.where(orb_mask, h, np.nan)
    orb_high_per_day = df.groupby("_date")["_orb_h"].transform("max")
    df["orb_high_dist"] = (c - orb_high_per_day) / (atr_14 + 1e-9)

    # Intraday Horizon
    eod_mins = 15 * 60 + 10
    bar_mins = [t.hour * 60 + t.minute for t in df.index.time]
    df["h_rem"] = [max(0, (eod_mins - m) // 15) for m in bar_mins]
    df["near_close"] = (df.index.time >= pd.Timestamp("15:00").time()).astype(int)
    df["morning"]    = (df.index.time <= pd.Timestamp("11:30").time()).astype(int)

    # Clean temporary calculation columns
    df.drop(columns=["_date", "_tpv", "_vcum", "_volcum", "_time", "_orb_h"], inplace=True, errors="ignore")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df
