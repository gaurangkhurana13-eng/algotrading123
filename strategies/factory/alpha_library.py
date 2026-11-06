"""
Agent 1: Alpha Researcher (alpha_library.py)
Stores concrete, non-correlated rule-based hypotheses (V7 to V100).
Every hypothesis specifies:
- entry_logic(df) -> pd.Series(bool)
- parameters (bounded, non-overfit)
- rationale & description
- Pine Script implementation template
"""

import numpy as np
import pandas as pd

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h = df['high']
    l = df['low']
    c = df['close']
    c_prev = c.shift(1)
    tr = np.maximum(h - l, np.maximum((h - c_prev).abs(), (l - c_prev).abs()))
    return tr.rolling(period).mean()

def get_shifted_vwap(df: pd.DataFrame) -> pd.Series:
    df_temp = df.copy()
    df_temp["_date"] = df_temp.index.date
    tp = (df_temp["high"] + df_temp["low"] + df_temp["close"]) / 3.0
    df_temp["_tpv"] = tp * df_temp["volume"]
    cum_tpv = df_temp.groupby("_date")["_tpv"].cumsum()
    cum_vol = df_temp.groupby("_date")["volume"].cumsum()
    return (cum_tpv / (cum_vol + 1e-9)).shift(1)

def compute_supertrend(df, period=10, mult=2.5):
    atr = calculate_atr(df, period)
    hl2 = (df['high'] + df['low']) / 2.0
    b_up = hl2 + mult * atr
    b_dn = hl2 - mult * atr
    f_up = b_up.copy()
    f_dn = b_dn.copy()
    direction = pd.Series(1, index=df.index, dtype=int)
    curr_dir = 1
    for i in range(1, len(df)):
        if b_dn.iloc[i] > f_dn.iloc[i-1] or df['close'].iloc[i-1] < f_dn.iloc[i-1]:
            f_dn.iloc[i] = b_dn.iloc[i]
        else:
            f_dn.iloc[i] = f_dn.iloc[i-1]
        if b_up.iloc[i] < f_up.iloc[i-1] or df['close'].iloc[i-1] > f_up.iloc[i-1]:
            f_up.iloc[i] = b_up.iloc[i]
        else:
            f_up.iloc[i] = f_up.iloc[i-1]
        if curr_dir == 1:
            if df['close'].iloc[i] < f_dn.iloc[i]:
                curr_dir = -1
        else:
            if df['close'].iloc[i] > f_up.iloc[i]:
                curr_dir = 1
        direction.iloc[i] = curr_dir
    return (direction == 1)

HYPOTHESES = {
    "V7": {
        "name": "SuperTrend Momentum Confluence & Dynamic Moonshot Ratchet",
        "description": "SuperTrend (10, 2.5) Bullish Trend Regime combined with MACD Zero-Line cross, Shifted Daily VWAP, Fast RSI 7, ADX 10, and Dynamic Asymmetric Moonshot Ratchet.",
        "params": {
            "st_period": 10,
            "st_mult": 2.5,
            "sl_mult": 1.00,
            "p1_trig": 0.85,
            "p1_lock": 0.40,
            "p2_trig": 1.70,
            "trail_dist": 0.65
        },
        "compute_signals": lambda df: _compute_v7(df)
    },
    "V8": {
        "name": "Keltner-Bollinger Squeeze Release + Chande Momentum Velocity",
        "description": "Bollinger Bands (20, 2.0) coiled inside Keltner Channels (20, 1.5 ATR), triggering on bullish squeeze release with Chande Momentum Oscillator > 10.",
        "params": {
            "bb_len": 20,
            "bb_mult": 2.0,
            "kc_mult": 1.5,
            "cmo_min": 10.0,
            "sl_mult": 1.00,
            "p1_trig": 0.85,
            "p1_lock": 0.40,
            "p2_trig": 1.70,
            "trail_dist": 0.65
        },
        "compute_signals": lambda df: _compute_v8(df)
    },
    "V9": {
        "name": "SuperTrend Regime + Stochastic RSI Bullish Pullback",
        "description": "SuperTrend (10, 2.5) Bullish Trend Regime combined with StochRSI (14, 14, 3, 3) oversold momentum crossover from below 30.",
        "params": {
            "st_period": 10,
            "st_mult": 2.5,
            "stoch_min": 30.0,
            "sl_mult": 1.00,
            "p1_trig": 0.85,
            "p1_lock": 0.40,
            "p2_trig": 1.70,
            "trail_dist": 0.65
        },
        "compute_signals": lambda df: _compute_v9(df)
    },
    "V10": {
        "name": "Ichimoku Kinko Hyo Cloud Breakout + Volume Surge",
        "description": "Tenkan-sen (9) > Kijun-sen (26) with price breaking above Senkou Span A & B Cloud with Shifted VWAP confirmation.",
        "params": {
            "tenkan": 9,
            "kijun": 26,
            "senkou_b": 52,
            "sl_mult": 1.00,
            "p1_trig": 0.85,
            "p1_lock": 0.40,
            "p2_trig": 1.70,
            "trail_dist": 0.65
        },
        "compute_signals": lambda df: _compute_v10(df)
    }
}

def _compute_v7(df):
    c = df["close"]
    h = df["high"]
    l = df["low"]
    atr14 = calculate_atr(df, 14)
    ema50 = c.ewm(span=50).mean()
    vwap = get_shifted_vwap(df)
    
    # Fast RSI 7
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
    
    st_bull = compute_supertrend(df, 10, 2.5)
    
    macd_cross = (macd > sig) & (macd.shift(1) <= sig.shift(1))
    entry = macd_cross & (macd > 0) & (c > ema50) & (rsi7 > 50.0) & (adx > 18.0) & (plus_di > minus_di) & (c > vwap) & st_bull
    return entry, atr14

def _compute_v8(df):
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]
    atr = calculate_atr(df, 14)
    vwap = get_shifted_vwap(df)
    ema50 = c.ewm(span=50).mean()
    
    # Bollinger Bands 20, 2.0
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    bb_up = sma20 + 2.0 * std20
    bb_dn = sma20 - 2.0 * std20
    
    # Keltner Channel 20, 1.5 ATR
    kc_up = sma20 + 1.5 * atr
    kc_dn = sma20 - 1.5 * atr
    
    # Squeeze: BB inside KC
    squeeze_on = (bb_up < kc_up) & (bb_dn > kc_dn)
    squeeze_release = squeeze_on.shift(1) & (~squeeze_on)
    
    # Chande Momentum Oscillator (14)
    diff = c.diff()
    cz_up = diff.clip(lower=0).rolling(14).sum()
    cz_dn = (-diff.clip(upper=0)).rolling(14).sum()
    cmo = 100 * ((cz_up - cz_dn) / (cz_up + cz_dn + 1e-9))
    
    entry = squeeze_release & (cmo > 10.0) & (c > vwap) & (c > ema50)
    return entry, atr

def _compute_v9(df):
    c = df["close"]
    h = df["high"]
    l = df["low"]
    atr = calculate_atr(df, 10)
    vwap = get_shifted_vwap(df)
    ema50 = c.ewm(span=50).mean()
    
    # SuperTrend 10, 2.5
    hl2 = (h + l) / 2.0
    b_up = hl2 + 2.5 * atr
    b_dn = hl2 - 2.5 * atr
    f_up = b_up.copy()
    f_dn = b_dn.copy()
    direction = pd.Series(1, index=df.index, dtype=int)
    curr_dir = 1
    for i in range(1, len(df)):
        if b_dn.iloc[i] > f_dn.iloc[i-1] or c.iloc[i-1] < f_dn.iloc[i-1]:
            f_dn.iloc[i] = b_dn.iloc[i]
        else:
            f_dn.iloc[i] = f_dn.iloc[i-1]
        if b_up.iloc[i] < f_up.iloc[i-1] or c.iloc[i-1] > f_up.iloc[i-1]:
            f_up.iloc[i] = b_up.iloc[i]
        else:
            f_up.iloc[i] = f_up.iloc[i-1]
        if curr_dir == 1:
            if c.iloc[i] < f_dn.iloc[i]:
                curr_dir = -1
        else:
            if c.iloc[i] > f_up.iloc[i]:
                curr_dir = 1
        direction.iloc[i] = curr_dir
    st_bull = (direction == 1)
    
    # StochRSI 14, 14, 3, 3
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = 100 - (100 / (1 + gain / (loss + 1e-9)))
    rsi_min = rsi.rolling(14).min()
    rsi_max = rsi.rolling(14).max()
    stoch_k = 100 * ((rsi - rsi_min) / (rsi_max - rsi_min + 1e-9))
    k_smooth = stoch_k.rolling(3).mean()
    d_smooth = k_smooth.rolling(3).mean()
    stoch_cross = (k_smooth > d_smooth) & (k_smooth.shift(1) <= d_smooth.shift(1)) & (k_smooth < 45.0)
    
    entry = st_bull & stoch_cross & (c > vwap) & (c > ema50)
    return entry, atr

def _compute_v10(df):
    c = df["close"]
    h = df["high"]
    l = df["low"]
    atr = calculate_atr(df, 14)
    vwap = get_shifted_vwap(df)
    
    # Ichimoku
    tenkan = (h.rolling(9).max() + l.rolling(9).min()) / 2.0
    kijun = (h.rolling(26).max() + l.rolling(26).min()) / 2.0
    span_a = ((tenkan + kijun) / 2.0).shift(26)
    span_b = ((h.rolling(52).max() + l.rolling(52).min()) / 2.0).shift(26)
    
    above_cloud = (c > span_a) & (c > span_b)
    tk_cross = (tenkan > kijun) & (tenkan.shift(1) <= kijun.shift(1))
    
    entry = above_cloud & tk_cross & (c > vwap)
    return entry, atr
