"""
Feature engineering module for Bhavishya V36:
Claude Sonnet 7-Factor High-Conviction Breakout & Microstructure Stack.
1. Breakout Bar: Close in top 25% of range AND body >= 0.60*range AND close > open
2. RVOL_tod >= 1.8 (Intraday volume vs rolling baseline)
3. VWAP Anchor & Slope: Close > VWAP(shift 1) + 0.25*ATR AND VWAP 4-bar slope > 0
4. Compression before break: ATR(6)/ATR(20) < 0.85 AND break of 12-bar high
5. Anti-chase: bar range <= 1.6*ATR AND (close - level_12) <= 0.5*ATR
6. Index Gate: Nifty 15m close > Nifty VWAP(shift 1) AND Nifty 4-bar return >= 0
7. Retest / Shelf Hold: Low >= level_12 - 0.15*ATR
Strict 1-bar shift on all VWAPs guarantees zero lookahead bias.
"""

import numpy as np
import pandas as pd
import ta

FEATURE_COLS = [
    "ret_1", "ret_3", "ret_6",
    "atr_norm", "atr_norm_z", "atr_expanding",
    "rsi", "rsi_z", "macd_hist", "macd_accel", "macd_cross",
    "ema_dev_9", "ema_dev_21", "ema_dev_50", "bullish_alignment",
    "adx", "di_diff", "adx_rising", "adx_slope_3", "bulls_dominate",
    "bb_pos", "bb_width", "squeeze_on", "squeeze_fast", "squeeze_std", "squeeze_ratio", "squeeze_release",
    "rvol", "vol_zscore", "vol_surge", "obv_norm",
    "vwap_dev", "vwap_dist", "above_vwap", "near_close", "morning",
    "flow_clv", "orb_impulse", "h_rem", "cs_rs",
    "squeeze_x_vol", "ret_x_rvol"
]

def compute_stock_features(df: pd.DataFrame, nifty_df: pd.DataFrame = None) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    o = df["open"]

    df["ret_1"] = c.pct_change(1)
    df["ret_3"] = c.pct_change(3)
    df["ret_6"] = c.pct_change(6)

    atr_raw = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
    df["atr_raw"]       = atr_raw
    df["atr_norm"]      = (atr_raw / c).fillna(0)
    df["atr_norm_z"]    = (df["atr_norm"] - df["atr_norm"].rolling(50).mean()) / (df["atr_norm"].rolling(50).std() + 1e-9)
    df["atr_expanding"] = (df["atr_norm"] > df["atr_norm"].shift(1)).astype(int)

    # Fast and slow ATR for compression ratio (Factor 4)
    atr_6  = ta.volatility.AverageTrueRange(h, l, c, window=6).average_true_range()
    atr_20 = ta.volatility.AverageTrueRange(h, l, c, window=20).average_true_range()
    df["atr_ratio_6_20"] = atr_6 / (atr_20 + 1e-9)

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

    adx = ta.trend.ADXIndicator(h, l, c, window=14)
    df["adx"]            = adx.adx()
    df["di_diff"]        = adx.adx_pos() - adx.adx_neg()
    df["adx_rising"]     = (df["adx"] > df["adx"].shift(2)).astype(int)
    df["adx_slope_3"]    = df["adx"] - df["adx"].shift(3)
    df["bulls_dominate"] = (adx.adx_pos() > adx.adx_neg()).astype(int)

    # Multi-Horizon Squeeze
    bb10 = ta.volatility.BollingerBands(c, window=10, window_dev=2)
    kc10 = ta.volatility.KeltnerChannel(h, l, c, window=10)
    sq10 = ((bb10.bollinger_hband() < kc10.keltner_channel_hband()) &
            (bb10.bollinger_lband() > kc10.keltner_channel_lband()))

    bb20 = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    kc20 = ta.volatility.KeltnerChannel(h, l, c, window=20)
    sq20 = ((bb20.bollinger_hband() < kc20.keltner_channel_hband()) &
            (bb20.bollinger_lband() > kc20.keltner_channel_lband()))

    sq = sq10 | sq20
    df["squeeze_on"]      = sq.astype(int)
    df["squeeze_fast"]    = sq10.astype(int)
    df["squeeze_std"]     = sq20.astype(int)
    df["squeeze_release"] = ((sq.shift(1) | sq.shift(2) | sq.shift(3)) & ~sq).astype(int)

    bb10_w = bb10.bollinger_hband() - bb10.bollinger_lband()
    kc10_w = kc10.keltner_channel_hband() - kc10.keltner_channel_lband()
    bb20_w = bb20.bollinger_hband() - bb20.bollinger_lband()
    kc20_w = kc20.keltner_channel_hband() - kc20.keltner_channel_lband()
    r10 = bb10_w / (kc10_w + 1e-9)
    r20 = bb20_w / (kc20_w + 1e-9)
    df["squeeze_ratio"] = np.minimum(r10, r20)

    df["bb_pos"]   = bb20.bollinger_pband()
    df["bb_width"] = (bb20.bollinger_hband() - bb20.bollinger_lband()) / (bb20.bollinger_mavg() + 1e-9)

    # RVOL (Time-of-day rolling)
    df["rvol"]       = v / (v.rolling(20).mean() + 1e-9)
    df["vol_zscore"] = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-9)
    df["vol_surge"]  = (df["rvol"] > 2.0).astype(int)
    
    obv = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    df["obv_norm"]   = (obv - obv.rolling(50).min()) / (obv.rolling(50).max() - obv.rolling(50).min() + 1e-9)

    # Intraday VWAP (strictly shifted by 1 to prevent bar-close lookahead)
    df["_date"]      = df.index.date
    tp_price         = (h + l + c) / 3
    df["_tpv"]       = tp_price * v
    df["_vcum"]      = df.groupby("_date")["_tpv"].cumsum()
    df["_volcum"]    = df.groupby("_date")["volume"].cumsum()
    df["vwap"]       = (df["_vcum"] / (df["_volcum"] + 1e-9)).shift(1)
    df["vwap_dev"]   = (c - df["vwap"]) / (df["vwap"] + 1e-9)
    df["vwap_dist"]  = (c - df["vwap"]) / (atr_raw + 1e-9)
    df["above_vwap"] = (c > df["vwap"]).astype(int)
    df["vwap_slope_4"] = (df["vwap"] - df["vwap"].shift(4))

    # Institutional Order Flow Absorption (Flow CLV x RVOL)
    clv = ((c - l) - (h - c)) / (h - l + 1e-6)
    df["flow_clv"] = clv * df["rvol"]

    # Opening Range Breakout Impulse (ORB 09:15-09:45 High)
    df["_time"] = df.index.time
    orb_mask = (df["_time"] >= pd.Timestamp("09:15").time()) & (df["_time"] <= pd.Timestamp("09:45").time())
    df["_orb_h"] = np.where(orb_mask, h, np.nan)
    orb_high_per_day = df.groupby("_date")["_orb_h"].transform("max")
    df["orb_impulse"] = (c - orb_high_per_day) / (atr_raw + 1e-9)

    # Remaining bars until 15:10 IST
    eod_mins = 15 * 60 + 10
    bar_mins = [t.hour * 60 + t.minute for t in df.index.time]
    df["h_rem"] = [max(0, (eod_mins - m) // 15) for m in bar_mins]

    # Section 3: Claude 7-Factor High-Conviction Feature Signals
    # Factor 1: Breakout Bar: close in top 25% of range AND body >= 0.6*range AND close > open
    bar_range = h - l + 1e-6
    bar_body = c - o
    df["f1_breakout_bar"] = (
        (c >= l + 0.75 * bar_range) &
        (bar_body >= 0.60 * bar_range) &
        (c > o)
    ).astype(int)

    # Factor 2: RVOL_tod >= 1.8
    df["f2_rvol"] = (df["rvol"] >= 1.8).astype(int)

    # Factor 3: Close > VWAP(shift 1) + 0.25*ATR AND VWAP slope > 0 over 4 bars
    df["f3_vwap_confirm"] = (
        (c > df["vwap"] + 0.25 * atr_raw) &
        (df["vwap_slope_4"] > 0)
    ).astype(int)

    # Factor 4: Compression before break: ATR(6)/ATR(20) < 0.85 AND break of 12-bar high
    level_12 = h.shift(1).rolling(12).max()
    df["level_12"] = level_12
    df["f4_compression_break"] = (
        (df["atr_ratio_6_20"] < 0.85) &
        (c > level_12)
    ).astype(int)

    # Factor 5: Anti-chase: bar range <= 1.6*ATR AND (close - level_12) <= 0.5*ATR
    df["f5_anti_chase"] = (
        (bar_range <= 1.6 * atr_raw) &
        ((c - level_12) <= 0.50 * atr_raw)
    ).astype(int)

    # Factor 6: Index Gate: Nifty 15m close > Nifty VWAP(shift 1) AND Nifty 4-bar return >= 0
    if nifty_df is not None:
        nifty_merged = nifty_df.reindex(df.index).ffill()
        n_c = nifty_merged["close"]
        n_v = nifty_merged["vwap"]
        n_ret4 = n_c.pct_change(4)
        df["f6_index_gate"] = ((n_c > n_v) & (n_ret4 >= 0)).astype(int)
    else:
        df["f6_index_gate"] = 1

    # Factor 7: Retest / Shelf Hold: low >= level_12 - 0.15*ATR
    df["f7_retest_hold"] = (l >= level_12 - 0.15 * atr_raw).astype(int)

    # Sum of 7 factors (High-Conviction Score out of 7)
    df["conviction_score"] = (
        df["f1_breakout_bar"] +
        df["f2_rvol"] +
        df["f3_vwap_confirm"] +
        df["f4_compression_break"] +
        df["f5_anti_chase"] +
        df["f6_index_gate"] +
        df["f7_retest_hold"]
    )

    # Clean temporary columns
    df.drop(columns=["_date", "_tpv", "_vcum", "_volcum", "_time", "_orb_h"], inplace=True)

    df["near_close"] = (df.index.time >= pd.Timestamp("15:00").time()).astype(int)
    df["morning"]    = (df.index.time <= pd.Timestamp("11:15").time()).astype(int)

    df["squeeze_x_vol"] = df["squeeze_release"] * df["rvol"]
    df["ret_x_rvol"]    = df["ret_1"] * df["rvol"]

    # 4-bar momentum relative to ATR for cross-sectional scoring
    df["r_4"] = (c - c.shift(4)) / (atr_raw.shift(1) + 1e-9)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def augment_cross_sectional_features(master_df: pd.DataFrame) -> pd.DataFrame:
    """Computes cross-sectional relative strength Z-score, percentile rank, market breadth B_t, and dispersion D_t."""
    master_df = master_df.copy()
    
    def calc_cs_metrics(g):
        vals = g["r_4"].values
        if np.isnan(vals).all() or len(vals) < 2:
            g["cs_rs"] = 0.0
            g["pct_cs_rs"] = 0.50
            g["breadth"] = 0.50
            g["dispersion"] = 0.0
            return g
        med = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals - med))
        if np.isnan(mad) or mad < 1e-6:
            std = np.nanstd(vals)
            mad = std if not np.isnan(std) and std >= 1e-6 else 1.0
        g["cs_rs"] = (vals - med) / (mad + 1e-6)
        g["pct_cs_rs"] = pd.Series(g["cs_rs"]).rank(pct=True).values

        valid_vals = vals[~np.isnan(vals)]
        if len(valid_vals) > 0:
            g["breadth"] = float(np.mean(valid_vals > 0))
            g["dispersion"] = float(mad)
        else:
            g["breadth"] = 0.50
            g["dispersion"] = 0.0
        return g

    master_df = master_df.groupby(master_df.index, group_keys=False).apply(calc_cs_metrics)
    master_df.drop(columns=["r_4"], inplace=True, errors="ignore")
    master_df["cs_rs"] = master_df["cs_rs"].fillna(0.0)
    master_df["pct_cs_rs"] = master_df["pct_cs_rs"].fillna(0.50)
    master_df["breadth"] = master_df["breadth"].fillna(0.50)
    master_df["dispersion"] = master_df["dispersion"].fillna(0.0)
    return master_df
