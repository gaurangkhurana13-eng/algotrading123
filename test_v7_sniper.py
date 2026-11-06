# ==============================================================================
# BHAVISHYA V7.0 — PURE BUY SNIPER PIPELINE (TEST HARNESS)
# ==============================================================================
import sys
import os
import json
import time
import numpy as np
import pandas as pd
import yfinance as yf
from catboost import CatBoostClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import ta

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

CONFIG = {
    "CAPITAL"         : 500.0,
    "LEVERAGE"        : 5.0,
    "TP_ATR_MULT"     : 2.0,
    "SL_ATR_MULT"     : 1.5,
    "MIN_ATR_NORM"    : 0.005,
    "HOLD_BARS"       : 12,
    "ADX_MIN_TREND"   : 22.0,
    "CONVICTION_THR"  : 0.75,
    "DAILY_TOP_N"     : 3,
    "MAX_DAILY_LOSS"  : 2,
    "N_FOLDS"         : 5,
    "EMBARGO_DAYS"    : 2,
    "MIN_TRAIN_DAYS"  : 25,
    "SEEDS"           : [1, 2, 3],
    "DEPTH"           : 4,
    "L2"              : 15,
    "ITERATIONS"      : 400,
    "LEARNING_RATE"   : 0.04,
    "MONTE_CARLO_SIMS": 10000,
    "UNIVERSE"        : [
        'ADANIPOWER.NS', 'ALOKINDS.NS', 'ASHOKLEY.NS', 'BANDHANBNK.NS', 'BANKBARODA.NS',
        'BANKINDIA.NS', 'BCLIND.NS', 'BPCL.NS', 'CANBK.NS', 'CENTRALBK.NS',
        'COALINDIA.NS', 'ELECTCAST.NS', 'EQUITASBNK.NS', 'GAIL.NS', 'GMRAIRPORT.NS',
        'GSFC.NS', 'HATHWAY.NS', 'HCC.NS', 'HUDCO.NS', 'IDBI.NS',
        'IDEA.NS', 'IDFCFIRSTB.NS', 'IFCI.NS', 'INDIANB.NS', 'IOB.NS',
        'IOC.NS', 'IRCON.NS', 'IRFC.NS', 'J&KBANK.NS', 'JAICORPLTD.NS',
        'JPPOWER.NS', 'JYOTHYLAB.NS', 'LLOYDSENGG.NS', 'MAHABANK.NS', 'MANALIPETC.NS',
        'MMTC.NS', 'MOREPENLAB.NS', 'MOTHERSON.NS', 'MRPL.NS', 'MTNL.NS',
        'NBCC.NS', 'NETWORK18.NS', 'NFL.NS', 'NHPC.NS', 'NIACL.NS',
        'NMDC.NS', 'NTPC.NS', 'ONGC.NS', 'PATANJALI.NS', 'PCJEWELLER.NS',
        'PFC.NS', 'PNB.NS', 'PTC.NS', 'RBLBANK.NS', 'RCF.NS',
        'RECLTD.NS', 'RENUKA.NS', 'RPOWER.NS', 'RVNL.NS', 'SAIL.NS',
        'SBIN.NS', 'SJVN.NS', 'SOUTHBANK.NS', 'SPIC.NS', 'SUBEXLTD.NS',
        'SUZLON.NS', 'SWSOLAR.NS', 'TATAPOWER.NS', 'TATASTEEL.NS', 'TEXRAIL.NS',
        'TRIDENT.NS', 'TTML.NS', 'UCOBANK.NS', 'UJJIVANSFB.NS', 'UNIONBANK.NS',
        'UTKARSHBNK.NS', 'VEDL.NS', 'YESBANK.NS'
    ]
}

BUYING_POWER = CONFIG["CAPITAL"] * CONFIG["LEVERAGE"]

def calc_dhan_charges(buy_price: float, sell_price: float, qty: int) -> float:
    buy_to   = buy_price  * qty
    sell_to  = sell_price * qty
    total_to = buy_to + sell_to
    brok  = min(20.0, buy_to * 0.0003) + min(20.0, sell_to * 0.0003)
    exc   = total_to * 0.0000307
    stt   = sell_to * 0.00025
    sebi  = total_to * 0.000001
    stamp = buy_to * 0.00003
    gst   = (brok + exc + sebi) * 0.18
    return round(brok + exc + stt + sebi + stamp + gst, 2)

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    df["ret_1"] = c.pct_change(1)
    df["ret_3"] = c.pct_change(3)
    df["ret_6"] = c.pct_change(6)

    atr_raw = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
    df["atr_norm"]      = (atr_raw / c).fillna(0)
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

    adx = ta.trend.ADXIndicator(h, l, c, window=14)
    df["adx"]        = adx.adx()
    df["di_diff"]    = adx.adx_pos() - adx.adx_neg()
    df["adx_rising"] = (df["adx"] > df["adx"].shift(2)).astype(int)
    df["bulls_dominate"] = (adx.adx_pos() > adx.adx_neg()).astype(int)

    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df["bb_pos"]   = bb.bollinger_pband()
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / (bb.bollinger_mavg() + 1e-9)
    kc = ta.volatility.KeltnerChannel(h, l, c, window=20)
    
    sq = ((bb.bollinger_hband() < kc.keltner_channel_hband()) &
          (bb.bollinger_lband() > kc.keltner_channel_lband()))
    df["squeeze_on"]      = sq.astype(int)
    df["squeeze_release"] = ((sq.shift(1) | sq.shift(2) | sq.shift(3)) & ~sq).astype(int)

    df["rvol"]       = v / (v.rolling(20).mean() + 1e-9)
    df["vol_zscore"] = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-9)
    df["vol_surge"]  = (df["rvol"] > 2.0).astype(int)
    
    obv = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    df["obv_norm"]   = (obv - obv.rolling(50).min()) / (obv.rolling(50).max() - obv.rolling(50).min() + 1e-9)

    df["_date"]      = df.index.date
    tp_price         = (h + l + c) / 3
    df["_tpv"]       = tp_price * v
    df["_vcum"]      = df.groupby("_date")["_tpv"].cumsum()
    df["_volcum"]    = df.groupby("_date")["volume"].cumsum()
    df["vwap"]       = (df["_vcum"] / (df["_volcum"] + 1e-9)).shift(1)
    df["vwap_dev"]   = (c - df["vwap"]) / (df["vwap"] + 1e-9)
    df["above_vwap"] = (c > df["vwap"]).astype(int)
    df.drop(columns=["_date","_tpv","_vcum","_volcum"], inplace=True)

    df["near_close"] = (df.index.time >= pd.Timestamp("15:00").time()).astype(int)
    df["morning"]    = (df.index.time <= pd.Timestamp("11:15").time()).astype(int)

    df["squeeze_x_vol"] = df["squeeze_release"] * df["rvol"]
    df["ret_x_rvol"]    = df["ret_1"] * df["rvol"]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

FEATURE_COLS = [
    "ret_1","ret_3","ret_6",
    "atr_norm","atr_norm_z","atr_expanding",
    "rsi","rsi_z","macd_hist","macd_accel","macd_cross",
    "ema_dev_9","ema_dev_21","ema_dev_50","bullish_alignment",
    "adx","di_diff","adx_rising","bulls_dominate",
    "bb_pos","bb_width","squeeze_on","squeeze_release",
    "rvol","vol_zscore","vol_surge","obv_norm",
    "vwap_dev","above_vwap","near_close","morning",
    "squeeze_x_vol","ret_x_rvol"
]

def apply_buy_labels(df: pd.DataFrame) -> pd.DataFrame:
    df   = df.copy()
    n    = len(df)
    c    = df["close"].values
    h    = df["high"].values
    l    = df["low"].values
    atr  = df["atr_norm"].values
    hold = CONFIG["HOLD_BARS"]
    tp_mult = CONFIG["TP_ATR_MULT"]
    sl_mult = CONFIG["SL_ATR_MULT"]
    min_atr = CONFIG["MIN_ATR_NORM"]

    y_buy       = np.zeros(n, dtype=int)
    target_pcts = np.zeros(n, dtype=float)
    sl_pcts     = np.zeros(n, dtype=float)

    for i in range(n - hold):
        entry = c[i]
        curr_atr = max(atr[i], min_atr)
        
        tp_dist = curr_atr * tp_mult
        sl_dist = curr_atr * sl_mult
        target_pcts[i] = tp_dist
        sl_pcts[i]     = sl_dist

        tp_price = entry * (1.0 + tp_dist)
        sl_price = entry * (1.0 - sl_dist)

        hit = False
        for j in range(1, hold + 1):
            hi = h[i + j]
            lo = l[i + j]
            if hi >= tp_price:
                y_buy[i] = 1
                hit = True
                break
            elif lo <= sl_price:
                y_buy[i] = 0
                hit = True
                break

        if not hit:
            ret = (c[i + hold] - entry) / entry
            y_buy[i] = 1 if ret >= tp_dist else 0

    df["y_buy"]      = y_buy
    df["target_pct"] = target_pcts
    df["sl_pct"]     = sl_pcts
    return df.iloc[:n - hold]

print("Script template ready.")
