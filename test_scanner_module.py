import numpy as np
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
import catboost as cb
from catboost import CatBoostClassifier, Pool

tv = TvDatafeed()

def scan_live_market(model, feat_cols, cat_features, symbol='NIFTY', exchange='NSE', threshold=0.50):
    print(f"\n📡 Fetching latest live data for {exchange}:{symbol} via tvDatafeed...")
    df_live = tv.get_hist(symbol=symbol, exchange=exchange, interval=Interval.in_15_minute, n_bars=200)
    if df_live is None or df_live.empty:
        print(f"❌ Failed to fetch live data for {exchange}:{symbol}")
        return None
        
    df_live.columns = [c.lower() for c in df_live.columns]
    if 'symbol' in df_live.columns:
        df_live = df_live.drop(columns=['symbol'])
    df_live.index = pd.to_datetime(df_live.index)
    df_live = df_live.sort_index()
    
    c, h, l, o = df_live['close'], df_live['high'], df_live['low'], df_live['open']
    v = df_live['volume'].fillna(0.0)
    
    # 1. Log returns
    for k in [1, 2, 3, 5, 10]:
        df_live[f'log_ret_{k}'] = np.log(c / c.shift(k))
        
    # 2. Session Awareness
    df_live['is_opening_range'] = ((df_live.index.hour == 9) & (df_live.index.minute.isin([15, 30]))).astype(int)
    dates = pd.Series(df_live.index.date, index=df_live.index)
    daily_first_open = df_live.groupby(dates)['open'].first()
    daily_last_close = df_live.groupby(dates)['close'].last()
    prev_close = daily_last_close.shift(1)
    daily_gap = (daily_first_open - prev_close) / (prev_close + 1e-8)
    df_live['overnight_gap'] = dates.map(daily_gap).fillna(0.0).astype(float)
    
    # 3. Microstructure
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(span=14, adjust=False).mean()
    df_live['atr_14'] = atr14
    df_live['body_size_atr'] = (c - o) / (atr14 + 1e-8)
    df_live['upper_wick_ratio'] = (h - np.maximum(o, c)) / (h - l + 1e-8)
    df_live['lower_wick_ratio'] = (np.minimum(o, c) - l) / (h - l + 1e-8)
    
    # 4. Momentum & Volatility
    def calc_rsi(s, n):
        diff = s.diff()
        up = diff.clip(lower=0).ewm(span=n, adjust=False).mean()
        dn = (-diff.clip(upper=0)).ewm(span=n, adjust=False).mean()
        return 100 - (100 / (1 + up / (dn + 1e-8)))
    df_live['rsi_14'] = calc_rsi(c, 14)
    df_live['rsi_7']  = calc_rsi(c, 7)
    
    min_rsi = df_live['rsi_14'].rolling(14).min()
    max_rsi = df_live['rsi_14'].rolling(14).max()
    stoch_rsi = (df_live['rsi_14'] - min_rsi) / (max_rsi - min_rsi + 1e-8)
    df_live['stoch_rsi_k'] = stoch_rsi.rolling(3).mean() * 100
    df_live['stoch_rsi_d'] = df_live['stoch_rsi_k'].rolling(3).mean()
    
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df_live['macd_hist_norm'] = (macd_line - signal_line) / (c + 1e-8)
    
    ma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df_live['bb_pct_b'] = (c - (ma20 - 2 * std20)) / (4 * std20 + 1e-8)
    df_live['bb_bandwidth'] = (4 * std20) / (ma20 + 1e-8)
    df_live['atr_norm'] = atr14 / (c + 1e-8)
    
    up_move = h - h.shift(1)
    dn_move = l.shift(1) - l
    p_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
    m_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
    p_di = 100 * pd.Series(p_dm, index=df_live.index).ewm(span=14).mean() / (atr14 + 1e-8)
    m_di = 100 * pd.Series(m_dm, index=df_live.index).ewm(span=14).mean() / (atr14 + 1e-8)
    dx = 100 * (p_di - m_di).abs() / (p_di + m_di + 1e-8)
    df_live['adx_14'] = dx.ewm(span=14).mean()
    
    vol_eff = v.replace(0, np.nan).fillna(1.0)
    typical_p = (h + l + c) / 3.0
    cum_tp_vol = (typical_p * vol_eff).groupby(dates).cumsum()
    cum_vol = vol_eff.groupby(dates).cumsum()
    vwap = cum_tp_vol / (cum_vol + 1e-8)
    df_live['vwap_dist'] = (c - vwap) / (vwap + 1e-8)
    
    vol_sma20 = v.rolling(20).mean()
    df_live['volume_ratio'] = np.where(vol_sma20 > 0, v / (vol_sma20 + 1e-8), 1.0)
    
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    df_live['dist_ema20'] = (c - ema20) / (ema20 + 1e-8)
    df_live['dist_ema50'] = (c - ema50) / (ema50 + 1e-8)
    
    df_live['hour'] = df_live.index.hour.astype(int)
    df_live['minute'] = df_live.index.minute.astype(int)
    df_live['dayofweek'] = df_live.index.dayofweek.astype(int)
    
    latest_candle = df_live.dropna().iloc[[-1]]
    latest_time = latest_candle.index[0]
    latest_price = latest_candle['close'].iloc[0]
    
    X_latest = latest_candle[feat_cols]
    live_pool = Pool(X_latest, cat_features=cat_features)
    
    probs = model.predict_proba(live_pool)[0]
    prob_bearish = probs[0]
    prob_sideways = probs[1]
    prob_bullish = probs[2]
    
    if prob_bullish >= threshold:
        signal = "🟢 STRONG BUY (BULLISH CONVICTION)"
        action = "LONG"
    elif prob_bearish >= threshold:
        signal = "🔴 STRONG SELL (BEARISH CONVICTION)"
        action = "SHORT"
    else:
        signal = "⚪ NO TRADE - WAIT (CHOPPY / LOW CONVICTION)"
        action = "WAIT"
        
    print("=" * 75)
    print(f"📊 LIVE SCANNER REPORT: {exchange}:{symbol} [15-Minute]")
    print("=" * 75)
    print(f"🕒 Latest Closed Candle: {latest_time}")
    print(f"💰 Current Price (LTP):   ₹{latest_price:,.2f}")
    print("-" * 75)
    print(f"📈 Bullish Prob (Class 2):  {prob_bullish*100:5.1f}%")
    print(f"📉 Bearish Prob (Class 0):  {prob_bearish*100:5.1f}%")
    print(f"↔️  Sideways Prob (Class 1): {prob_sideways*100:5.1f}%")
    print(f"🎯 Threshold Trigger:       {threshold*100:.0f}%")
    print("-" * 75)
    print(f"⚡ ACTIONABLE SIGNAL:       {signal}")
    print("=" * 75)
    return {
        "timestamp": latest_time,
        "price": latest_price,
        "prob_bullish": prob_bullish,
        "prob_bearish": prob_bearish,
        "prob_sideways": prob_sideways,
        "signal": signal,
        "action": action
    }

print("Scanner defined successfully!")
