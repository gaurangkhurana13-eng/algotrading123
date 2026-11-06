import numpy as np
import pandas as pd
from tvDatafeed import TvDatafeed, Interval

tv = TvDatafeed()
df = tv.get_hist(symbol='NIFTY', exchange='NSE', interval=Interval.in_15_minute, n_bars=1000)
df.columns = [c.lower() for c in df.columns]
if 'symbol' in df.columns:
    df = df.drop(columns=['symbol'])

df.index = pd.to_datetime(df.index)
df = df.sort_index()

c, h, l, o = df['close'], df['high'], df['low'], df['open']
v = df['volume'].fillna(0.0)

# 1. Log returns
for k in [1, 2, 3, 5, 10]:
    df[f'log_ret_{k}'] = np.log(c / c.shift(k))

# 2. Session Awareness
df['is_opening_range'] = ((df.index.hour == 9) & (df.index.minute.isin([15, 30]))).astype(int)

# Overnight gap
dates = pd.Series(df.index.date, index=df.index)
daily_first_open = df.groupby(dates)['open'].first()
daily_last_close = df.groupby(dates)['close'].last()
prev_close = daily_last_close.shift(1)
daily_gap = (daily_first_open - prev_close) / (prev_close + 1e-8)
df['overnight_gap'] = dates.map(daily_gap).fillna(0.0).astype(float)

# 3. Microstructure & ATR
tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
atr14 = tr.ewm(span=14, adjust=False).mean()
df['atr_14'] = atr14
df['body_size_atr'] = (c - o) / (atr14 + 1e-8)
df['upper_wick_ratio'] = (h - np.maximum(o, c)) / (h - l + 1e-8)
df['lower_wick_ratio'] = (np.minimum(o, c) - l) / (h - l + 1e-8)

# 4. Momentum & Volatility
def calc_rsi(s, n):
    diff = s.diff()
    up = diff.clip(lower=0).ewm(span=n, adjust=False).mean()
    dn = (-diff.clip(upper=0)).ewm(span=n, adjust=False).mean()
    return 100 - (100 / (1 + up / (dn + 1e-8)))

df['rsi_14'] = calc_rsi(c, 14)
df['rsi_7']  = calc_rsi(c, 7)

# Stoch RSI
min_rsi = df['rsi_14'].rolling(14).min()
max_rsi = df['rsi_14'].rolling(14).max()
stoch_rsi = (df['rsi_14'] - min_rsi) / (max_rsi - min_rsi + 1e-8)
df['stoch_rsi_k'] = stoch_rsi.rolling(3).mean() * 100
df['stoch_rsi_d'] = df['stoch_rsi_k'].rolling(3).mean()

# MACD
ema12 = c.ewm(span=12, adjust=False).mean()
ema26 = c.ewm(span=26, adjust=False).mean()
macd_line = ema12 - ema26
signal_line = macd_line.ewm(span=9, adjust=False).mean()
df['macd_hist_norm'] = (macd_line - signal_line) / (c + 1e-8)

# Bollinger Bands
ma20 = c.rolling(20).mean()
std20 = c.rolling(20).std()
df['bb_pct_b'] = (c - (ma20 - 2 * std20)) / (4 * std20 + 1e-8)
df['bb_bandwidth'] = (4 * std20) / (ma20 + 1e-8)
df['atr_norm'] = atr14 / (c + 1e-8)

# ADX 14
up_move = h - h.shift(1)
dn_move = l.shift(1) - l
p_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
m_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
p_di = 100 * pd.Series(p_dm, index=df.index).ewm(span=14).mean() / (atr14 + 1e-8)
m_di = 100 * pd.Series(m_dm, index=df.index).ewm(span=14).mean() / (atr14 + 1e-8)
dx = 100 * (p_di - m_di).abs() / (p_di + m_di + 1e-8)
df['adx_14'] = dx.ewm(span=14).mean()

# VWAP distance
vol_eff = v.replace(0, np.nan).fillna(1.0)
typical_p = (h + l + c) / 3.0
cum_tp_vol = (typical_p * vol_eff).groupby(dates).cumsum()
cum_vol = vol_eff.groupby(dates).cumsum()
vwap = cum_tp_vol / (cum_vol + 1e-8)
df['vwap_dist'] = (c - vwap) / (vwap + 1e-8)

# 5. Volume Dynamics
vol_sma20 = v.rolling(20).mean()
df['volume_ratio'] = np.where(vol_sma20 > 0, v / (vol_sma20 + 1e-8), 1.0)

# 6. Moving Average Distances
ema20 = c.ewm(span=20, adjust=False).mean()
ema50 = c.ewm(span=50, adjust=False).mean()
df['dist_ema20'] = (c - ema20) / (ema20 + 1e-8)
df['dist_ema50'] = (c - ema50) / (ema50 + 1e-8)

# 7. Time Features
df['hour'] = df.index.hour.astype(int)
df['minute'] = df.index.minute.astype(int)
df['dayofweek'] = df.index.dayofweek.astype(int)

# Cell 4: Target Creation & EOD Filter
N = 1
threshold = 0.25 * df['atr_14']
future_close = df['close'].shift(-N)

conditions = [
    future_close > (df['close'] + threshold),
    future_close < (df['close'] - threshold)
]
df['target'] = np.select(conditions, [2, 0], default=1)

# EOD Filter: Drop candles whose target reaches into next day
future_date = dates.shift(-N)
is_cross_day = (future_date != dates) | future_date.isna()
df = df[~is_cross_day]

df_clean = df.dropna()
print("Clean dataset shape after dropping NaNs and EOD bars:", df_clean.shape)
print("Target distribution:\n", df_clean['target'].value_counts(normalize=True))
