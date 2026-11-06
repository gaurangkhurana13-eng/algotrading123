import numpy as np
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
import catboost as cb
from catboost import CatBoostClassifier, Pool

tv = TvDatafeed()
df = tv.get_hist(symbol='NIFTY', exchange='NSE', interval=Interval.in_15_minute, n_bars=1000)
df.columns = [c.lower() for c in df.columns]
if 'symbol' in df.columns:
    df = df.drop(columns=['symbol'])
df.index = pd.to_datetime(df.index)
df = df.sort_index()

c, h, l, o = df['close'], df['high'], df['low'], df['open']
v = df['volume'].fillna(0.0)

for k in [1, 2, 3, 5, 10]:
    df[f'log_ret_{k}'] = np.log(c / c.shift(k))

df['is_opening_range'] = ((df.index.hour == 9) & (df.index.minute.isin([15, 30]))).astype(int)
dates = pd.Series(df.index.date, index=df.index)
daily_first_open = df.groupby(dates)['open'].first()
daily_last_close = df.groupby(dates)['close'].last()
prev_close = daily_last_close.shift(1)
daily_gap = (daily_first_open - prev_close) / (prev_close + 1e-8)
df['overnight_gap'] = dates.map(daily_gap).fillna(0.0).astype(float)

tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
atr14 = tr.ewm(span=14, adjust=False).mean()
df['atr_14'] = atr14
df['body_size_atr'] = (c - o) / (atr14 + 1e-8)
df['upper_wick_ratio'] = (h - np.maximum(o, c)) / (h - l + 1e-8)
df['lower_wick_ratio'] = (np.minimum(o, c) - l) / (h - l + 1e-8)

def calc_rsi(s, n):
    diff = s.diff()
    up = diff.clip(lower=0).ewm(span=n, adjust=False).mean()
    dn = (-diff.clip(upper=0)).ewm(span=n, adjust=False).mean()
    return 100 - (100 / (1 + up / (dn + 1e-8)))

df['rsi_14'] = calc_rsi(c, 14)
df['rsi_7']  = calc_rsi(c, 7)

min_rsi = df['rsi_14'].rolling(14).min()
max_rsi = df['rsi_14'].rolling(14).max()
stoch_rsi = (df['rsi_14'] - min_rsi) / (max_rsi - min_rsi + 1e-8)
df['stoch_rsi_k'] = stoch_rsi.rolling(3).mean() * 100
df['stoch_rsi_d'] = df['stoch_rsi_k'].rolling(3).mean()

ema12 = c.ewm(span=12, adjust=False).mean()
ema26 = c.ewm(span=26, adjust=False).mean()
macd_line = ema12 - ema26
signal_line = macd_line.ewm(span=9, adjust=False).mean()
df['macd_hist_norm'] = (macd_line - signal_line) / (c + 1e-8)

ma20 = c.rolling(20).mean()
std20 = c.rolling(20).std()
df['bb_pct_b'] = (c - (ma20 - 2 * std20)) / (4 * std20 + 1e-8)
df['bb_bandwidth'] = (4 * std20) / (ma20 + 1e-8)
df['atr_norm'] = atr14 / (c + 1e-8)

up_move = h - h.shift(1)
dn_move = l.shift(1) - l
p_dm = np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0)
m_dm = np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0)
p_di = 100 * pd.Series(p_dm, index=df.index).ewm(span=14).mean() / (atr14 + 1e-8)
m_di = 100 * pd.Series(m_dm, index=df.index).ewm(span=14).mean() / (atr14 + 1e-8)
dx = 100 * (p_di - m_di).abs() / (p_di + m_di + 1e-8)
df['adx_14'] = dx.ewm(span=14).mean()

vol_eff = v.replace(0, np.nan).fillna(1.0)
typical_p = (h + l + c) / 3.0
cum_tp_vol = (typical_p * vol_eff).groupby(dates).cumsum()
cum_vol = vol_eff.groupby(dates).cumsum()
vwap = cum_tp_vol / (cum_vol + 1e-8)
df['vwap_dist'] = (c - vwap) / (vwap + 1e-8)

vol_sma20 = v.rolling(20).mean()
df['volume_ratio'] = np.where(vol_sma20 > 0, v / (vol_sma20 + 1e-8), 1.0)

ema20 = c.ewm(span=20, adjust=False).mean()
ema50 = c.ewm(span=50, adjust=False).mean()
df['dist_ema20'] = (c - ema20) / (ema20 + 1e-8)
df['dist_ema50'] = (c - ema50) / (ema50 + 1e-8)

df['hour'] = df.index.hour.astype(int)
df['minute'] = df.index.minute.astype(int)
df['dayofweek'] = df.index.dayofweek.astype(int)

N = 1
threshold = 0.25 * df['atr_14']
future_close = df['close'].shift(-N)
conditions = [
    future_close > (df['close'] + threshold),
    future_close < (df['close'] - threshold)
]
df['target'] = np.select(conditions, [2, 0], default=1)

future_date = dates.shift(-N)
is_cross_day = (future_date != dates) | future_date.isna()
df = df[~is_cross_day].dropna()

raw_cols = ['open', 'high', 'low', 'close', 'volume', 'atr_14', 'target']
feature_cols = [col for col in df.columns if col not in raw_cols]
cat_features = ['hour', 'minute', 'dayofweek', 'is_opening_range']

n = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

train_data = df.iloc[:train_end]
val_data   = df.iloc[train_end:val_end]
test_data  = df.iloc[val_end:]

X_train, y_train = train_data[feature_cols], train_data['target']
X_val, y_val     = val_data[feature_cols], val_data['target']
X_test, y_test   = test_data[feature_cols], test_data['target']

train_pool = Pool(X_train, y_train, cat_features=cat_features)
val_pool   = Pool(X_val, y_val, cat_features=cat_features)
test_pool  = Pool(X_test, y_test, cat_features=cat_features)

primary_model = CatBoostClassifier(iterations=300, depth=5, learning_rate=0.03, loss_function='MultiClass', eval_metric='MultiClass', auto_class_weights='Balanced', random_seed=42, verbose=False)
primary_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=40, verbose=False)

# Cell 9: Stage-2 Meta-Labeling
def build_meta_dataset(df_subset, model, primary_threshold=0.52):
    subset_pool = Pool(df_subset[feature_cols], cat_features=cat_features)
    probs = model.predict_proba(subset_pool) # [P(0), P(1), P(2)]
    
    # Calculate Prediction Entropy: - sum(P * log(P))
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)
    
    rows = []
    for idx, (original_idx, row) in enumerate(df_subset.iterrows()):
        p_short = probs[idx, 0]
        p_long  = probs[idx, 2]
        
        signal = 0
        conf = 0.0
        if p_long >= primary_threshold:
            signal = 1  # Long
            conf = p_long
        elif p_short >= primary_threshold:
            signal = -1 # Short
            conf = p_short
            
        if signal == 0:
            continue
            
        actual_target = row['target']
        is_win = 1 if ((signal == 1 and actual_target == 2) or (signal == -1 and actual_target == 0)) else 0
        
        feat_dict = {c: row[c] for c in feature_cols}
        feat_dict['primary_signal'] = signal
        feat_dict['stage1_prob']    = conf
        feat_dict['entropy']        = entropy[idx]
        feat_dict['prob_times_adx'] = conf * (row['adx_14'] / 50.0)
        feat_dict['prob_over_entropy'] = conf / (entropy[idx] + 1e-4)
        feat_dict['meta_target']    = is_win
        feat_dict['future_ret']     = (row['close'] if signal==1 else row['close']) # will be used in backtest
        rows.append(feat_dict)
        
    return pd.DataFrame(rows)

meta_val_df  = build_meta_dataset(val_data, primary_model, primary_threshold=0.45) # lower threshold for demonstration
meta_test_df = build_meta_dataset(test_data, primary_model, primary_threshold=0.45)

print(f"Meta Val Set: {len(meta_val_df)} setups")
print(f"Meta Test Set: {len(meta_test_df)} setups")
