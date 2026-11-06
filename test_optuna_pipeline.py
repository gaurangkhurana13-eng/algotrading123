import numpy as np
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
import catboost as cb
from catboost import CatBoostClassifier, Pool
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

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

# Non feature columns
raw_cols = ['open', 'high', 'low', 'close', 'volume', 'atr_14', 'target']
feat_cols = [col for col in df.columns if col not in raw_cols]
cat_features = ['hour', 'minute', 'dayofweek', 'is_opening_range']

# Time Series Split: 70% Train, 15% Val, 15% Test
n_total = len(df)
train_end_idx = int(0.70 * n_total)
val_end_idx   = int(0.85 * n_total)

train_df = df.iloc[:train_end_idx]
val_df   = df.iloc[train_end_idx:val_end_idx]
test_df  = df.iloc[val_end_idx:]

X_train, y_train = train_df[feat_cols], train_df['target']
X_val, y_val     = val_df[feat_cols], val_df['target']
X_test, y_test   = test_df[feat_cols], test_df['target']

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

train_pool = Pool(X_train, y_train, cat_features=cat_features)
val_pool   = Pool(X_val, y_val, cat_features=cat_features)
test_pool  = Pool(X_test, y_test, cat_features=cat_features)

def objective(trial):
    params = {
        'loss_function': 'MultiClass',
        'eval_metric': 'MultiClass',
        'auto_class_weights': 'Balanced',
        'depth': trial.suggest_int('depth', 4, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
        'random_strength': trial.suggest_float('random_strength', 1.0, 5.0),
        'iterations': 300,
        'early_stopping_rounds': 50,
        'verbose': False,
        'random_seed': 42
    }
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool, verbose=False)
    preds = model.predict_proba(val_pool)
    from sklearn.metrics import log_loss
    return log_loss(y_val, preds)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=3)
print("Optuna best trial:", study.best_trial.value)
print("Optuna best params:", study.best_trial.params)

# Train best model
best_params = study.best_trial.params
best_params.update({
    'loss_function': 'MultiClass',
    'eval_metric': 'MultiClass',
    'auto_class_weights': 'Balanced',
    'iterations': 400,
    'early_stopping_rounds': 50,
    'verbose': False,
    'random_seed': 42
})
model = CatBoostClassifier(**best_params)
model.fit(train_pool, eval_set=val_pool, verbose=False)

# Cell 7: Probability Filtering
probs = model.predict_proba(test_pool)
# probs: [P(Class 0), P(Class 1), P(Class 2)]
results = []
for th in [0.40, 0.45, 0.50, 0.55, 0.60]:
    long_mask = probs[:, 2] >= th
    short_mask = probs[:, 0] >= th
    
    long_trades = np.where(long_mask)[0]
    short_trades = np.where(short_mask)[0]
    
    total_trades = len(long_trades) + len(short_trades)
    if total_trades == 0:
        results.append({"Threshold": th, "Trades": 0, "Wins": 0, "Losses": 0, "Win Rate %": "0.0%"})
        continue
        
    long_wins = np.sum(y_test.iloc[long_trades] == 2)
    short_wins = np.sum(y_test.iloc[short_trades] == 0)
    total_wins = long_wins + short_wins
    total_losses = total_trades - total_wins
    win_rate = (total_wins / total_trades) * 100
    results.append({
        "Threshold": f">={th*100:.0f}%",
        "Trades": total_trades,
        "Wins": total_wins,
        "Losses": total_losses,
        "Win Rate %": f"{win_rate:.1f}%"
    })

print(pd.DataFrame(results).to_string(index=False))
print("Test completed successfully!")
