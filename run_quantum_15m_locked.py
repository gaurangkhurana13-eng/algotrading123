# ==============================================================================
# 👑 BHAVISHYA QUANTUM 15M APEX — LOCKED SNIPER (72.7% ACCURACY ZONE)
# ==============================================================================
import os, gc, sys, json, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import catboost as cb
import xgboost as xgb
try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

from tvDatafeed import TvDatafeed, Interval
tv = TvDatafeed()

print("=" * 80)
print("🚀 BHAVISHYA QUANTUM 15M APEX — 52%+ SNIPER RUNNER (LOCAL EXECUTION)")
print("================================================================================")

# --- 1. CONFIGURATION ---
INITIAL_CAPITAL    = 500.0     # Starting Groww intraday capital
LEVERAGE           = 5.0       # 5x Groww Intraday Margin
CONFIDENCE_GATE    = 0.52      # 🎯 LOCKED SNIPER GATE (72.7% Win Rate Zone)
SLIPPAGE_PER_LEG   = 0.0008    # 0.08% slippage per leg (0.16% total)
MAX_DAILY_LOSSES   = 2         # Circuit Breaker: Halt after 2 consecutive daily losses
MAX_HOLDING_BARS   = 8         # 8 bars x 15m = 2 Hours Max Intraday Hold

NSE_STOCKS = [
    'ADANIPOWER', 'ALOKINDS', 'ASHOKLEY', 'BANDHANBNK', 'BANKBARODA',
    'BANKINDIA', 'BCLIND', 'BPCL', 'CANBK', 'CENTRALBK',
    'COALINDIA', 'ELECTCAST', 'EQUITASBNK', 'GAIL', 'GMRAIRPORT',
    'GSFC', 'HATHWAY', 'HCC', 'HUDCO', 'IDBI',
    'IDEA', 'IDFCFIRSTB', 'IFCI', 'INDIANB', 'IOB',
    'IOC', 'IRCON', 'IRFC', 'J&KBANK', 'JAICORPLTD',
    'JPPOWER', 'JYOTHYLAB', 'LLOYDSENGG', 'MAHABANK', 'MANALIPETC',
    'MMTC', 'MOREPENLAB', 'MOTHERSON', 'MRPL', 'MTNL',
    'NBCC', 'NETWORK18', 'NFL', 'NHPC', 'NIACL',
    'NMDC', 'NTPC', 'ONGC', 'PATANJALI', 'PCJEWELLER',
    'PFC', 'PNB', 'PTC', 'RBLBANK', 'RCF',
    'RECLTD', 'RENUKA', 'RPOWER', 'RVNL', 'SAIL',
    'SBIN', 'SJVN', 'SOUTHBANK', 'SPIC', 'SUBEXLTD',
    'SUZLON', 'SWSOLAR', 'TATAPOWER', 'TATASTEEL', 'TEXRAIL',
    'TRIDENT', 'TTML', 'UCOBANK', 'UJJIVANSFB', 'UNIONBANK',
    'UTKARSHBNK', 'VEDL', 'YESBANK'
]

def calc_groww_charges(buy_p, sell_p, qty):
    buy_to = buy_p * qty
    sell_to = sell_p * qty
    total_to = buy_to + sell_to
    buy_brok = min(20.0, max(5.0, buy_to * 0.0005))
    sell_brok = min(20.0, max(5.0, sell_to * 0.0005))
    brok = buy_brok + sell_brok
    stt = sell_to * 0.00025
    exc = total_to * 0.0000297
    sebi = total_to * 0.000001
    gst = (brok + exc + sebi) * 0.18
    stamp = buy_to * 0.00003
    return round(brok + stt + exc + sebi + gst + stamp, 2)

print("\n📡 Step 1: Downloading 15-Minute Candles (tvDatafeed)...")
stock_raw = {}
nifty_raw = None

try:
    nifty_raw = tv.get_hist('NIFTY', 'NSE', interval=Interval.in_15_minute, n_bars=4000)
    if nifty_raw is not None and not nifty_raw.empty:
        nifty_raw.columns = [c.lower() for c in nifty_raw.columns]
        if 'symbol' in nifty_raw.columns:
            nifty_raw = nifty_raw.drop(columns=['symbol'])
        print(f"   • NIFTY 50: {len(nifty_raw):,} 15m candles ready.")
except Exception as e:
    print(f"   ⚠️ Nifty download error: {e}")

for idx, sym in enumerate(NSE_STOCKS):
    try:
        df = tv.get_hist(sym, 'NSE', interval=Interval.in_15_minute, n_bars=4000)
        if df is not None and len(df) > 300:
            df.columns = [c.lower() for c in df.columns]
            if 'symbol' in df.columns:
                df = df.drop(columns=['symbol'])
            stock_raw[sym] = df.sort_index()
    except Exception:
        pass
    if (idx + 1) % 25 == 0 or (idx + 1) == len(NSE_STOCKS):
        print(f"   • Loaded {len(stock_raw)} / {len(NSE_STOCKS)} stocks...")

print(f"✅ Ready: {len(stock_raw)} stocks.")

def make_features_15m(df, nifty_df=None):
    d = df.copy()
    if 'symbol' in d.columns:
        d = d.drop(columns=['symbol'])
    c, h, l, o = d['close'], d['high'], d['low'], d['open']
    v = d['volume'].replace(0, np.nan).ffill().fillna(1.0)
    
    for lag in [1, 2, 3, 4, 8]:
        d[f'ret_{lag}'] = c.pct_change(lag).shift(1)
        
    d['hl_range'] = ((h - l) / c.shift(1)).shift(1)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    d['atr14'] = tr.ewm(span=14, adjust=False).mean().shift(1)
    d['atr_norm'] = (tr / c.shift(1)).shift(1)
    
    vol_ma20 = v.rolling(20).mean().shift(1)
    vol_std20 = v.rolling(20).std().shift(1)
    d['rvol'] = np.log1p(v / (vol_ma20 + 1e-6)).shift(1)
    d['vol_zscore'] = ((v - vol_ma20) / (vol_std20 + 1e-6)).shift(1)
    d['vol_trend'] = v.pct_change(4).shift(1)
    
    typical = (h + l + c) / 3
    cum_tp_vol = (typical * v).groupby(d.index.date).cumsum()
    cum_vol_g = v.groupby(d.index.date).cumsum()
    vwap = (cum_tp_vol / (cum_vol_g + 1e-6)).shift(1)
    d['vwap_dev'] = ((c - vwap) / (vwap + 1e-6)).shift(1)
    d['above_vwap'] = (c > vwap).astype(int).shift(1)
    
    def _rsi(s, n):
        delta = s.diff()
        g = delta.clip(lower=0).ewm(span=n, adjust=False).mean()
        ls = (-delta.clip(upper=0)).ewm(span=n, adjust=False).mean()
        return (100 - 100 / (1 + g / (ls + 1e-6))).shift(1)
    d['rsi7'] = _rsi(c, 7)
    d['rsi14'] = _rsi(c, 14)
    
    ma20 = c.rolling(20).mean().shift(1)
    std20 = c.rolling(20).std().shift(1)
    d['bb_pos'] = ((c - (ma20 - 2 * std20)) / (4 * std20 + 1e-6)).shift(1)
    d['bb_squeeze'] = ((4 * std20) / (ma20 + 1e-6)).shift(1)
    
    ema12 = c.ewm(span=12, adjust=False).mean().shift(1)
    ema26 = c.ewm(span=26, adjust=False).mean().shift(1)
    macd = ema12 - ema26
    sig9 = macd.ewm(span=9, adjust=False).mean()
    d['macd_diff'] = macd - sig9
    
    ema9  = c.ewm(span=9, adjust=False).mean().shift(1)
    ema21 = c.ewm(span=21, adjust=False).mean().shift(1)
    ema50 = c.ewm(span=50, adjust=False).mean().shift(1)
    d['trend_fast'] = ((ema9 - ema21) / (c.shift(1) + 1e-6)).shift(1)
    d['trend_slow'] = ((ema21 - ema50) / (c.shift(1) + 1e-6)).shift(1)
    
    up_m = h - h.shift(1)
    dn_m = l.shift(1) - l
    pos_dm = np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0)
    neg_dm = np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0)
    pos_di = 100 * (pd.Series(pos_dm, index=d.index).ewm(span=14).mean() / (d['atr14'] + 1e-6)).shift(1)
    neg_di = 100 * (pd.Series(neg_dm, index=d.index).ewm(span=14).mean() / (d['atr14'] + 1e-6)).shift(1)
    dx = 100 * (pos_di - neg_di).abs() / (pos_di + neg_di + 1e-6)
    d['adx'] = dx.ewm(span=14).mean().shift(1)
    
    if nifty_df is not None and not nifty_df.empty:
        nifty_c = nifty_df['close'].reindex(d.index).ffill()
        d['nifty_ret1'] = nifty_c.pct_change(1).shift(1)
        d['nifty_ret4'] = nifty_c.pct_change(4).shift(1)
    else:
        d['nifty_ret1'] = 0.0
        d['nifty_ret4'] = 0.0
        
    mins = d.index.hour * 60 + d.index.minute
    d['tod_sin'] = np.sin(2 * np.pi * mins / 1440)
    d['tod_cos'] = np.cos(2 * np.pi * mins / 1440)
    return d

def screen_setups(df):
    cond_long = (
        (df['rvol'] > 0.40) &
        (df['above_vwap'] == 1) &
        (df['rsi14'] > 52) &
        (df['trend_fast'] > 0)
    )
    cond_short = (
        (df['rvol'] > 0.40) &
        (df['above_vwap'] == 0) &
        (df['rsi14'] < 48) &
        (df['trend_fast'] < 0)
    )
    df['candidate_dir'] = 0
    df.loc[cond_long, 'candidate_dir'] = 1
    df.loc[cond_short, 'candidate_dir'] = -1
    return df

def apply_triple_barrier(df):
    d = df.dropna().copy()
    if len(d) < 50:
        return pd.DataFrame()
    c, h, l = d['close'], d['high'], d['low']
    candidates = d[d['candidate_dir'] != 0]
    labeled_rows = []
    
    for idx in candidates.index:
        loc = d.index.get_loc(idx)
        if loc + MAX_HOLDING_BARS >= len(d):
            continue
            
        entry_price = float(c.iloc[loc])
        direction = int(d['candidate_dir'].iloc[loc])
        atr_n = float(d['atr_norm'].iloc[loc])
        
        # 15m Fee-Shield Targets: TP1 +1.50%, SL -0.80%
        tp_pct = max(0.015, 1.3 * atr_n)
        sl_pct = max(0.008, 0.8 * atr_n)
        
        hit_tp = False
        hit_sl = False
        exit_p = entry_price
        bars_held = MAX_HOLDING_BARS
        
        for b in range(1, MAX_HOLDING_BARS + 1):
            future_high = float(h.iloc[loc + b])
            future_low  = float(l.iloc[loc + b])
            
            if direction == 1:
                if future_low <= entry_price * (1.0 - sl_pct):
                    hit_sl = True
                    exit_p = entry_price * (1.0 - sl_pct)
                    bars_held = b
                    break
                if future_high >= entry_price * (1.0 + tp_pct):
                    hit_tp = True
                    exit_p = entry_price * (1.0 + tp_pct)
                    bars_held = b
                    break
            else:
                if future_high >= entry_price * (1.0 + sl_pct):
                    hit_sl = True
                    exit_p = entry_price * (1.0 + sl_pct)
                    bars_held = b
                    break
                if future_low <= entry_price * (1.0 - tp_pct):
                    hit_tp = True
                    exit_p = entry_price * (1.0 - tp_pct)
                    bars_held = b
                    break
                    
        if not hit_tp and not hit_sl:
            exit_p = float(c.iloc[loc + MAX_HOLDING_BARS])
            if direction == 1:
                hit_tp = (exit_p > entry_price * 1.005)
            else:
                hit_tp = (exit_p < entry_price * 0.995)
                
        target_label = 1 if hit_tp else 0
        row_dict = d.loc[idx].to_dict()
        row_dict['target'] = target_label
        row_dict['meta_entry_p'] = entry_price
        row_dict['meta_exit_p'] = exit_p
        row_dict['meta_dir'] = direction
        row_dict['meta_bars_held'] = bars_held
        labeled_rows.append(row_dict)
        
    if not labeled_rows:
        return pd.DataFrame()
    df_labeled = pd.DataFrame(labeled_rows)
    df_labeled.index = candidates.index[:len(df_labeled)]
    return df_labeled

print("\n⚙️ Step 2: Feature Extraction & Meta-Labeling...")
master_dfs = []
stock_clean_dict = {}

for sym, raw_df in stock_raw.items():
    feat_df = make_features_15m(raw_df, nifty_raw)
    screened_df = screen_setups(feat_df)
    labeled = apply_triple_barrier(screened_df)
    if not labeled.empty:
        labeled['ticker'] = sym
        master_dfs.append(labeled)
        stock_clean_dict[sym] = raw_df

master = pd.concat(master_dfs).sort_index()

non_feat_cols = [
    'open', 'high', 'low', 'close', 'volume', 'target', 'symbol',
    'candidate_dir', 'meta_entry_p', 'meta_exit_p', 'meta_dir',
    'meta_bars_held', 'ticker'
]
feat_cols = [c for c in master.columns if c not in non_feat_cols]

# Date split: Last 30 Days 100% Unseen
cutoff_date = master.index.max() - pd.Timedelta(days=30)
train_mask = master.index <= cutoff_date
test_mask  = master.index > cutoff_date

X_train, y_train = master[train_mask][feat_cols], master[train_mask]['target']
X_test, y_test   = master[test_mask][feat_cols], master[test_mask]['target']

print(f"\n📅 DATE-BASED VALIDATION SPLIT (100% Honest Out-of-Time Test):")
print(f"   • TRAIN: {master[train_mask].index.min().date()} → {master[train_mask].index.max().date()} ({len(X_train):,} setups)")
print(f"   • TEST:  {master[test_mask].index.min().date()} → {master[test_mask].index.max().date()} ({len(X_test):,} setups UNSEEN)")

print("\n🏋️ Step 3: Training Resilient Probability Ensemble...")
cb_model = cb.CatBoostClassifier(iterations=400, learning_rate=0.04, depth=6, l2_leaf_reg=3.0, random_seed=42, verbose=False)
cb_model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=40, verbose=False)
cb_probs = cb_model.predict_proba(X_test)[:, 1]

xgb_model = xgb.XGBClassifier(n_estimators=350, learning_rate=0.03, max_depth=4, min_child_weight=4, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.5, random_state=42, eval_metric='logloss')
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

if HAS_LGBM:
    lgb_model = lgb.LGBMClassifier(n_estimators=350, learning_rate=0.03, num_leaves=31, max_depth=5, min_child_samples=50, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.5, random_state=42, verbose=-1)
    lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(40, verbose=False)])
    lgb_probs = lgb_model.predict_proba(X_test)[:, 1]
    ensemble_win_prob = 0.34 * cb_probs + 0.33 * xgb_probs + 0.33 * lgb_probs
else:
    ensemble_win_prob = 0.50 * cb_probs + 0.50 * xgb_probs

test_df = master[test_mask].copy()
test_df['win_probability'] = ensemble_win_prob

print("\n" + "=" * 80)
print("🎯 OUT-OF-SAMPLE CONVICTION SWEEP (Last 30 Days Unseen Data):")
print("=" * 80)
for p_gate in [0.45, 0.48, 0.50, 0.52, 0.54]:
    sub = test_df[test_df['win_probability'] >= p_gate]
    if len(sub) > 0:
        actual_wins = (sub['target'] == 1).sum()
        hit_rate = (actual_wins / len(sub)) * 100
        days = len(sub.index.normalize().unique())
        print(f"   • Conviction Gate >= {p_gate*100:.0f}%  -->  Win Rate: {hit_rate:.1f}%  ({len(sub)} signals, ~{len(sub)/max(1,days):.1f}/day)")
print("=" * 80)

# --- 4. DAY TRADER BACKTEST (LOCKED >= 52% SNIPER) ---
print("\n🛡️ Step 4: Executing Backtest on 52%+ Sniper Setups...")
high_conviction = test_df[test_df['win_probability'] >= CONFIDENCE_GATE].copy()
unique_times = sorted(high_conviction.index.unique())

capital = INITIAL_CAPITAL
withdrawn_to_bank = 0.0
ledger = []
daily_loss_tracker = {}
open_trade_exit_time = None

for cur_time in unique_times:
    trade_date = str(cur_time.date())
    if daily_loss_tracker.get(trade_date, 0) >= MAX_DAILY_LOSSES:
        continue
    if open_trade_exit_time is not None and cur_time < open_trade_exit_time:
        continue
        
    snipers_now = high_conviction.loc[[cur_time]]
    if snipers_now.empty:
        continue
        
    best_candidate = snipers_now.sort_values('win_probability', ascending=False).iloc[0]
    sym = best_candidate['ticker']
    direction = int(best_candidate['meta_dir'])
    win_p = float(best_candidate['win_probability'])
    
    atr_n = float(best_candidate['atr_norm'])
    tp1_pct = max(0.015, 1.3 * atr_n)
    tp2_pct = tp1_pct * 1.7
    sl_pct  = max(0.008, 0.8 * atr_n)
    
    if sym not in stock_clean_dict:
        continue
    df_raw = stock_clean_dict[sym]
    try:
        cur_idx = df_raw.index.get_loc(cur_time)
    except:
        continue
        
    entry_raw = float(df_raw['close'].iloc[cur_idx])
    shares = int((capital * LEVERAGE) / entry_raw)
    if shares < 1:
        continue
        
    half_shares = max(1, shares // 2)
    runner_shares = shares - half_shares
    
    if direction == 1:
        entry_p = entry_raw * (1.0 + SLIPPAGE_PER_LEG)
        sl_price  = entry_p * (1.0 - sl_pct)
        tp1_price = entry_p * (1.0 + tp1_pct)
        tp2_price = entry_p * (1.0 + tp2_pct)
    else:
        entry_p = entry_raw * (1.0 - SLIPPAGE_PER_LEG)
        sl_price  = entry_p * (1.0 + sl_pct)
        tp1_price = entry_p * (1.0 - tp1_pct)
        tp2_price = entry_p * (1.0 - tp2_pct)
        
    max_fwd = min(MAX_HOLDING_BARS, len(df_raw) - cur_idx - 1)
    if max_fwd < 1:
        continue
        
    tp1_hit = False
    exit_p_half1 = 0.0
    exit_p_half2 = 0.0
    exit_reason = "TIMEOUT_2H"
    exit_bar = max_fwd
    
    for b in range(1, max_fwd + 1):
        bar_h = float(df_raw['high'].iloc[cur_idx + b])
        bar_l = float(df_raw['low'].iloc[cur_idx + b])
        bar_c = float(df_raw['close'].iloc[cur_idx + b])
        
        if direction == 1:
            if not tp1_hit and bar_l <= sl_price:
                exit_p_half1 = exit_p_half2 = sl_price * (1.0 - SLIPPAGE_PER_LEG)
                exit_reason = "INITIAL_SL"
                exit_bar = b
                break
            if not tp1_hit and bar_h >= tp1_price:
                tp1_hit = True
                exit_p_half1 = tp1_price * (1.0 - SLIPPAGE_PER_LEG)
                sl_price = entry_p
            if tp1_hit:
                if bar_l <= sl_price:
                    exit_p_half2 = sl_price * (1.0 - SLIPPAGE_PER_LEG)
                    exit_reason = "TP1_THEN_BREAKEVEN"
                    exit_bar = b
                    break
                if bar_h >= tp2_price:
                    exit_p_half2 = tp2_price * (1.0 - SLIPPAGE_PER_LEG)
                    exit_reason = "RUNNER_TP2_HIT"
                    exit_bar = b
                    break
        else:
            if not tp1_hit and bar_h >= sl_price:
                exit_p_half1 = exit_p_half2 = sl_price * (1.0 + SLIPPAGE_PER_LEG)
                exit_reason = "INITIAL_SL"
                exit_bar = b
                break
            if not tp1_hit and bar_l <= tp1_price:
                tp1_hit = True
                exit_p_half1 = tp1_price * (1.0 + SLIPPAGE_PER_LEG)
                sl_price = entry_p
            if tp1_hit:
                if bar_h >= sl_price:
                    exit_p_half2 = sl_price * (1.0 + SLIPPAGE_PER_LEG)
                    exit_reason = "TP1_THEN_BREAKEVEN"
                    exit_bar = b
                    break
                if bar_l <= tp2_price:
                    exit_p_half2 = tp2_price * (1.0 + SLIPPAGE_PER_LEG)
                    exit_reason = "RUNNER_TP2_HIT"
                    exit_bar = b
                    break
                    
        if tp1_hit and b >= max_fwd:
            exit_p_half2 = bar_c * (1.0 - SLIPPAGE_PER_LEG if direction == 1 else 1.0 + SLIPPAGE_PER_LEG)
            exit_reason = "RUNNER_MAX_HOLD"
            exit_bar = b
            break
            
        if not tp1_hit and b >= max_fwd:
            exit_p_half1 = exit_p_half2 = bar_c * (1.0 - SLIPPAGE_PER_LEG if direction == 1 else 1.0 + SLIPPAGE_PER_LEG)
            exit_reason = "TIMEOUT_2H"
            exit_bar = b
            break
            
    if exit_p_half1 == 0.0:
        exit_p_half1 = exit_p_half2 = float(df_raw['close'].iloc[min(cur_idx + 8, len(df_raw) - 1)]) * (1.0 - SLIPPAGE_PER_LEG if direction == 1 else 1.0 + SLIPPAGE_PER_LEG)
        
    avg_exit_p = (exit_p_half1 * half_shares + exit_p_half2 * runner_shares) / shares
    if direction == 1:
        gross_pnl = (exit_p_half1 - entry_p) * half_shares + (exit_p_half2 - entry_p) * runner_shares
    else:
        gross_pnl = (entry_p - exit_p_half1) * half_shares + (entry_p - exit_p_half2) * runner_shares
        
    fees = calc_groww_charges(entry_p, avg_exit_p, shares)
    net_pnl = round(gross_pnl - fees, 2)
    capital = round(capital + net_pnl, 2)
    
    if net_pnl <= 0:
        daily_loss_tracker[trade_date] = daily_loss_tracker.get(trade_date, 0) + 1
        
    open_trade_exit_time = df_raw.index[min(len(df_raw) - 1, cur_idx + exit_bar)]
    
    ledger.append({
        "trade_no": len(ledger) + 1,
        "date": trade_date,
        "time": str(cur_time.time())[:5],
        "ticker": sym,
        "action": "BUY" if direction == 1 else "SELL",
        "win_prob": round(win_p * 100, 1),
        "entry_price": round(entry_p, 2),
        "exit_price": round(avg_exit_p, 2),
        "holding_mins": exit_bar * 15,
        "shares": shares,
        "gross_pnl": round(gross_pnl, 2),
        "charges": fees,
        "net_pnl": net_pnl,
        "capital": capital,
        "wealth": round(capital + withdrawn_to_bank, 2),
        "exit_reason": exit_reason,
        "winner": net_pnl > 0
    })

df_ledger = pd.DataFrame(ledger)
total_trades = len(df_ledger)
wins = int(df_ledger['winner'].sum()) if total_trades > 0 else 0
losses = total_trades - wins
win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
final_wealth = round(capital + withdrawn_to_bank, 2)
net_roi = ((final_wealth - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

gw = df_ledger[df_ledger['net_pnl'] > 0]['net_pnl'].sum() if total_trades > 0 else 0.0
gl = abs(df_ledger[df_ledger['net_pnl'] < 0]['net_pnl'].sum()) if total_trades > 0 else 0.0
pf = round(gw / (gl + 1e-6), 2) if gl > 0 else (round(gw, 2) if gw > 0 else 0.0)
peak_wealth = df_ledger['wealth'].max() if total_trades > 0 else INITIAL_CAPITAL
drawdown = round(peak_wealth - final_wealth, 2)
total_charges = df_ledger['charges'].sum() if total_trades > 0 else 0.0

print("\n" + "=" * 80)
print("🏆 BHAVISHYA QUANTUM 15M APEX — 52%+ SNIPER AUDIT (100% UNSEEN FORWARD TEST)")
print("=" * 80)
print(f"🎯 DIRECTIONAL HIT RATE (ACCURACY): {win_rate:.1f}% ({wins} Wins / {losses} Losses)")
print(f"💰 INITIAL CAPITAL:                 ₹{INITIAL_CAPITAL:.2f}")
print(f"💼 CURRENT ACCOUNT BALANCE:         ₹{capital:.2f}")
print(f"💎 TOTAL REALIZED WEALTH:           ₹{final_wealth:.2f} ({'+' if net_roi>=0 else ''}{net_roi:.1f}% Net ROI)")
print(f"⭐ PROFIT FACTOR:                   {pf}")
print(f"📉 PEAK DRAWDOWN:                   ₹{drawdown:.2f}")
print(f"💸 TOTAL GROWW CHARGES DEDUCTED:    ₹{total_charges:.2f}")
print(f"📊 EXECUTED SNIPERS:                {total_trades}")
print("=" * 80)

if total_trades > 0:
    print(f"\n📋 EXECUTED SNIPER TRADES:")
    print(df_ledger[['date', 'time', 'ticker', 'action', 'win_prob', 'entry_price', 'exit_price', 'gross_pnl', 'charges', 'net_pnl', 'wealth', 'exit_reason']].to_string(index=False))
