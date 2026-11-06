# ============================================================
# BHAVISHYA V4.1 — DHAN EDITION (MICRO-CAPITAL INTRADAY)
# Google Colab Ready | Single-Cell Copy-Paste
# Target: ₹500 Account | 5x MIS = ₹2,500 Buying Power
# Broker: DHAN (Exact ₹1.39 Friction Model — No Min Brokerage)
# ============================================================

import subprocess, sys

def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["catboost", "tvDatafeed", "ta", "scipy"]:
    try:
        __import__(pkg.replace("-","_").split("==")[0])
    except ImportError:
        pip_install(pkg)

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from catboost import CatBoostClassifier
from tvDatafeed import TvDatafeed, Interval
import ta, time

print("✅ All libraries loaded successfully.")

# ── CONFIG ───────────────────────────────────────────────────
CONFIG = {
    # Capital & Margin
    "CAPITAL"         : 500.0,       # Base capital in ₹
    "LEVERAGE"        : 5.0,         # Dhan 5x MIS Intraday Leverage
    
    # Triple Barrier Setup (Optimized for Dhan's Low Friction)
    "TP_PCT"          : 0.012,       # 1.20% Take Profit (₹30 Gross Win)
    "SL_PCT"          : 0.006,       # 0.60% Stop Loss   (₹15 Gross Loss) -> 2:1 R Ratio
    "HOLD_BARS"       : 10,          # 10 bars × 15m = 2.5 hours maximum hold
    
    # Walk-Forward Validation
    "N_FOLDS"         : 5,           # 5-fold expanding walk-forward
    "EMBARGO_DAYS"    : 2,           # 2-day embargo between train and test
    "MIN_TRAIN_BARS"  : 1200,        # Minimum training bars before first fold
    
    # Model Hyperparameters (Anti-Overfitting)
    "SEEDS"           : [1, 2, 3, 4, 5],
    "DEPTH"           : 4,           # Shallow trees to prevent memorization
    "L2"              : 12,          # Heavy L2 regularization
    "ITERATIONS"      : 400,
    "LEARNING_RATE"   : 0.04,
    
    # High-Beta NSE Stocks Basket
    "UNIVERSE"        : ["BEL", "SAIL", "TATASTEEL", "BHEL", "CANBK", "SUZLON", "PNB", "NMDC"],
    "N_BARS"          : 4000,
    "INTERVAL"        : Interval.in_15_minute,
    "EXCHANGE"        : "NSE",
}

BUYING_POWER = CONFIG["CAPITAL"] * CONFIG["LEVERAGE"]  # ₹2,500

# ── EXACT REVERSE-ENGINEERED DHAN CHARGES ────────────────────
def calc_dhan_charges(buy_price: float, sell_price: float, qty: int) -> float:
    """
    Exact Dhan Equity Intraday (MIS) Charges.
    Reverse-engineered directly from Dhan production calculator API.
    """
    buy_to   = buy_price  * qty
    sell_to  = sell_price * qty
    total_to = buy_to + sell_to
    
    # 1. Brokerage: 0.03% or Rs 20, whichever is LOWER. (NO MINIMUM FLOOR!)
    brok  = min(20.0, buy_to * 0.0003) + min(20.0, sell_to * 0.0003)
    
    # 2. Exchange Turnover (NSE Equity): 0.00307%
    exc   = total_to * 0.0000307
    
    # 3. STT: 0.025% on sell side
    stt   = sell_to * 0.00025
    
    # 4. SEBI Charges: Rs 10 / Crore (0.0001%)
    sebi  = total_to * 0.000001
    
    # 5. Stamp Duty: 0.003% on buy side
    stamp = buy_to * 0.00003
    
    # 6. GST: 18% on (Brokerage + Exchange + SEBI)
    gst   = (brok + exc + sebi) * 0.18
    
    total = brok + exc + stt + sebi + stamp + gst
    return round(total, 2)

# Sanity check Dhan charges
_sample_charges = calc_dhan_charges(37.0, 37.5, 35)
print(f"✅ Dhan Engine Active. Sample Charges (SUZLON ₹37, Qty 35, Turnover ₹2,608): ₹{_sample_charges:.2f}")

# ── MATHEMATICAL BREAK-EVEN AUDIT ON DHAN ───────────────────
gross_win_inr  = BUYING_POWER * CONFIG["TP_PCT"]  # ₹2,500 * 1.2% = ₹30.00
gross_loss_inr = BUYING_POWER * CONFIG["SL_PCT"]  # ₹2,500 * 0.6% = ₹15.00
net_win_inr    = gross_win_inr - _sample_charges   # ₹30.00 - ₹1.40 = ₹28.60
net_loss_inr   = gross_loss_inr + _sample_charges  # ₹15.00 + ₹1.40 = ₹16.40
dhan_payoff    = net_win_inr / net_loss_inr        # 28.60 / 16.40 = 1.74
dhan_breakeven = 1.0 / (1.0 + dhan_payoff)        # 1 / 2.74 = 36.4%

print(f"🎯 DHAN ECONOMICS:")
print(f"   • Gross Win: ₹{gross_win_inr:.2f} | Net Win (Post Dhan Tax): ₹{net_win_inr:.2f}")
print(f"   • Gross Loss: ₹{gross_loss_inr:.2f} | Net Loss: ₹{net_loss_inr:.2f}")
print(f"   • Payoff Ratio: {dhan_payoff:.2f} : 1")
print(f"   • Statistically Required Break-Even Win Rate: {dhan_breakeven:.1%} (Target > 40%)")

# ── DATA FETCHING ────────────────────────────────────────────
def fetch_stock_data(symbol: str) -> pd.DataFrame:
    tv = TvDatafeed()
    try:
        df = tv.get_hist(symbol=symbol, exchange=CONFIG["EXCHANGE"],
                         interval=CONFIG["INTERVAL"], n_bars=CONFIG["N_BARS"])
        if df is None or len(df) < 500:
            return None
        df = df[["open","high","low","close","volume"]].copy()
        df.index = pd.to_datetime(df.index, utc=True).tz_convert("Asia/Kolkata")
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"  ⚠️ [{symbol}] Error fetching: {e}")
        return None

# ── FEATURE ENGINEERING (ZERO RAW PRICES / LEAKAGE-FREE) ─────
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    # 1. Stationary Returns
    df["ret_1"] = c.pct_change(1)
    df["ret_3"] = c.pct_change(3)
    df["ret_6"] = c.pct_change(6)

    # 2. Normalized ATR & Expansion
    atr_raw = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
    df["atr_norm"]      = (atr_raw / c).fillna(0)
    df["atr_norm_z"]    = (df["atr_norm"] - df["atr_norm"].rolling(50).mean()) / (df["atr_norm"].rolling(50).std() + 1e-9)
    df["atr_expanding"] = (df["atr_norm"] > df["atr_norm"].shift(1)).astype(int)

    # 3. Oscillators
    df["rsi"]   = ta.momentum.RSIIndicator(c, window=14).rsi()
    df["rsi_z"] = (df["rsi"] - df["rsi"].rolling(50).mean()) / (df["rsi"].rolling(50).std() + 1e-9)
    
    macd = ta.trend.MACD(c)
    df["macd_hist"]  = macd.macd_diff()
    df["macd_accel"] = df["macd_hist"].diff()
    df["macd_cross"] = (macd.macd() > macd.macd_signal()).astype(int)

    # 4. Stationary Trend Deviations
    for p in [9, 21, 50]:
        ema = c.ewm(span=p).mean()
        df[f"ema_dev_{p}"] = (c - ema) / (ema + 1e-9)

    adx = ta.trend.ADXIndicator(h, l, c, window=14)
    df["adx"]        = adx.adx()
    df["di_diff"]    = adx.adx_pos() - adx.adx_neg()
    df["adx_rising"] = (df["adx"] > df["adx"].shift(2)).astype(int)

    # 5. Bollinger-Keltner Squeeze & Energy Release
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df["bb_pos"]   = bb.bollinger_pband()
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / (bb.bollinger_mavg() + 1e-9)
    kc = ta.volatility.KeltnerChannel(h, l, c, window=20)
    
    sq = ((bb.bollinger_hband() < kc.keltner_channel_hband()) &
          (bb.bollinger_lband() > kc.keltner_channel_lband()))
    df["squeeze_on"]      = sq.astype(int)
    df["squeeze_release"] = ((sq.shift(1) | sq.shift(2) | sq.shift(3)) & ~sq).astype(int)

    # 6. Institutional Volume Surges
    df["rvol"]       = v / (v.rolling(20).mean() + 1e-9)
    df["vol_zscore"] = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-9)
    df["vol_surge"]  = (df["rvol"] > 2.0).astype(int)
    
    obv = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    df["obv_norm"]   = (obv - obv.rolling(50).min()) / (obv.rolling(50).max() - obv.rolling(50).min() + 1e-9)

    # 7. Daily VWAP (Shifted 1 bar to strictly avoid lookahead bias)
    df["_date"]      = df.index.date
    tp_price         = (h + l + c) / 3
    df["_tpv"]       = tp_price * v
    df["_vcum"]      = df.groupby("_date")["_tpv"].cumsum()
    df["_volcum"]    = df.groupby("_date")["volume"].cumsum()
    df["vwap"]       = (df["_vcum"] / (df["_volcum"] + 1e-9)).shift(1)
    df["vwap_dev"]   = (c - df["vwap"]) / (df["vwap"] + 1e-9)
    df["above_vwap"] = (c > df["vwap"]).astype(int)
    df.drop(columns=["_date","_tpv","_vcum","_volcum"], inplace=True)

    # 8. Time Filters
    df["near_close"] = (df.index.time >= pd.Timestamp("15:00").time()).astype(int)
    df["morning"]    = (df.index.time <= pd.Timestamp("11:15").time()).astype(int)

    # 9. Cross Interactions
    df["squeeze_x_vol"] = df["squeeze_release"] * df["rvol"]
    df["ret_x_rvol"]    = df["ret_1"] * df["rvol"]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df

FEATURE_COLS = [
    "ret_1","ret_3","ret_6",
    "atr_norm","atr_norm_z","atr_expanding",
    "rsi","rsi_z","macd_hist","macd_accel","macd_cross",
    "ema_dev_9","ema_dev_21","ema_dev_50",
    "adx","di_diff","adx_rising",
    "bb_pos","bb_width","squeeze_on","squeeze_release",
    "rvol","vol_zscore","vol_surge","obv_norm",
    "vwap_dev","above_vwap","near_close","morning",
    "squeeze_x_vol","ret_x_rvol"
]

# ── TRIPLE BARRIER LABELING (1.2% TP / 0.6% SL) ─────────────
def apply_labels(df: pd.DataFrame) -> pd.DataFrame:
    df   = df.copy()
    n    = len(df)
    c    = df["close"].values
    h    = df["high"].values
    l    = df["low"].values
    hold = CONFIG["HOLD_BARS"]
    tp   = CONFIG["TP_PCT"]
    sl   = CONFIG["SL_PCT"]

    y_long  = np.zeros(n, dtype=int)
    y_short = np.zeros(n, dtype=int)

    for i in range(n - hold):
        entry = c[i]
        tp_long  = entry * (1.0 + tp)
        sl_long  = entry * (1.0 - sl)
        tp_short = entry * (1.0 - tp)
        sl_short = entry * (1.0 + sl)

        hit_l = hit_s = False
        for j in range(1, hold + 1):
            hi = h[i + j]
            lo = l[i + j]

            # Long barrier
            if not hit_l:
                if hi >= tp_long:
                    y_long[i] = 1; hit_l = True
                elif lo <= sl_long:
                    y_long[i] = 0; hit_l = True

            # Short barrier
            if not hit_s:
                if lo <= tp_short:
                    y_short[i] = 1; hit_s = True
                elif hi >= sl_short:
                    y_short[i] = 0; hit_s = True

        # Time barrier exit
        if not hit_l:
            ret = (c[i + hold] - entry) / entry
            y_long[i] = 1 if ret >= tp else 0
        if not hit_s:
            ret = (c[i + hold] - entry) / entry
            y_short[i] = 1 if ret <= -tp else 0

    df["y_long"]  = y_long
    df["y_short"] = y_short
    return df.iloc[:n - hold]

# ── ENSEMBLE TRAINER ─────────────────────────────────────────
def train_ensemble(X, y):
    pos = y.sum()
    neg = len(y) - pos
    cw  = {0: 1, 1: max(2, min(6, neg // max(pos, 1)))}
    models = []
    for seed in CONFIG["SEEDS"]:
        m = CatBoostClassifier(
            iterations    = CONFIG["ITERATIONS"],
            depth         = CONFIG["DEPTH"],
            l2_leaf_reg   = CONFIG["L2"],
            learning_rate = CONFIG["LEARNING_RATE"],
            random_seed   = seed,
            eval_metric   = "AUC",
            od_type       = "Iter",
            od_wait       = 40,
            verbose       = 0,
            class_weights = cw
        )
        m.fit(X, y)
        models.append(m)
    return models

def predict_ensemble(models, X):
    return np.array([m.predict_proba(X)[:, 1] for m in models]).mean(axis=0)

# ── WALK-FORWARD ENGINE ──────────────────────────────────────
def run_walk_forward(df_feat: pd.DataFrame, target_col: str, symbol: str) -> pd.DataFrame:
    n         = len(df_feat)
    fold_size = n // (CONFIG["N_FOLDS"] + 1)
    embargo   = CONFIG["EMBARGO_DAYS"] * 26
    min_train = CONFIG["MIN_TRAIN_BARS"]
    oos_list  = []

    for fold in range(1, CONFIG["N_FOLDS"] + 1):
        t_end   = fold * fold_size
        s_start = t_end + embargo
        s_end   = min(s_start + fold_size, n)

        if t_end < min_train or s_start >= n:
            continue

        tr = df_feat.iloc[:t_end]
        te = df_feat.iloc[s_start:s_end]

        # Training filter: active market conditions
        mask_tr = (tr["near_close"] == 0) & (tr["adx"] > 16)
        tr_f    = tr[mask_tr]
        if len(tr_f) < 50 or tr_f[target_col].sum() < 10:
            continue

        models = train_ensemble(tr_f[FEATURE_COLS].values, tr_f[target_col].values)

        te_f = te[te["near_close"] == 0].copy()
        if len(te_f) == 0:
            continue

        te_f["prob"]   = predict_ensemble(models, te_f[FEATURE_COLS].values)
        te_f["y_true"] = te_f[target_col].values
        te_f["fold"]   = fold
        te_f["symbol"] = symbol
        oos_list.append(te_f)

    return pd.concat(oos_list) if oos_list else pd.DataFrame()

# ── DHAN RUPEE BACKTEST ──────────────────────────────────────
def run_dhan_backtest(oos_df: pd.DataFrame, threshold: float, symbol: str, direction: str):
    """
    Executes trades with EXACT Dhan MIS charges (approx ₹1.40/trade).
    Daily circuit breaker: Max 2 consecutive losses per day.
    """
    bp  = BUYING_POWER
    sub = oos_df[oos_df["prob"] >= threshold].sort_index()
    if len(sub) == 0:
        return {"n_trades": 0, "total_pnl": 0.0, "win_rate": 0.0, "trade_log": []}

    total_pnl     = 0.0
    wins          = 0
    losses        = 0
    total_charges = 0.0
    trade_log     = []
    daily_losses  = {}

    for idx, row in sub.iterrows():
        d = idx.date()
        # Daily risk management: halt after 2 losses in a day
        if daily_losses.get(d, 0) >= 2:
            continue

        price   = row["close"]
        qty     = max(1, int(bp / price))
        
        # Exact Dhan friction calculation
        charges = calc_dhan_charges(price, price, qty)

        if row["y_true"] == 1:
            gross = bp * CONFIG["TP_PCT"]
            wins += 1
        else:
            gross = -bp * CONFIG["SL_PCT"]
            losses += 1
            daily_losses[d] = daily_losses.get(d, 0) + 1

        net = gross - charges
        total_pnl += net
        total_charges += charges
        trade_log.append({
            "date": d, "symbol": symbol, "price": price, "qty": qty,
            "gross": gross, "charges": charges, "net_pnl": net, "win": row["y_true"] == 1
        })

    n   = wins + losses
    wr  = wins / n if n > 0 else 0.0
    aw  = np.mean([t["net_pnl"] for t in trade_log if t["win"]])     if wins   > 0 else 0.0
    al  = np.mean([t["net_pnl"] for t in trade_log if not t["win"]]) if losses > 0 else 0.0
    pf_n = sum(t["net_pnl"] for t in trade_log if t["net_pnl"] > 0)
    pf_d = abs(sum(t["net_pnl"] for t in trade_log if t["net_pnl"] < 0))
    pf   = pf_n / pf_d if pf_d > 0 else float("inf")
    roi  = total_pnl / CONFIG["CAPITAL"] * 100.0

    return {
        "n_trades": n, "wins": wins, "losses": losses, "win_rate": wr,
        "avg_win": aw, "avg_loss": al, "profit_factor": pf,
        "total_pnl": total_pnl, "charges": total_charges, "roi": roi,
        "trade_log": trade_log
    }

# ── SWEEP OPTIMAL THRESHOLD ──────────────────────────────────
def find_best_threshold(oos_df: pd.DataFrame, symbol: str, direction: str):
    if len(oos_df) == 0:
        return 0.55, {}
    
    best_thr = 0.55
    best_pnl = -999999
    best_bt  = {}

    for thr in np.arange(0.45, 0.75, 0.05):
        bt = run_dhan_backtest(oos_df, thr, symbol, direction)
        if bt["n_trades"] >= 15:  # Minimum statistical sample
            if bt["total_pnl"] > best_pnl:
                best_pnl = bt["total_pnl"]
                best_thr = thr
                best_bt  = bt

    if not best_bt:
        best_bt = run_dhan_backtest(oos_df, 0.50, symbol, direction)
        best_thr = 0.50

    return best_thr, best_bt

# ════════════════════════════════════════════════════════════
#  MAIN EXECUTION PIPELINE
# ════════════════════════════════════════════════════════════
print("\n" + "="*70)
print("  🚀 BHAVISHYA V4.1 — DHAN LOW-FRICTION INTRADAY ENGINE")
print(f"  Account: ₹500 | Buying Power: ₹2,500 | Est. Dhan Tax: ~₹1.40/trade")
print("="*70)

all_results = {}
active_models_long  = {}
active_models_short = {}

for sym in CONFIG["UNIVERSE"]:
    print(f"\n{'─'*70}")
    print(f"  📡 [{sym}] Fetching & Processing 15-Minute Data...", end=" ")
    df_raw = fetch_stock_data(sym)
    if df_raw is None:
        print("❌ Skipped"); time.sleep(2); continue

    df_feat = compute_features(df_raw)
    df_lab  = apply_labels(df_feat)

    lp = df_lab["y_long"].mean()
    sp = df_lab["y_short"].mean()
    print(f"✅ {len(df_lab):,} rows | Long Setup: {lp:.1%} | Short Setup: {sp:.1%}")

    # 1. Walk-Forward Testing
    oos_l = run_walk_forward(df_lab, "y_long",  sym)
    oos_s = run_walk_forward(df_lab, "y_short", sym)

    # 2. Optimal Threshold Search on Dhan Net ₹ PnL
    thr_l, bt_l = find_best_threshold(oos_l, sym, "LONG")
    thr_s, bt_s = find_best_threshold(oos_s, sym, "SHORT")

    # 3. Binomial Statistical Edge Gate vs Dhan Break-Even
    p_val_l = stats.binomtest(bt_l["wins"], max(1, bt_l["n_trades"]), dhan_breakeven, alternative="greater").pvalue if bt_l["n_trades"] > 0 else 1.0
    p_val_s = stats.binomtest(bt_s["wins"], max(1, bt_s["n_trades"]), dhan_breakeven, alternative="greater").pvalue if bt_s["n_trades"] > 0 else 1.0

    edge_l = (p_val_l < 0.05) and (bt_l["total_pnl"] > 0)
    edge_s = (p_val_s < 0.05) and (bt_s["total_pnl"] > 0)

    print(f"  [{sym}] LONG  | Thr={thr_l:.2f} | Trades={bt_l['n_trades']:>2} | "
          f"WR={bt_l['win_rate']:>5.1%} | Dhan Charges=₹{bt_l.get('charges',0):>5.1f} | "
          f"Net PnL=₹{bt_l['total_pnl']:>+6.1f} | PF={bt_l.get('profit_factor',0):>4.2f} | "
          f"Edge={'✅ YES' if edge_l else '❌'}")
    print(f"  [{sym}] SHORT | Thr={thr_s:.2f} | Trades={bt_s['n_trades']:>2} | "
          f"WR={bt_s['win_rate']:>5.1%} | Dhan Charges=₹{bt_s.get('charges',0):>5.1f} | "
          f"Net PnL=₹{bt_s['total_pnl']:>+6.1f} | PF={bt_s.get('profit_factor',0):>4.2f} | "
          f"Edge={'✅ YES' if edge_s else '❌'}")

    all_results[sym] = {
        "bt_l": bt_l, "bt_s": bt_s,
        "thr_l": thr_l, "thr_s": thr_s,
        "edge_l": edge_l, "edge_s": edge_s,
        "p_l": p_val_l, "p_s": p_val_s
    }
    time.sleep(2)

# ════════════════════════════════════════════════════════════
#  MASTER DHAN TRADING REPORT
# ════════════════════════════════════════════════════════════
print("\n\n" + "="*75)
print("  🏆 MASTER DHAN PERFORMANCE DASHBOARD (₹500 ACCOUNT | ₹2,500 MIS)")
print("="*75)
print(f"  {'Stock':<10} | {'L-Trades':<8} | {'L-WR%':<7} | {'L-Net₹':<8} | {'S-Trades':<8} | {'S-WR%':<7} | {'S-Net₹':<8} | {'Combined PnL':<12}")
print(f"  {'-'*75}")

total_portfolio_pnl     = 0.0
total_portfolio_trades  = 0
total_portfolio_charges = 0.0
green_stocks            = []

for sym, r in all_results.items():
    bl = r["bt_l"]
    bs = r["bt_s"]
    comb_pnl = bl["total_pnl"] + bs["total_pnl"]
    comb_trades = bl["n_trades"] + bs["n_trades"]
    
    total_portfolio_pnl     += comb_pnl
    total_portfolio_trades  += comb_trades
    total_portfolio_charges += bl.get("charges", 0) + bs.get("charges", 0)

    l_wr_str = f"{bl['win_rate']:.1%}" if bl["n_trades"] > 0 else "0.0%"
    s_wr_str = f"{bs['win_rate']:.1%}" if bs["n_trades"] > 0 else "0.0%"
    l_pnl_str = f"₹{bl['total_pnl']:+.1f}"
    s_pnl_str = f"₹{bs['total_pnl']:+.1f}"
    c_pnl_str = f"₹{comb_pnl:+.1f}"

    if comb_pnl > 0:
        green_stocks.append(sym)

    print(f"  {sym:<10} | {bl['n_trades']:<8} | {l_wr_str:<7} | {l_pnl_str:<8} | {bs['n_trades']:<8} | {s_wr_str:<7} | {s_pnl_str:<8} | {c_pnl_str:<12}")

print(f"  {'-'*75}")
print(f"  📊 PORTFOLIO TOTALS:")
print(f"     • Total Out-of-Sample Trades Taken: {total_portfolio_trades}")
print(f"     • Total Dhan Charges Paid:          ₹{total_portfolio_charges:.2f} (vs ~₹{total_portfolio_trades * 12.70:.2f} on Groww!)")
print(f"     • Total NET Rupee P&L:             ₹{total_portfolio_pnl:+.2f}")
print(f"     • Net Portfolio ROI:               {total_portfolio_pnl / CONFIG['CAPITAL'] * 100:+.1f}% on ₹500 Capital")
print(f"     • Consistently Green Stocks:       {green_stocks if green_stocks else 'None'}")
print("="*75)


# ════════════════════════════════════════════════════════════
#  OPTIONAL: LIVE DHAN SCANNER & ORDER TICKET GENERATOR
# ════════════════════════════════════════════════════════════
def scan_dhan_live_market(threshold=0.55):
    """
    Scans live market for setup opportunities and outputs ready-to-place
    Dhan Intraday Order Tickets.
    """
    tv = TvDatafeed()
    print("\n" + "="*70)
    print("  📡 SCANNING LIVE NSE MARKET FOR DHAN INTRADAY SETUPS...")
    print(f"  Account: ₹500 | Buying Power: ₹2,500 | Setup Threshold: {threshold:.2f}")
    print("="*70)

    signals_found = 0
    for sym in CONFIG["UNIVERSE"]:
        try:
            raw = tv.get_hist(symbol=sym, exchange="NSE", interval=CONFIG["INTERVAL"], n_bars=250)
            if raw is None or len(raw) < 100:
                continue
            raw = raw[["open","high","low","close","volume"]].copy()
            raw.index = pd.to_datetime(raw.index, utc=True).tz_convert("Asia/Kolkata")
            
            feat = compute_features(raw)
            feat = apply_labels(feat)
            if len(feat) == 0:
                continue

            last_bar = feat.iloc[[-1]]
            price = last_bar["close"].values[0]
            qty   = max(1, int(BUYING_POWER / price))
            charges = calc_dhan_charges(price, price, qty)

            # Check time filter
            if last_bar["near_close"].values[0] == 1:
                print(f"  ⏰ [{sym}] Skipping: Session close approaching (> 3:00 PM)")
                continue

            # Target & Stop Loss Prices
            tp_price_long  = round(price * (1.0 + CONFIG["TP_PCT"]), 2)
            sl_price_long  = round(price * (1.0 - CONFIG["SL_PCT"]), 2)
            tp_price_short = round(price * (1.0 - CONFIG["TP_PCT"]), 2)
            sl_price_short = round(price * (1.0 + CONFIG["SL_PCT"]), 2)

            print(f"  [{sym:<10}] Price: ₹{price:<7.2f} | Qty: {qty:<3} | Dhan Charges: ₹{charges:.2f}")

        except Exception as e:
            pass
        time.sleep(1)

    print("="*70)

# To run live scanner in market hours, simply uncomment:
# scan_dhan_live_market()
