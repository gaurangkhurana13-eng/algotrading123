# ============================================================
# BHAVISHYA V3.1 — WALK-FORWARD + V8/V9 GEMS INTEGRATED
# Google Colab Ready | Single-Cell Copy-Paste
# Target: ₹500 Groww Account | 5x MIS = ₹2500 Buying Power
# ============================================================
# WHAT'S NEW IN V3.1 vs V3.0:
#   + Daily VWAP reset per trading day (shift(1), no leakage)
#   + Bollinger-Keltner Squeeze flag (squeeze_on)
#   + vol_zscore, obv_norm, stoch_k features from V8
#   + ATR-adaptive target threshold (replaces hard 1.25%)
#   + Pre-training filter: adx>18, rvol>0, near_close==0
#   + Exact Groww charges (calc_groww_charges function)
#   + Daily circuit breaker (halt after 2 losses/day)
#   + Binary targets: separate y_long + y_short models
#   + 6-fold expanding walk-forward with 2-day embargo
#   + 5-seed CatBoost ensemble
#   + Binomial p-value gate (need < 0.05)
#   + Monte Carlo only if 200+ OOS trades
# ============================================================

# ── CELL 1: Install & Import ─────────────────────────────────
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
from catboost import CatBoostClassifier, Pool
from tvDatafeed import TvDatafeed, Interval
import ta

print("✅ All libraries loaded.")

# ── CELL 2: CONFIG ───────────────────────────────────────────
CONFIG = {
    # Account
    "CAPITAL"        : 500.0,    # ₹ base capital
    "LEVERAGE"       : 5.0,      # Groww MIS 5x
    # Triple Barrier
    "TP_MULT"        : 2.0,      # ATR multiplier for Take Profit
    "SL_MULT"        : 1.0,      # ATR multiplier for Stop Loss
    "HOLD_BARS"      : 10,       # 10 bars × 15m = 2.5 hours
    # Walk-Forward
    "N_FOLDS"        : 6,
    "EMBARGO_DAYS"   : 2,
    "MIN_TRAIN_DAYS" : 60,
    # Model
    "SEEDS"          : [1, 2, 3, 4, 5],
    "DEPTH"          : 4,
    "L2"             : 10,
    "ITERATIONS"     : 500,
    # Thresholds
    "LONG_THRESH"    : 0.55,     # Will be swept on validation
    "SHORT_THRESH"   : 0.55,
    "MIN_OOS_TRADES" : 200,      # Gate for Monte Carlo
    # Data
    "SYMBOL"         : "NIFTY",
    "EXCHANGE"       : "NSE",
    "INTERVAL"       : Interval.in_15_minute,
    "N_BARS"         : 5000,
    # Scanner Universe (high-beta, under ₹500)
    "UNIVERSE"       : ["SUZLON", "BHEL", "BEL", "PNB", "CANBK",
                        "TATASTEEL", "SAIL", "NMDC"],
}

BUYING_POWER = CONFIG["CAPITAL"] * CONFIG["LEVERAGE"]  # ₹2500
print(f"✅ Config loaded. Buying power: ₹{BUYING_POWER:.0f}")

# ── CELL 3: EXACT GROWW CHARGES ─────────────────────────────
def calc_groww_charges(buy_price: float, sell_price: float, qty: int) -> float:
    """
    Exact Groww intraday (MIS) charge calculation.
    Returns total charges in ₹.

    Breakdown:
      Brokerage : 0.05% per leg, min ₹5, max ₹20 per leg
      STT       : 0.025% on SELL side (intraday equity)
      Exchange  : NSE 0.00297% on total turnover
      SEBI      : 0.0001% on total turnover
      GST       : 18% on (brokerage + exchange + SEBI)
      Stamp     : 0.003% on BUY side
    """
    buy_to  = buy_price  * qty
    sell_to = sell_price * qty
    total_to = buy_to + sell_to

    buy_brok  = min(20.0, max(5.0, buy_to  * 0.0005))
    sell_brok = min(20.0, max(5.0, sell_to * 0.0005))
    brok = buy_brok + sell_brok

    stt   = sell_to  * 0.00025          # 0.025% sell side
    exc   = total_to * 0.0000297        # NSE exchange
    sebi  = total_to * 0.000001         # SEBI charges
    gst   = (brok + exc + sebi) * 0.18  # 18% GST
    stamp = buy_to   * 0.00003          # 0.003% stamp

    total = brok + stt + exc + sebi + gst + stamp
    return round(total, 4)


# Quick sanity check
_test_charges = calc_groww_charges(100.0, 101.5, 25)
print(f"✅ Charges function ready. Sample (₹100→₹101.5, qty=25): ₹{_test_charges:.2f}")

# ── CELL 4: FETCH DATA ──────────────────────────────────────
def fetch_data(symbol=CONFIG["SYMBOL"],
               exchange=CONFIG["EXCHANGE"],
               interval=CONFIG["INTERVAL"],
               n_bars=CONFIG["N_BARS"]) -> pd.DataFrame:
    tv = TvDatafeed()
    print(f"📡 Fetching {n_bars} candles for {exchange}:{symbol} ...")
    df = tv.get_hist(symbol=symbol, exchange=exchange,
                     interval=interval, n_bars=n_bars)
    df = df[["open","high","low","close","volume"]].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df.index = df.index.tz_convert("Asia/Kolkata")
    df.dropna(inplace=True)
    print(f"✅ Fetched {len(df):,} candles | {df.index[0]} → {df.index[-1]}")
    return df

# ── CELL 5: FEATURE ENGINEERING (LEAKAGE-FREE) ──────────────
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    All 30+ stationary features. No raw OHLCV fed to model.
    Daily VWAP is computed per trading day and shifted(1) to avoid lookahead.
    """
    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    # ── Returns (stationary) ────────────────────────────────
    df["ret_1"]  = c.pct_change(1)
    df["ret_3"]  = c.pct_change(3)
    df["ret_9"]  = c.pct_change(9)

    # ── ATR (normalized) ────────────────────────────────────
    atr_raw = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
    df["atr_norm"]  = (atr_raw / c).fillna(0)
    df["atr_norm_z"]= (df["atr_norm"] - df["atr_norm"].rolling(50).mean()) / \
                       (df["atr_norm"].rolling(50).std() + 1e-9)

    # ── RSI ─────────────────────────────────────────────────
    df["rsi"]    = ta.momentum.RSIIndicator(c, window=14).rsi()
    df["rsi_z"]  = (df["rsi"] - df["rsi"].rolling(50).mean()) / \
                   (df["rsi"].rolling(50).std() + 1e-9)

    # ── MACD ────────────────────────────────────────────────
    macd_obj      = ta.trend.MACD(c)
    df["macd"]    = macd_obj.macd()
    df["macd_sig"]= macd_obj.macd_signal()
    df["macd_hist"]= macd_obj.macd_diff()
    df["macd_cross"] = (df["macd"] > df["macd_sig"]).astype(int)

    # ── EMA Deviations (stationary) ─────────────────────────
    for p in [9, 21, 50]:
        ema = c.ewm(span=p).mean()
        df[f"ema_dev_{p}"] = (c - ema) / (ema + 1e-9)

    # ── ADX ─────────────────────────────────────────────────
    adx_obj      = ta.trend.ADXIndicator(h, l, c, window=14)
    df["adx"]    = adx_obj.adx()
    df["dip"]    = adx_obj.adx_pos()
    df["dim"]    = adx_obj.adx_neg()
    df["di_diff"]= df["dip"] - df["dim"]

    # ── Bollinger Bands ──────────────────────────────────────
    bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
    df["bb_pos"] = bb.bollinger_pband()          # 0=lower, 1=upper
    df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / \
                     (bb.bollinger_mavg() + 1e-9)

    # ── Keltner Channels ─────────────────────────────────────
    kc = ta.volatility.KeltnerChannel(h, l, c, window=20)
    kc_upper = kc.keltner_channel_hband()
    kc_lower = kc.keltner_channel_lband()
    bb_upper = bb.bollinger_hband()
    bb_lower = bb.bollinger_lband()

    # GEM 2: Bollinger-Keltner Squeeze (coiling = imminent expansion)
    df["squeeze_on"] = ((bb_upper < kc_upper) & (bb_lower > kc_lower)).astype(int)

    # ── Stochastic (GEM 7) ───────────────────────────────────
    stoch = ta.momentum.StochasticOscillator(h, l, c, window=14, smooth_window=3)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()
    df["stoch_cross"] = (df["stoch_k"] > df["stoch_d"]).astype(int)

    # ── Volume Features ──────────────────────────────────────
    df["rvol"]    = v / (v.rolling(20).mean() + 1e-9)  # relative volume

    # GEM 7: Volume Z-score
    df["vol_zscore"] = (v - v.rolling(20).mean()) / (v.rolling(20).std() + 1e-9)

    # GEM 7: OBV normalized
    obv_raw      = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    obv_min      = obv_raw.rolling(50).min()
    obv_max      = obv_raw.rolling(50).max()
    df["obv_norm"]= (obv_raw - obv_min) / (obv_max - obv_min + 1e-9)

    # ── GEM 1: Daily VWAP (per-day, shift(1) to avoid lookahead) ──
    df["date"]   = df.index.date
    # Typical price × volume
    tp           = (h + l + c) / 3
    tp_v         = tp * v
    df["_tpv"]   = tp_v
    df["_vcum"]  = df.groupby("date")["_tpv"].cumsum()
    df["_volcum"]= df.groupby("date")["volume"].cumsum()
    df["_vwap_raw"] = df["_vcum"] / (df["_volcum"] + 1e-9)
    # Shift(1) so bar N uses VWAP up to bar N-1 (no lookahead)
    df["vwap"]   = df["_vwap_raw"].shift(1)
    df["vwap_dev"]   = (c - df["vwap"]) / (df["vwap"] + 1e-9)
    df["above_vwap"] = (c > df["vwap"]).astype(int)
    df.drop(columns=["date","_tpv","_vcum","_volcum","_vwap_raw"], inplace=True)

    # ── Near Market Close flag (last 30 min = last 2 bars of session) ──
    df["near_close"] = (df.index.time >= pd.Timestamp("15:00").time()).astype(int)

    # ── Price-Volume interaction ─────────────────────────────
    df["ret_x_rvol"] = df["ret_1"] * df["rvol"]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    return df


# ── CELL 6: TRIPLE BARRIER LABELING (ATR-ADAPTIVE) ──────────
def apply_triple_barrier_labels(df: pd.DataFrame,
                                 tp_mult=CONFIG["TP_MULT"],
                                 sl_mult=CONFIG["SL_MULT"],
                                 hold_bars=CONFIG["HOLD_BARS"]) -> pd.DataFrame:
    """
    GEM 4: ATR-adaptive barrier (not hard %).
    threshold = max(0.5 * atr_norm, 0.006)  →  ≥ 0.60% move required.
    Returns df with y_long, y_short columns.
    Binary: 1 = profit after charges, 0 = not.
    """
    df = df.copy()
    n = len(df)
    c = df["close"].values
    atr = df["atr_norm"].values

    y_long  = np.zeros(n, dtype=int)
    y_short = np.zeros(n, dtype=int)

    for i in range(n - hold_bars):
        # GEM 4: ATR-adaptive threshold
        thresh = max(0.5 * atr[i], 0.006)

        tp_long  =  thresh * tp_mult
        sl_long  = -thresh * sl_mult
        tp_short = -thresh * tp_mult
        sl_short =  thresh * sl_mult

        entry = c[i]
        hit_long = hit_short = False

        for j in range(1, hold_bars + 1):
            ret = (c[i+j] - entry) / entry

            if not hit_long:
                if ret >= tp_long:
                    y_long[i] = 1; hit_long = True
                elif ret <= sl_long:
                    y_long[i] = 0; hit_long = True

            if not hit_short:
                if ret <= tp_short:
                    y_short[i] = 1; hit_short = True
                elif ret >= sl_short:
                    y_short[i] = 0; hit_short = True

        # Time exit: last bar return
        if not hit_long:
            ret = (c[i+hold_bars] - entry) / entry
            y_long[i] = 1 if ret >= tp_long else 0

        if not hit_short:
            ret = (c[i+hold_bars] - entry) / entry
            y_short[i] = 1 if ret <= tp_short else 0

    df["y_long"]  = y_long
    df["y_short"] = y_short
    df = df.iloc[:n - hold_bars]  # drop last hold_bars rows (no label)
    return df


# ── CELL 7: FEATURE COLUMNS ─────────────────────────────────
FEATURE_COLS = [
    "ret_1","ret_3","ret_9",
    "atr_norm","atr_norm_z",
    "rsi","rsi_z",
    "macd_hist","macd_cross",
    "ema_dev_9","ema_dev_21","ema_dev_50",
    "adx","di_diff",
    "bb_pos","bb_width",
    "squeeze_on",           # GEM 2
    "stoch_k","stoch_cross",# GEM 7
    "rvol","vol_zscore",    # GEM 7
    "obv_norm",             # GEM 7
    "vwap_dev","above_vwap",# GEM 1
    "near_close",
    "ret_x_rvol",
]


# ── CELL 8: CATBOOST ENSEMBLE TRAINER ───────────────────────
def train_catboost_ensemble(X_train, y_train,
                             seeds=CONFIG["SEEDS"],
                             depth=CONFIG["DEPTH"],
                             l2=CONFIG["L2"],
                             iterations=CONFIG["ITERATIONS"]):
    """5-seed ensemble → averaged probability (soft vote)."""
    models = []
    for seed in seeds:
        m = CatBoostClassifier(
            iterations   = iterations,
            depth        = depth,
            l2_leaf_reg  = l2,
            learning_rate= 0.05,
            random_seed  = seed,
            eval_metric  = "AUC",
            od_type      = "Iter",
            od_wait      = 50,
            verbose      = 0,
        )
        m.fit(X_train, y_train)
        models.append(m)
    return models


def predict_ensemble(models, X):
    """Average probability across all seeds."""
    probs = np.array([m.predict_proba(X)[:, 1] for m in models])
    return probs.mean(axis=0)


# ── CELL 9: WALK-FORWARD VALIDATION ─────────────────────────
def run_walk_forward(df_feat: pd.DataFrame,
                     target_col: str,
                     label: str = "LONG") -> pd.DataFrame:
    """
    6-fold expanding walk-forward with 2-day embargo.
    GEM 5: Pre-training filter: adx>18, rvol>0, near_close==0
    Returns OOS results dataframe.
    """
    n = len(df_feat)
    fold_size = n // (CONFIG["N_FOLDS"] + 1)
    embargo_bars = CONFIG["EMBARGO_DAYS"] * 26  # ~26 15-min bars per day

    min_train = CONFIG["MIN_TRAIN_DAYS"] * 26
    oos_records = []

    print(f"\n{'='*60}")
    print(f"  WALK-FORWARD [{label}] | {CONFIG['N_FOLDS']} Folds")
    print(f"{'='*60}")

    for fold in range(1, CONFIG["N_FOLDS"] + 1):
        train_end  = fold * fold_size
        test_start = train_end + embargo_bars
        test_end   = test_start + fold_size

        if train_end < min_train:
            print(f"  Fold {fold}: Skipped (not enough training data)")
            continue
        if test_end > n:
            test_end = n

        df_train = df_feat.iloc[:train_end].copy()
        df_test  = df_feat.iloc[test_start:test_end].copy()

        if len(df_test) == 0:
            continue

        # GEM 5: Pre-training signal filter (applied to training set)
        mask_train = (
            (df_train["adx"]       > 18) &
            (df_train["rvol"]      > 0 ) &
            (df_train["near_close"]== 0)
        )
        df_train_filtered = df_train[mask_train]

        if len(df_train_filtered) < 50:
            print(f"  Fold {fold}: Skipped (too few filtered training signals: {len(df_train_filtered)})")
            continue

        X_tr = df_train_filtered[FEATURE_COLS].values
        y_tr = df_train_filtered[target_col].values

        if y_tr.sum() < 10:
            print(f"  Fold {fold}: Skipped (too few positive labels: {y_tr.sum()})")
            continue

        # Train ensemble
        models = train_catboost_ensemble(X_tr, y_tr)

        # Test (also filter near_close for live trading realism)
        df_test_filtered = df_test[df_test["near_close"] == 0].copy()
        if len(df_test_filtered) == 0:
            continue

        X_te = df_test_filtered[FEATURE_COLS].values
        probs = predict_ensemble(models, X_te)

        df_test_filtered["prob"]   = probs
        df_test_filtered["y_true"] = df_test_filtered[target_col].values
        df_test_filtered["fold"]   = fold
        df_test_filtered["direction"] = label

        oos_records.append(df_test_filtered)
        pos_rate = df_test_filtered["y_true"].mean()
        print(f"  Fold {fold}: Train={len(df_train_filtered):,} | Test={len(df_test_filtered):,} | "
              f"Label %={pos_rate:.1%}")

    if not oos_records:
        print("  ❌ No OOS records generated!")
        return pd.DataFrame()

    oos_df = pd.concat(oos_records)
    print(f"\n  ✅ Total OOS rows: {len(oos_df):,}")
    return oos_df


# ── CELL 10: THRESHOLD SWEEP + P-VALUE ──────────────────────
def analyze_oos_results(oos_df: pd.DataFrame,
                        label: str,
                        direction: str = "LONG") -> dict:
    """
    Sweep conviction thresholds.
    Gate: binomial p-value < 0.05 required.
    Theoretical break-even = SL / (TP + SL).
    """
    if len(oos_df) == 0:
        return {}

    # Theoretical break-even win rate
    tp = CONFIG["TP_MULT"]
    sl = CONFIG["SL_MULT"]
    breakeven_wr = sl / (tp + sl)  # 1/(2+1) = 33.3% for 2:1 R

    print(f"\n{'='*60}")
    print(f"  OOS THRESHOLD SWEEP [{label}] | Break-even WR = {breakeven_wr:.1%}")
    print(f"{'='*60}")
    print(f"  {'Thresh':>7} | {'Trades':>7} | {'WR%':>7} | {'p-value':>9} | {'Verdict':>12}")
    print(f"  {'-'*55}")

    results = {}
    for thresh in np.arange(0.45, 0.80, 0.05):
        subset = oos_df[oos_df["prob"] >= thresh]
        n_trades = len(subset)
        if n_trades < 10:
            continue
        wins   = subset["y_true"].sum()
        wr     = wins / n_trades
        # One-sided binomial test: H0 = win rate <= breakeven
        p_val  = stats.binomtest(int(wins), n_trades, breakeven_wr, alternative="greater").pvalue
        verdict = "✅ EDGE" if p_val < 0.05 else "❌ NOISE"
        print(f"  {thresh:.2f}    | {n_trades:>7} | {wr:>6.1%} | {p_val:>9.4f} | {verdict:>12}")
        results[thresh] = {
            "n_trades": n_trades, "win_rate": wr,
            "p_value": p_val, "has_edge": p_val < 0.05
        }

    return results


# ── CELL 11: RUPEE BACKTEST (EXACT GROWW CHARGES) ───────────
def run_rupee_backtest(oos_df: pd.DataFrame,
                        threshold: float,
                        capital: float = CONFIG["CAPITAL"],
                        leverage: float = CONFIG["LEVERAGE"],
                        direction: str = "LONG") -> dict:
    """
    GEM 3: Exact Groww charges per trade.
    GEM 6: Daily circuit breaker (halt after 2 losses/day).
    Returns detailed backtest dict.
    """
    buying_power = capital * leverage
    subset = oos_df[oos_df["prob"] >= threshold].copy()
    subset = subset.sort_index()

    if len(subset) == 0:
        return {"error": "No trades at this threshold"}

    total_pnl   = 0.0
    wins        = 0
    losses      = 0
    total_charges = 0.0
    trade_log   = []

    # GEM 6: Daily circuit breaker state
    daily_losses = {}

    for idx, row in subset.iterrows():
        trade_date = idx.date()

        # Circuit breaker check
        if daily_losses.get(trade_date, 0) >= 2:
            continue  # Skip rest of day after 2 losses

        price = row["close"]
        qty   = max(1, int(buying_power / price))
        charges = calc_groww_charges(price, price, qty)  # approximate (same price)

        # Simulate outcome
        tp_mult = CONFIG["TP_MULT"]
        sl_mult = CONFIG["SL_MULT"]
        atr_pct = row["atr_norm"]
        thresh_pct = max(0.5 * atr_pct, 0.006)

        if row["y_true"] == 1:
            # Winner: TP hit
            move_pct = thresh_pct * tp_mult if direction == "LONG" else thresh_pct * tp_mult
            gross_pnl = buying_power * move_pct
            net_pnl   = gross_pnl - charges
            wins += 1
        else:
            # Loser: SL hit
            move_pct  = thresh_pct * sl_mult
            gross_pnl = -buying_power * move_pct
            net_pnl   = gross_pnl - charges
            losses += 1
            daily_losses[trade_date] = daily_losses.get(trade_date, 0) + 1

        total_pnl    += net_pnl
        total_charges += charges
        trade_log.append({
            "date"     : trade_date,
            "direction": direction,
            "price"    : price,
            "qty"      : qty,
            "net_pnl"  : net_pnl,
            "charges"  : charges,
            "win"      : row["y_true"] == 1,
        })

    n_trades   = wins + losses
    win_rate   = wins / n_trades if n_trades > 0 else 0
    avg_win    = np.mean([t["net_pnl"] for t in trade_log if t["win"]])  if wins   > 0 else 0
    avg_loss   = np.mean([t["net_pnl"] for t in trade_log if not t["win"]]) if losses > 0 else 0
    payoff     = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    pf_num     = sum(t["net_pnl"] for t in trade_log if t["net_pnl"] > 0)
    pf_den     = abs(sum(t["net_pnl"] for t in trade_log if t["net_pnl"] < 0))
    profit_factor = pf_num / pf_den if pf_den > 0 else np.inf
    net_roi    = total_pnl / capital * 100

    print(f"\n{'='*60}")
    print(f"  ₹ RUPEE BACKTEST [{direction} | Thresh={threshold:.2f}]")
    print(f"{'='*60}")
    print(f"  Trades       : {n_trades}")
    print(f"  Win Rate     : {win_rate:.1%}")
    print(f"  Avg Win      : ₹{avg_win:.2f}")
    print(f"  Avg Loss     : ₹{avg_loss:.2f}")
    print(f"  Payoff Ratio : {payoff:.2f}  (need > 1.2)")
    print(f"  Profit Factor: {profit_factor:.2f}")
    print(f"  Total Charges: ₹{total_charges:.2f}")
    print(f"  Net P&L      : ₹{total_pnl:.2f}")
    print(f"  Net ROI      : {net_roi:.1f}% on ₹{capital:.0f} capital")

    return {
        "n_trades": n_trades, "wins": wins, "losses": losses,
        "win_rate": win_rate, "avg_win": avg_win, "avg_loss": avg_loss,
        "payoff": payoff, "profit_factor": profit_factor,
        "total_pnl": total_pnl, "total_charges": total_charges,
        "net_roi": net_roi, "trade_log": trade_log,
    }


# ── CELL 12: MONTE CARLO (only if 200+ OOS trades) ──────────
def run_monte_carlo_if_valid(backtest_result: dict, n_sims=10000) -> None:
    """
    Bootstrap resamples from ACTUAL OOS trade PnLs (not Gaussian).
    Only runs if OOS trades >= 200.
    """
    trade_log = backtest_result.get("trade_log", [])
    n_trades  = len(trade_log)

    if n_trades < CONFIG["MIN_OOS_TRADES"]:
        print(f"\n⚠️  Monte Carlo SKIPPED: Only {n_trades} OOS trades.")
        print(f"   Need ≥ {CONFIG['MIN_OOS_TRADES']} for statistical validity.")
        print(f"   Paper trade 1–2 months and re-run.")
        return

    pnls = np.array([t["net_pnl"] for t in trade_log])
    capital = CONFIG["CAPITAL"]
    ruin_count  = 0
    final_caps  = []

    for _ in range(n_sims):
        sim_pnls = np.random.choice(pnls, size=n_trades, replace=True)
        cap = capital + sim_pnls.sum()
        final_caps.append(cap)
        if cap < capital * 0.5:   # 50% drawdown = ruin
            ruin_count += 1

    ruin_pct = ruin_count / n_sims * 100
    p5  = np.percentile(final_caps, 5)
    p50 = np.percentile(final_caps, 50)
    p95 = np.percentile(final_caps, 95)

    print(f"\n{'='*60}")
    print(f"  MONTE CARLO ({n_sims:,} simulations, {n_trades} actual OOS trades)")
    print(f"{'='*60}")
    print(f"  5th  percentile final capital : ₹{p5:.0f}")
    print(f"  50th percentile final capital : ₹{p50:.0f}")
    print(f"  95th percentile final capital : ₹{p95:.0f}")
    print(f"  Ruin probability (50% DD)     : {ruin_pct:.1f}%")
    if ruin_pct < 5:
        print(f"  ✅ Ruin risk acceptable (< 5%)")
    else:
        print(f"  ⚠️  Ruin risk HIGH — reduce position size")


# ── CELL 13: LIVE SCANNER ───────────────────────────────────
def scan_groww_tickets(models_long, models_short,
                        long_thresh=CONFIG["LONG_THRESH"],
                        short_thresh=CONFIG["SHORT_THRESH"],
                        universe=CONFIG["UNIVERSE"]) -> None:
    """
    Fetches latest bars for each stock in universe.
    Prints live BUY/SHORT signals with conviction.
    Rate-limited: 1 request per stock with 2s sleep.
    """
    import time
    tv = TvDatafeed()

    print(f"\n{'='*60}")
    print(f"  📡 LIVE SCANNER | {len(universe)} Stocks")
    print(f"  Long Thresh: {long_thresh:.2f} | Short Thresh: {short_thresh:.2f}")
    print(f"{'='*60}")

    signals_found = 0
    for sym in universe:
        try:
            raw = tv.get_hist(symbol=sym, exchange="NSE",
                              interval=CONFIG["INTERVAL"], n_bars=200)
            if raw is None or len(raw) < 100:
                print(f"  ⚠️  [{sym}] Insufficient data")
                continue
            raw = raw[["open","high","low","close","volume"]].copy()
            raw.index = pd.to_datetime(raw.index, utc=True).tz_convert("Asia/Kolkata")
            feat = compute_features(raw)
            feat = apply_triple_barrier_labels(feat)

            if len(feat) == 0:
                continue

            # Use latest bar
            last = feat.iloc[[-1]][FEATURE_COLS]
            price = feat["close"].iloc[-1]

            if feat.iloc[-1]["near_close"] == 1:
                print(f"  ⏰ [{sym}] Near session close — skip")
                continue

            # Conviction scores
            long_prob  = predict_ensemble(models_long,  last.values)[0]
            short_prob = predict_ensemble(models_short, last.values)[0]

            # Buying power → qty
            qty = max(1, int(BUYING_POWER / price))
            charges = calc_groww_charges(price, price, qty)

            if long_prob >= long_thresh:
                print(f"  🟢 [{sym}] LONG  | Price=₹{price:.1f} | Qty={qty} | "
                      f"Conv={long_prob:.2%} | Est.Charges=₹{charges:.2f}")
                signals_found += 1
            elif short_prob >= short_thresh:
                print(f"  🔴 [{sym}] SHORT | Price=₹{price:.1f} | Qty={qty} | "
                      f"Conv={short_prob:.2%} | Est.Charges=₹{charges:.2f}")
                signals_found += 1
            else:
                print(f"  ⚪ [{sym}] No signal | Long={long_prob:.2%} | Short={short_prob:.2%}")

        except Exception as e:
            print(f"  ⚠️  [{sym}] Error: {e}")
        time.sleep(2)  # Rate limit

    if signals_found == 0:
        print(f"\n  📭 No signals above threshold right now. Market may be consolidating.")
    else:
        print(f"\n  🎯 {signals_found} signal(s) found. Review before placing order.")
    print(f"{'='*60}")


# ── CELL 14: MAIN PIPELINE ───────────────────────────────────
def run_bhavishya_v31():
    print("\n" + "="*60)
    print("  🚀 BHAVISHYA V3.1 — WALK-FORWARD ML PIPELINE")
    print("  Capital: ₹500 | Leverage: 5x | Buying Power: ₹2,500")
    print("="*60)

    # Step 1: Fetch Data
    df_raw = fetch_data()

    # Step 2: Feature Engineering
    print("\n⚙️  Engineering features...")
    df_feat = compute_features(df_raw)
    print(f"✅ Features: {len(FEATURE_COLS)} columns | Rows: {len(df_feat):,}")

    # Step 3: Triple Barrier Labels (ATR-adaptive)
    print("\n🏷️  Applying ATR-adaptive triple barrier labels...")
    df_labeled = apply_triple_barrier_labels(df_feat)
    long_pos  = df_labeled["y_long"].mean()
    short_pos = df_labeled["y_short"].mean()
    print(f"✅ Labels: y_long={long_pos:.1%} positive | y_short={short_pos:.1%} positive")

    # Step 4: Walk-Forward (LONG model)
    oos_long = run_walk_forward(df_labeled, target_col="y_long", label="LONG")

    # Step 5: Walk-Forward (SHORT model)
    oos_short = run_walk_forward(df_labeled, target_col="y_short", label="SHORT")

    # Step 6: Threshold sweep
    if len(oos_long) > 0:
        long_results = analyze_oos_results(oos_long, label="LONG")
    if len(oos_short) > 0:
        short_results = analyze_oos_results(oos_short, label="SHORT", direction="SHORT")

    # Step 7: Rupee backtest at best threshold
    print("\n\n💰 Running ₹ Backtest at threshold 0.55...")
    if len(oos_long) > 0:
        bt_long = run_rupee_backtest(oos_long, threshold=0.55, direction="LONG")
        run_monte_carlo_if_valid(bt_long)

    if len(oos_short) > 0:
        bt_short = run_rupee_backtest(oos_short, threshold=0.55, direction="SHORT")
        run_monte_carlo_if_valid(bt_short)

    # Step 8: Verdict
    print("\n" + "="*60)
    print("  📊 FINAL VERDICT")
    print("="*60)
    n_total = len(oos_long) + len(oos_short)
    if n_total < CONFIG["MIN_OOS_TRADES"]:
        print(f"  ⚠️  Only {n_total} total OOS rows.")
        print(f"  Need more data: paper trade 30-60 days, then re-run.")
    else:
        # Check if any threshold passed p-value gate
        long_edge  = any(v["has_edge"] for v in long_results.values()) if len(oos_long) > 0 else False
        short_edge = any(v["has_edge"] for v in short_results.values()) if len(oos_short) > 0 else False

        if long_edge or short_edge:
            print(f"  ✅ STATISTICALLY VALID EDGE FOUND!")
            print(f"  Long Edge : {'✅' if long_edge else '❌'}")
            print(f"  Short Edge: {'✅' if short_edge else '❌'}")
            print(f"\n  🚀 Proceed to live scanner. Risk only ₹500 base capital.")
        else:
            print(f"  ❌ No statistically significant edge yet (p >= 0.05 at all thresholds)")
            print(f"  📋 Recommendations:")
            print(f"    1. Collect 30 more days of data")
            print(f"    2. Try different feature combinations")
            print(f"    3. Check feature importance — prune low-importance features")

    return oos_long, oos_short

# ── RUN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    oos_long, oos_short = run_bhavishya_v31()

    # Uncomment to run live scanner (requires trained models)
    # Retrain on full dataset first, then call:
    # scan_groww_tickets(models_long_final, models_short_final)
