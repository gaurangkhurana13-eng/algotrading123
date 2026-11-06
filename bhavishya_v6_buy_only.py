# ============================================================
# BHAVISHYA V6.0 — PURE BUY-ONLY (LONG SNIPER ENGINE)
# Google Colab Ready | Single-Cell Copy-Paste
# Strictly: BUY First → Hit Target (Sell with Profit) / SL
# ZERO SHORT SELLING. 100% Traditional Buy & Sell.
# Account: ₹500 Base Capital | 5x MIS Leverage = ₹2,500 Buying Power
# Broker: DHAN (Exact ₹1.39 Friction Model — No Min Brokerage)
# ============================================================

import subprocess, sys

def pip_install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

for pkg in ["catboost", "yfinance", "ta", "scipy"]:
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
import yfinance as yf
import ta, time

print("✅ All Buy-Only libraries loaded successfully.")

# ── 1. CONFIG: STRICT BUY-ONLY ARCHITECTURE ──────────────────
CONFIG = {
    # Capital & Leverage
    "CAPITAL"         : 500.0,       # Base capital in ₹
    "LEVERAGE"        : 5.0,         # Dhan 5x MIS Leverage = ₹2,500 Buying Power
    
    # ATR Volatility-Adaptive Barriers for BUY TRADES
    "TP_ATR_MULT"     : 2.0,         # Target = Entry + (2.0 * ATR)
    "SL_ATR_MULT"     : 1.5,         # Stop Loss = Entry - (1.5 * ATR)
    "MIN_ATR_NORM"    : 0.005,       # Minimum 0.5% move threshold
    "HOLD_BARS"       : 12,          # 12 bars × 15m = 3 hours maximum holding window
    
    # Strict Bullish Filters (To prevent falling knives in correcting markets)
    "ADX_MIN_TREND"   : 22.0,        # Minimum trend strength
    "CONVICTION_THR"  : 0.70,        # Conviction Gate: >= 70% confidence to BUY
    "MAX_DAILY_LOSS"  : 2,           # Circuit breaker: Halt after 2 losses in a day
    
    # Walk-Forward Validation
    "N_FOLDS"         : 5,           # 5-fold expanding walk-forward
    "EMBARGO_DAYS"    : 2,           # 2-day embargo
    "MIN_TRAIN_DAYS"  : 25,          # Minimum training trading days
    
    # Model Hyperparameters
    "SEEDS"           : [1, 2, 3],   # 3-seed ensemble
    "DEPTH"           : 4,           # Shallow trees
    "L2"              : 15,          # Heavy regularization
    "ITERATIONS"      : 400,
    "LEARNING_RATE"   : 0.04,
    "MONTE_CARLO_SIMS": 10000,
    
    # 78 Liquid High-Beta NSE Stocks Basket
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

BUYING_POWER = CONFIG["CAPITAL"] * CONFIG["LEVERAGE"]  # ₹2,500
print(f"✅ Pure BUY-ONLY Engine Active. Universe: {len(CONFIG['UNIVERSE'])} Stocks | Buying Power: ₹{BUYING_POWER:.0f}")

# ── 2. EXACT REVERSE-ENGINEERED DHAN CHARGES ─────────────────
def calc_dhan_charges(buy_price: float, sell_price: float, qty: int) -> float:
    buy_to   = buy_price  * qty
    sell_to  = sell_price * qty
    total_to = buy_to + sell_to
    
    # 1. Brokerage: 0.03% or Rs 20, whichever is LOWER. (NO MINIMUM FLOOR!)
    brok  = min(20.0, buy_to * 0.0003) + min(20.0, sell_to * 0.0003)
    exc   = total_to * 0.0000307
    stt   = sell_to * 0.00025
    sebi  = total_to * 0.000001
    stamp = buy_to * 0.00003
    gst   = (brok + exc + sebi) * 0.18
    return round(brok + exc + stt + sebi + stamp + gst, 2)

_sample_charges = calc_dhan_charges(37.0, 37.5, 35)
print(f"✅ Dhan Engine Active. Sample Charges (SUZLON ₹37, Qty 35, Turnover ₹2,608): ₹{_sample_charges:.2f}")

# ── 3. BATCH PARALLEL DATA DOWNLOAD (YAHOO FINANCE) ──────────
def download_78_stocks():
    print(f"\n📡 Batch Downloading 60-Day 15-Minute Data for {len(CONFIG['UNIVERSE'])} Stocks...")
    t0 = time.time()
    try:
        raw = yf.download(CONFIG["UNIVERSE"], period="60d", interval="15m", progress=False, group_by="ticker")
        elapsed = time.time() - t0
        print(f"✅ Downloaded in {elapsed:.1f} seconds.")
        return raw
    except Exception as e:
        print(f"❌ Download error: {e}")
        return None

# ── 4. FEATURE ENGINEERING: BULLISH MOMENTUM FOCUS ───────────
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]

    # 1. Stationary Returns (Shifted 1 bar to strictly avoid lookahead)
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

    # 4. Stationary Trend Deviations & Moving Averages
    ema9  = c.ewm(span=9).mean()
    ema21 = c.ewm(span=21).mean()
    ema50 = c.ewm(span=50).mean()
    
    df["ema_dev_9"]  = (c - ema9) / (ema9 + 1e-9)
    df["ema_dev_21"] = (c - ema21) / (ema21 + 1e-9)
    df["ema_dev_50"] = (c - ema50) / (ema50 + 1e-9)
    
    # BULLISH ALIGNMENT: 9 EMA > 21 EMA & Price > 50 EMA
    df["bullish_alignment"] = ((ema9 > ema21) & (c > ema50)).astype(int)

    # ADX Trend Filter
    adx = ta.trend.ADXIndicator(h, l, c, window=14)
    df["adx"]        = adx.adx()
    df["di_diff"]    = adx.adx_pos() - adx.adx_neg()
    df["adx_rising"] = (df["adx"] > df["adx"].shift(2)).astype(int)
    # Plus DI > Minus DI confirms buyers dominate
    df["bulls_dominate"] = (adx.adx_pos() > adx.adx_neg()).astype(int)

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
    "ema_dev_9","ema_dev_21","ema_dev_50","bullish_alignment",
    "adx","di_diff","adx_rising","bulls_dominate",
    "bb_pos","bb_width","squeeze_on","squeeze_release",
    "rvol","vol_zscore","vol_surge","obv_norm",
    "vwap_dev","above_vwap","near_close","morning",
    "squeeze_x_vol","ret_x_rvol"
]

# ── 5. PURE BUY TRIPLE BARRIER LABELING ───────────────────────
def apply_buy_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    PURE BUY ONLY:
    Entry at Close[i] (BUY).
    Target Hit = Exit at Close[i] + (2.0 * ATR) -> WIN!
    Stop Loss Hit = Exit at Close[i] - (1.5 * ATR) -> LOSS.
    """
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

            # Check if Target hit first
            if hi >= tp_price:
                y_buy[i] = 1
                hit = True
                break
            # Check if Stop Loss hit
            elif lo <= sl_price:
                y_buy[i] = 0
                hit = True
                break

        # Time barrier exit (3 hours)
        if not hit:
            ret = (c[i + hold] - entry) / entry
            y_buy[i] = 1 if ret >= tp_dist else 0

    df["y_buy"]      = y_buy
    df["target_pct"] = target_pcts
    df["sl_pct"]     = sl_pcts
    return df.iloc[:n - hold]

# ── 6. PROCESS 78 STOCKS MATRIX ──────────────────────────────
def build_master_dataset(raw_data):
    print("\n⚙️  Engineering Bullish Features & BUY Labels Across 78 Stocks...")
    dfs = []
    skipped = 0
    for ticker in CONFIG["UNIVERSE"]:
        try:
            if ticker in raw_data:
                df_s = raw_data[ticker].dropna().copy()
            else:
                continue

            if len(df_s) < 200:
                skipped += 1
                continue

            df_s.columns = [col.lower() for col in df_s.columns]
            df_feat = compute_features(df_s)
            df_lab  = apply_buy_labels(df_feat)
            df_lab["ticker"] = ticker.replace(".NS","")
            dfs.append(df_lab)
        except Exception:
            skipped += 1
            continue

    if not dfs:
        raise ValueError("No stocks successfully processed!")

    master_df = pd.concat(dfs).sort_index()
    print(f"✅ Master Matrix Assembled: {len(master_df):,} samples across {len(dfs)} stocks.")
    print(f"   • BUY Setups (Class 1): {master_df['y_buy'].mean():.1%}")
    return master_df

# ── 7. ENSEMBLE CATBOOST TRAINER ─────────────────────────────
def train_ensemble(X, y):
    pos = y.sum()
    neg = len(y) - pos
    cw  = {0: 1, 1: max(2, min(5, neg // max(pos, 1)))}
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
            od_wait       = 35,
            verbose       = 0,
            class_weights = cw
        )
        m.fit(X, y)
        models.append(m)
    return models

def predict_ensemble(models, X):
    return np.array([m.predict_proba(X)[:, 1] for m in models]).mean(axis=0)

# ── 8. EXPANDING WALK-FORWARD (STRICT BULLISH REGIME ONLY) ───
def run_walk_forward_pooled(df_pool: pd.DataFrame):
    """
    EXPANDING WALK-FORWARD (STRICT BUY ONLY):
    Strict filter for taking BUY trades:
      - ADX >= 22 (Trending)
      - Above VWAP == 1 (Buyers in control)
      - Bulls Dominate == 1 (+DI > -DI)
    """
    df_pool = df_pool.sort_index()
    unique_dates = pd.Series(df_pool.index.date).drop_duplicates().tolist()
    n_dates = len(unique_dates)
    fold_dates = n_dates // (CONFIG["N_FOLDS"] + 1)
    embargo_days = CONFIG["EMBARGO_DAYS"]
    
    oos_list = []
    print(f"\n{'='*70}\n  WALK-FORWARD POOLED [BUY ONLY] | Strict Bullish Gate (Above VWAP + Trending)\n{'='*70}")

    for fold in range(1, CONFIG["N_FOLDS"] + 1):
        train_end_idx = fold * fold_dates
        test_start_idx = train_end_idx + embargo_days
        test_end_idx = min(test_start_idx + fold_dates, n_dates)

        if train_end_idx < CONFIG["MIN_TRAIN_DAYS"] or test_start_idx >= n_dates:
            continue

        train_cutoff    = unique_dates[train_end_idx]
        test_start_date = unique_dates[test_start_idx]
        test_end_date   = unique_dates[test_end_idx - 1]

        tr = df_pool[df_pool.index.date < train_cutoff]
        te = df_pool[(df_pool.index.date >= test_start_date) & (df_pool.index.date <= test_end_date)]

        # Train on bullish setups
        tr_f = tr[(tr["near_close"] == 0) & (tr["adx"] >= CONFIG["ADX_MIN_TREND"]) & (tr["above_vwap"] == 1)]
        pos_labels = tr_f["y_buy"].sum()

        if len(tr_f) < 400 or pos_labels < 40:
            print(f"  Fold {fold}: Skipped (not enough bullish setups: {len(tr_f)})")
            continue

        models = train_ensemble(tr_f[FEATURE_COLS].values, tr_f["y_buy"].values)

        # Test strictly on Bullish Setups
        te_f = te[(te["near_close"] == 0) & (te["adx"] >= CONFIG["ADX_MIN_TREND"]) & (te["above_vwap"] == 1)].copy()
        if len(te_f) == 0:
            continue

        te_f["prob"]   = predict_ensemble(models, te_f[FEATURE_COLS].values)
        te_f["y_true"] = te_f["y_buy"].values
        te_f["fold"]   = fold
        oos_list.append(te_f)

        print(f"  Fold {fold}: Train={len(tr_f):,} (Pos={int(pos_labels)}) | "
              f"Bullish Test Bars={len(te_f):,} | Label%={te_f['y_true'].mean():.1%}")

    if not oos_list:
        print("  ❌ No out-of-sample data generated!")
        return pd.DataFrame()
    
    oos = pd.concat(oos_list).sort_index()
    print(f"  ✅ Out-of-Sample Matrix Ready: {len(oos):,} high-probability bullish setups.")
    return oos

# ── 9. PURE BUY RUPEE BACKTEST (DHAN FEES + ATR P&L) ─────────
def run_buy_backtest(oos_df: pd.DataFrame, threshold: float):
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
        # Circuit breaker: max 2 losses in a day
        if daily_losses.get(d, 0) >= CONFIG["MAX_DAILY_LOSS"]:
            continue

        price   = row["close"]
        qty     = max(1, int(bp / price))
        charges = calc_dhan_charges(price, price, qty)

        tp_pct = row["target_pct"]
        sl_pct = row["sl_pct"]

        if row["y_true"] == 1:
            # BUY WIN: Price rose to Target!
            gross = bp * tp_pct
            wins += 1
        else:
            # BUY LOSS: Price fell to Stop Loss!
            gross = -bp * sl_pct
            losses += 1
            daily_losses[d] = daily_losses.get(d, 0) + 1

        net = gross - charges
        total_pnl += net
        total_charges += charges
        trade_log.append({
            "datetime": idx, "ticker": row["ticker"], "price": price,
            "qty": qty, "gross": gross, "charges": charges,
            "net_pnl": net, "win": row["y_true"] == 1,
            "tp_pct": tp_pct, "sl_pct": sl_pct
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
        "avg_win": aw, "avg_loss": al, "payoff": abs(aw / al) if al != 0 else 0,
        "profit_factor": pf, "total_pnl": total_pnl, "charges": total_charges,
        "roi": roi, "trade_log": trade_log
    }

# ── 10. 10,000-RUN MONTE CARLO STRESS TEST ───────────────────
def run_monte_carlo(trade_log: list, n_sims=CONFIG["MONTE_CARLO_SIMS"]):
    n_trades = len(trade_log)
    if n_trades < 15:
        print(f"\n⚠️  Monte Carlo Skipped: {n_trades} trades is too small to resample.")
        return

    pnls    = np.array([t["net_pnl"] for t in trade_log])
    capital = CONFIG["CAPITAL"]
    
    final_capitals = []
    max_drawdowns  = []
    ruin_count     = 0

    print(f"\n{'='*70}\n  🎲 RUNNING {n_sims:,} MONTE CARLO BOOTSTRAP SIMULATIONS\n{'='*70}")
    
    for _ in range(n_sims):
        sim_pnls = np.random.choice(pnls, size=n_trades, replace=True)
        equity_curve = capital + np.cumsum(sim_pnls)
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve) / peak
        max_dd = np.max(drawdown) * 100.0
        max_drawdowns.append(max_dd)

        final_cap = equity_curve[-1]
        final_capitals.append(final_cap)
        if np.min(equity_curve) <= capital * 0.50:
            ruin_count += 1

    ruin_prob  = ruin_count / n_sims * 100.0
    p5         = np.percentile(final_capitals, 5)
    p50        = np.percentile(final_capitals, 50)
    p95        = np.percentile(final_capitals, 95)
    avg_max_dd = np.mean(max_drawdowns)

    print(f"  • Actual Out-of-Sample BUY Trades Resampled: {n_trades}")
    print(f"  • 5th Percentile Capital (Worst 5% Luck):  ₹{p5:.2f}")
    print(f"  • 50th Percentile Capital (Median Outcome): ₹{p50:.2f}")
    print(f"  • 95th Percentile Capital (Best 5% Luck):   ₹{p95:.2f}")
    print(f"  • Average Maximum Simulated Drawdown:      {avg_max_dd:.1f}%")
    print(f"  • Probability of 50% Account Ruin:         {ruin_prob:.2f}%")
    
    if ruin_prob < 5.0 and p50 > capital:
        print(f"  ✅ INSTITUTIONAL VERDICT: PASSED (Ruin Risk < 5% | Edge Confirmed)")
    else:
        print(f"  ⚠️  INSTITUTIONAL VERDICT: FAILED (High Ruin Risk or Inadequate Edge)")

# ── 11. MASTER RUN ENGINE ────────────────────────────────────
def run_bhavishya_buy_only():
    print("\n" + "="*70)
    print("  🚀 STARTING BHAVISHYA V6.0 PURE BUY-ONLY PIPELINE")
    print("="*70)

    # 1. Download
    raw_data = download_78_stocks()
    if raw_data is None:
        return

    # 2. Bullish Features & Labels
    master_matrix = build_master_dataset(raw_data)

    # 3. Expanding Walk-Forward with Strict Bullish Filter
    oos_buy = run_walk_forward_pooled(master_matrix)

    # 4. Conviction Threshold Sweep
    print(f"\n🎯 AUDITING CONVICTION THRESHOLDS (BUY CONVICTION >= 65%):")
    print(f"  {'Thresh':>7} | {'BUY Trades':>12} | {'BUY Win Rate':>14}")
    print(f"  {'-'*40}")
    for thr in [0.65, 0.70, 0.75, 0.80]:
        sub = oos_buy[oos_buy["prob"] >= thr]
        wr = sub["y_true"].mean() if len(sub) > 0 else 0
        print(f"  {thr:>6.2f}  | {len(sub):>12} | {wr:>13.1%}")

    # 5. Execute Backtest at Configured Conviction Threshold
    best_thr = CONFIG["CONVICTION_THR"]  # 0.70
    bt_buy   = run_buy_backtest(oos_buy, best_thr)

    print("\n" + "="*75)
    print(f"  🏆 PURE BUY-ONLY DHAN PERFORMANCE AUDIT (CONVICTION >= {best_thr*100:.0f}%)")
    print("="*75)
    print(f"  • Total Unseen BUY Trades Taken:  {bt_buy['n_trades']}")
    print(f"  • BUY Win Rate:                   {bt_buy['win_rate']:.1%}")
    print(f"  • Average Win on BUY:             ₹{bt_buy['avg_win']:.2f}")
    print(f"  • Average Loss on BUY:            ₹{bt_buy['avg_loss']:.2f}")
    print(f"  • Profit Factor:                  {bt_buy['profit_factor']:.2f}")
    print(f"  • Total Dhan Brokerage & Taxes:   ₹{bt_buy['charges']:.2f}")
    print(f"  • NET Realized Rupee Profit:      ₹{bt_buy['total_pnl']:+.2f}")
    print(f"  • Net Return on ₹500 Capital:     {bt_buy['roi']:+.1f}%")
    print("="*75)

    # 6. Monte Carlo 10,000 Simulations
    run_monte_carlo(bt_buy["trade_log"])

if __name__ == "__main__":
    run_bhavishya_buy_only()
