import os, sys, pickle
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

from framework import calc_dhan_charges, calculate_atr

print("=" * 100)
print("BHAVISHYA V42 EXIT & DYNAMIC R-MULTIPLE ARCHITECT STUDY")
print("Target multiples: 1.5R, 2.0R, 2.5R, 3.0R vs Early Lock (+0.40 ATR @ 0.85 ATR MFE)")
print("=" * 100)

# Folds configuration matching benchmark_and_certify_v42.py
folds_def = [
    (0, "Fold 0 (2021 Full Year)", ROOT / 'data/2021-22/universe_30_15m.pkl', "2021-01-01", "2021-12-31"),
    (1, "Fold 1 (2026 Bull Horizon)", 'COMPOSITE_1', "2026-01-01", "2026-12-31"),
    (2, "Fold 2 (2022 Bear Market)", ROOT / 'data/2022-23/universe_30_15m.pkl', "2022-01-01", "2022-12-31"),
    (3, "Fold 3 (2023 Full Year)", ROOT / 'data/2023-24/universe_30_15m.pkl', "2023-01-02", "2023-12-29"),
    (4, "Fold 4 (2023 H1 Virgin)", ROOT / 'data/2023-24/universe_30_15m.pkl', "2023-01-01", "2023-04-30"),
    (5, "Fold 5 (2023 Mid 4M)", ROOT / 'data/2023-24/universe_30_15m.pkl', "2023-05-01", "2023-08-31"),
    (6, "Fold 6 (2023 H2 Virgin)", ROOT / 'data/2023-24/universe_30_15m.pkl', "2023-09-01", "2023-12-31"),
    (7, "Fold 7 (2024 H1 Pre-Election)", ROOT / 'data/2024-25/universe_14_15m.pkl', "2024-01-01", "2024-06-30"),
    (8, "Fold 8 (2024 H2 Crash FII)", ROOT / 'data/2024-25/universe_14_15m.pkl', "2024-07-01", "2024-12-31"),
    (9, "Fold 9 (2025 H1 Choppy)", ROOT / 'data/2025-26/universe_14_15m.pkl', "2025-01-01", "2025-07-31"),
    (10, "Fold 10 (2025 Q4)", ROOT / 'data/2025-26/universe_14_15m.pkl', "2025-10-01", "2025-12-31"),
    (11, "Fold 11 (2025 Last 5M)", ROOT / 'data/2025-26/universe_14_15m.pkl', "2025-08-01", "2025-12-31"),
    (12, "Fold 12 (20 Virgin Companies)", 'COMPOSITE_12', "2026-01-01", "2026-10-31"),
    (13, "Fold 13 (2026 Unseen Midcaps)", 'COMPOSITE_13', "2026-01-01", "2026-12-31"),
    (14, "Fold 14 (2020 Full COVID Year)", ROOT / 'data/2020_FULL/universe_30_15m.pkl', "2020-01-01", "2020-12-31"),
    (15, "Fold 15 (2019 Unseen Year)", ROOT / 'data/2019_UNSEEN/universe_30_15m.pkl', "2019-01-01", "2019-12-31"),
    (16, "Fold 16 (2018 Unseen Year)", ROOT / 'data/2018_UNSEEN/universe_30_15m.pkl', "2018-01-01", "2018-12-31"),
    (17, "Fold 17 (2017 Unseen Year)", ROOT / 'data/2017_UNSEEN/universe_30_15m.pkl', "2017-01-01", "2017-12-31"),
    (18, "Fold 18 (2016 H2 Unseen)", ROOT / 'data/2016_H2_UNSEEN/universe_30_15m.pkl', "2016-07-01", "2016-12-31"),
]

def load_pool_by_spec(spec):
    if spec == 'COMPOSITE_1':
        u1 = pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))
        u2 = pickle.load(open(ROOT / 'data/unseen_high_beta_10_15m_1y.pkl', 'rb'))
        u3 = pickle.load(open(ROOT / 'data/unseen_10_15m_1y.pkl', 'rb'))
        return {**u1, **u2, **u3}
    elif spec == 'COMPOSITE_12':
        b1 = pickle.load(open(ROOT / 'data/unseen_10_15m_1y.pkl', 'rb'))
        b3 = pickle.load(open(ROOT / 'data/unseen_batch3_10_15m_1y.pkl', 'rb'))
        return {**b1, **b3}
    elif spec == 'COMPOSITE_13':
        return pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))
    else:
        return pickle.load(open(spec, 'rb'))

# Load trades from v42_19_folds_master_results.pkl
master_data = pickle.load(open(ROOT / 'scripts/v42_19_folds_master_results.pkl', 'rb'))
trades_list = master_data['all_trades']
print(f"Total historical V42 trades loaded: {len(trades_list)}")

# Map trades to folds
# We know the count of trades per fold from df_matrix
fold_counts = master_data['df_matrix']['v42_trades'].tolist()

trades_by_fold = []
cur_idx = 0
for fid, count in enumerate(fold_counts):
    fold_trades = trades_list[cur_idx : cur_idx + count]
    trades_by_fold.append((fid, fold_trades))
    cur_idx += count

print("Trades mapped to folds successfully.")

# Prepare dataframe cache for each fold
fold_pool_cache = {}
for fid, lbl, spec, s_dt, e_dt in folds_def:
    pool = load_pool_by_spec(spec)
    # clean ticker keys
    clean_pool = {}
    for k, df in pool.items():
        clean = k.replace('.NS', '')
        if clean not in clean_pool or not k.endswith('.NS'):
            clean_pool[clean] = df
    fold_pool_cache[fid] = clean_pool

# Now let's build trade bar sequences
trade_contexts = []
slippage = 0.0005

for fid, fold_trades in trades_by_fold:
    clean_pool = fold_pool_cache[fid]
    for t in fold_trades:
        sym = t['symbol']
        if sym not in clean_pool:
            continue
        df_sym = clean_pool[sym]
        entry_t = pd.Timestamp(t['entry_time'])
        entry_p = t['entry_price']
        initial_sl = t['sl']
        shares = t['shares']
        risk = entry_p - initial_sl
        entry_atr = risk / 0.90 # Since active_sl = entry_p - 0.90 * entry_atr
        
        # Extract bars from entry_t to EOD on that same date
        cur_date = entry_t.date()
        sub_df = df_sym.loc[str(cur_date)].copy()
        
        # Bars strictly after entry_t up to 14:45
        bars_after = sub_df[(sub_df.index > entry_t) & (sub_df.index.time <= pd.Timestamp('14:45').time())]
        
        trade_contexts.append({
            'trade_orig': t,
            'fid': fid,
            'symbol': sym,
            'entry_t': entry_t,
            'entry_p': entry_p,
            'initial_sl': initial_sl,
            'risk': risk,
            'entry_atr': entry_atr,
            'shares': shares,
            'bars': bars_after,
            'orig_net_pnl': t['net_pnl'],
            'orig_exit_price': t['exit_price'],
            'orig_exit_reason': t['exit_reason'],
        })

print(f"Total valid trade contexts extracted with subsequent bar history: {len(trade_contexts)}")

# PART 1: MFE DISTRIBUTION STUDY
print("\n" + "=" * 80)
print("PART 1: MAXIMUM FAVORABLE EXCURSION (MFE) DISTRIBUTION OF V42 ENTRIES")
print("=" * 80)

mfes_r = []
mfes_atr = []
for tc in trade_contexts:
    bars = tc['bars']
    if len(bars) == 0:
        max_h = tc['entry_p']
    else:
        max_h = bars['high'].max()
    mfe_price = max_h - tc['entry_p']
    mfe_r = mfe_price / tc['risk']
    mfe_atr = mfe_price / tc['entry_atr']
    mfes_r.append(mfe_r)
    mfes_atr.append(mfe_atr)

mfes_r = np.array(mfes_r)
mfes_atr = np.array(mfes_atr)

print(f"Total Trades Evaluated: {len(mfes_r)}")
print(f"Average Potential MFE: {mfes_r.mean():.2f}R ({mfes_atr.mean():.2f} ATR)")
print(f"Median Potential MFE:  {np.median(mfes_r):.2f}R ({np.median(mfes_atr):.2f} ATR)")
print(f"Max Potential MFE:     {mfes_r.max():.2f}R ({mfes_atr.max():.2f} ATR)")
print("\nPercentage of Trades Reaching Potential Thresholds (Before EOD):")
for r_level in [0.5, 0.85/0.9, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    pct = (mfes_r >= r_level).mean() * 100
    atr_equiv = r_level * 0.90
    print(f"  MFE >= {r_level:.2f}R (~{atr_equiv:.2f} ATR): {pct:5.1f}% of trades ({int((mfes_r >= r_level).sum())} trades)")

# PART 2: TESTING DYNAMIC R-MULTIPLE STRATEGIES ON EXISTING ENTRIES
print("\n" + "=" * 80)
print("PART 2: TESTING DYNAMIC R-MULTIPLE TARGETS (1.5R, 2R, 2.5R, 3R) VS V42 BASELINE")
print("=" * 80)

def simulate_trade_replay(tc, exit_mode: str, target_r: float = 2.0):
    entry_p = tc['entry_p']
    initial_sl = tc['initial_sl']
    risk = tc['risk']
    entry_atr = tc['entry_atr']
    shares = tc['shares']
    bars = tc['bars']
    
    if len(bars) == 0:
        # exited on same bar at entry
        exit_fill = entry_p * (1.0 - slippage)
        gross = round((exit_fill - entry_p) * shares, 2)
        chg = round(calc_dhan_charges(entry_p, exit_fill, shares, is_intraday=True), 2)
        return {"net_pnl": gross - chg, "gross_pnl": gross, "charges": chg, "exit_price": exit_fill, "reason": "SAME_BAR", "realized_r": 0.0}

    active_sl = initial_sl
    high_seen = entry_p
    target_price = round(entry_p + target_r * risk, 2)
    
    be_price = round(entry_p + 0.05 * entry_atr, 2)
    
    exit_fill = None
    exit_reason = None
    realized_r = 0.0
    
    for bar_idx, (ts, row) in enumerate(bars.iterrows()):
        cur_h = row['high']
        cur_l = row['low']
        cur_c = row['close']
        high_seen = max(high_seen, cur_h)
        mfe_r = (high_seen - entry_p) / risk
        mfe_atr = (high_seen - entry_p) / entry_atr
        
        # --- Update Stop Loss based on Architecture ---
        if exit_mode == 'BASELINE_V42':
            # V42 original ratchet logic
            if mfe_atr >= 3.50:
                stop_ladder = high_seen - (0.18 * entry_atr)
            elif mfe_atr >= 2.25:
                stop_ladder = high_seen - (0.20 * entry_atr)
            elif mfe_atr >= 1.25:
                stop_ladder = high_seen - (0.28 * entry_atr)
            elif mfe_atr >= 0.85:
                # The Early Lock (+0.40 ATR)
                stop_ladder = entry_p + (0.40 * entry_atr)
            elif mfe_atr >= 0.45:
                stop_ladder = entry_p + (0.05 * entry_atr)
            else:
                stop_ladder = entry_p - (0.90 * entry_atr)
            active_sl = max(active_sl, round(stop_ladder, 2))
            
        elif exit_mode == 'FIXED_R_NO_BE':
            # Pure fixed target with initial SL
            pass # active_sl remains initial_sl
            
        elif exit_mode == 'FIXED_R_WITH_BE':
            # Move to BE (+0.05 ATR) when MFE >= 0.45 ATR (covers costs), no early 0.85 ATR lock
            if mfe_atr >= 0.45:
                active_sl = max(active_sl, be_price)
                
        elif exit_mode == 'DYNAMIC_R_LADDER_2R':
            # BE @ 0.45 ATR (0.5R), Lock 1.0R @ 1.5R MFE, Target @ 2.0R
            if mfe_r >= 1.50:
                active_sl = max(active_sl, round(entry_p + 1.0 * risk, 2))
            elif mfe_atr >= 0.45:
                active_sl = max(active_sl, be_price)

        elif exit_mode == 'DYNAMIC_R_LADDER_2.5R':
            # BE @ 0.45 ATR, Lock 1.0R @ 1.5R, Lock 1.75R @ 2.0R, Target @ 2.5R
            if mfe_r >= 2.00:
                active_sl = max(active_sl, round(entry_p + 1.75 * risk, 2))
            elif mfe_r >= 1.50:
                active_sl = max(active_sl, round(entry_p + 1.0 * risk, 2))
            elif mfe_atr >= 0.45:
                active_sl = max(active_sl, be_price)

        elif exit_mode == 'DYNAMIC_R_LADDER_3R':
            # BE @ 0.45 ATR, Lock 1.0R @ 1.5R, Lock 1.8R @ 2.2R, Trail with 0.5R buffer, Target @ 3.0R
            if mfe_r >= 2.50:
                trail = high_seen - (0.50 * risk)
                active_sl = max(active_sl, round(max(entry_p + 2.0 * risk, trail), 2))
            elif mfe_r >= 2.00:
                active_sl = max(active_sl, round(entry_p + 1.5 * risk, 2))
            elif mfe_r >= 1.50:
                active_sl = max(active_sl, round(entry_p + 1.0 * risk, 2))
            elif mfe_atr >= 0.45:
                active_sl = max(active_sl, be_price)
                
        elif exit_mode == 'DYNAMIC_ATR_TRAIL':
            # Exit.py style dynamic trailing: BE @ 0.45 ATR, then once MFE >= 1.2R, trail stop at high_seen - 1.0 * entry_atr
            if mfe_r >= 1.20:
                trail_stop = high_seen - (1.0 * entry_atr)
                active_sl = max(active_sl, round(trail_stop, 2))
            elif mfe_atr >= 0.45:
                active_sl = max(active_sl, be_price)

        # --- Check Target Hit (for target-based modes) ---
        if 'TARGET' in exit_mode or 'FIXED' in exit_mode or 'LADDER' in exit_mode:
            if cur_h >= target_price:
                exit_fill = target_price
                exit_reason = f"TARGET_{target_r}R"
                break
                
        # --- Check Stop Hit ---
        if cur_l <= active_sl:
            raw_fill = min(cur_h, active_sl) * (1.0 - slippage)
            exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
            exit_reason = "STOP"
            break
            
        # --- Check EOD ---
        if ts.time() >= pd.Timestamp("14:45").time():
            raw_fill = cur_c * (1.0 - slippage)
            exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
            exit_reason = "EOD"
            break
            
    if exit_fill is None:
        last_c = bars.iloc[-1]['close']
        raw_fill = last_c * (1.0 - slippage)
        exit_fill = round(raw_fill, 2)
        exit_reason = "EOD"

    gross = round((exit_fill - entry_p) * shares, 2)
    chg = round(calc_dhan_charges(entry_p, exit_fill, shares, is_intraday=True), 2)
    net = round(gross - chg, 2)
    realized_r = round((exit_fill - entry_p) / risk, 2)
    
    return {
        "net_pnl": net,
        "gross_pnl": gross,
        "charges": chg,
        "exit_price": exit_fill,
        "reason": exit_reason,
        "realized_r": realized_r
    }

# Architectures to evaluate
architectures = [
    ("1. V42 Baseline (+0.40 ATR @ 0.85 ATR)", "BASELINE_V42", 0.0),
    ("2. Fixed 1.5R Target (with BE @ 0.45 ATR)", "FIXED_R_WITH_BE", 1.5),
    ("3. Fixed 2.0R Target (with BE @ 0.45 ATR)", "FIXED_R_WITH_BE", 2.0),
    ("4. Fixed 2.5R Target (with BE @ 0.45 ATR)", "FIXED_R_WITH_BE", 2.5),
    ("5. Fixed 3.0R Target (with BE @ 0.45 ATR)", "FIXED_R_WITH_BE", 3.0),
    ("6. Fixed 2.0R Target (Pure, No BE)", "FIXED_R_NO_BE", 2.0),
    ("7. Fixed 3.0R Target (Pure, No BE)", "FIXED_R_NO_BE", 3.0),
    ("8. Dynamic Ladder 2.0R (BE + Lock 1R @ 1.5R + Tgt 2R)", "DYNAMIC_R_LADDER_2R", 2.0),
    ("9. Dynamic Ladder 2.5R (BE + Lock 1R + Lock 1.75R + Tgt 2.5R)", "DYNAMIC_R_LADDER_2.5R", 2.5),
    ("10. Dynamic Ladder 3.0R (BE + Lock 1R + Lock 1.8R + Trail + Tgt 3R)", "DYNAMIC_R_LADDER_3R", 3.0),
    ("11. Dynamic ATR Trailing Stop (Trail High - 1.0 ATR @ 1.2R MFE)", "DYNAMIC_ATR_TRAIL", 0.0),
]

summary_rows = []

for name, mode, tgt_r in architectures:
    res_list = [simulate_trade_replay(tc, mode, tgt_r) for tc in trade_contexts]
    df_res = pd.DataFrame(res_list)
    
    n_trades = len(df_res)
    wins = df_res[df_res['net_pnl'] > 0]
    losses = df_res[df_res['net_pnl'] < 0]
    be = df_res[df_res['net_pnl'] == 0]
    
    wr = len(wins) / n_trades * 100.0
    tot_net = df_res['net_pnl'].sum()
    gross_pnl = df_res['gross_pnl'].sum()
    tot_charges = df_res['charges'].sum()
    
    avg_win = wins['net_pnl'].mean() if len(wins) > 0 else 0.0
    med_win = wins['net_pnl'].median() if len(wins) > 0 else 0.0
    avg_loss = losses['net_pnl'].mean() if len(losses) > 0 else 0.0
    
    gp = wins['net_pnl'].sum() if len(wins) > 0 else 0.0
    gl = abs(losses['net_pnl'].sum()) if len(losses) > 0 else 1.0
    pf = gp / gl if gl > 0 else 999.0
    
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0
    
    # Calculate Drawdown on these trades
    equity_curve = [500.0]
    for p in df_res['net_pnl']:
        equity_curve.append(equity_curve[-1] + p)
    eq_s = pd.Series(equity_curve)
    peak = eq_s.cummax()
    dd_s = (peak - eq_s) / peak * 100.0
    max_dd = dd_s.max()
    
    avg_realized_r_wins = wins['realized_r'].mean() if len(wins) > 0 else 0.0
    
    # Target exits count
    tgt_exits = df_res['reason'].str.startswith('TARGET').sum() if 'reason' in df_res else 0
    
    summary_rows.append({
        "Architecture": name,
        "Win Rate (%)": round(wr, 1),
        "Avg Win (Rs)": round(avg_win, 2),
        "Median Win (Rs)": round(med_win, 2),
        "Avg Loss (Rs)": round(avg_loss, 2),
        "Win/Loss Ratio": round(win_loss_ratio, 2),
        "Avg Realized R (Wins)": round(avg_realized_r_wins, 2),
        "Profit Factor": round(pf, 2),
        "Total Net PnL (Rs)": round(tot_net, 2),
        "Gross PnL (Rs)": round(gross_pnl, 2),
        "Charges (Rs)": round(tot_charges, 2),
        "Max DD (%)": round(max_dd, 1),
        "Target Hits": int(tgt_exits)
    })

df_summary = pd.DataFrame(summary_rows)
print(df_summary.to_string(index=False))

# Save summary to pickle and csv for reporting
df_summary.to_csv(ROOT / "reports/exit_architecture_comparison.csv", index=False)
pickle.dump(df_summary, open(ROOT / "reports/exit_architecture_comparison.pkl", "wb"))
print("\nSaved comparison report to reports/exit_architecture_comparison.csv")
