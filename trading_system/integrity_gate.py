"""
BHAVISHYA INTEGRITY GATE
Run this AFTER any new version is built and BEFORE claiming any results.
Usage: python integrity_gate.py --version V31

Checks performed:
1. Look-ahead bias in features
2. Label future leakage
3. Walk-forward embargo enforced
4. Training / test date separation (no overlap)
5. Hyperparameter tuning on held-out test data (Optuna leakage)
6. Dhan charges applied (not zero)
7. No impossibly clean results (flags if WR > 85% or PF > 20 on < 20 trades)
8. OOS cache reuse in final test (forbidden)
"""

import sys
import os
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path("C:/Users/admin/.gemini/antigravity/scratch/new_trade/trading_system")

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

results = []

def check(label, passed, detail="", warn=False):
    tag = PASS if passed else (WARN if warn else FAIL)
    results.append((tag, label, detail))
    print(f"  {tag}  {label}")
    if detail:
        print(f"         => {detail}")

def run_gate(version: str):
    print("=" * 65)
    print(f"  BHAVISHYA INTEGRITY GATE — {version}")
    print("=" * 65)

    vdir = BASE / version
    src  = vdir / "src"
    bt   = vdir / "backtests"
    trades_csv = vdir / "trades" / f"{version.lower()}_trades.csv"

    # ── 1. TRADES FILE EXISTS ─────────────────────────────────────
    print("\n[A] Trades File")
    exists = trades_csv.exists()
    check("Trades CSV exists", exists, str(trades_csv))
    if not exists:
        print("\nNo trades file found. Cannot continue audit.")
        return summarize()

    df = pd.read_csv(trades_csv)
    check("At least 1 trade recorded", len(df) > 0, f"{len(df)} trades")

    # ── 2. DHAN CHARGES ──────────────────────────────────────────
    print("\n[B] Realistic Brokerage & Charges")
    if "charges" in df.columns:
        zero_charges = (df["charges"] == 0).sum()
        check("All trades have non-zero charges", zero_charges == 0,
              f"{zero_charges} trades have zero charges")
        avg_charge = df["charges"].mean()
        check("Average charge is realistic (Rs 1-5 per trade)",
              0.5 <= avg_charge <= 6.0, f"Avg = Rs {avg_charge:.2f}")
    else:
        check("Charges column present", False, "Column 'charges' missing from trades CSV")

    # ── 3. IMPOSSIBLY CLEAN RESULTS ───────────────────────────────
    print("\n[C] Statistical Sanity")
    n = len(df)
    wins = int(df["winner"].sum()) if "winner" in df.columns else 0
    losses = n - wins
    wr = wins / n if n > 0 else 0
    avg_win  = df[df["winner"] == True]["net_pnl"].mean() if wins > 0 else 0
    avg_loss = df[df["winner"] == False]["net_pnl"].mean() if losses > 0 else 0
    tot_profit = df[df["net_pnl"] > 0]["net_pnl"].sum()
    tot_loss   = abs(df[df["net_pnl"] < 0]["net_pnl"].sum())
    pf = (tot_profit / tot_loss) if tot_loss > 0 else 999

    print(f"         Trades={n}, WR={wr*100:.1f}%, AvgWin=Rs{avg_win:.2f}, AvgLoss=Rs{avg_loss:.2f}, PF={pf:.2f}")

    check("Trade count >= 10 (enough to be meaningful)", n >= 10, f"n={n}")
    check("Win rate <= 90% (human plausibility)", wr <= 0.90,
          f"WR={wr*100:.1f}% — suspicious if > 90%",
          warn=(0.80 < wr <= 0.90))
    check("Profit factor <= 20 (human plausibility)", pf <= 20.0,
          f"PF={pf:.2f} — extremely suspicious if > 20",
          warn=(10 < pf <= 20))
    check("Average loss is non-trivial (< -Rs 0.50)", avg_loss < -0.50,
          f"AvgLoss=Rs{avg_loss:.2f} — if near zero, SL may not be working")

    if "exit_reason" in df.columns:
        sl_hits = (df["exit_reason"].str.contains("SL_HIT|STOP", na=False)).sum()
        check("At least 1 structural SL hit exists (real trading has real losses)",
              sl_hits >= 1,
              f"SL hits found: {sl_hits} — 0 SL hits in real trading is a red flag",
              warn=(sl_hits == 0))

    # ── 4. LOOK-AHEAD BIAS IN FEATURES ───────────────────────────
    print("\n[D] Look-Ahead Bias — Features")
    feat_file = src / "features.py"
    if feat_file.exists():
        feat_src = feat_file.read_text(encoding="utf-8")
        vwap_shifted = ".shift(1)" in feat_src and "vwap" in feat_src.lower()
        check("VWAP uses shift(1) before bar close", vwap_shifted)
        # Check for dangerous forward-fill or future access patterns
        bad_patterns = ["shift(-", ".shift(-1)", "future", "look_ahead"]
        found_bad = [p for p in bad_patterns if p in feat_src]
        check("No negative shift (future data access)", len(found_bad) == 0,
              f"Found: {found_bad}" if found_bad else "")
    else:
        check("features.py exists", False, str(feat_file))

    # ── 5. LABEL FUTURE LEAKAGE ───────────────────────────────────
    print("\n[E] Label Integrity")
    label_file = src / "labels.py"
    if label_file.exists():
        lbl_src = label_file.read_text(encoding="utf-8")
        future_trim = "n - hold_bars" in lbl_src or "iloc[:-hold_bars]" in lbl_src or "iloc[:n - hold_bars]" in lbl_src
        check("Labels trim rows that peek into future", future_trim)
        uses_future_bars = ("i + j" in lbl_src or "i+j" in lbl_src)
        check("Triple barrier correctly iterates forward bars (i+j)", uses_future_bars)
    else:
        check("labels.py exists", False, str(label_file))

    # ── 6. WALK-FORWARD EMBARGO ───────────────────────────────────
    print("\n[F] Walk-Forward Separation")
    strat_file = src / "strategy.py"
    if strat_file.exists():
        strat_src = strat_file.read_text(encoding="utf-8")
        embargo_enforced = "embargo_days" in strat_src
        test_after_train = "test_start_idx = train_end_idx + embargo_days" in strat_src
        check("Embargo days enforced between train and test", embargo_enforced)
        check("Test period starts AFTER embargo gap", test_after_train)

        # Check train/test date overlap using the trades CSV
        raw_data = Path("C:/Users/admin/.gemini/antigravity/scratch/new_trade/data/universe_78_15m.pkl")
        if raw_data.exists():
            df_raw = pd.read_pickle(raw_data)
            all_dates = sorted(list(set(df_raw.index.date)))
            n_dates = len(all_dates)
            n_folds = 5
            fold_dates = n_dates // (n_folds + 1)

            trade_dates = set(pd.to_datetime(df["timestamp"], utc=True).dt.date.unique())
            leakage_found = False
            for fold in range(1, n_folds + 1):
                train_end = fold * fold_dates
                if train_end >= len(all_dates): break
                train_cutoff = all_dates[train_end]
                # Trades that happened before or on training cutoff for this fold would be leakage
                # (But in WF, each fold has different cutoff - we just check gross violation)
                trades_in_train = [d for d in trade_dates if d < train_cutoff]
                # Only flag if ALL trades are in the earliest training period
                if fold == 1 and len(trades_in_train) == len(trade_dates):
                    leakage_found = True
            check("No gross train/test date overlap detected", not leakage_found)
    else:
        check("strategy.py exists", False, str(strat_file))

    # ── 7. OPTUNA LEAKAGE CHECK ───────────────────────────────────
    print("\n[G] Hyperparameter Optimization Leakage")
    oos_cache = bt / "oos_df_cache.pkl"
    run_script = bt / f"run_{version.lower()}.py"
    opt_script = bt / f"optimize_{version.lower()}.py"

    if oos_cache.exists() and run_script.exists() and opt_script.exists():
        run_src = run_script.read_text(encoding="utf-8")
        opt_src = opt_script.read_text(encoding="utf-8")
        run_uses_cache = "oos_df_cache" in run_src
        opt_uses_cache = "cached_oos.exists()" in opt_src

        if opt_uses_cache:
            # Optuna used cache - check if final run also used same cache
            check("Final run_VXX.py does NOT reuse Optuna's OOS cache",
                  not run_uses_cache,
                  "Optuna tuned params on cached OOS. If final test uses same dates, results are inflated.")
            if not run_uses_cache:
                # Optuna cache exists, final run retrains fresh.
                # BUT: did Optuna tune on the same OOS dates the final run tests on?
                # This is a partial leakage (hyperparameter overfitting)
                check("WARNING: Optuna tuned hyperparams on the same OOS dates as final test",
                      False,
                      "Partial hyperparameter leakage. True unseen test requires separate holdout.",
                      warn=True)
    elif opt_script.exists():
        check("WARNING: Optuna script found - verify it used a separate tuning set", False,
              "Ensure Optuna ran on a DIFFERENT date range than the final test.", warn=True)
    else:
        check("No Optuna optimization script found (params set manually or by prior logic)", True,
              "No Optuna leakage risk")

    # ── 8. FINAL SUMMARY ─────────────────────────────────────────
    return summarize()


def summarize():
    print("\n" + "=" * 65)
    print("  INTEGRITY GATE SUMMARY")
    print("=" * 65)
    passed  = [r for r in results if r[0] == PASS]
    warned  = [r for r in results if r[0] == WARN]
    failed  = [r for r in results if r[0] == FAIL]

    print(f"  PASSED : {len(passed)}")
    print(f"  WARNED : {len(warned)}")
    print(f"  FAILED : {len(failed)}")

    if failed:
        print("\n  FAILED CHECKS:")
        for _, label, detail in failed:
            print(f"    - {label}")
            if detail:
                print(f"      {detail}")

    if warned:
        print("\n  WARNINGS (requires human judgment):")
        for _, label, detail in warned:
            print(f"    - {label}")
            if detail:
                print(f"      {detail}")

    if not failed and not warned:
        print("\n  ALL CHECKS PASSED. Results can be trusted as genuine.")
    elif not failed and warned:
        print("\n  No hard failures, but WARNINGS exist. Review before claiming results.")
    else:
        print("\n  HARD FAILURES FOUND. Do NOT publish these results as genuine.")

    print("=" * 65)
    return len(failed) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bhavishya Integrity Gate")
    parser.add_argument("--version", required=True, help="e.g. V31")
    args = parser.parse_args()
    success = run_gate(args.version)
    sys.exit(0 if success else 1)
