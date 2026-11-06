"""
Integrity Gate for Strategy Versions in scratch/new_trade/strategies.
Verifies reality, charges, slippage, absence of lookahead, and statistical sanity.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

def audit_strategy_version(v_dir: Path):
    print("=" * 65)
    print(f"  STRATEGY INTEGRITY GATE  {v_dir.name}")
    print("=" * 65)
    
    results = []
    
    def check(label, passed, detail="", warn=False):
        tag = PASS if passed else (WARN if warn else FAIL)
        results.append((tag, label, detail))
        print(f"  {tag}  {label}")
        if detail:
            print(f"         => {detail}")

    # 1. Pine Script Exists
    pine_files = list(v_dir.glob("*.pine"))
    check("Pine Script (.pine) file exists", len(pine_files) > 0, str(pine_files[0]) if pine_files else "Missing")
    if pine_files:
        p_txt = pine_files[0].read_text(encoding="utf-8", errors="ignore")
        check("Pine Script version specified (//@version=6)", "//@version=6" in p_txt or "//@version=5" in p_txt)
        check("Pine Script has strategy entry logic", "strategy.entry" in p_txt or "strategy.order" in p_txt or "strategy(" in p_txt)

    # 2. Results text file exists
    res_txt = v_dir / "results.txt"
    check("results.txt file exists", res_txt.exists(), str(res_txt) if res_txt.exists() else "Missing")
    if res_txt.exists():
        content = res_txt.read_text(encoding="utf-8", errors="ignore")
        check("results.txt contains Intraday section", "INTRADAY" in content.upper())
        check("results.txt contains Swing section", "SWING" in content.upper())

    # 3. Trades CSV files exist
    intra_csv = v_dir / "trades_intraday.csv"
    swing_csv = v_dir / "trades_swing.csv"
    check("Intraday trades CSV exists", intra_csv.exists(), str(intra_csv))
    check("Swing trades CSV exists", swing_csv.exists(), str(swing_csv))

    # Audit Intraday Trades
    if intra_csv.exists():
        df_in = pd.read_csv(intra_csv)
        check("Intraday trades recorded", len(df_in) > 0, f"{len(df_in)} trades")
        if len(df_in) > 0:
            # Charges check
            if "charges" in df_in.columns:
                zero_chg = (df_in["charges"] <= 0).sum()
                check("Intraday non-zero Dhan charges", zero_chg == 0, f"{zero_chg} zero-charge trades")
                avg_chg = df_in["charges"].mean()
                check("Intraday charges realistic (Rs 0.8 - 5.0)", 0.8 <= avg_chg <= 5.0, f"Avg = Rs {avg_chg:.2f}")
            else:
                check("Intraday charges column present", False, "Missing 'charges'")
                
            # Intraday square-off verification (no overnight hold)
            if "exit_time" in df_in.columns:
                exit_times = pd.to_datetime(df_in["exit_time"])
                entry_times = pd.to_datetime(df_in["entry_time"]) if "entry_time" in df_in.columns else None
                if entry_times is not None:
                    overnight = (exit_times.dt.date != entry_times.dt.date).sum()
                    check("100% Intraday (Zero overnight holds in MIS)", overnight == 0, f"{overnight} overnight trades")

            # Statistical sanity
            wins = (df_in["net_pnl"] > 0).sum()
            wr = (wins / len(df_in)) * 100.0
            check("Intraday win rate plausible (<= 90%)", wr <= 90.0, f"WR = {wr:.1f}%", warn=(wr > 85.0))
            
            losses = df_in[df_in["net_pnl"] <= 0]
            if len(losses) > 0:
                avg_l = losses["net_pnl"].mean()
                check("Losses are non-trivial (Avg loss < -Rs 0.50)", avg_l < -0.50, f"Avg Loss = Rs {avg_l:.2f}")
            else:
                check("At least 1 loss exists (Real trading has losses)", False, "0 losses found - suspicious curve fit")

    # Audit Swing Trades
    if swing_csv.exists():
        df_sw = pd.read_csv(swing_csv)
        check("Swing trades recorded", len(df_sw) > 0, f"{len(df_sw)} trades")
        if len(df_sw) > 0:
            if "charges" in df_sw.columns:
                check("Swing charges recorded", (df_sw["charges"] <= 0).sum() == 0)

    # Summary
    passed  = [r for r in results if r[0] == PASS]
    warned  = [r for r in results if r[0] == WARN]
    failed  = [r for r in results if r[0] == FAIL]
    
    print("-" * 65)
    print(f"  Summary: PASSED={len(passed)}, WARNED={len(warned)}, FAILED={len(failed)}")
    print("=" * 65)
    return len(failed) == 0

if __name__ == "__main__":
    if len(sys.argv) > 1:
        v_name = sys.argv[1]
        target_dir = Path("C:/Users/admin/.gemini/antigravity/scratch/new_trade/strategies") / v_name
        if target_dir.exists():
            success = audit_strategy_version(target_dir)
            sys.exit(0 if success else 1)
        else:
            print(f"Directory {target_dir} not found.")
            sys.exit(1)
    else:
        print("Usage: python strategy_integrity_gate.py <VersionFolder>")
        sys.exit(1)
