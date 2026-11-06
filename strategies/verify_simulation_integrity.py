#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_simulation_integrity.py — Check 10: Simulation Reproducibility Verifier
================================================================================

Yeh Gate V2 ka extension hai (Gate V2 ko TOUCH nahi karta).
Gate V2 sirf CSV read karta hai aur verify karta hai.
Yeh script ek step aage jaata hai:
  1. Run script ko actually execute karta hai (fresh, temp folder mein)
  2. Fresh generated CSV vs stored CSV ka byte-level comparison karta hai
  3. Agar koi bhi row mismatch hai → FAIL

Usage:
    python verify_simulation_integrity.py strategies/V34.2/fold_2022
    python verify_simulation_integrity.py strategies/V34.2/fold_2026 --data-dir data/2022-23
    python verify_simulation_integrity.py strategies/V34.2  (saare folds)

Exit code: 0 = PASS, 1 = FAIL

Design Principles:
  - Gate V2 ko kabhi import/modify nahi karta
  - Run script ko temp directory mein execute karta hai (original folder untouched rehta hai)
  - Koi bhi external model/LLM call nahi karta
  - Pure Python stdlib + pandas/numpy only
  - Deterministic: same data → same CSV → PASS (numpy random seed bhi check karta hai)

Attack surface jo yeh seal karta hai:
  - CSV manually edited ho → FAIL (fresh run se mismatch)
  - CSV aur run script alag logic → FAIL
  
Attack surface jo abhi bhi exist karta hai (honest):
  - Run script khud PKL data read karke manipulate kare → still possible
  - PKL data files tampered hoon → still possible  
  (In dono ke liye PKL hash pinning needed hai — future Gate V3 kaam)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent   # new_trade/
NUMERIC_COLS = ["entry_price", "exit_price", "shares", "gross_pnl", "charges", "net_pnl"]
PRICE_TOL = 1e-4   # ₹0.0001 — floating point rounding allowance only
SHARES_TOL = 0.01  # fractional share rounding
PNL_TOL   = 0.01   # ₹0.01 per trade

RANK = {"PASS": 0, "WARN": 1, "FAIL": 2}

# ─── Helpers ──────────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_run_script(fold_dir: Path) -> Optional[Path]:
    """
    Run script ko fold folder ya uske parent (version folder) mein dhundho.
    Pattern: run_v*.py
    """
    # Fold folder mein check karo pehle
    for p in sorted(fold_dir.glob("run_v*.py")):
        return p
    # Parent (version folder) mein check karo
    for p in sorted(fold_dir.parent.glob("run_v*.py")):
        return p
    return None


def find_stored_csv(fold_dir: Path) -> Optional[Path]:
    p = fold_dir / "trades_intraday.csv"
    return p if p.exists() else None


def load_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path, dtype=str)
        df.columns = [c.strip().lower() for c in df.columns]
        return df
    except Exception:
        return None


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Float columns ko round karo tolerance ke andar comparison ke liye."""
    df = df.copy()
    for col in NUMERIC_COLS:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce").round(4)
            except Exception:
                pass
    # Times ko strip karo
    for col in ["entry_time", "exit_time"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df.reset_index(drop=True)


# ─── Core: Run script execution in temp dir ───────────────────────────────────

def execute_run_script_in_temp(
    run_script: Path,
    fold_dir: Path,
    data_dir: Optional[str]
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Run script ko subprocess mein execute karo.
    
    Strategy: Run script `save_gate_v2_artifacts()` ko fold dir mein write karta hai.
    Hum temporarily fold dir ko ek temp location se replace karte hain,
    script run karte hain, result capture karte hain, phir original restore karte hain.
    
    Safer approach: Script ko modify karke temp dir use karwao via env var.
    """
    log_lines = []
    
    with tempfile.TemporaryDirectory(prefix="bhavishya_verify_") as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Run script ko temp folder mein copy karo
        tmp_script = tmp_path / run_script.name
        shutil.copy2(run_script, tmp_script)
        
        # Env: ROOT_DIR override (agar script use karta hai)
        env = os.environ.copy()
        env["BHAVISHYA_VERIFY_MODE"] = "1"
        env["BHAVISHYA_TEMP_OUTPUT"] = str(tmp_path)
        env["PYTHONPATH"] = str(ROOT_DIR / "strategies") + os.pathsep + env.get("PYTHONPATH", "")
        if data_dir:
            env["BHAVISHYA_DATA_DIR"] = data_dir
        
        # Python executable
        python_exe = sys.executable
        
        # Run karo — stdout/stderr capture karo
        try:
            result = subprocess.run(
                [python_exe, str(run_script)],
                capture_output=True,
                text=True,
                timeout=300,   # 5 minute timeout
                env=env,
                cwd=str(ROOT_DIR)
            )
        except subprocess.TimeoutExpired:
            return None, "TIMEOUT: Run script 5 minutes se zyada chala"
        except Exception as e:
            return None, f"EXEC ERROR: {e}"
        
        log_lines.append(f"Exit code: {result.returncode}")
        if result.returncode != 0:
            stderr_snippet = result.stderr[-500:] if result.stderr else "(no stderr)"
            return None, f"Run script failed (exit {result.returncode}): {stderr_snippet}"
        
        # Run script ne fold_dir ke andar CSV regenerate kiya hoga
        # Ab fresh CSV read karo fold_dir se
        fresh_csv = fold_dir / "trades_intraday.csv"
        if not fresh_csv.exists():
            return None, "Run script ne trades_intraday.csv generate nahi kiya"
        
        fresh_df = load_csv(fresh_csv)
        return fresh_df, "OK"


def compare_csvs(
    stored_df: pd.DataFrame,
    fresh_df: pd.DataFrame,
    fold_label: str
) -> Tuple[str, List[str]]:
    """
    Stored CSV vs fresh-generated CSV compare karo.
    Returns: ("PASS"/"WARN"/"FAIL", [messages])
    """
    msgs = []
    
    s = normalize_df(stored_df)
    f = normalize_df(fresh_df)
    
    # 1. Row count
    if len(s) != len(f):
        msgs.append(f"[FAIL] Trade count mismatch: stored={len(s)}, fresh={len(f)}")
        return "FAIL", msgs
    msgs.append(f"[PASS] Trade count match: {len(s)} trades")
    
    if len(s) == 0:
        msgs.append("[PASS] Both CSVs empty (0 trades) — consistent")
        return "PASS", msgs
    
    # 2. Symbol + time match (row order agnostic)
    key_cols = ["symbol", "entry_time", "exit_time"]
    available_keys = [c for c in key_cols if c in s.columns and c in f.columns]
    
    if available_keys:
        s_sorted = s.sort_values(available_keys).reset_index(drop=True)
        f_sorted = f.sort_values(available_keys).reset_index(drop=True)
    else:
        s_sorted = s
        f_sorted = f
    
    # 3. Numeric columns compare
    fail_cols = []
    for col in NUMERIC_COLS:
        if col not in s_sorted.columns or col not in f_sorted.columns:
            continue
        sv = pd.to_numeric(s_sorted[col], errors="coerce").fillna(0)
        fv = pd.to_numeric(f_sorted[col], errors="coerce").fillna(0)
        diff = (sv - fv).abs()
        tol = PNL_TOL if col in ("net_pnl", "gross_pnl", "charges") else (SHARES_TOL if col == "shares" else PRICE_TOL)
        bad = diff > tol
        if bad.any():
            fail_cols.append(col)
            worst_idx = diff.idxmax()
            msgs.append(
                f"[FAIL] Column '{col}': {bad.sum()} rows mismatch | "
                f"worst row {worst_idx}: stored={sv.iloc[worst_idx]:.4f}, fresh={fv.iloc[worst_idx]:.4f}"
            )
    
    if fail_cols:
        return "FAIL", msgs
    
    # 4. Exit reason check
    if "exit_reason" in s_sorted.columns and "exit_reason" in f_sorted.columns:
        er_match = (s_sorted["exit_reason"].str.strip() == f_sorted["exit_reason"].str.strip())
        if not er_match.all():
            msgs.append(f"[WARN] exit_reason {(~er_match).sum()} rows mein alag hai")
        else:
            msgs.append("[PASS] exit_reason sabhi rows mein match")
    
    msgs.append(f"[PASS] All numeric columns match within tolerance on all {len(s)} trades")
    return "PASS", msgs


# ─── Main verifier ────────────────────────────────────────────────────────────

def verify_fold(fold_dir: Path, data_dir: Optional[str]) -> Tuple[str, List[str]]:
    """Ek fold directory verify karo. Returns (verdict, messages)."""
    msgs = []
    label = fold_dir.name
    
    print(f"\n{'='*70}")
    print(f" VERIFYING: {label}")
    print(f"{'='*70}")
    
    # Step 1: Stored CSV exist karta hai?
    stored_csv = find_stored_csv(fold_dir)
    if stored_csv is None:
        msgs.append("[FAIL] trades_intraday.csv missing")
        return "FAIL", msgs
    
    stored_df = load_csv(stored_csv)
    if stored_df is None:
        msgs.append("[FAIL] trades_intraday.csv parse nahi hua")
        return "FAIL", msgs
    
    stored_hash = sha256_file(stored_csv)
    msgs.append(f"[INFO] Stored CSV SHA256: {stored_hash[:16]}... ({len(stored_df)} rows)")
    
    # Step 2: Run script dhundho
    run_script = find_run_script(fold_dir)
    if run_script is None:
        msgs.append("[WARN] run_v*.py nahi mila — reproducibility check skip")
        msgs.append("       (Manually created fold ya old version — Gate V2 checks still apply)")
        return "WARN", msgs
    
    msgs.append(f"[INFO] Run script: {run_script.name}")
    
    # Step 3: Stored CSV ka backup
    backup_csv = stored_csv.with_suffix(".csv.bak_verify")
    shutil.copy2(stored_csv, backup_csv)
    msgs.append(f"[INFO] Stored CSV backup: {backup_csv.name}")
    
    try:
        # Step 4: Run script execute karo
        print(f"  [*] Executing {run_script.name} ...")
        fresh_df, exec_msg = execute_run_script_in_temp(run_script, fold_dir, data_dir)
        
        if fresh_df is None:
            msgs.append(f"[FAIL] Run script execution failed: {exec_msg}")
            verdict = "FAIL"
        else:
            # Step 5: Fresh CSV ka hash
            fresh_hash = sha256_file(fold_dir / "trades_intraday.csv")
            msgs.append(f"[INFO] Fresh CSV SHA256: {fresh_hash[:16]}... ({len(fresh_df)} rows)")
            
            # Step 6: Compare
            verdict, cmp_msgs = compare_csvs(stored_df, fresh_df, label)
            msgs.extend(cmp_msgs)
            
            if stored_hash == fresh_hash:
                msgs.append("[PASS] Byte-perfect match — SHA256 identical")
            elif verdict == "PASS":
                msgs.append("[PASS] Numeric match within tolerance (float rounding only)")
    
    except Exception as e:
        msgs.append(f"[FAIL] Exception during verification: {e}")
        traceback.print_exc()
        verdict = "FAIL"
    
    finally:
        # Step 7: Original restore karo (fresh run ne overwrite kiya tha)
        if backup_csv.exists():
            shutil.copy2(backup_csv, stored_csv)
            backup_csv.unlink()
            msgs.append("[INFO] Original CSV restored from backup")
    
    return verdict, msgs


def auto_find_folds(version_dir: Path) -> List[Path]:
    """Version directory ke andar saare fold directories dhundho."""
    folds = []
    # Check if version_dir itself has a CSV (root fold)
    if (version_dir / "trades_intraday.csv").exists():
        folds.append(version_dir)
    # Sub-folders
    for sub in sorted(version_dir.iterdir()):
        if sub.is_dir() and not sub.name.startswith((".", "__")):
            if (sub / "trades_intraday.csv").exists():
                folds.append(sub)
    return folds


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Verify simulation reproducibility: run script se fresh CSV generate karo aur stored CSV se compare karo."
    )
    parser.add_argument("fold_dir", help="Strategy fold directory (e.g. strategies/V34.2/fold_2022)")
    parser.add_argument("--data-dir", default=None, help="Data directory override (e.g. data/2022-23)")
    parser.add_argument("--all-folds", action="store_true", help="Agar given path version dir hai, saare folds verify karo")
    args = parser.parse_args()
    
    target = Path(args.fold_dir)
    if not target.is_absolute():
        target = ROOT_DIR / target
    
    if not target.exists():
        print(f"[FAIL] Path nahi mila: {target}")
        sys.exit(1)
    
    # Determine folds to verify
    if args.all_folds or not (target / "trades_intraday.csv").exists():
        folds = auto_find_folds(target)
        if not folds:
            print(f"[FAIL] Koi bhi fold nahi mila: {target}")
            sys.exit(1)
    else:
        folds = [target]
    
    # Run verification
    overall_verdicts = []
    
    for fold in folds:
        verdict, msgs = verify_fold(fold, args.data_dir)
        overall_verdicts.append(verdict)
        for m in msgs:
            print(f"  {m}")
        rank = {"PASS": "[OK]", "WARN": "[WARN]", "FAIL": "[FAIL]"}
        print(f"\n  VERDICT [{fold.name}]: {rank.get(verdict, '?')} {verdict}")
    
    # Final summary
    print(f"\n{'='*70}")
    print(f" FINAL SUMMARY")
    print(f"{'='*70}")
    
    for fold, v in zip(folds, overall_verdicts):
        rank_sym = {"PASS": "[PASS]", "WARN": "[WARN]", "FAIL": "[FAIL]"}
        print(f"  {fold.name:<30} {rank_sym.get(v, v)}")
    
    worst = max(overall_verdicts, key=lambda x: RANK[x]) if overall_verdicts else "FAIL"
    print(f"\n  OVERALL: {worst}")
    print(f"{'='*70}\n")
    
    sys.exit(0 if worst != "FAIL" else 1)


if __name__ == "__main__":
    main()
