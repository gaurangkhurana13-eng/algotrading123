"""
Comprehensive Forensic Scanner for all subfolders in new_trade/trading_system and real_versions.
Scans every single version, reads trade logs, checks parameters, inspects code for lookahead and charges,
and validates whether reported results are real, overfitted, or biased.
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path("C:/Users/admin/.gemini/antigravity/scratch/new_trade")
TS_DIR = BASE_DIR / "trading_system"
RV_DIR = BASE_DIR / "real_versions"

def scan_version_folder(vdir: Path, name: str):
    info = {
        "version": name,
        "path": str(vdir),
        "has_trades_csv": False,
        "trades_csv_path": None,
        "num_trades": 0,
        "win_rate": None,
        "net_pnl": None,
        "profit_factor": None,
        "avg_charge": None,
        "has_zero_charges": None,
        "has_slippage": None,
        "has_features_code": False,
        "vwap_shifted": None,
        "has_labels_code": False,
        "label_future_trimmed": None,
        "has_strategy_code": False,
        "embargo_present": None,
        "verdict": "UNKNOWN",
        "notes": []
    }

    # Find trades csv
    trades_candidates = list(vdir.glob("**/trades/*.csv")) + list(vdir.glob("**/*trades*.csv")) + list(vdir.glob("*.csv"))
    trades_candidates = [p for p in trades_candidates if "rejected" not in p.name.lower()]
    
    if trades_candidates:
        t_path = trades_candidates[0]
        info["has_trades_csv"] = True
        info["trades_csv_path"] = str(t_path)
        try:
            df = pd.read_csv(t_path)
            info["num_trades"] = len(df)
            if len(df) > 0:
                # Check charges
                if "charges" in df.columns:
                    info["avg_charge"] = round(df["charges"].mean(), 2)
                    info["has_zero_charges"] = bool((df["charges"] == 0).sum() > 0)
                else:
                    info["has_zero_charges"] = True
                    info["notes"].append("No 'charges' column found in trades CSV")

                # Check pnl
                pnl_col = "net_pnl" if "net_pnl" in df.columns else ("pnl" if "pnl" in df.columns else None)
                if pnl_col:
                    tot_pnl = df[pnl_col].sum()
                    info["net_pnl"] = round(tot_pnl, 2)
                    wins = df[df[pnl_col] > 0]
                    losses = df[df[pnl_col] < 0]
                    wr = (len(wins) / len(df)) * 100 if len(df) > 0 else 0
                    info["win_rate"] = round(wr, 1)
                    tot_win = wins[pnl_col].sum()
                    tot_loss = abs(losses[pnl_col].sum())
                    pf = (tot_win / tot_loss) if tot_loss > 0 else (999.0 if tot_win > 0 else 0.0)
                    info["profit_factor"] = round(pf, 2)
                else:
                    info["notes"].append("No PnL column found")

                # Check slippage
                if "entry_price" in df.columns and "raw_price" in df.columns:
                    slip_diff = (df["entry_price"] - df["raw_price"]).abs().mean()
                    info["has_slippage"] = bool(slip_diff > 1e-4)
                elif "slippage" in df.columns:
                    info["has_slippage"] = bool(df["slippage"].mean() > 0)
                else:
                    info["has_slippage"] = "Not explicit in CSV"

        except Exception as e:
            info["notes"].append(f"Error parsing trades CSV: {e}")
    else:
        info["notes"].append("No trades CSV found")

    # Code checks
    feat_files = list(vdir.glob("**/features.py")) + list(vdir.glob("*features*.py"))
    if feat_files:
        info["has_features_code"] = True
        try:
            content = feat_files[0].read_text(encoding="utf-8", errors="ignore")
            info["vwap_shifted"] = bool(".shift(1)" in content and "vwap" in content.lower())
            if not info["vwap_shifted"]:
                info["notes"].append("VWAP is NOT shifted by 1 (Lookahead risk!)")
        except Exception as e:
            pass

    label_files = list(vdir.glob("**/labels.py")) + list(vdir.glob("*labels*.py"))
    if label_files:
        info["has_labels_code"] = True
        try:
            content = label_files[0].read_text(encoding="utf-8", errors="ignore")
            info["label_future_trimmed"] = bool("hold_bars" in content and ("n - hold_bars" in content or "iloc[:" in content))
        except Exception as e:
            pass

    strat_files = list(vdir.glob("**/strategy.py")) + list(vdir.glob("**/backtester.py")) + list(vdir.glob("*strategy*.py"))
    if strat_files:
        info["has_strategy_code"] = True
        try:
            content = ""
            for sf in strat_files:
                content += sf.read_text(encoding="utf-8", errors="ignore") + "\n"
            info["embargo_present"] = bool("embargo" in content.lower())
        except Exception as e:
            pass

    # Determine Verdict
    if not info["has_trades_csv"]:
        info["verdict"] = "NO_TRADES"
    elif info["has_zero_charges"] is True and (info["avg_charge"] is None or info["avg_charge"] < 0.5):
        info["verdict"] = "ZERO_CHARGES_UNREALISTIC"
    elif info["win_rate"] is not None and info["win_rate"] > 88.0 and info["num_trades"] > 20:
        info["verdict"] = "SUSPICIOUS_OVERFIT"
    elif info["vwap_shifted"] is False:
        info["verdict"] = "LOOKAHEAD_BIAS"
    elif info["win_rate"] is not None and info["win_rate"] >= 50.0 and info["net_pnl"] is not None and info["net_pnl"] > 0:
        info["verdict"] = "GENUINE_PROFITABLE"
    elif info["net_pnl"] is not None and info["net_pnl"] <= 0:
        info["verdict"] = "GENUINE_LOSS"
    else:
        info["verdict"] = "MIXED/INCOMPLETE"

    return info

def main():
    all_results = []
    
    # 1. Scan trading_system
    if TS_DIR.exists():
        subdirs = sorted([d for d in TS_DIR.iterdir() if d.is_dir() and not d.name.startswith(".") and d.name != "__pycache__"])
        for sd in subdirs:
            res = scan_version_folder(sd, f"TS/{sd.name}")
            all_results.append(res)

    # 2. Scan real_versions
    if RV_DIR.exists():
        subdirs = sorted([d for d in RV_DIR.iterdir() if d.is_dir() and not d.name.startswith(".") and d.name != "__pycache__"])
        for sd in subdirs:
            res = scan_version_folder(sd, f"RV/{sd.name}")
            all_results.append(res)

    # Output table
    print(f"{'Version':<18} | {'Trades':<6} | {'WR %':<6} | {'Net PnL':<10} | {'PF':<6} | {'Avg Chg':<7} | {'VWAP Shift':<10} | {'Verdict'}")
    print("-" * 95)
    for r in all_results:
        wr_str = f"{r['win_rate']}%" if r['win_rate'] is not None else "N/A"
        pnl_str = f"Rs {r['net_pnl']}" if r['net_pnl'] is not None else "N/A"
        pf_str = f"{r['profit_factor']}" if r['profit_factor'] is not None else "N/A"
        chg_str = f"Rs {r['avg_charge']}" if r['avg_charge'] is not None else "N/A"
        vwap_str = str(r['vwap_shifted']) if r['vwap_shifted'] is not None else "N/A"
        print(f"{r['version']:<18} | {r['num_trades']:<6} | {wr_str:<6} | {pnl_str:<10} | {pf_str:<6} | {chg_str:<7} | {vwap_str:<10} | {r['verdict']}")

    # Save to json for deep inspection
    out_json = BASE_DIR / "full_version_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved full detailed audit to {out_json}")

if __name__ == "__main__":
    main()
