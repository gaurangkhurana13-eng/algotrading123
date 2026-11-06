"""
Orchestrator & Factory Manager (orchestrator.py)
Master State Machine managing the 8 Specialized Agents:
1. Alpha Researcher
2. Data & Universe Engineer
3. Strategy Developer
4. Backtest Engineer
5. Independent Auditor (strategy_evaluate_gate.py)
6. Validation Specialist (Unseen Walk-Forward)
7. Risk & Cost Analyst
8. Release & Documentation Agent

State Machine: QUEUED -> RUNNING -> AUDITING -> VALIDATING -> APPROVED / REJECTED
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

FACTORY_DIR = Path(__file__).parent
STRATEGIES_DIR = FACTORY_DIR.parent
sys.path.insert(0, str(FACTORY_DIR))
sys.path.insert(0, str(STRATEGIES_DIR))

from alpha_library import HYPOTHESES
from data_manager import create_data_manifest
from strategy_builder import build_strategy_version_files
from auditor import run_independent_audit
from validator import run_oos_validation
from releaser import check_risk_and_costs, create_run_manifest, update_master_readme

STATE_FILE = STRATEGIES_DIR / "factory_state.json"

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "current_candidate": "V7",
        "state": "IDLE",
        "approved": ["V5", "V6"],
        "rejected": [],
        "retries": {}
    }

def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")

def process_candidate(version_tag: str) -> bool:
    print("\n" + "=" * 80)
    print(f"  FACTORY ORCHESTRATOR: PROCESSING CANDIDATE {version_tag}")
    print("=" * 80)
    
    if version_tag not in HYPOTHESES:
        print(f"No hypothesis registered for {version_tag}. Halting.")
        return False
        
    hypothesis = HYPOTHESES[version_tag]
    v_dir = STRATEGIES_DIR / version_tag
    v_dir.mkdir(parents=True, exist_ok=True)
    
    # ── STAGE 1 & 2: Freeze Data Manifest ────────────────────────
    print(f"[Stage 1 & 2] Freezing data & creating manifest for {version_tag}...")
    create_data_manifest(v_dir)
    
    # ── STAGE 3: Build Strategy Files (Python + Pine v6) ───────────
    print(f"[Stage 3] Strategy Developer building Python & Pine Script v6...")
    build_strategy_version_files(v_dir, version_tag, hypothesis)
    
    # ── STAGE 4: Execute Backtests ────────────────────────────────
    print(f"[Stage 4] Backtest Engineer executing 15m simulations (MIS + Swing)...")
    runner_script = v_dir / f"run_{version_tag.lower()}.py"
    res = subprocess.run([sys.executable, str(runner_script)], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Execution Error in {runner_script.name}:\n{res.stderr}")
        return False
        
    # Check if trades were produced
    intra_csv = v_dir / "trades_intraday.csv"
    if not intra_csv.exists() or len(pd.read_csv(intra_csv)) == 0:
        print("Zero trades generated. Rejecting candidate.")
        create_run_manifest(version_tag, v_dir, status="REJECTED_ZERO_TRADES")
        return False
        
    # ── STAGE 5: Independent Sanity Gate ─────────────────────────
    print(f"[Stage 5] Independent Auditor running strategy_evaluate_gate.py...")
    audit_passed, audit_log = run_independent_audit(version_tag, v_dir)
    if not audit_passed:
        print(f"AUDIT GATE FAILED for {version_tag}!")
        print(audit_log[-1200:])
        create_run_manifest(version_tag, v_dir, status="REJECTED_AUDIT_FAIL")
        return False
    print("Audit Gate PASSED (100% Genuine, 0 Failures).")
    
    # ── STAGE 6: Reserved Unseen OOS Validation ───────────────────
    print(f"[Stage 6] Validation Specialist running Reserved Unseen OOS Walk-Forward...")
    val_passed, val_log = run_oos_validation(version_tag, v_dir, hypothesis)
    if not val_passed:
        print(f"OOS VALIDATION FAILED for {version_tag} (Curve-fitting detected)!")
        print(val_log)
        create_run_manifest(version_tag, v_dir, status="REJECTED_OOS_FAIL")
        return False
    print("Unseen OOS Walk-Forward Validation PASSED.")
    
    # ── STAGE 7: Risk Check & Release Documentation ───────────────
    print(f"[Stage 7] Risk Analyst & Release Documenter finalizing...")
    risk_ok, risk_msg = check_risk_and_costs(v_dir)
    if not risk_ok:
        print(f"Risk Check Failed: {risk_msg}")
        create_run_manifest(version_tag, v_dir, status="REJECTED_RISK_FAIL")
        return False
        
    create_run_manifest(version_tag, v_dir, status="APPROVED")
    update_master_readme(version_tag, v_dir, hypothesis)
    print(f"CANDIDATE {version_tag} OFFICIALLY APPROVED & SIGNED OFF!")
    return True

if __name__ == "__main__":
    import pandas as pd
    v = sys.argv[1] if len(sys.argv) > 1 else "V7"
    success = process_candidate(v)
    sys.exit(0 if success else 1)
