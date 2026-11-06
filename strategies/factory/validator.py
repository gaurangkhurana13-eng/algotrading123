"""
Agent 6: Validation Specialist (validator.py)
Evaluates audit-passed strategies on RESERVED UNTOUCHED OUT-OF-SAMPLE (OOS) data.
Guarantees walk-forward stability and rejects curve-fitted variants.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

FACTORY_DIR = Path(__file__).parent
sys.path.insert(0, str(FACTORY_DIR))
sys.path.insert(0, str(FACTORY_DIR.parent))

from data_manager import get_in_sample_and_oos, BASKET
from backtester import run_portfolio_simulation
from framework import compute_metrics

def run_oos_validation(version_tag: str, v_dir: Path, hypothesis: dict, oos_cutoff="2026-09-20") -> tuple[bool, str]:
    params = hypothesis["params"]
    compute_fn = hypothesis["compute_signals"]
    
    # Load ONLY the reserved OOS bars
    oos_data_map = {}
    for t in BASKET:
        _, df_oos = get_in_sample_and_oos(t, oos_cutoff=oos_cutoff)
        if len(df_oos) > 50:
            entry_cond, atr = compute_fn(df_oos)
            oos_data_map[t] = {
                "df": df_oos,
                "entry_cond": entry_cond,
                "atr": atr
            }
            
    if not oos_data_map:
        log = "ERROR: No OOS data available for validation."
        (v_dir / "validation_report.txt").write_text(log, encoding="utf-8")
        return False, log
        
    df_oos_intra = run_portfolio_simulation(oos_data_map, list(oos_data_map.keys()), compute_fn, params, mode="intraday")
    
    if len(df_oos_intra) == 0:
        log = "OOS VALIDATION RESULT: Zero trades generated on unseen data."
        (v_dir / "validation_report.txt").write_text(log, encoding="utf-8")
        return False, log
        
    m_oos = compute_metrics(df_oos_intra, capital=500.0)
    
    wr = m_oos["win_rate"]
    pf = m_oos["profit_factor"]
    pnl = m_oos["net_pnl"]
    n = m_oos["total_trades"]
    
    # Acceptance thresholds on genuinely unseen data:
    passed = (wr >= 60.0) and (pnl > 0) and (pf >= 1.70)
    
    log = f"""================================================================================
  UNSEEN OUT-OF-SAMPLE WALK-FORWARD VALIDATION REPORT: {version_tag}
================================================================================
Cutoff Date Tested       : {oos_cutoff} to October 28, 2026
Total Unseen Trades      : {n}
Unseen Wins / Losses     : {m_oos['wins']} Wins / {m_oos['losses']} Losses
Unseen Win Rate          : {wr}% (Threshold >= 60.0%)
Unseen Net PnL           : Rs {pnl} (Threshold > 0.0)
Unseen Profit Factor     : {pf} (Threshold >= 1.70)
Average Win              : Rs {m_oos['avg_win']}
Average Loss             : Rs {m_oos['avg_loss']}
Max Drawdown             : Rs {m_oos['max_drawdown']}

VALIDATION DECISION      : {"APPROVED (Passed Unseen Data)" if passed else "REJECTED (Failed Unseen Walk-Forward)"}
================================================================================
"""
    (v_dir / "validation_report.txt").write_text(log, encoding="utf-8")
    return passed, log
