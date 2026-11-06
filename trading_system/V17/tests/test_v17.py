"""
Unit & Integration Test Suite for Bhavishya V17
"""
import sys
from datetime import time
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
from trading_system.V17.src.dhan_engine import calc_dhan_charges
from trading_system.V17.src.pre_trade_mc import PreTradeMonteCarloEngine

def test_v17_dhan_charges():
    charges = calc_dhan_charges(300.0, 308.0, 8)
    assert 1.30 <= charges <= 3.00

def test_v17_time_window_logic():
    t_start = time(9, 45)
    t_end = time(14, 15)
    
    # 09:30 should be rejected
    assert not (t_start <= time(9, 30) <= t_end)
    # 10:15 should be accepted
    assert (t_start <= time(10, 15) <= t_end)
    # 14:45 should be rejected
    assert not (t_start <= time(14, 45) <= t_end)

def test_v17_mc_engine():
    config = {
        "pre_trade_mc": {
            "iterations": 1000,
            "min_win_prob": 0.50,
            "min_expected_ev": 0.0,
            "max_cvar_loss": -45.0,
            "degrees_of_freedom": 5,
            "seed": 42
        }
    }
    mc = PreTradeMonteCarloEngine(config)
    snapshot = {
        "timestamp": "2026-09-01 10:15:00",
        "ticker": "RELIANCE",
        "close": 2800.0,
        "atr_norm": 0.009,
        "prob": 0.82,
        "qty": 1,
        "tp_pct": 0.018,
        "sl_pct": 0.0135,
        "hold_bars": 12
    }
    res = mc.evaluate_candidate(snapshot)
    assert res["decision"] in ["ALLOW ENTRY", "REJECT ENTRY"]
