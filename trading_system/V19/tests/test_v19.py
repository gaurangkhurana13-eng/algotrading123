"""
Unit & Integration Test Suite for Bhavishya V19
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
from trading_system.V19.src.dhan_engine import calc_dhan_charges
from trading_system.V19.src.backtester import compute_half_kelly_budget
from trading_system.V19.src.pre_trade_mc import PreTradeMonteCarloEngine

def test_v19_dhan_charges():
    charges = calc_dhan_charges(500.0, 515.0, 4)
    assert 1.30 <= charges <= 3.00

def test_v19_half_kelly_budget():
    b_75 = compute_half_kelly_budget(0.75, base_budget=20.0)
    b_85 = compute_half_kelly_budget(0.85, base_budget=20.0)
    assert 18.0 <= b_75 <= 35.0
    assert 18.0 <= b_85 <= 35.0
    assert b_85 > b_75, f"Higher probability should yield higher budget: {b_85} vs {b_75}"

def test_v19_mc_engine():
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
        "ticker": "HDFCBANK",
        "close": 1650.0,
        "atr_norm": 0.010,
        "prob": 0.83,
        "qty": 1,
        "tp_pct": 0.020,
        "sl_pct": 0.015,
        "hold_bars": 12
    }
    res = mc.evaluate_candidate(snapshot)
    assert res["decision"] in ["ALLOW ENTRY", "REJECT ENTRY"]
