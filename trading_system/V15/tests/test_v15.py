"""
Unit & Integration Test Suite for Bhavishya V15
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
from trading_system.V15.src.dhan_engine import calc_dhan_charges
from trading_system.V15.src.pre_trade_mc import PreTradeMonteCarloEngine

def test_v15_dhan_charges():
    charges = calc_dhan_charges(200.0, 206.0, 10)
    assert 1.30 <= charges <= 3.00

def test_v15_scale_out_logic():
    qty = 10
    qty_1 = max(1, qty // 2)
    qty_2 = qty - qty_1
    assert qty_1 == 5
    assert qty_2 == 5

def test_v15_mc_engine():
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
        "ticker": "WIPRO",
        "close": 500.0,
        "atr_norm": 0.012,
        "prob": 0.78,
        "qty": 4,
        "tp_pct": 0.0192,
        "sl_pct": 0.0180,
        "hold_bars": 12
    }
    res = mc.evaluate_candidate(snapshot)
    assert res["decision"] in ["ALLOW ENTRY", "REJECT ENTRY"]
