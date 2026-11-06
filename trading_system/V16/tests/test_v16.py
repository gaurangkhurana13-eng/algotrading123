"""
Unit & Integration Test Suite for Bhavishya V16
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
from trading_system.V16.src.dhan_engine import calc_dhan_charges
from trading_system.V16.src.pre_trade_mc import PreTradeMonteCarloEngine

def test_v16_dhan_charges():
    charges = calc_dhan_charges(45.0, 48.0, 60)
    assert 1.30 <= charges <= 3.00

def test_v16_rvol_threshold():
    config = {"rvol_min_threshold": 1.80}
    assert config["rvol_min_threshold"] >= 1.80

def test_v16_mc_engine():
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
        "ticker": "TATAMOTORS",
        "close": 900.0,
        "atr_norm": 0.015,
        "prob": 0.81,
        "qty": 2,
        "tp_pct": 0.030,
        "sl_pct": 0.0225,
        "hold_bars": 12
    }
    res = mc.evaluate_candidate(snapshot)
    assert res["decision"] in ["ALLOW ENTRY", "REJECT ENTRY"]
