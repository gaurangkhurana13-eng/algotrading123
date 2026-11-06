"""
Unit & Integration Test Suite for Bhavishya V14
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
from trading_system.V14.src.dhan_engine import calc_dhan_charges
from trading_system.V14.src.backtester import get_tiered_risk_budget
from trading_system.V14.src.pre_trade_mc import PreTradeMonteCarloEngine

def test_v14_dhan_charges():
    charges = calc_dhan_charges(120.0, 125.0, 20)
    assert 1.30 <= charges <= 3.00

def test_v14_tiered_milestone():
    tiers = [
        {"max_capital": 750.0, "risk_budget": 20.0},
        {"max_capital": 1000.0, "risk_budget": 30.0},
        {"max_capital": 999999.0, "risk_budget": 40.0}
    ]
    assert get_tiered_risk_budget(500.0, tiers) == 20.0
    assert get_tiered_risk_budget(749.0, tiers) == 20.0
    assert get_tiered_risk_budget(750.0, tiers) == 30.0
    assert get_tiered_risk_budget(950.0, tiers) == 30.0
    assert get_tiered_risk_budget(1050.0, tiers) == 40.0

def test_v14_mc_engine():
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
        "ticker": "TCS",
        "close": 3000.0,
        "atr_norm": 0.008,
        "prob": 0.80,
        "qty": 1,
        "tp_pct": 0.016,
        "sl_pct": 0.012,
        "hold_bars": 12
    }
    res = mc.evaluate_candidate(snapshot)
    assert res["decision"] in ["ALLOW ENTRY", "REJECT ENTRY"]
