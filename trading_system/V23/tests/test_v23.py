"""
Unit tests for Bhavishya V23: Pre-Trade Monte Carlo Drift Shrinkage & Microstructure Friction Penalty
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
import numpy as np
from trading_system.V23.src.pre_trade_mc import PreTradeMonteCarloEngine
from trading_system.V23.src.dhan_engine import calc_dhan_charges

def test_pre_trade_mc_shrinkage_calculation():
    config = {
        "pre_trade_mc": {
            "iterations": 1000,
            "min_win_prob": 0.50,
            "min_expected_ev": 0.0,
            "max_cvar_loss": -45.0,
            "degrees_of_freedom": 5,
            "drift_shrinkage_lambda": 0.65,
            "spread_drag_per_bar": 0.0002,
            "prior_base_rate": 0.274,
            "seed": 42
        }
    }
    mc = PreTradeMonteCarloEngine(config)
    assert mc.drift_shrinkage == 0.65
    assert mc.spread_drag_per_bar == 0.0002
    assert mc.prior_base_rate == 0.274

    snapshot = {
        "timestamp": "2026-02-15 10:15:00",
        "ticker": "RELIANCE",
        "close": 2500.0,
        "atr_norm": 0.012,
        "prob": 0.48,
        "qty": 2,
        "tp_pct": 0.024,
        "sl_pct": 0.018,
        "hold_bars": 12
    }

    eval_res = mc.evaluate_candidate(snapshot)
    assert "decision" in eval_res
    assert eval_res["decision"] in ["ALLOW ENTRY", "REJECT", "INSUFFICIENT EVIDENCE"]
    assert "shrunk_drift" in eval_res["stats"]
    assert "spread_drag" in eval_res["stats"]
    assert eval_res["stats"]["spread_drag"] == 0.0002

def test_insufficient_evidence():
    config = {"pre_trade_mc": {}}
    mc = PreTradeMonteCarloEngine(config)
    res = mc.evaluate_candidate({"ticker": "TCS"})
    assert res["decision"] == "INSUFFICIENT EVIDENCE"

def test_dhan_charges():
    c = calc_dhan_charges(100.0, 105.0, 10)
    assert c > 0.0
    assert c < 5.0
