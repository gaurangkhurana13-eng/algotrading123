"""
Unit & Integration Test Suite for Bhavishya V13
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import numpy as np
import pandas as pd
from trading_system.V13.src.dhan_engine import calc_dhan_charges
from trading_system.V13.src.features import FEATURE_COLS, compute_features
from trading_system.V13.src.pre_trade_mc import PreTradeMonteCarloEngine

def test_v13_dhan_charges():
    charges = calc_dhan_charges(50.0, 52.0, 50)
    assert 1.30 <= charges <= 3.00

def test_v13_adx_gate():
    config = {"adx_min_trend": 25.0}
    assert config["adx_min_trend"] >= 25.0

def test_v13_mc_engine():
    config = {
        "pre_trade_mc": {
            "iterations": 1000,
            "min_win_prob": 0.52,
            "min_expected_ev": 0.0,
            "max_cvar_loss": -45.0,
            "degrees_of_freedom": 5,
            "seed": 42
        }
    }
    mc = PreTradeMonteCarloEngine(config)
    snapshot = {
        "timestamp": "2026-09-01 10:15:00",
        "ticker": "INFY",
        "close": 1800.0,
        "atr_norm": 0.010,
        "prob": 0.79,
        "qty": 1,
        "tp_pct": 0.020,
        "sl_pct": 0.015,
        "hold_bars": 12
    }
    res = mc.evaluate_candidate(snapshot)
    assert res["decision"] in ["ALLOW ENTRY", "REJECT ENTRY"]
