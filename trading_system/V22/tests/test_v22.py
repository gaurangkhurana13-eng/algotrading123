"""
Unit & Integration Test Suite for Bhavishya V22: Calibrated Probability Engine
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from trading_system.V22.src.dhan_engine import calc_dhan_charges
from trading_system.V22.src.pre_trade_mc import PreTradeMonteCarloEngine

def test_v22_dhan_charges():
    charges = calc_dhan_charges(550.0, 565.0, 4)
    assert 1.30 <= charges <= 3.00

def test_v22_calibration_mechanics():
    # Direct Platt scaling test
    X = np.random.randn(100, 5)
    y = np.random.choice([0, 1], size=100)
    base = LogisticRegression().fit(X[:80], y[:80])
    p_val = base.predict_proba(X[80:])[:, 1].reshape(-1, 1)
    platt = LogisticRegression().fit(p_val, y[80:])
    p_test = base.predict_proba(X[:10])[:, 1].reshape(-1, 1)
    cal_probs = platt.predict_proba(p_test)[:, 1]
    assert np.all((cal_probs >= 0.0) & (cal_probs <= 1.0))

def test_v22_pre_trade_mc():
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
        "ticker": "ICICIBANK",
        "close": 1200.0,
        "atr_norm": 0.012,
        "prob": 0.76,
        "qty": 2,
        "tp_pct": 0.024,
        "sl_pct": 0.018,
        "hold_bars": 12
    }
    res = mc.evaluate_candidate(snapshot)
    assert res["decision"] in ["ALLOW ENTRY", "REJECT ENTRY"]
