"""
Unit tests for Bhavishya V29:
Spatio-Temporal Horizon Protection, Dynamic First-Passage Scaling & Margin-Synchronized Half-Kelly.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
import numpy as np
import pandas as pd
from trading_system.V29.src.backtester import compute_continuous_half_kelly
from trading_system.V29.src.dhan_engine import calc_dhan_charges
from trading_system.V29.src.pre_trade_mc import PreTradeMonteCarloEngine

def test_v29_continuous_half_kelly():
    # P_cal = 0.48 -> min budget 22.0
    b_min = compute_continuous_half_kelly(0.48, 22.0, 35.0)
    assert np.isclose(b_min, 22.0)

    # P_cal = 0.58 -> max budget 35.0
    b_max = compute_continuous_half_kelly(0.58, 22.0, 35.0)
    assert np.isclose(b_max, 35.0)

    # P_cal = 0.53 -> intermediate budget ~28.5
    b_mid = compute_continuous_half_kelly(0.53, 22.0, 35.0)
    assert b_mid > 25.0 and b_mid < 32.0

def test_v29_horizon_truncated_mc():
    config = {
        "pre_trade_mc": {
            "iterations": 500,
            "min_win_prob": 0.52,
            "min_expected_ev": 0.0,
            "max_cvar_loss": -45.0,
            "degrees_of_freedom": 5,
            "seed": 42
        }
    }
    mc = PreTradeMonteCarloEngine(config)
    snapshot = {
        "timestamp": "2026-09-29 11:45:00",
        "ticker": "LLOYDSENGG",
        "close": 100.0,
        "atr_norm": 0.015,
        "prob": 0.55,
        "qty": 15,
        "tp_pct": 0.035, # 2.35 * atr
        "sl_pct": 0.0225, # 1.50 * atr
        "hold_bars": 8 # Horizon truncated
    }
    res = mc.evaluate_candidate(snapshot)
    assert res["decision"] in ["ALLOW ENTRY", "REJECT"]
    assert "win_prob" in res["stats"]

def test_v29_dhan_brokerage():
    c = calc_dhan_charges(100.0, 104.0, 20)
    assert c > 0.0
    assert c < 5.0
