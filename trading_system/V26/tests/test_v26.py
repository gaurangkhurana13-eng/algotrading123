"""
Unit tests for Bhavishya V26: In-Loop Event Queueing & Post-MC Daily Slot Execution
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
import numpy as np
import pandas as pd
from trading_system.V26.src.labels import apply_triple_barrier_labels
from trading_system.V26.src.pre_trade_mc import PreTradeMonteCarloEngine
from trading_system.V26.src.dhan_engine import calc_dhan_charges

def test_v26_labels_invariant_sl():
    n = 50
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "close": close,
        "high": close + 0.5,
        "low": close - 0.5,
        "atr_norm": np.full(n, 0.01),
        "atr_norm_z": np.full(n, 0.0),
        "squeeze_on": np.zeros(n)
    })
    labeled = apply_triple_barrier_labels(df, hold_bars=5)
    assert "target_pct" in labeled.columns
    assert "sl_pct" in labeled.columns
    # Check invariant 1.5 * ATR stop loss
    assert np.isclose(labeled["sl_pct"].iloc[0], 0.015)

def test_v26_pre_trade_mc():
    config = {
        "pre_trade_mc": {
            "iterations": 500,
            "min_win_prob": 0.50,
            "min_expected_ev": 0.0,
            "max_cvar_loss": -45.0,
            "degrees_of_freedom": 5,
            "seed": 42
        }
    }
    mc = PreTradeMonteCarloEngine(config)
    snapshot = {
        "timestamp": "2026-02-15 10:15:00",
        "ticker": "INFY",
        "close": 1500.0,
        "atr_norm": 0.012,
        "prob": 0.52,
        "qty": 3,
        "tp_pct": 0.024,
        "sl_pct": 0.018,
        "hold_bars": 12
    }
    res = mc.evaluate_candidate(snapshot)
    assert res["decision"] in ["ALLOW ENTRY", "REJECT"]

def test_v26_dhan_brokerage():
    charges = calc_dhan_charges(500.0, 520.0, 10)
    assert charges > 0.0
    assert charges < 10.0
