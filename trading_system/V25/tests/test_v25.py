"""
Unit tests for Bhavishya V25: Asymmetric Quadratic MFE Ratchet & Continuous Stagnation Time-Decay Engine
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
import numpy as np
import pandas as pd
from trading_system.V25.src.labels import apply_triple_barrier_labels
from trading_system.V25.src.nifty_filter import compute_nifty_regime
from trading_system.V25.src.pre_trade_mc import PreTradeMonteCarloEngine
from trading_system.V25.src.dhan_engine import calc_dhan_charges

def test_quadratic_mfe_labels():
    n = 60
    np.random.seed(42)
    close = 200.0 + np.cumsum(np.random.randn(n) * 0.4)
    df = pd.DataFrame({
        "close": close,
        "high": close + 0.6,
        "low": close - 0.4,
        "atr_norm": np.full(n, 0.008)
    })
    labeled = apply_triple_barrier_labels(df, hold_bars=10)
    assert "target_pct" in labeled.columns
    assert "realized_pct" in labeled.columns
    assert "y_buy" in labeled.columns
    assert len(labeled) == 50

def test_cvar_scale_factor():
    config = {
        "pre_trade_mc": {
            "iterations": 500,
            "min_win_prob": 0.45,
            "min_expected_ev": 0.0,
            "max_cvar_loss": -45.0,
            "degrees_of_freedom": 5,
            "seed": 42
        }
    }
    mc = PreTradeMonteCarloEngine(config)
    snapshot = {
        "timestamp": "2026-02-15 10:15:00",
        "ticker": "COALINDIA",
        "close": 400.0,
        "atr_norm": 0.010,
        "prob": 0.48,
        "qty": 5,
        "tp_pct": 0.024,
        "sl_pct": 0.014,
        "hold_bars": 12
    }
    res = mc.evaluate_candidate(snapshot)
    assert res["decision"] in ["ALLOW ENTRY", "REJECT"]
    assert "scale_factor" in res
    assert 0.35 <= res["scale_factor"] <= 1.0

def test_dhan_brokerage_v25():
    charges = calc_dhan_charges(350.0, 360.0, 7)
    assert charges > 0.0
    assert charges < 5.0
