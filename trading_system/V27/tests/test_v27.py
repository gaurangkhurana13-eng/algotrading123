"""
Unit tests for Bhavishya V27: Two-Tier Execution & Uncapped Post-MC Architecture
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
import numpy as np
import pandas as pd
from trading_system.V27.src.labels import apply_triple_barrier_labels
from trading_system.V27.src.pre_trade_mc import PreTradeMonteCarloEngine
from trading_system.V27.src.dhan_engine import calc_dhan_charges
from trading_system.V27.src.backtester import compute_half_kelly_budget

def test_v27_labels_regime_targets_and_invariant_sl():
    n = 60
    np.random.seed(42)
    close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "close": close,
        "high": close + 0.6,
        "low": close - 0.6,
        "atr_norm": np.full(n, 0.01),
        "atr_norm_z": np.full(n, -1.0), # Compression
        "squeeze_on": np.ones(n)
    })
    labeled = apply_triple_barrier_labels(df, hold_bars=5)
    assert "target_pct" in labeled.columns
    assert "sl_pct" in labeled.columns
    # Squeeze compression expands target to 2.4 * ATR
    assert np.isclose(labeled["target_pct"].iloc[0], 0.024)
    # Stop loss remains invariant 1.5 * ATR
    assert np.isclose(labeled["sl_pct"].iloc[0], 0.015)

def test_v27_half_kelly():
    b1 = compute_half_kelly_budget(0.55, 20.0, 1.333, 18.0, 35.0)
    b2 = compute_half_kelly_budget(0.46, 20.0, 1.333, 18.0, 35.0)
    assert b1 > b2
    assert b1 <= 35.0
    assert b2 >= 18.0

def test_v27_pre_trade_mc():
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
        "timestamp": "2026-09-29 11:45:00",
        "ticker": "LLOYDSENGG",
        "close": 100.0,
        "atr_norm": 0.015,
        "prob": 0.55,
        "qty": 15,
        "tp_pct": 0.036,
        "sl_pct": 0.0225,
        "hold_bars": 12
    }
    res = mc.evaluate_candidate(snapshot)
    assert res["decision"] in ["ALLOW ENTRY", "REJECT"]
    assert "win_prob" in res["stats"]

def test_v27_dhan_charges():
    c = calc_dhan_charges(100.0, 103.0, 20)
    assert c > 0.0
    assert c < 5.0
