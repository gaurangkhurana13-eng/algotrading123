"""
Unit tests for Bhavishya V28:
Multi-Horizon Squeeze, Conviction-Gated Target Expansion & Bimodal Asymmetric Risk Sizing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pytest
import numpy as np
import pandas as pd
from trading_system.V28.src.features import compute_features, FEATURE_COLS
from trading_system.V28.src.labels import apply_triple_barrier_labels
from trading_system.V28.src.pre_trade_mc import PreTradeMonteCarloEngine
from trading_system.V28.src.dhan_engine import calc_dhan_charges
from trading_system.V28.src.backtester import compute_tier1a_budget

def test_v28_multi_horizon_squeeze():
    n = 60
    np.random.seed(42)
    c = 100.0 + np.cumsum(np.random.randn(n) * 0.1) # very tight coil
    df = pd.DataFrame({
        "open": c, "high": c + 0.1, "low": c - 0.1, "close": c,
        "volume": np.full(n, 100000.0)
    }, index=pd.date_range("2026-09-01 09:15", periods=n, freq="15min"))
    feats = compute_features(df)
    assert "squeeze_fast" in feats.columns
    assert "squeeze_std" in feats.columns
    assert "squeeze_ratio" in feats.columns
    assert "adx_slope_3" in feats.columns
    for col in FEATURE_COLS:
        assert col in feats.columns

def test_v28_bimodal_sizing():
    # Tier 1A Elite (P >= 0.55): budget between 28 and 35
    b_elite = compute_tier1a_budget(0.56, 32.0, 1.333, 28.0, 35.0)
    assert b_elite >= 28.0
    assert b_elite <= 35.0

def test_v28_invariant_sl():
    n = 50
    close = np.full(n, 100.0)
    df = pd.DataFrame({
        "close": close, "high": close + 0.5, "low": close - 0.5,
        "atr_norm": np.full(n, 0.01), "atr_norm_z": np.zeros(n),
        "squeeze_on": np.zeros(n)
    })
    labeled = apply_triple_barrier_labels(df, hold_bars=5)
    # Check invariant 1.5 * ATR stop loss
    assert np.isclose(labeled["sl_pct"].iloc[0], 0.015)

def test_v28_dhan_brokerage():
    c = calc_dhan_charges(150.0, 155.0, 10)
    assert c > 0.0
    assert c < 5.0
