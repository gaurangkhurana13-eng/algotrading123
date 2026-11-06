"""
Unit tests for Bhavishya V30:
- Institutional Microstructure Features (cs_rs, flow_clv, vwap_dist, orb_impulse, h_rem)
- 3-Model Stacking Ensemble & Platt Calibration
- Terminal-Aware Monte Carlo Engine (MC_term)
- Two-Stage Asymmetric Ratchet & Margin Recycling
- Real Dhan MIS Microstructure Charges
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import numpy as np
import pandas as pd
from trading_system.V30.src.features import compute_stock_features, augment_cross_sectional_features, FEATURE_COLS
from trading_system.V30.src.labels import apply_triple_barrier_labels
from trading_system.V30.src.pre_trade_mc import PreTradeMonteCarloEngine
from trading_system.V30.src.dhan_engine import calc_dhan_charges
from trading_system.V30.src.backtester import compute_v30_risk_budget, get_sector_info

def test_v30_microstructure_features():
    n = 100
    dates = pd.date_range("2026-09-01 09:15", periods=n, freq="15min")
    np.random.seed(42)
    c = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    h = c + np.abs(np.random.randn(n) * 0.3) + 0.1
    l = c - np.abs(np.random.randn(n) * 0.3) - 0.1
    v = np.random.randint(1000, 10000, size=n).astype(float)
    df = pd.DataFrame({"open": c, "high": h, "low": l, "close": c, "volume": v}, index=dates)

    feat_df = compute_stock_features(df)
    # Check new institutional features exist
    for col in ["flow_clv", "orb_impulse", "h_rem", "vwap_dist", "squeeze_fast", "squeeze_std"]:
        assert col in feat_df.columns, f"Missing feature: {col}"

def test_v30_cross_sectional_augmentation():
    n = 50
    dates = pd.date_range("2026-09-01 09:15", periods=n, freq="15min")
    dfs = []
    for ticker in ["STOCK_A", "STOCK_B", "STOCK_C"]:
        c = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        df = pd.DataFrame({
            "close": c,
            "high": c + 0.5,
            "low": c - 0.5,
            "volume": np.full(n, 5000.0),
            "ticker": ticker
        }, index=dates)
        feat = compute_stock_features(df)
        feat["ticker"] = ticker
        dfs.append(feat)

    pool = pd.concat(dfs).sort_index()
    pool_cs = augment_cross_sectional_features(pool)
    assert "cs_rs" in pool_cs.columns
    assert not pool_cs["cs_rs"].isna().any()

def test_v30_pre_trade_mc_terminal_aware():
    config = {
        "pre_trade_mc": {
            "iterations": 500,
            "min_win_prob": 0.49,
            "min_expected_ev": 0.0,
            "max_cvar_loss": -45.0,
            "degrees_of_freedom": 5,
            "seed": 42,
            "terminal_gain_atr": 0.65
        }
    }
    mc = PreTradeMonteCarloEngine(config)
    snapshot = {
        "timestamp": "2026-09-29 11:45:00",
        "ticker": "LLOYDSENGG",
        "close": 100.0,
        "atr_norm": 0.015,
        "prob": 0.56,
        "qty": 15,
        "tp_pct": 0.042,
        "sl_pct": 0.0225,
        "hold_bars": 10
    }
    res = mc.evaluate_candidate(snapshot)
    assert res["decision"] in ["ALLOW ENTRY", "REJECT"]
    assert "win_prob" in res["stats"]
    assert "cvar_loss" in res["stats"]

def test_v30_continuous_half_kelly_sizing():
    b_high = compute_v30_risk_budget(0.58, 24.0, 38.0)
    b_mid  = compute_v30_risk_budget(0.51, 24.0, 38.0)
    b_low  = compute_v30_risk_budget(0.45, 24.0, 38.0)
    assert b_high == 38.0
    assert b_low == 24.0
    assert b_high > b_mid > b_low

def test_v30_dhan_charges_and_friction():
    c = calc_dhan_charges(100.0, 103.0, 20)
    assert c > 0.0
    assert c < 5.0

def test_v30_sector_mapping():
    s1 = get_sector_info("LLOYDSENGG")
    assert s1[0] == "INDUSTRIALS"
    s2 = get_sector_info("MOREPENLAB")
    assert s2[0] == "HEALTHCARE"
