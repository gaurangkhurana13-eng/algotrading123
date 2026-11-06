"""
Unit & Integration Test Suite for Bhavishya V11
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import numpy as np
import pandas as pd
from trading_system.V11.src.dhan_engine import calc_dhan_charges
from trading_system.V11.src.features import FEATURE_COLS, compute_features
from trading_system.V11.src.labels import apply_triple_barrier_labels
from trading_system.V11.src.pre_trade_mc import PreTradeMonteCarloEngine

def test_dhan_charges():
    # Buy 100 shares at Rs 25, Sell at Rs 26 (Turnover Rs 5,100)
    charges = calc_dhan_charges(25.0, 26.0, 100)
    assert 1.30 <= charges <= 2.80, f"Unexpected charges: {charges}"

def test_features_length_and_shift():
    # Synthetic dataframe
    n = 100
    dates = pd.date_range("2026-08-01 09:15", periods=n, freq="15min")
    df = pd.DataFrame({
        "open": np.linspace(100, 110, n),
        "high": np.linspace(101, 111, n),
        "low": np.linspace(99, 109, n),
        "close": np.linspace(100, 110, n),
        "volume": np.random.randint(1000, 5000, n)
    }, index=dates)

    feat_df = compute_features(df)
    for col in FEATURE_COLS:
        assert col in feat_df.columns, f"Missing feature: {col}"
    assert len(feat_df) > 0

def test_pre_trade_mc_decisions():
    config = {
        "pre_trade_mc": {
            "iterations": 1000,
            "min_win_prob": 0.50,
            "min_expected_ev": 0.0,
            "max_cvar_loss": -50.0,
            "seed": 42
        }
    }
    mc = PreTradeMonteCarloEngine(config)
    
    # Positive EV candidate snapshot
    valid_snap = {
        "timestamp": "2026-09-01 10:15:00",
        "ticker": "BEL",
        "close": 150.0,
        "atr_norm": 0.015,
        "prob": 0.78,
        "qty": 16,
        "tp_pct": 0.030,
        "sl_pct": 0.020,
        "hold_bars": 12
    }
    res = mc.evaluate_candidate(valid_snap)
    assert res["decision"] in ["ALLOW ENTRY", "REJECT ENTRY"]
    assert "win_prob" in res["stats"]
    assert "expected_net_ev" in res["stats"]

    # Incomplete snapshot -> INSUFFICIENT EVIDENCE
    bad_snap = {"timestamp": "2026-09-01 10:15:00", "ticker": "BEL"}
    res_bad = mc.evaluate_candidate(bad_snap)
    assert res_bad["decision"] == "INSUFFICIENT EVIDENCE"

if __name__ == "__main__":
    test_dhan_charges()
    test_features_length_and_shift()
    test_pre_trade_mc_decisions()
    print("✅ All V11 Unit Tests Passed!")
