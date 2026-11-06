"""
Unit & Integration Test Suite for Bhavishya V12
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import numpy as np
import pandas as pd
from trading_system.V12.src.dhan_engine import calc_dhan_charges
from trading_system.V12.src.features import FEATURE_COLS, compute_features
from trading_system.V12.src.pre_trade_mc import PreTradeMonteCarloEngine

def test_dhan_charges():
    charges = calc_dhan_charges(100.0, 103.0, 25)
    assert 1.30 <= charges <= 3.50, f"Unexpected charges: {charges}"

def test_dynamic_volatility_sizing():
    # If risk budget is Rs 20 and SL distance is Rs 4 per share, qty should be 5
    price = 100.0
    atr_norm = 0.02666
    sl_atr_mult = 1.5
    risk_budget = 20.0
    per_share_sl = price * (sl_atr_mult * atr_norm)  # ~4.0
    qty = max(1, min(int(risk_budget / per_share_sl), int(2500 / price)))
    assert qty == 5
    dollar_risk = qty * per_share_sl
    assert dollar_risk <= 20.05

def test_students_t_fat_tail_mc():
    config = {
        "pre_trade_mc": {
            "iterations": 1000,
            "min_win_prob": 0.50,
            "min_expected_ev": 0.0,
            "max_cvar_loss": -40.0,
            "degrees_of_freedom": 5,
            "seed": 42
        }
    }
    mc = PreTradeMonteCarloEngine(config)
    snapshot = {
        "timestamp": "2026-09-01 10:15:00",
        "ticker": "SBIN",
        "close": 800.0,
        "atr_norm": 0.012,
        "prob": 0.77,
        "qty": 2,
        "tp_pct": 0.024,
        "sl_pct": 0.018,
        "hold_bars": 12
    }
    res = mc.evaluate_candidate(snapshot)
    assert res["decision"] in ["ALLOW ENTRY", "REJECT ENTRY", "INSUFFICIENT EVIDENCE"]
    assert "stats" in res
    assert "win_prob" in res["stats"]
    assert "cvar_95_loss" in res["stats"]
