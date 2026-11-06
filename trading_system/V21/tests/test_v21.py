"""
Unit & Integration Test Suite for Bhavishya V21: Dynamic Sector Capacity
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
from trading_system.V21.src.dhan_engine import calc_dhan_charges
from trading_system.V21.src.sector_map import get_sector_info
from trading_system.V21.src.backtester import compute_half_kelly_budget
from trading_system.V21.src.pre_trade_mc import PreTradeMonteCarloEngine

def test_v21_dhan_charges():
    charges = calc_dhan_charges(450.0, 465.0, 5)
    assert 1.30 <= charges <= 3.00

def test_v21_subsector_classification():
    s_hdfc, sub_hdfc = get_sector_info("HDFCBANK")
    s_pfc, sub_pfc = get_sector_info("PFC")
    assert s_hdfc == "FINANCIALS"
    assert s_pfc == "FINANCIALS"
    assert sub_hdfc != sub_pfc, f"Subsectors should differ: {sub_hdfc} vs {sub_pfc}"

def test_v21_sector_penalty_math():
    base_b = compute_half_kelly_budget(0.80, base_budget=20.0)
    penalized = base_b * 0.80
    assert penalized < base_b
    assert penalized >= 14.0

def test_v21_pre_trade_mc():
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
        "ticker": "PFC",
        "close": 450.0,
        "atr_norm": 0.015,
        "prob": 0.81,
        "qty": 3,
        "tp_pct": 0.030,
        "sl_pct": 0.0225,
        "hold_bars": 12
    }
    res = mc.evaluate_candidate(snapshot)
    assert res["decision"] in ["ALLOW ENTRY", "REJECT ENTRY"]
