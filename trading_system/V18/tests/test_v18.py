"""
Unit & Integration Test Suite for Bhavishya V18
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
from trading_system.V18.src.dhan_engine import calc_dhan_charges
from trading_system.V18.src.sector_map import get_sector
from trading_system.V18.src.pre_trade_mc import PreTradeMonteCarloEngine

def test_v18_dhan_charges():
    charges = calc_dhan_charges(400.0, 410.0, 5)
    assert 1.30 <= charges <= 3.00

def test_v18_sector_mapping():
    assert get_sector("HDFCBANK") == "FINANCIALS"
    assert get_sector("SBIN.NS") == "FINANCIALS"
    assert get_sector("INFY") == "IT"
    assert get_sector("TATAMOTORS") == "AUTO"
    assert get_sector("UNKNOWNSTOCK") == "GENERAL"

def test_v18_mc_engine():
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
        "ticker": "LT",
        "close": 3500.0,
        "atr_norm": 0.008,
        "prob": 0.81,
        "qty": 1,
        "tp_pct": 0.016,
        "sl_pct": 0.012,
        "hold_bars": 12
    }
    res = mc.evaluate_candidate(snapshot)
    assert res["decision"] in ["ALLOW ENTRY", "REJECT ENTRY"]
