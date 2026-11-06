"""
Unit & Integration Test Suite for Bhavishya V20: Quant Apex Master System
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
from trading_system.V20.src.dhan_engine import calc_dhan_charges
from trading_system.V20.src.sector_map import get_sector
from trading_system.V20.src.backtester import compute_half_kelly_budget
from trading_system.V20.src.pre_trade_mc import PreTradeMonteCarloEngine

def test_v20_dhan_charges():
    charges = calc_dhan_charges(600.0, 618.0, 4)
    assert 1.30 <= charges <= 3.00

def test_v20_sector_classification():
    assert get_sector("SBIN") == "FINANCIALS"
    assert get_sector("TCS") == "IT"
    assert get_sector("MARUTI") == "AUTO"

def test_v20_half_kelly_bounds():
    b = compute_half_kelly_budget(0.80, base_budget=20.0, min_b=18.0, max_b=35.0)
    assert 18.0 <= b <= 35.0

def test_v20_pre_trade_mc():
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
        "ticker": "SBIN",
        "close": 820.0,
        "atr_norm": 0.012,
        "prob": 0.82,
        "qty": 2,
        "tp_pct": 0.024,
        "sl_pct": 0.018,
        "hold_bars": 12
    }
    res = mc.evaluate_candidate(snapshot)
    assert res["decision"] in ["ALLOW ENTRY", "REJECT ENTRY"]
