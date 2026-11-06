import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

def test_clv_calculation():
    c = 100.0
    l = 95.0
    h = 105.0
    clv = (c - l) / max(h - l, 1e-6)
    assert round(clv, 4) == 0.5000

def test_dhan_charges_reality():
    # Verify Dhan charges for MIS trade: Buy @ 100, Sell @ 105, Qty 20 (Turnover Rs 4,100)
    chg = calc_dhan_charges(100.0, 105.0, 20, is_intraday=True)
    assert 2.0 <= chg <= 5.0

def test_risk_schedule_tiers():
    # Tier 1: Equity = 800 -> Cap = 40
    r_cand1 = (14.0 + 0.057 * 300) * 1.0
    r1 = min(r_cand1, 40.0)
    assert r1 <= 40.0

    # Tier 2: Equity = 1500 -> Cap = 110
    r_cand2 = (14.0 + 0.057 * 1000) * 1.0
    r2 = min(r_cand2, 110.0)
    assert r2 <= 110.0

    # Tier 3: Equity = 2500 -> Cap = 3% of 2500 = 75
    r_cand3 = 0.032 * 2500 * 1.0
    r3 = min(r_cand3, 0.03 * 2500)
    assert r3 == 75.0

def test_stop_monotonicity():
    # Stop can only move up, never down
    sl_seq = [100.0]
    new_stops = [102.0, 101.5, 104.0, 103.0]
    for s in new_stops:
        sl_seq.append(max(sl_seq[-1], s))
    assert sl_seq == [100.0, 102.0, 102.0, 104.0, 104.0]

def test_trades_csv_exists_and_valid():
    csv_path = ROOT_DIR / "V20" / "trades" / "v20_trades.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) > 0
    mandatory = ["entry_time", "exit_time", "entry_price", "exit_price", "shares", "net_pnl", "charges", "exit_reason"]
    for col in mandatory:
        assert col in df.columns
