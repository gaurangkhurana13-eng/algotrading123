import pickle
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "V21" / "src"))
from engine import BhavishyaV21Engine

with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    univ = pickle.load(f)

core_10 = [
    "PCJEWELLER.NS", "RPOWER.NS", "DELTACORP.NS", "RITES.NS",
    "IDEA.NS", "MTNL.NS", "ZEEL.NS", "UTKARSHBNK.NS", "IFCI.NS", "NCC.NS"
]

dfs = {}
for t in core_10:
    if t in univ:
        df = univ[t].copy()
        if "timestamp" not in df.columns:
            df["timestamp"] = df.index
        clean = t.replace(".NS", "")
        dfs[clean] = df

# Test relaxing tick_to_atr to <= 0.85 (or removing it since NATR >= 0.0056 handles tax drag)
from engine import compute_v21_features

# Let's inspect features with relaxed tick_to_atr
def compute_v21_features_relaxed(df: pd.DataFrame, tick_cap: float = 0.85) -> pd.DataFrame:
    d = compute_v21_features(df)
    # override microstructure_ok and entry_signal
    c = d["close"].astype(float)
    microstructure_ok = (
        (d["tick_to_atr"] <= tick_cap) &     # relaxed for sub-Rs 50 high-beta stocks
        (d["natr_14"] >= 0.0056) &           # Caps Dhan statutory tax drag <= 18.9% of 1R
        (d["wbr_20"] <= 0.57) &              # Blocks PSU/cyclical pin-bar chop automatically
        (d["er_10"] >= 0.26) &               # Requires directional fractal efficiency
        (d["rvol_20"] >= 1.18) &             # Requires institutional volume expansion
        (d["clv"] >= 0.66) &                 # Closes in top 34% of bar
        (d["bar_range_atr"] >= 0.45) &
        (d["bar_range_atr"] <= 1.80)         # Blocks exhausted climax blow-off bars
    )
    macd_ok = (d["macd"] > 0.0) & (d["macd"] > d["macd_signal"]) & (d["macd_hist"] > d["macd_hist"].shift(1))
    mom_ok = (d["rsi_7"] >= 52.0) & (d["rsi_7"] <= 78.5) & (d["adx_10"] >= 18.0) & (d["plus_di"] > d["minus_di"])
    regime_ok = (c > d["ema_50"]) & (d["ema_50"] >= d["ema_110"] * 0.997)
    vwap_ok = ((c - d["vwap_shifted"]) >= 0.08 * d["atr_14"]) & ((c - d["vwap_shifted"]) <= 2.40 * d["atr_14"])
    session_ok = (d["time_hm"] >= 930) & (d["time_hm"] <= 1430)
    d["entry_signal"] = macd_ok & mom_ok & regime_ok & vwap_ok & microstructure_ok & session_ok
    return d

print("Testing relaxed tick_to_atr across various caps...")
for cap in [0.25, 0.45, 0.85, 1.0]:
    # monkey-patch compute_v21_features in engine
    import engine
    engine.compute_v21_features = lambda df, c=cap: compute_v21_features_relaxed(df, c)
    
    eng = BhavishyaV21Engine(
        initial_capital=500.0,
        leverage=5.0,
        hard_unit_risk_cap=102.0,
        physical_stop_atr=0.84,
        pyramid_scale_ratio=0.70,
        max_slots_cap=3
    )
    trades_df, summ = eng.run_backtest(dfs)
    print(f"\nCap {cap}: Trades = {summ.get('total_trades', 0)}, WR = {summ.get('win_rate_pct', 0)}%, Net = Rs {summ.get('net_profit_rs', 0)}, PF = {summ.get('net_profit_factor', 0)}, MaxDD = {summ.get('max_drawdown_pct', 0)}%")
