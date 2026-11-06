import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "scripts"))
from test_v19_direct import raw_dfs, prepare_v19_features

print("=== DIAGNOSING V19 FILTER DROPS ===")
total_base_signals = 0
dropped_clv = 0
dropped_wbr = 0
dropped_natr = 0
dropped_rvol = 0
dropped_tick = 0
total_passed = 0

for sym, df in raw_dfs.items():
    d = prepare_v19_features(df)
    c = d["close"]
    macd_bull = (d["macd"] > 0.0) & (d["macd"] > d["macd_signal"]) & (d["macd_hist"] > d["macd_hist"].shift(1))
    mom_bull = (d["rsi_7"] > 52.0) & (d["rsi_7"] < 78.0) & (d["adx_10"] > 18.0) & (d["plus_di"] > d["minus_di"])
    regime_bull = (c > d["ema_50"]) & (d["ema_50"] >= d["ema_110"] * 0.997)
    vwap_band = ((c - d["vwap_shifted"]) >= 0.10 * d["atr_14"]) & ((c - d["vwap_shifted"]) <= 2.35 * d["atr_14"])
    time_ok = (d["time_hm"] >= 930) & (d["time_hm"] <= 1430)
    
    base_sig = macd_bull & mom_bull & regime_bull & vwap_band & time_ok
    total_base_signals += base_sig.sum()
    
    # Check drop per filter
    dropped_clv += (base_sig & (d["clv"] < 0.68)).sum()
    dropped_wbr += (base_sig & (d["wbr_20"] > 0.56)).sum()
    dropped_natr += (base_sig & (d["natr_14"] < 0.0058)).sum()
    dropped_rvol += (base_sig & (d["rvol_20"] < 1.20)).sum()
    dropped_tick += (base_sig & (d["tick_to_atr"] > 0.14)).sum()
    total_passed += d["entry_signal"].sum()

print(f"Total Base Signals   : {total_base_signals}")
print(f"Dropped by CLV < 0.68: {dropped_clv} ({(dropped_clv/max(1, total_base_signals))*100:.1f}%)")
print(f"Dropped by WBR > 0.56: {dropped_wbr} ({(dropped_wbr/max(1, total_base_signals))*100:.1f}%)")
print(f"Dropped by NATR<0.0058: {dropped_natr} ({(dropped_natr/max(1, total_base_signals))*100:.1f}%)")
print(f"Dropped by RVOL < 1.2: {dropped_rvol} ({(dropped_rvol/max(1, total_base_signals))*100:.1f}%)")
print(f"Dropped by Tick >0.14: {dropped_tick} ({(dropped_tick/max(1, total_base_signals))*100:.1f}%)")
print(f"Total V19 Passed     : {total_passed}")
