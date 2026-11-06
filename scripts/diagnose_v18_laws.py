import pickle
import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
from build_and_simulate_v18 import data_map, UNIVERSE_20

print("=== FORENSIC BREAKDOWN OF V18 ADMISSIONS LAWS ===")
for sym in UNIVERSE_20:
    t_data = data_map[sym]
    c = t_data["c"]
    atr = t_data["atr14"]
    v = t_data["v"]
    df = t_data["df"]
    
    # Calculate components
    turnover = (c * v).rolling(20).median()
    direction = (c - c.shift(10)).abs()
    volatility = c.diff().abs().rolling(10).sum()
    er10 = direction / (volatility + 1e-9)
    
    # Check technical entry condition before laws
    ema50 = c.ewm(span=50).mean()
    ema110 = c.ewm(span=110).mean()
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    sig = macd.ewm(span=9).mean()
    macd_cross = (macd > sig) & (macd.shift(1) <= sig.shift(1))
    
    tech_signals = macd_cross & (macd > 0) & (c > ema50) & (c > ema110) & (c > t_data["vwap_lag"]) & (t_data["rsi7"] > 50.0)
    
    l1 = (c >= 15.0) & (atr > 0) & ((0.05 / (atr + 1e-9)) <= 0.038)
    l2 = ((atr / (c + 1e-9)) * 100.0) >= 0.78
    l3 = turnover >= 6_000_000.0
    l4 = er10 >= 0.36
    
    total_tech = int(tech_signals.sum())
    passed_l1 = int((tech_signals & l1).sum())
    passed_l2 = int((tech_signals & l2).sum())
    passed_l3 = int((tech_signals & l3).sum())
    passed_l4 = int((tech_signals & l4).sum())
    passed_all = int((tech_signals & l1 & l2 & l3 & l4).sum())
    
    print(f"{sym:<15} Tech Signals={total_tech:<3} | L1(Tick/P)={passed_l1:<3} | L2(NATR)={passed_l2:<3} | L3(Turnover)={passed_l3:<3} | L4(ER10)={passed_l4:<3} | ALL={passed_all:<3}")
