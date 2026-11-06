import pickle
import pandas as pd
import numpy as np

with open(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\data\universe_78_15m_1y.pkl", "rb") as f:
    u = pickle.load(f)

df = u['MTNL.NS'].loc['2026-10-01 10:30:00']
print("MTNL Bar on 2026-10-01 10:30:00:")
print(df)
vol = df['volume']
print(f"Volume: {vol}, 5% volume: {int(vol * 0.05)}")
