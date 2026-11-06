import pickle
import pandas as pd
import numpy as np

with open(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\data\universe_78_15m_1y.pkl", "rb") as f:
    u = pickle.load(f)

df = u['TEXRAIL.NS'].loc['2026-09-25 09:15:00':'2026-09-25 11:00:00']
print("TEXRAIL bars on 2026-09-25:")
print(df)
