import pickle
import pandas as pd
import numpy as np

with open(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\data\universe_78_15m_1y.pkl", "rb") as f:
    u = pickle.load(f)

df = u['TEXRAIL.NS'].loc['2026-09-20':'2026-09-26']
c = df['close']
ema20 = c.ewm(span=20).mean()
ema50 = c.ewm(span=50).mean()
ema110 = c.ewm(span=110).mean()

print(df[['open', 'high', 'low', 'close', 'volume']])
print("\nDaily closes for TEXRAIL in late Sep:")
print(df.resample('D').last().dropna()[['close', 'volume']])
