import sys
import os
import pandas as pd
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

cache_path = r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\data\universe_78_15m.pkl"
raw_data = pd.read_pickle(cache_path)
print("Cache loaded.")
