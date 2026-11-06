import os, sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tvDatafeed import TvDatafeed, Interval
import catboost as cb
from catboost import CatBoostClassifier, Pool
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("Imports successful!")

# Test TvDatafeed download
tv = TvDatafeed()
df_raw = tv.get_hist(symbol='NIFTY', exchange='NSE', interval=Interval.in_15_minute, n_bars=1000)
print(f"Downloaded NIFTY: {len(df_raw)} candles")
print("Columns:", df_raw.columns.tolist())
