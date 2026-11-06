import os, sys, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import catboost as cb
from catboost import CatBoostClassifier, Pool

# Verify numpy vectorized Monte Carlo simulation speed and logic
np.random.seed(42)
test_returns = np.array([0.015, -0.008, 0.022, -0.007, 0.012, -0.008, 0.018, 0.025, -0.008, 0.014])
num_sims = 10000
path_len = 200
starting_capital = 100000.0

import time
t0 = time.time()

# Vectorized random choice with replacement + noise
random_indices = np.random.randint(0, len(test_returns), size=(num_sims, path_len))
sim_returns = test_returns[random_indices] + np.random.normal(0, 0.0003, size=(num_sims, path_len))

# Compounding capital curves
multipliers = 1.0 + sim_returns
equity_paths = np.empty((num_sims, path_len + 1))
equity_paths[:, 0] = starting_capital
equity_paths[:, 1:] = starting_capital * np.cumprod(multipliers, axis=1)

# Drawdown calculation
running_max = np.maximum.accumulate(equity_paths, axis=1)
drawdown_paths = (running_max - equity_paths) / running_max
max_drawdowns = np.max(drawdown_paths, axis=1)

t1 = time.time()
print(f"10,000 Monte Carlo paths completed in {t1 - t0:.3f} seconds!")
print(f"Median Final Equity: ₹{np.median(equity_paths[:, -1]):,.2f}")
print(f"95th Percentile Max Drawdown: {np.percentile(max_drawdowns, 95)*100:.2f}%")
print(f"Probability of Ruin (>20% DD): {np.mean(max_drawdowns > 0.20)*100:.2f}%")
