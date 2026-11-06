import sys, pickle
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

from framework import calc_dhan_charges, calculate_atr
from scripts.test_v42_prototype import compute_vwap_shifted, compute_er10, compute_adx_di, SECTOR_MAP, SYM_TO_SECTOR, V29_30_UNIVERSE

p_2026 = pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))
p_highbeta = pickle.load(open(ROOT / 'data/unseen_high_beta_10_15m_1y.pkl', 'rb'))
p_unseen10 = pickle.load(open(ROOT / 'data/unseen_10_15m_1y.pkl', 'rb'))
pool_2026 = {**p_2026, **p_highbeta, **p_unseen10}

# Let's inspect position sizing dynamics
# In V42, initial_equity = 500.0, raw_r = 14.0 at start
# Let's see what happens if initial risk is 5% (Rs 25) vs 2.8% (Rs 14)
# On a Rs 500 account, Rs 25 risk allows buying more shares using the available 5x MIS leverage!
print("=" * 80)
print("INSPECTING POSITION SIZING BOTTLENECK ON ₹500 CAPITAL")
print("=" * 80)

# Let's verify Dhan charges calculation on Dhan Calculator values:
# Buy 100 shares at Rs 20 = Rs 2000 buy value
# Sell 100 shares at Rs 21 = Rs 2100 sell value
# Turnover = Rs 4100
b_val = 2000.0
s_val = 2100.0
t_over = b_val + s_val
brk = min(20.0, b_val * 0.0003) + min(20.0, s_val * 0.0003) # 0.60 + 0.63 = 1.23
stt = s_val * 0.00025 # 0.525
exch = t_over * 0.0000345 # 0.141
sebi = t_over * 0.000001 # 0.0041
stamp = b_val * 0.00003 # 0.06
gst = 0.18 * (brk + exch + sebi) # 0.18 * (1.23 + 0.141 + 0.0041) = 0.2475
total_dhan = brk + stt + exch + sebi + stamp + gst
print(f"Dhan Official Formula on ₹2,000 buy / ₹2,100 sell:")
print(f"Brokerage: ₹{brk:.2f} (0.03% on both legs)")
print(f"STT: ₹{stt:.2f}")
print(f"Exchange: ₹{exch:.2f}")
print(f"SEBI: ₹{sebi:.4f}")
print(f"Stamp Duty: ₹{stamp:.2f}")
print(f"GST: ₹{gst:.2f}")
print(f"TOTAL TAX & CHARGES: ₹{total_dhan:.2f}")
print(f"Framework calc_dhan_charges output: ₹{calc_dhan_charges(20.0, 21.0, 100):.2f}")
