import sys, pickle
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
SKILL_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\day_trader_skill")

print("Checking day_trader_skill components:")
for p in SKILL_DIR.glob("trader-*"):
    if p.is_dir():
        print(f"  {p.name}: {list(p.glob('*.py'))}")
