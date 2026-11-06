from datasets import load_dataset
import pandas as pd

try:
    print("Testing Hugging Face Indian Stock Market dataset info...")
    # Stream first few rows without downloading everything
    ds = load_dataset("xxparthparekhxx/indian-stock-market-minute-data", streaming=True)
    print("Dataset loaded successfully! Splits:", ds.keys())
    # Inspect first example
    sample = next(iter(ds['train'])) if 'train' in ds else next(iter(ds[list(ds.keys())[0]]))
    print("Sample record keys:", sample.keys())
    print("Sample data:", sample)
except Exception as e:
    print("Hugging Face dataset test error:", e)
