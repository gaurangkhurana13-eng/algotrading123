import pandas as pd
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

print("Downloading README for xxparthparekhxx/indian-stock-market-minute-data...")
try:
    readme_path = hf_hub_download(repo_id="xxparthparekhxx/indian-stock-market-minute-data", filename="README.md", repo_type="dataset")
    with open(readme_path, "r", encoding="utf-8") as f:
        print("=== README ===")
        print(f.read()[:1500])
except Exception as e:
    print("Error reading README:", e)

print("\nDownloading sample of minute/train-00000.parquet...")
try:
    f_path = hf_hub_download(repo_id="xxparthparekhxx/indian-stock-market-minute-data", filename="minute/train-00000.parquet", repo_type="dataset")
    print(f"Downloaded to {f_path}")
    
    # Read schema and metadata
    parquet_file = pq.ParquetFile(f_path)
    print("Num rows:", parquet_file.metadata.num_rows)
    print("Schema:\n", parquet_file.schema)
    
    # Read first 100 rows
    df = pd.read_parquet(f_path)
    print("\nHead:")
    print(df.head())
    print("\nColumns:", df.columns.tolist())
    if 'ticker' in df.columns or 'Symbol' in df.columns or 'symbol' in df.columns:
        col = 'ticker' if 'ticker' in df.columns else ('Symbol' if 'Symbol' in df.columns else 'symbol')
        print("Unique tickers in chunk 0:", df[col].unique()[:20])
    if 'date' in df.columns or 'Date' in df.columns or 'timestamp' in df.columns or 'datetime' in df.columns:
        dcol = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()][0]
        print(f"Date range in chunk 0: {df[dcol].min()} to {df[dcol].max()}")
except Exception as e:
    print("Error reading parquet:", e)
