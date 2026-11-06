import sys
from pathlib import Path
import pandas as pd

SOURCE_DIR = Path(r"C:\Users\admin\.cache\kagglehub\datasets\debashis74017\stock-market-data-nifty-100-stocks-15-min-data\versions\2")
csv_path = SOURCE_DIR / "SBIN_15minute.csv"

df = pd.read_csv(csv_path)
df.columns = [c.lower() for c in df.columns]
date_col = 'datetime' if 'datetime' in df.columns else 'date'
df['dt'] = pd.to_datetime(df[date_col])

print(f"Overall date range: {df['dt'].min()} to {df['dt'].max()}")
h1_2016 = df[(df['dt'] >= '2016-01-01') & (df['dt'] <= '2016-06-30 23:59:59')]
print(f"2016 H1 rows count: {len(h1_2016)}")
if len(h1_2016) > 0:
    print(f"2016 H1 range: {h1_2016['dt'].min()} to {h1_2016['dt'].max()}")
    print(f"Unique trading days in 2016 H1: {len(set(h1_2016['dt'].dt.date))}")
