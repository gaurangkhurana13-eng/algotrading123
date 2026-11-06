import duckdb
import pandas as pd

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")

shard_url = "https://huggingface.co/datasets/xxparthparekhxx/indian-stock-market-minute-data/resolve/main/minute/train-00004.parquet"

# Pull 1 day of minute data for PCJEWELLER
df_1m = con.execute(f"""
    SELECT timestamp, open, high, low, close, volume
    FROM '{shard_url}'
    WHERE symbol = 'PCJEWELLER'
      AND timestamp >= '2025-01-02 09:00:00'
      AND timestamp <= '2025-01-02 16:00:00'
    ORDER BY timestamp
""").df()

df_1m['datetime'] = pd.to_datetime(df_1m['timestamp']).dt.tz_localize(None)
df_1m = df_1m.set_index('datetime')

# Resample to 15m starting from 9:15
df_15m = df_1m.resample('15min', origin='start_day', offset='15min').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}).dropna()

print(f"Total 15m bars on 2025-01-02: {len(df_15m)}")
print("First 3 bars:")
print(df_15m.head(3))
print("Last 3 bars:")
print(df_15m.tail(3))
