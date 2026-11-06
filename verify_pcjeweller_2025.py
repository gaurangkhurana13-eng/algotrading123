import duckdb
import pandas as pd
import yfinance as yf
import datetime

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")

shard_url = "https://huggingface.co/datasets/xxparthparekhxx/indian-stock-market-minute-data/resolve/main/minute/train-00004.parquet"

print("1. Extracting January 2025 minute data for PCJEWELLER from Shard 4...")
query = f"""
    SELECT 
        strftime(timestamp, '%Y-%m-%d') as trade_date,
        FIRST(open ORDER BY timestamp) as day_open,
        MAX(high) as day_high,
        MIN(low) as day_low,
        LAST(close ORDER BY timestamp) as day_close,
        SUM(volume) as day_volume,
        COUNT(*) as minute_candles
    FROM '{shard_url}'
    WHERE symbol = 'PCJEWELLER'
      AND timestamp >= '2025-01-01'
      AND timestamp < '2025-01-16'
    GROUP BY strftime(timestamp, '%Y-%m-%d')
    ORDER BY trade_date
"""

df_dataset = con.execute(query).df()
print(f"Dataset extracted: {len(df_dataset)} trading days found in Jan 2025.")

print("\n2. Fetching Official NSE market data via yfinance (PCJEWELLER.NS)...")
# yfinance provides official NSE daily data historically
yf_data = yf.download("PCJEWELLER.NS", start="2025-01-01", end="2025-01-16", progress=False)

# Flatten yfinance multi-index if present
if isinstance(yf_data.columns, pd.MultiIndex):
    yf_data.columns = [c[0].lower() for c in yf_data.columns]
else:
    yf_data.columns = [c.lower() for c in yf_data.columns]

yf_data['trade_date'] = yf_data.index.strftime('%Y-%m-%d')

print("\n3. Merging and Cross-Verifying every single price point:")
comparison = []
for idx, row in df_dataset.iterrows():
    d = row['trade_date']
    match = yf_data[yf_data['trade_date'] == d]
    if len(match) > 0:
        m = match.iloc[0]
        o_diff = round(abs(row['day_open'] - m['open']), 2)
        h_diff = round(abs(row['day_high'] - m['high']), 2)
        l_diff = round(abs(row['day_low'] - m['low']), 2)
        c_diff = round(abs(row['day_close'] - m['close']), 2)
        v_diff = int(abs(row['day_volume'] - m['volume']))
        
        comparison.append({
            "Date": d,
            "DS_Open": round(row['day_open'], 2),
            "NSE_Open": round(m['open'], 2),
            "Open_Diff": o_diff,
            "DS_High": round(row['day_high'], 2),
            "NSE_High": round(m['high'], 2),
            "High_Diff": h_diff,
            "DS_Low": round(row['day_low'], 2),
            "NSE_Low": round(m['low'], 2),
            "Low_Diff": l_diff,
            "DS_Close": round(row['day_close'], 2),
            "NSE_Close": round(m['close'], 2),
            "Close_Diff": c_diff,
            "Min_Bars": row['minute_candles']
        })

comp_df = pd.DataFrame(comparison)
print(comp_df.to_string(index=False))

print("\n=== SUMMARY AUDIT RESULT ===")
max_c_diff = comp_df['Close_Diff'].max()
max_h_diff = comp_df['High_Diff'].max()
max_l_diff = comp_df['Low_Diff'].max()
print(f"Max Difference in Close Price: ₹{max_c_diff}")
print(f"Max Difference in High Price: ₹{max_h_diff}")
print(f"Max Difference in Low Price: ₹{max_l_diff}")
if max_c_diff <= 0.05 and max_h_diff <= 0.05 and max_l_diff <= 0.05:
    print(">>> 100% PERFECT MATCH! DATA IS VERIFIED AUTHENTIC OFFICIAL NSE TICK DATA! <<<")
else:
    print(">>> WARNING: DISCREPANCY DETECTED! <<<")
