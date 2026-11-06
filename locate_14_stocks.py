import duckdb

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")

symbols = [
    'PCJEWELLER', 'RPOWER', 'ZEEL', 'DELTACORP', 'IDEA', 
    'MTNL', 'UTKARSHBNK', 'IFCI', 'NCC', 'TEXRAIL', 
    'JPPOWER', 'PTC', 'NIACL', 'RITES'
]

base_url = "https://huggingface.co/datasets/xxparthparekhxx/indian-stock-market-minute-data/resolve/main/minute/"

print("Mapping 14 stocks across 8 shards...")
stock_map = {}

for i in range(8):
    shard_url = f"{base_url}train-0000{i}.parquet"
    print(f"Scanning shard {i}...")
    try:
        found = con.execute(f"""
            SELECT symbol, MIN(timestamp) as min_ts, MAX(timestamp) as max_ts, COUNT(*) as rows
            FROM '{shard_url}'
            WHERE symbol IN ({','.join([f"'{s}'" for s in symbols])})
            GROUP BY symbol
        """).df()
        
        for idx, row in found.iterrows():
            sym = row['symbol']
            stock_map[sym] = {
                "shard": i,
                "shard_url": shard_url,
                "min_ts": str(row['min_ts']),
                "max_ts": str(row['max_ts']),
                "rows": int(row['rows'])
            }
            print(f"  -> Found {sym} in shard {i} ({row['rows']} rows, {row['min_ts']} to {row['max_ts']})")
            
        if len(stock_map) == len(symbols):
            print("\nAll 14 stocks successfully mapped!")
            break
    except Exception as e:
        print(f"Error on shard {i}: {e}")

print("\n--- FINAL MAPPING OF 14 STOCKS ---")
for s in symbols:
    if s in stock_map:
        info = stock_map[s]
        print(f"{s:<12} | Shard {info['shard']} | Rows: {info['rows']} | {info['min_ts'][:10]} to {info['max_ts'][:10]}")
    else:
        print(f"{s:<12} | NOT FOUND YET")
