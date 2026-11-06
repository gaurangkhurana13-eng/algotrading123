import duckdb

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")

print("Searching for PCJEWELLER in Hugging Face parquet shards...")
base_url = "https://huggingface.co/datasets/xxparthparekhxx/indian-stock-market-minute-data/resolve/main/minute/"

for i in range(8):
    shard_url = f"{base_url}train-0000{i}.parquet"
    print(f"Checking shard {i}...")
    try:
        # Check if PCJEWELLER exists in this shard
        res = con.execute(f"""
            SELECT symbol, MIN(timestamp) as min_ts, MAX(timestamp) as max_ts, COUNT(*) as rows
            FROM '{shard_url}'
            WHERE symbol = 'PCJEWELLER'
            GROUP BY symbol
        """).df()
        if len(res) > 0:
            print(f"FOUND PCJEWELLER in Shard {i}!")
            print(res)
            break
        else:
            # Print sample symbols to understand partitioning
            sample = con.execute(f"SELECT DISTINCT symbol FROM '{shard_url}' LIMIT 3").df()
            print(f"Shard {i} sample symbols: {sample['symbol'].tolist()}")
    except Exception as e:
        print(f"Error checking shard {i}: {e}")
