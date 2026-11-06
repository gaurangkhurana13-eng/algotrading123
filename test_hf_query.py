import duckdb

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")

url_shard_0 = "https://huggingface.co/datasets/xxparthparekhxx/indian-stock-market-minute-data/resolve/main/minute/train-00000.parquet"
url_shard_7 = "https://huggingface.co/datasets/xxparthparekhxx/indian-stock-market-minute-data/resolve/main/minute/train-00007.parquet"

print("Querying Schema from Shard 0...")
schema = con.execute(f"DESCRIBE SELECT * FROM '{url_shard_0}' LIMIT 5").df()
print(schema)

print("\nSampling 5 rows from Shard 0:")
sample = con.execute(f"SELECT * FROM '{url_shard_0}' LIMIT 5").df()
print(sample)

print("\nChecking symbols & date min/max from Shard 0:")
stats0 = con.execute(f"""
    SELECT 
        MIN(timestamp) as min_ts, 
        MAX(timestamp) as max_ts, 
        COUNT(*) as total_rows, 
        COUNT(DISTINCT symbol) as num_symbols
    FROM '{url_shard_0}'
""").df()
print(stats0)
