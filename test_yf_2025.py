import yfinance as yf

df = yf.download("PCJEWELLER.NS", start="2025-01-01", end="2025-03-01", interval="15m")
print(f"2025 yfinance fetched rows: {len(df)}")
