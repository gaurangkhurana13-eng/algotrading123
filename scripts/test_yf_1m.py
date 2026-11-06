import yfinance as yf

ticker = yf.Ticker("PCJEWELLER.NS")
df_yf = ticker.history(period="60d", interval="1m")
print("YFinance 1m:", len(df_yf))
if len(df_yf) > 0:
    print(df_yf.index.min(), df_yf.index.max())
