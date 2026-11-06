import{j as e}from"./vendor-react-DzkUCm50.js";import{G as n,P as t,S as a,C as s,N as r,R as i,a as d}from"./guideKit-DF5ZXJWM.js";import"./SEOHead-ByQ1FCOZ.js";const o=[{id:"read",label:"Read the file"},{id:"timestamp",label:"The time column"},{id:"client",label:"Fetch the latest rows"},{id:"inspect",label:"Inspect the data"},{id:"resample",label:"Change the timeframe"}];function c(){return e.jsxs(n,{slug:"load-in-python",title:"Load a dataset in Python",description:"Read a Databank CSV or Parquet file into pandas, use its UTC time column, fetch the latest rows with your API key and resample candles.",kicker:"Databank guide",nav:o,intro:e.jsx(t,{children:"Download CSV or Parquet from the catalogue. Parquet keeps column types in the file; CSV is useful for spreadsheets and other text readers. Both load into a pandas DataFrame."}),children:[e.jsxs(a,{id:"read",title:"Read the file",children:[e.jsxs(t,{children:["Install the readers with"," ",e.jsx("code",{className:"font-mono text-[12px]",children:"pip install pandas pyarrow"}),", then use the filename from your download."]}),e.jsx(s,{code:`import pandas as pd

df = pd.read_parquet("AAPL_us_stocks_5m_2026-10-08_2026-10-14.parquet")
# For a CSV export:
# df = pd.read_csv("AAPL_us_stocks_5m_2026-10-08_2026-10-14.csv")
df.head()
df.dtypes`}),e.jsx(t,{children:"Candle files contain time, open, high, low, close, volume and trades. The filename identifies the symbol, market, timeframe and date range."})]}),e.jsxs(a,{id:"timestamp",title:"The time column",children:[e.jsxs(t,{children:["The ",e.jsx("code",{className:"font-mono text-[12px]",children:"time"})," column is the bar opening time in UTC. Normalize it to a UTC datetime, then set it as the index."]}),e.jsx(s,{code:`df["time"] = pd.to_datetime(df["time"], utc=True)
df = df.set_index("time").sort_index()

window = df.loc["2026-10-08":"2026-10-14"]
new_york = df.tz_convert("America/New_York")`}),e.jsx(r,{children:"Keep timestamps in UTC when joining instruments. Convert to a local zone for display. Daily, weekly and monthly exports follow the exchange trading day; their date boundaries can differ from UTC calendar dates."})]}),e.jsxs(a,{id:"client",title:"Fetch the latest rows with the API",children:[e.jsxs(t,{children:["The preview endpoint returns the latest ten rows at your chosen timeframe. Store your key in the LSE_API_KEY environment variable and send it as a header. Install the HTTP client with"," ",e.jsx("code",{className:"font-mono text-[12px]",children:"pip install requests"}),"."]}),e.jsx(s,{code:`import os
import requests
import pandas as pd

response = requests.get(
    "https://download.londonstrategicedge.com/v3/preview",
    headers={"X-API-Key": os.environ["LSE_API_KEY"]},
    params={"market": "us_stocks", "symbol": "AAPL",
            "timeframe": "5m", "adjusted": "1"},
    timeout=30,
)
response.raise_for_status()
df = pd.DataFrame(response.json()["rows"])
df["time"] = pd.to_datetime(df["time"], utc=True)
df = df.set_index("time").sort_index()`}),e.jsxs(t,{children:["Find your key under"," ",e.jsx("a",{href:"/data#billing",className:"text-[#1a56db] underline-offset-2 hover:underline dark:text-[#6ea8fe]",children:"Billing in the Databank"}),". A failed request raises an error before the response is read as data. For an export covering a date range, use the catalogue or the calls in"," ",e.jsx("a",{href:"/data#api",className:"text-[#1a56db] underline-offset-2 hover:underline dark:text-[#6ea8fe]",children:"Data documentation"}),". CME Globex, Nasdaq TotalView and OPRA require their exchange plans for Trades and Quotes."]})]}),e.jsxs(a,{id:"inspect",title:"Inspect the data",children:[e.jsx(t,{children:"Check the columns, range and missing values before using the data."}),e.jsx(s,{code:`df.describe()
df.isna().sum()
df.index.min(), df.index.max()`}),e.jsx(t,{children:"Review gaps against the exchange calendar and the date range you requested. A missing value can come from joining frames with different coverage."})]}),e.jsxs(a,{id:"resample",title:"Change the timeframe",children:[e.jsx(t,{children:"You can combine smaller intraday candles into larger candles locally. This example combines five-minute candles into UTC clock hours."}),e.jsx(s,{code:`hourly = df.resample("1h").agg({
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "trades": "sum",
}).dropna(subset=["open", "close"])`}),e.jsx(t,{children:"For daily, weekly or monthly candles, request the timeframe from the catalogue so the exchange trading day determines the groups. Local resampling into UTC days can give different results."})]}),e.jsx(i,{items:[{href:"https://pandas.pydata.org/docs/user_guide/timeseries.html",label:"pandas time series",note:"Index by time, slice dates, resample and convert zones"},{href:"https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html",label:"pandas.read_parquet",note:"Reader parameters and engine choice"},{href:"/data#api",label:"Data documentation",note:"Preview, export and resumed download calls"}]}),e.jsx(d,{items:[{href:"/data/guides/train-a-model",title:"Train a model",blurb:"Build features from the DataFrame and fit a first model."},{href:"/data/guides/parquet-to-csv",title:"Convert Parquet to CSV",blurb:"Convert a saved Parquet file when a tool needs text."}]})]})}export{c as default};
