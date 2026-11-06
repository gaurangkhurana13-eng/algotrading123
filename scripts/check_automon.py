import requests

url = "https://www.kaggle.com/api/v1/datasets/view/automon/nifty-500-stocks-multi-timeframe-ohlcv-data"
headers = {'User-Agent': 'Mozilla/5.0'}
r = requests.get(url, headers=headers, timeout=10)
if r.status_code == 200:
    data = r.json()
    print("Title:", data.get("title"))
    print("Total Bytes:", data.get("totalBytes"))
    print("Description snippet:", data.get("description")[:500].encode('ascii', errors='ignore').decode('ascii'))
else:
    print("Status:", r.status_code)
