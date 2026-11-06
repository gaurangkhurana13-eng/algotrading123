import requests

headers = {'User-Agent': 'Mozilla/5.0'}

queries = [
    "nifty 500",
    "nse 15m",
    "nse intraday",
    "nifty 2021",
    "indian stocks minute",
    "nse historical minute"
]

for q in queries:
    url = f"https://www.kaggle.com/api/v1/datasets/list?search={requests.utils.quote(q)}"
    r = requests.get(url, headers=headers, timeout=10)
    if r.status_code == 200:
        items = r.json()
        print(f"\nQuery '{q}': found {len(items)}")
        for it in items[:6]:
            print(f"  - {it.get('ref')}: {it.get('title')}")
