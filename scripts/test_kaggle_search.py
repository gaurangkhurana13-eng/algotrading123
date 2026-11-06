import requests

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
}

queries = ["15min candle data", "nifty 100 15 min", "nifty 500 intraday", "prabhavsemwal", "nse minute"]

print("--- Testing Kaggle Public Search API ---")
for q in queries:
    url = f"https://www.kaggle.com/api/v1/datasets/list?search={requests.utils.quote(q)}"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        print(f"Query '{q}': status {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            print(f"Results count: {len(data)}")
            for item in data[:5]:
                print(f"  - ref: {item.get('ref')}, title: {item.get('title')}")
        else:
            print("  Response:", r.text[:200])
    except Exception as e:
        print(f"Error for '{q}': {e}")
