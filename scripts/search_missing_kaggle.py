import requests

headers = {'User-Agent': 'Mozilla/5.0'}
missing = ['ELECTCAST', 'IFCI', 'IRCON', 'JKBANK', 'MANALIPETC', 'MTNL', 'PCJEWELLER', 'RPOWER', 'TEXRAIL']

print("--- Searching Kaggle API for individual missing tickers ---")
for sym in missing:
    url = f"https://www.kaggle.com/api/v1/datasets/list?search={sym}"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            data = r.json()
            print(f"\nSymbol '{sym}': found {len(data)} datasets")
            for item in data[:3]:
                print(f"  - {item.get('ref')}: {item.get('title')}")
    except Exception as e:
        print(f"Error {sym}: {e}")
