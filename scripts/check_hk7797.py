import requests

url = "https://www.kaggle.com/api/v1/datasets/list-files/hk7797/stock-market-india"
headers = {'User-Agent': 'Mozilla/5.0'}

try:
    r = requests.get(url, headers=headers, timeout=10)
    print("Status:", r.status_code)
    if r.status_code == 200:
        data = r.json()
        print("Total files:", len(data))
        file_names = [f.get('name') for f in data]
        print("First 20 files:", file_names[:20])
        
        missing = ['ELECTCAST', 'IFCI', 'IRCON', 'JKBANK', 'MANALIPETC', 'MTNL', 'PCJEWELLER', 'RPOWER', 'TEXRAIL']
        matched = [f for f in file_names if any(m.lower() in f.lower() for m in missing)]
        print(f"Matched missing stocks: {matched}")
    else:
        print(r.text[:300])
except Exception as e:
    print("Error:", e)
