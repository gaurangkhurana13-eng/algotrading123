import urllib.request
import json

url = "https://openweb-search.dhan.co/DhanSearch"
payload = {
    "Data": {
        "searchterm": "SUZLON",
        "inst": "E",
        "exch": "NSE"
    }
}

req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0"
})

try:
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        print("Search response for SUZLON:")
        print(json.dumps(data, indent=2)[:1000])
except Exception as e:
    print("Search error:", e)
