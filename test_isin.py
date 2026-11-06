import urllib.request
import json

url = "https://open-web-scanx.dhan.co/scanx/owrtscrdt_isin"
payload = {"Data": {"isin": "INE040H01021"}} # SUZLON

req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={
    "Content-Type": "application/json",
    "User-Agent": "Mozilla/5.0"
})

try:
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        print("Response for SUZLON isin:")
        print(json.dumps(data, indent=2))
except Exception as e:
    print("Error:", e)
