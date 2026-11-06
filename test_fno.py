import urllib.request
import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

url = "https://openweb-api.dhan.co/brokerage"

# Test F&O on Dhan
def query_fno(buy_price, sell_price, qty, instrument="FUTIDX", sec_id=2):
    payload = {
        "source": "N",
        "data": {
            "exchange": "NSE",
            "segment": "D",
            "txn_type": "S",
            "qty": str(qty),
            "window": "ONLY_BUY",
            "security_id": sec_id,
            "sell_price": f"{sell_price:.2f}",
            "buy_price": f"{buy_price:.2f}",
            "product": "I",
            "instrument": instrument,
            "exchange1": ""
        }
    }
    headers = {"Content-Type": "application/json", "User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data["data"][0] if data.get("status") == "success" else data
    except Exception as e:
        return {"error": str(e)}

print("=== FUTURES TEST ===")
res_fut = query_fno(24000, 24050, 25, instrument="FUTIDX")
print(json.dumps(res_fut, indent=2))

print("\n=== OPTIONS TEST ===")
res_opt = query_fno(100, 110, 25, instrument="OPTIDX")
print(json.dumps(res_opt, indent=2))
