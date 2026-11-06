import urllib.request
import json

url = "https://openweb-api.dhan.co/brokerage"

def query_brokerage_exact(buy_price, sell_price, qty, product="I", exchange="NSE", sec_id=12018):
    # Notice: when buy and sell are both > 0:
    # txn_type = "S", window = "ONLY_BUY"
    payload = {
        "source": "N",
        "data": {
            "exchange": exchange,
            "segment": "E",
            "txn_type": "S",
            "qty": str(qty),
            "window": "ONLY_BUY",
            "security_id": sec_id,
            "sell_price": f"{sell_price:.2f}",
            "buy_price": f"{buy_price:.2f}",
            "product": product,
            "instrument": "EQ",
            "exchange1": ""
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }
    
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data
    except Exception as e:
        return {"error": str(e)}

print("=== TEST A: SUZLON Intraday (Product I) - Buy 37, Sell 38, Qty 100 ===")
res_a = query_brokerage_exact(37.0, 38.0, 100, product="I", sec_id=12018)
print(json.dumps(res_a, indent=2))

print("=== TEST B: SUZLON Delivery/Swing (Product C) - Buy 37, Sell 38, Qty 100 ===")
res_b = query_brokerage_exact(37.0, 38.0, 100, product="C", sec_id=12018)
print(json.dumps(res_b, indent=2))

print("=== TEST C: Micro Trade - SUZLON Buy 37, Sell 38, Qty 35 (Turnover ~2625) ===")
res_c = query_brokerage_exact(37.0, 38.0, 35, product="I", sec_id=12018)
print(json.dumps(res_c, indent=2))
