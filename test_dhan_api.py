import urllib.request
import json

url = "https://openweb-api.dhan.co/brokerage"

def query_brokerage(buy_price, sell_price, qty, product="I", segment="E", exchange="NSE", sec_id="1333"):
    # product: "I" for Intraday, "C" for Delivery (Cash)
    payload = {
        "source": "N",
        "data": {
            "exchange": exchange,
            "segment": segment,
            "txn_type": "S" if (sell_price > 0 and buy_price > 0) else ("B" if buy_price > 0 else "S"),
            "qty": str(qty),
            "window": "ONLY_BUY" if (buy_price > 0 and sell_price == 0) else "BOTH",
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

# Test 1: Equity Intraday (Product "I")
print("=== TEST 1: Intraday (Product I) - 100 Buy, 101.5 Sell, Qty 25 (Turnover ~5037) ===")
res1 = query_brokerage(100.0, 101.5, 25, product="I")
print(json.dumps(res1, indent=2))

# Test 2: Equity Delivery (Product "C") - 100 Buy, 101.5 Sell, Qty 25
print("\n=== TEST 2: Delivery / Swing (Product C) - 100 Buy, 101.5 Sell, Qty 25 ===")
res2 = query_brokerage(100.0, 101.5, 25, product="C")
print(json.dumps(res2, indent=2))

# Test 3: Micro capital trade - SUZLON 70 Buy, 71 Sell, Qty 35 (Turnover ~4935)
print("\n=== TEST 3: Intraday Micro - SUZLON 70 Buy, 71 Sell, Qty 35 ===")
res3 = query_brokerage(70.0, 71.0, 35, product="I")
print(json.dumps(res3, indent=2))
