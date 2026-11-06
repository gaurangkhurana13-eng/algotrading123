import urllib.request
import json
import pandas as pd
import sys

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

url = "https://openweb-api.dhan.co/brokerage"

def get_dhan_brokerage(buy_price, sell_price, qty, product="I", exchange="NSE", sec_id=12018):
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
        "User-Agent": "Mozilla/5.0"
    }
    
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            if data.get("status") == "success" and data.get("data"):
                return data["data"][0]
            return {"error": data.get("message")}
    except Exception as e:
        return {"error": str(e)}

test_cases_intraday = [
    (37.0, 37.5, 35, "INR 2,600 (INR 500 cap 5x MIS)"),
    (70.0, 71.0, 35, "INR 4,935 (INR 1,000 cap 5x MIS)"),
    (100.0, 101.5, 25, "INR 5,037 (INR 1,000 cap 5x MIS)"),
    (100.0, 102.0, 50, "INR 10,100 (INR 2,000 cap 5x MIS)"),
    (200.0, 204.0, 62, "INR 25,048 (INR 5,000 cap 5x MIS)"),
    (500.0, 510.0, 50, "INR 50,500 (INR 10,000 cap 5x MIS)"),
    (1000.0, 1015.0, 100, "INR 201,500 (Large trade)"),
    (2000.0, 2020.0, 200, "INR 804,000 (Very large trade)")
]

rows_intraday = []
for bp, sp, q, lbl in test_cases_intraday:
    res = get_dhan_brokerage(bp, sp, q, product="I")
    buy_to = bp * q
    sell_to = sp * q
    total_to = buy_to + sell_to
    rows_intraday.append({
        "Label": lbl,
        "Turnover": total_to,
        "Brokerage": res.get("BROKERAGE"),
        "Exchange": res.get("EXCHANGE_CHARGES"),
        "STT": res.get("STT_CHARGES"),
        "SEBI": res.get("SEBI_CHARGES"),
        "Stamp": res.get("STAMP_DUTY"),
        "GST": res.get("GST_CHARGES"),
        "Total_Charges": res.get("TOTAL_CHARGES_TAX"),
        "Breakeven_Price": res.get("BREAKEVEN_PRICE")
    })

df_intraday = pd.DataFrame(rows_intraday)
print("=== DHAN INTRADAY CHARGES ANALYSIS ===")
print(df_intraday.to_string())

test_cases_delivery = [
    (37.0, 40.0, 35, "SUZLON Swing (Qty 35)"),
    (100.0, 110.0, 25, "Mid Swing (Qty 25)"),
    (500.0, 550.0, 10, "Large Swing (Qty 10)"),
    (1000.0, 1100.0, 50, "Big Swing (Qty 50)")
]

rows_delivery = []
for bp, sp, q, lbl in test_cases_delivery:
    res = get_dhan_brokerage(bp, sp, q, product="C")
    buy_to = bp * q
    sell_to = sp * q
    total_to = buy_to + sell_to
    rows_delivery.append({
        "Label": lbl,
        "Turnover": total_to,
        "Brokerage": res.get("BROKERAGE"),
        "DP_Charges": res.get("DP_CHARGES"),
        "Exchange": res.get("EXCHANGE_CHARGES"),
        "STT": res.get("STT_CHARGES"),
        "SEBI": res.get("SEBI_CHARGES"),
        "Stamp": res.get("STAMP_DUTY"),
        "GST": res.get("GST_CHARGES"),
        "Total_Charges": res.get("TOTAL_CHARGES_TAX")
    })

df_delivery = pd.DataFrame(rows_delivery)
print("\n=== DHAN DELIVERY / SWING CHARGES ANALYSIS ===")
print(df_delivery.to_string())
