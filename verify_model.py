import urllib.request
import json
import math

def calc_dhan_charges_model(buy_price: float, sell_price: float, qty: int, product="I"):
    buy_to = buy_price * qty
    sell_to = sell_price * qty
    total_to = buy_to + sell_to
    
    if product == "I": # Intraday Equity
        # Brokerage: 0.03% or Rs 20 whichever is lower, per executed order. NO MINIMUM!
        buy_brok = min(20.0, buy_to * 0.0003)
        sell_brok = min(20.0, sell_to * 0.0003)
        brokerage = buy_brok + sell_brok
        
        # Exchange turnover charge (NSE): approx 0.00307% (0.00297% - 0.00325%)
        # Dhan actual in API: total_to * 0.0000307
        exchange_charges = total_to * 0.0000307
        
        # STT: 0.025% on sell turnover, rounded to nearest rupee
        stt = round(sell_to * 0.00025)
        
        # SEBI: Rs 10 per crore (0.0001%)
        sebi = total_to * 0.000001
        
        # Stamp duty: 0.003% on buy side, rounded
        stamp = round(buy_to * 0.00003)
        
        # GST: 18% on (Brokerage + Exchange + SEBI)
        gst = (brokerage + exchange_charges + sebi) * 0.18
        
        total = brokerage + exchange_charges + stt + sebi + stamp + gst
        return {
            "brokerage": round(brokerage, 4),
            "exchange": round(exchange_charges, 4),
            "stt": stt,
            "sebi": round(sebi, 4),
            "stamp": stamp,
            "gst": round(gst, 4),
            "total": round(total, 2)
        }
    elif product == "C": # Delivery / Swing
        brokerage = 0.0 # Free delivery
        dp_charges = 14.75 # Rs 12.50 + 18% GST on selling
        exchange_charges = total_to * 0.0000307
        stt = round(total_to * 0.001) # 0.1% on buy and sell
        sebi = total_to * 0.000001
        stamp = round(buy_to * 0.00015) # 0.015% on buy
        gst = (brokerage + exchange_charges + sebi) * 0.18
        total = brokerage + dp_charges + exchange_charges + stt + sebi + stamp + gst
        return {
            "brokerage": 0.0,
            "dp_charges": dp_charges,
            "exchange": round(exchange_charges, 4),
            "stt": stt,
            "sebi": round(sebi, 4),
            "stamp": stamp,
            "gst": round(gst, 4),
            "total": round(total, 2)
        }

# Compare model vs actual Dhan API
import test_exact

test_cases = [
    (37.0, 37.5, 35),
    (70.0, 71.0, 35),
    (100.0, 101.5, 25),
    (200.0, 204.0, 62),
    (500.0, 510.0, 50)
]

print("=== COMPARISON: OUR REVERSE-ENGINEERED FORMULA VS DHAN LIVE API ===")
for bp, sp, q in test_cases:
    m = calc_dhan_charges_model(bp, sp, q, "I")
    a = test_exact.query_brokerage_exact(bp, sp, q, product="I", sec_id=12018)["data"][0]
    print(f"Trade: Buy {bp}, Sell {sp}, Qty {q} (Turnover: {bp*q + sp*q:.0f})")
    print(f"  Brokerage: Model = {m['brokerage']:.3f} | Dhan API = {a['BROKERAGE']:.3f}")
    print(f"  Total:     Model = {m['total']:.2f}  | Dhan API = {a['TOTAL_CHARGES_TAX']:.2f}")
    print(f"  Difference: {abs(m['total'] - a['TOTAL_CHARGES_TAX']):.2f} Rs\n")
