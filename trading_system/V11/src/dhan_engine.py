"""
Dhan Equity Intraday Transaction Cost Engine
Reverse-engineered exact formula matching Dhan production API:
- Brokerage: 0.03% or Rs 20 (whichever lower per leg, zero minimum floor)
- Exchange TurnOver Charges (NSE): 0.00307%
- Securities Transaction Tax (STT): 0.025% on sell turnover
- SEBI Turnover Charges: 0.0001%
- Stamp Duty: 0.003% on buy turnover
- GST: 18% on (Brokerage + Exchange + SEBI)
"""

def calc_dhan_charges(buy_price: float, sell_price: float, qty: int) -> float:
    buy_to   = float(buy_price)  * qty
    sell_to  = float(sell_price) * qty
    total_to = buy_to + sell_to
    
    brok  = min(20.0, buy_to * 0.0003) + min(20.0, sell_to * 0.0003)
    exc   = total_to * 0.0000307
    stt   = sell_to * 0.00025
    sebi  = total_to * 0.000001
    stamp = buy_to * 0.00003
    gst   = (brok + exc + sebi) * 0.18
    return round(brok + exc + stt + sebi + stamp + gst, 2)
