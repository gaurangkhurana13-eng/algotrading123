"""
Dhan MIS Execution Engine for PC Jeweller Intraday Cash Segment.
Strictly calculates brokerage, STT, exchange turnover fees, GST, SEBI turnover fees, and stamp duty.
"""

def calc_dhan_charges(buy_price: float, sell_price: float, qty: int) -> float:
    if qty <= 0:
        return 0.0
    turnover = (buy_price + sell_price) * qty
    brokerage = min(20.0, turnover * 0.0003)
    stt = (sell_price * qty) * 0.00025
    exchange_charges = turnover * 0.0000325
    gst = (brokerage + exchange_charges) * 0.18
    sebi_charges = turnover * 0.000001
    stamp_duty = (buy_price * qty) * 0.00003
    total = brokerage + stt + exchange_charges + gst + sebi_charges + stamp_duty
    return round(total, 2)
