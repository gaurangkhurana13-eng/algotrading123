"""
Sector classification for 78 NSE universe tickers.
Prevents correlated sector concentration risk.
"""

SECTOR_MAP = {
    # Banking & Financials
    "HDFCBANK": "FINANCIALS", "ICICIBANK": "FINANCIALS", "SBIN": "FINANCIALS",
    "KOTAKBANK": "FINANCIALS", "AXISBANK": "FINANCIALS", "INDUSINDBK": "FINANCIALS",
    "BAJFINANCE": "FINANCIALS", "BAJAJFINSV": "FINANCIALS", "CHOLAFIN": "FINANCIALS",
    "MUTHOOTFIN": "FINANCIALS", "PFC": "FINANCIALS", "RECLTD": "FINANCIALS",
    "SHRIRAMFIN": "FINANCIALS", "HDFCLIFE": "FINANCIALS", "SBILIFE": "FINANCIALS",
    # IT & Software
    "INFY": "IT", "TCS": "IT", "HCLTECH": "IT", "WIPRO": "IT",
    "TECHM": "IT", "LTIM": "IT", "COFORGE": "IT", "PERSISTENT": "IT",
    # Auto & Ancillaries
    "TATAMOTORS": "AUTO", "MARUTI": "AUTO", "M&M": "AUTO", "BAJAJ-AUTO": "AUTO",
    "HEROMOTOCO": "AUTO", "EICHERMOT": "AUTO", "BHARATFORG": "AUTO", "TVSMOTOR": "AUTO",
    # Metals & Mining
    "TATASTEEL": "METALS", "JSWSTEEL": "METALS", "HINDALCO": "METALS",
    "VEDL": "METALS", "NMDC": "METALS", "SAIL": "METALS", "JINDALSTEL": "METALS",
    # Energy, Oil & Power
    "RELIANCE": "ENERGY", "ONGC": "ENERGY", "NTPC": "ENERGY", "POWERGRID": "ENERGY",
    "COALINDIA": "ENERGY", "BPCL": "ENERGY", "IOC": "ENERGY", "TATAPOWER": "ENERGY",
    "ADANIGREEN": "ENERGY", "ADANIPOWER": "ENERGY",
    # Pharma & Healthcare
    "SUNPHARMA": "PHARMA", "DRREDDY": "PHARMA", "CIPLA": "PHARMA",
    "DIVISLAB": "PHARMA", "APOLLOHOSP": "PHARMA", "LUPIN": "PHARMA", "TORNTPHARM": "PHARMA",
    # FMCG & Consumption
    "ITC": "FMCG", "HINDUNILVR": "FMCG", "NESTLEIND": "FMCG",
    "BRITANNIA": "FMCG", "TATACONSUM": "FMCG", "DABUR": "FMCG", "MARICO": "FMCG",
    # Industrials & Infrastructure
    "LT": "INDUSTRIALS", "ADANIENT": "INDUSTRIALS", "ADANIPORTS": "INDUSTRIALS",
    "SIEMENS": "INDUSTRIALS", "ABB": "INDUSTRIALS", "BEL": "INDUSTRIALS", "HAL": "INDUSTRIALS",
    "BHEL": "INDUSTRIALS"
}

def get_sector(ticker: str) -> str:
    cleaned = ticker.replace(".NS", "").upper()
    return SECTOR_MAP.get(cleaned, "GENERAL")
