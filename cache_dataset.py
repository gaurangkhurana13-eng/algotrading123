# Cache 78 stocks 60-day 15m data locally
import sys
import os
import time
import pandas as pd
import yfinance as yf

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

UNIVERSE = [
    'ADANIPOWER.NS', 'ALOKINDS.NS', 'ASHOKLEY.NS', 'BANDHANBNK.NS', 'BANKBARODA.NS',
    'BANKINDIA.NS', 'BCLIND.NS', 'BPCL.NS', 'CANBK.NS', 'CENTRALBK.NS',
    'COALINDIA.NS', 'ELECTCAST.NS', 'EQUITASBNK.NS', 'GAIL.NS', 'GMRAIRPORT.NS',
    'GSFC.NS', 'HATHWAY.NS', 'HCC.NS', 'HUDCO.NS', 'IDBI.NS',
    'IDEA.NS', 'IDFCFIRSTB.NS', 'IFCI.NS', 'INDIANB.NS', 'IOB.NS',
    'IOC.NS', 'IRCON.NS', 'IRFC.NS', 'J&KBANK.NS', 'JAICORPLTD.NS',
    'JPPOWER.NS', 'JYOTHYLAB.NS', 'LLOYDSENGG.NS', 'MAHABANK.NS', 'MANALIPETC.NS',
    'MMTC.NS', 'MOREPENLAB.NS', 'MOTHERSON.NS', 'MRPL.NS', 'MTNL.NS',
    'NBCC.NS', 'NETWORK18.NS', 'NFL.NS', 'NHPC.NS', 'NIACL.NS',
    'NMDC.NS', 'NTPC.NS', 'ONGC.NS', 'PATANJALI.NS', 'PCJEWELLER.NS',
    'PFC.NS', 'PNB.NS', 'PTC.NS', 'RBLBANK.NS', 'RCF.NS',
    'RECLTD.NS', 'RENUKA.NS', 'RPOWER.NS', 'RVNL.NS', 'SAIL.NS',
    'SBIN.NS', 'SJVN.NS', 'SOUTHBANK.NS', 'SPIC.NS', 'SUBEXLTD.NS',
    'SUZLON.NS', 'SWSOLAR.NS', 'TATAPOWER.NS', 'TATASTEEL.NS', 'TEXRAIL.NS',
    'TRIDENT.NS', 'TTML.NS', 'UCOBANK.NS', 'UJJIVANSFB.NS', 'UNIONBANK.NS',
    'UTKARSHBNK.NS', 'VEDL.NS', 'YESBANK.NS'
]
INDEX_TICKER = '^NSEI'

CACHE_DIR = r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\data"
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, "universe_78_15m.pkl")

print(f"Downloading 60-day 15m candles for {len(UNIVERSE)} stocks + Nifty 50...")
t0 = time.time()
all_tickers = UNIVERSE + [INDEX_TICKER]
raw = yf.download(all_tickers, period="60d", interval="15m", progress=False, group_by="ticker")
elapsed = time.time() - t0
print(f"Downloaded in {elapsed:.1f}s. Saving to {CACHE_FILE}...")
raw.to_pickle(CACHE_FILE)
print(f"Successfully cached dataset! File size: {os.path.getsize(CACHE_FILE):,} bytes.")
