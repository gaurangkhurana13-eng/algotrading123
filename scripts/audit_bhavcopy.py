import os
import glob
import pickle
from datetime import datetime
import pandas as pd
import numpy as np

# Load universe pickle
universe_file = r'C:\Users\admin\.gemini\antigravity\scratch\new_trade\data\2021-22\universe_30_15m.pkl'
with open(universe_file, 'rb') as f:
    universe = pickle.load(f)

test_dates = [
    '2021-01-15',
    '2021-03-23',
    '2021-05-18',
    '2021-06-25',
    '2021-08-17',
    '2021-09-29',
    '2021-11-22',
    '2021-12-15'
]

core_stocks = ['TATASTEEL', 'PNB', 'CENTRALBK', 'DELTACORP', 'ZEEL', 'IOB', 'NMDC']

all_data = []

for dt_str in test_dates:
    dt = datetime.strptime(dt_str, '%Y-%m-%d')
    fname_tag = dt.strftime('%d%b%Y').upper()
    bhav_file = f'data/bhavcopy_audit/cm{fname_tag}bhav.csv'
    
    bhav_df = pd.read_csv(bhav_file)
    bhav_df.columns = [c.strip() for c in bhav_df.columns]
    bhav_df['SYMBOL'] = bhav_df['SYMBOL'].str.strip()
    bhav_df['SERIES'] = bhav_df['SERIES'].str.strip()
    
    for symbol in core_stocks:
        stock_df = universe[symbol]
        bars = stock_df.loc[dt_str]
        
        c_open = float(bars.iloc[0]['open'])
        c_high = float(bars['high'].max())
        c_low = float(bars['low'].min())
        c_close = float(bars.iloc[-1]['close'])
        c_vol = int(bars['volume'].sum())
        bar_count = len(bars)
        
        bhav_match = bhav_df[(bhav_df['SYMBOL'] == symbol) & (bhav_df['SERIES'] == 'EQ')]
        if len(bhav_match) == 0:
            print(f"NOT FOUND: {symbol} on {dt_str}")
            continue
            
        bhav_row = bhav_match.iloc[0]
        b_open = float(bhav_row['OPEN'])
        b_high = float(bhav_row['HIGH'])
        b_low = float(bhav_row['LOW'])
        b_close = float(bhav_row['CLOSE'])
        b_vol = int(bhav_row['TOTTRDQTY'])
        
        diff_open = c_open - b_open
        pct_open = (diff_open / b_open) * 100.0
        
        diff_high = c_high - b_high
        pct_high = (diff_high / b_high) * 100.0
        
        diff_low = c_low - b_low
        pct_low = (diff_low / b_low) * 100.0
        
        diff_close = c_close - b_close
        pct_close = (diff_close / b_close) * 100.0
        
        diff_vol = c_vol - b_vol
        pct_vol = (diff_vol / b_vol) * 100.0
        coverage_vol = (c_vol / b_vol) * 100.0
        
        all_data.append({
            'date': dt_str,
            'symbol': symbol,
            'bars': bar_count,
            'b_open': b_open, 'c_open': c_open, 'diff_open': diff_open, 'pct_open': pct_open,
            'b_high': b_high, 'c_high': c_high, 'diff_high': diff_high, 'pct_high': pct_high,
            'b_low': b_low, 'c_low': c_low, 'diff_low': diff_low, 'pct_low': pct_low,
            'b_close': b_close, 'c_close': c_close, 'diff_close': diff_close, 'pct_close': pct_close,
            'b_vol': b_vol, 'c_vol': c_vol, 'diff_vol': diff_vol, 'pct_vol': pct_vol,
            'vol_coverage': coverage_vol
        })

df = pd.DataFrame(all_data)

# Also test ZEEL merger date 2021-09-14
import requests, io, zipfile
url = 'https://archives.nseindia.com/content/historical/EQUITIES/2021/SEP/cm14SEP2021bhav.csv.zip'
headers = {'User-Agent': 'Mozilla/5.0'}
r = requests.get(url, headers=headers)
if r.status_code == 200:
    z = zipfile.ZipFile(io.BytesIO(r.content))
    bhav_sep14 = pd.read_csv(z.open(z.namelist()[0]))
    bhav_sep14.columns = [c.strip() for c in bhav_sep14.columns]
    bhav_sep14['SYMBOL'] = bhav_sep14['SYMBOL'].str.strip()
    bhav_sep14['SERIES'] = bhav_sep14['SERIES'].str.strip()
    row_z = bhav_sep14[(bhav_sep14['SYMBOL'] == 'ZEEL') & (bhav_sep14['SERIES'] == 'EQ')].iloc[0]
    bars_z = universe['ZEEL'].loc['2021-09-14']
    print("\n--- ZEEL High-Volatility Merger Day (2021-09-14) Forensic Verification ---")
    print(f"Bhavcopy: Open={row_z['OPEN']}, High={row_z['HIGH']}, Low={row_z['LOW']}, Close={row_z['CLOSE']}, Vol={int(row_z['TOTTRDQTY']):,}")
    print(f"Calculated: Open={bars_z.iloc[0]['open']}, High={bars_z['high'].max()}, Low={bars_z['low'].min()}, Close={bars_z.iloc[-1]['close']}, Vol={int(bars_z['volume'].sum()):,}")
    print(f"Diff%: Open={(bars_z.iloc[0]['open'] - row_z['OPEN'])/row_z['OPEN']*100:.2f}%, High={(bars_z['high'].max() - row_z['HIGH'])/row_z['HIGH']*100:.2f}%, Low={(bars_z['low'].min() - row_z['LOW'])/row_z['LOW']*100:.2f}%, Close={(bars_z.iloc[-1]['close'] - row_z['CLOSE'])/row_z['CLOSE']*100:.2f}%, Vol Cov={bars_z['volume'].sum()/row_z['TOTTRDQTY']*100:.2f}%")

df.to_csv('data/bhavcopy_audit/audit_comparison_results.csv', index=False)
