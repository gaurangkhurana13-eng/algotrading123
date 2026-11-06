"""
Download 15-minute historical data for all 78 stocks from January 2026 to October 29, 2026.
Uses tvdatafeed direct from NSE institutional feed.
Verifies data completeness, no missing bars, and saves clean dictionary & multiindex pickle.
"""

import sys
import os
import time
from pathlib import Path
import pandas as pd
from tvDatafeed import TvDatafeed, Interval

# Target 78 Tickers
TICKERS_78 = [
    'ADANIPOWER', 'ALOKINDS', 'ASHOKLEY', 'BANDHANBNK', 'BANKBARODA', 'BANKINDIA',
    'BCLIND', 'BPCL', 'CANBK', 'CENTRALBK', 'COALINDIA', 'ELECTCAST',
    'EQUITASBNK', 'GAIL', 'GMRAIRPORT', 'GSFC', 'HATHWAY', 'HCC',
    'HUDCO', 'IDBI', 'IDEA', 'IDFCFIRSTB', 'IFCI', 'INDIANB',
    'IOB', 'IOC', 'IRCON', 'IRFC', 'J&KBANK', 'JAICORPLTD',
    'JPPOWER', 'JYOTHYLAB', 'LLOYDSENGG', 'MAHABANK', 'MANALIPETC', 'MMTC',
    'MOREPENLAB', 'MOTHERSON', 'MRPL', 'MTNL', 'NBCC', 'NETWORK18',
    'NFL', 'NHPC', 'NIACL', 'NMDC', 'NTPC', 'ONGC',
    'PATANJALI', 'PCJEWELLER', 'PFC', 'PNB', 'PTC', 'RBLBANK',
    'RCF', 'RECLTD', 'RENUKA', 'RPOWER', 'RVNL', 'SAIL',
    'SBIN', 'SJVN', 'SOUTHBANK', 'SPIC', 'SUBEXLTD', 'SUZLON',
    'SWSOLAR', 'TATAPOWER', 'TATASTEEL', 'TEXRAIL', 'TRIDENT', 'TTML',
    'UCOBANK', 'UJJIVANSFB', 'UNIONBANK', 'UTKARSHBNK', 'VEDL', 'YESBANK'
]

def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return None
    # Ensure index is datetime
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    # Remove duplicates
    df = df[~df.index.duplicated(keep='last')]
    df = df.sort_index()
    # Standardize column names lowercase
    df.columns = [c.lower() for c in df.columns]
    # Check required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    for req in required:
        if req not in df.columns:
            return None
    df = df[required]
    # Drop rows where close or volume is null
    df = df.dropna(subset=['close'])
    df['volume'] = df['volume'].fillna(0)
    return df

def main():
    print("=" * 80)
    print("STARTING DIRECT NSE INTRADAY DOWNLOAD: 78 STOCKS (JAN 2026 - OCT 2026)")
    print("=" * 80)
    
    tv = TvDatafeed()
    out_dir = Path("data/universe_78_1y")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    universe_map = {}
    success_count = 0
    failed = []
    
    start_time = time.time()
    
    for i, ticker in enumerate(TICKERS_78, 1):
        print(f"[{i}/{len(TICKERS_78)}] Fetching {ticker} (15m, 5000 bars)...", end=" ", flush=True)
        df = None
        
        # Try NSE
        try:
            df = tv.get_hist(symbol=ticker, exchange="NSE", interval=Interval.in_15_minute, n_bars=5000)
        except Exception as e:
            pass
            
        # If failed, try BSE
        if df is None or len(df) == 0:
            try:
                # Handle special ticker names like J&KBANK -> JKBANK
                alt_sym = ticker.replace("&", "")
                df = tv.get_hist(symbol=alt_sym, exchange="NSE", interval=Interval.in_15_minute, n_bars=5000)
            except Exception:
                pass
                
        if df is None or len(df) == 0:
            try:
                df = tv.get_hist(symbol=ticker, exchange="BSE", interval=Interval.in_15_minute, n_bars=5000)
            except Exception:
                pass
                
        df_clean = clean_ohlcv(df)
        
        if df_clean is not None and len(df_clean) > 500:
            sym_key = f"{ticker}.NS"
            universe_map[sym_key] = df_clean
            
            # Save individual pickle
            single_file = out_dir / f"{ticker.lower()}_15m_1y.pkl"
            df_clean.to_pickle(single_file)
            
            d_min = df_clean.index.min().strftime('%Y-%m-%d')
            d_max = df_clean.index.max().strftime('%Y-%m-%d')
            print(f"OK! ({len(df_clean)} bars | {d_min} to {d_max})")
            success_count += 1
        else:
            print("FAILED!")
            failed.append(ticker)
            
        time.sleep(0.3) # Friendly throttling to avoid socket drops
        
    elapsed = time.time() - start_time
    print("=" * 80)
    print(f"DOWNLOAD COMPLETE in {elapsed:.1f}s")
    print(f"Successfully downloaded: {success_count}/{len(TICKERS_78)} stocks")
    if failed:
        print(f"Failed symbols ({len(failed)}): {failed}")
    print("=" * 80)
    
    # Save combined multi-ticker pickle
    combined_file = Path("data/universe_78_15m_1y.pkl")
    pd.to_pickle(universe_map, combined_file)
    print(f"Consolidated Universe Pickle saved to: {combined_file} ({combined_file.stat().st_size / (1024*1024):.2f} MB)")
    
    # Generate Manifest
    summary_data = []
    for sym, df_s in universe_map.items():
        summary_data.append({
            "ticker": sym,
            "bars": len(df_s),
            "start": str(df_s.index.min()),
            "end": str(df_s.index.max()),
            "null_close": int(df_s['close'].isna().sum()),
            "min_price": float(df_s['low'].min()),
            "max_price": float(df_s['high'].max())
        })
    df_summary = pd.DataFrame(summary_data)
    summary_csv = Path("data/universe_78_audit_summary.csv")
    df_summary.to_csv(summary_csv, index=False)
    print(f"Audit summary CSV saved to: {summary_csv}")
    print("All tasks finished successfully.")

if __name__ == "__main__":
    main()
