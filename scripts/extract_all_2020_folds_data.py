import os, sys, pickle
from pathlib import Path
import pandas as pd

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
KAGGLE_DIR = Path(r"C:\Users\admin\.cache\kagglehub\datasets\yeashusemwal\nse-stocks-15min-data-from-2019-03-01\versions\5")

core_27 = [
    'CANBK', 'DELTACORP', 'FEDERALBNK', 'GMRINFRA', 'GSFC', 'IDFCFIRSTB', 'IOC',
    'JYOTHYLAB', 'MANAPPURAM', 'MOTHERSON', 'NBCC', 'NCC', 'NATIONALUM', 'NMDC',
    'POONAWALLA', 'PNB', 'RBLBANK', 'RCF', 'SAIL', 'SUZLON', 'TATACHEM',
    'TATASTEEL', 'TATAPOWER', 'TRIDENT', 'UNIONBANK', 'YESBANK', 'ZEEL'
]

periods = [
    ('2020_FIRST4M', '2020-01-01', '2020-04-30 23:59:59'),
    ('2020_LAST4M', '2020-09-01', '2020-12-31 23:59:59'),
    ('2020_FULL', '2020-01-01', '2020-12-31 23:59:59')
]

for pname, s_dt, e_dt in periods:
    out_dir = ROOT / "data" / pname
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== EXTRACTING DATASET FOR {pname} ({s_dt} to {e_dt}) ===")
    
    universe = {}
    audit = []
    for sym in core_27:
        csv_path = KAGGLE_DIR / f"{sym}.csv"
        df = pd.read_csv(csv_path)
        df.columns = [c.lower() for c in df.columns]
        date_col = 'datetime' if 'datetime' in df.columns else 'date'
        df['datetime'] = pd.to_datetime(df[date_col])
        
        sub = df[(df['datetime'] >= s_dt) & (df['datetime'] <= e_dt)].copy()
        sub = sub.sort_values('datetime').set_index('datetime')
        sub = sub[['open', 'high', 'low', 'close', 'volume']].copy()
        sub = sub.between_time('09:15', '15:15').dropna()
        
        # Save individual CSV
        sub.to_csv(out_dir / f"{sym}.csv")
        
        # Save individual PKL for Gate 1 Check 2
        with open(out_dir / f"{sym.lower()}_15m_1y.pkl", 'wb') as f:
            pickle.dump(sub, f)
            
        universe[sym] = sub
        universe[f"{sym}.NS"] = sub
        
        days = len(set(sub.index.date))
        bars = len(sub)
        audit.append({'symbol': sym, 'bars': bars, 'days': days})
        
    # Save master universe
    with open(out_dir / "universe_30_15m.pkl", 'wb') as f:
        pickle.dump(universe, f)
        
    df_audit = pd.DataFrame(audit)
    print(f"[{pname}] Saved {len(universe)//2} stocks to {out_dir}")
    print(f"  Bars summary: min={df_audit['bars'].min()}, max={df_audit['bars'].max()}, days={df_audit['days'].iloc[0]}")
