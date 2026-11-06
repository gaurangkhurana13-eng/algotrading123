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

pname = '2019_UNSEEN'
s_dt = '2019-01-01'
e_dt = '2019-12-31 23:59:59'

out_dir = ROOT / "data" / pname
out_dir.mkdir(parents=True, exist_ok=True)
print(f"=== EXTRACTING DATASET FOR {pname} ({s_dt} to {e_dt}) ===")

universe = {}
audit = []
for sym in core_27:
    csv_path = KAGGLE_DIR / f"{sym}.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found!")
        continue
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
    audit.append({'symbol': sym, 'bars': bars, 'days': days, 'start': sub.index.min(), 'end': sub.index.max()})

# Save master universe
with open(out_dir / "universe_30_15m.pkl", 'wb') as f:
    pickle.dump(universe, f)
    
df_audit = pd.DataFrame(audit)
print(f"[{pname}] Successfully saved {len(universe)//2} stocks to {out_dir}")
print(f"  Bars summary: min={df_audit['bars'].min()}, max={df_audit['bars'].max()}, min_days={df_audit['days'].min()}, max_days={df_audit['days'].max()}")
print(f"  Date span: {df_audit['start'].min()} to {df_audit['end'].max()}")
