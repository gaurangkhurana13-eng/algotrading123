from pathlib import Path
import pandas as pd

k_dir = Path(r"C:\Users\admin\.cache\kagglehub\datasets\yeashusemwal\nse-stocks-15min-data-from-2019-03-01\versions\5")
csvs = list(k_dir.glob("*.csv"))
print(f"Total CSV files in Kaggle dataset: {len(csvs)}")

core_27 = [
    'CANBK', 'DELTACORP', 'FEDERALBNK', 'GMRINFRA', 'GSFC', 'IDFCFIRSTB', 'IOC',
    'JYOTHYLAB', 'MANAPPURAM', 'MOTHERSON', 'NBCC', 'NCC', 'NATIONALUM', 'NMDC',
    'POONAWALLA', 'PNB', 'RBLBANK', 'RCF', 'SAIL', 'SUZLON', 'TATACHEM',
    'TATASTEEL', 'TATAPOWER', 'TRIDENT', 'UNIONBANK', 'YESBANK', 'ZEEL'
]

present = [s for s in core_27 if (k_dir / f"{s}.csv").exists()]
print(f"Core stocks present: {len(present)} / {len(core_27)}")

results = []
for s in present:
    df = pd.read_csv(k_dir / f"{s}.csv")
    dcol = 'Datetime' if 'Datetime' in df.columns else 'datetime'
    sub = df[df[dcol].str.startswith('2019')]
    if len(sub) > 0:
        results.append({
            'symbol': s,
            'min_date': sub[dcol].min(),
            'max_date': sub[dcol].max(),
            'count': len(sub)
        })
    else:
        results.append({
            'symbol': s,
            'min_date': None,
            'max_date': None,
            'count': 0
        })

res_df = pd.DataFrame(results)
print(res_df.to_string())
