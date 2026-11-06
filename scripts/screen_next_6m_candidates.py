import pickle
import pandas as pd
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    univ = pickle.load(f)

# Screen 78 stocks for candidates in price bracket Rs 15 - Rs 250 with high daily volume and high NATR
candidates = []
for t, df in univ.items():
    if len(df) < 1000:
        continue
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]
    
    avg_price = c.tail(500).mean()
    if avg_price < 15.0 or avg_price > 250.0:
        continue
        
    avg_vol = v.tail(500).mean()
    daily_turnover_approx = (avg_vol * 25 * avg_price) / 1e7 # in Crores per day
    if daily_turnover_approx < 5.0: # Minimum Rs 5 Crore daily turnover
        continue
        
    # Calculate 15m NATR
    tr = (h - l).tail(500)
    natr = (tr / c.tail(500)).mean() * 100
    
    candidates.append({
        "Ticker": t.replace(".NS", ""),
        "Current Price (Rs)": round(avg_price, 1),
        "Daily Turnover (Cr)": round(daily_turnover_approx, 1),
        "15m NATR (%)": round(natr, 2)
    })

df_cand = pd.DataFrame(candidates).sort_values("15m NATR (%)", ascending=False)
print("=== TOP HIGH-BETA LIQUID CANDIDATES (RS 15 - RS 250) ===")
print(df_cand.head(20).to_string(index=False))
