import pickle
import pandas as pd
import numpy as np

DATA_FILE = r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\data\unseen_high_beta_10_15m_1y.pkl"
with open(DATA_FILE, "rb") as f:
    univ = pickle.load(f)

print("Analyzing 2026 Full-Year Regime & Volatility Characteristics of 10 Unseen Stocks:\n")
rows = []
for sym, df in univ.items():
    c = df['close']
    v = df['volume']
    start_p = c.iloc[0]
    end_p = c.iloc[-1]
    net_change_pct = (end_p - start_p) / start_p * 100
    daily_ret = c.resample('1D').last().dropna().pct_change()
    ann_vol = daily_ret.std() * np.sqrt(252) * 100
    
    # 200 EMA slope or trend ratio
    ema200 = c.ewm(span=200).mean()
    pct_above_ema = (c > ema200).mean() * 100
    
    rows.append({
        "symbol": sym.replace(".NS", ""),
        "start_price": round(start_p, 2),
        "end_price": round(end_p, 2),
        "year_return_%": round(net_change_pct, 1),
        "ann_volatility_%": round(ann_vol, 1),
        "pct_bars_above_ema200": round(pct_above_ema, 1)
    })

df_diag = pd.DataFrame(rows)
print(df_diag.to_string(index=False))
