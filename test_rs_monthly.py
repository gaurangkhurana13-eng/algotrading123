import pickle
import pandas as pd
import numpy as np
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")

# Load 2024 & 2025-26 data
with open(ROOT_DIR / "data" / "2024-25" / "universe_14_15m.pkl", "rb") as f:
    u24 = pickle.load(f)
with open(ROOT_DIR / "data" / "2025-26" / "universe_14_15m.pkl", "rb") as f:
    u25 = pickle.load(f)

tickers = [
    "PCJEWELLER.NS", "RPOWER.NS", "DELTACORP.NS", "RITES.NS",
    "IDEA.NS", "MTNL.NS", "ZEEL.NS", "UTKARSHBNK.NS", "IFCI.NS", "NCC.NS",
    "TEXRAIL.NS", "JPPOWER.NS", "PTC.NS", "NIACL.NS"
]

data_daily = {}
for t in tickers:
    raw_s = t.replace(".NS", "")
    df24 = u24.get(t, u24.get(raw_s))
    df25 = u25.get(t, u25.get(raw_s))
    df = pd.concat([df24, df25]).sort_index() if df24 is not None else df25.copy()
    df = df[~df.index.duplicated(keep="last")]
    # Resample to daily
    daily = df.resample("1D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum"
    }).dropna()
    data_daily[t] = daily

# Inspect RS on last day of May 2025 (for June 2025), June (for July), etc.
test_dates = [
    "2025-05-30", "2025-06-30", "2025-07-31", "2025-08-29",
    "2025-09-30", "2025-10-31", "2025-11-28", "2025-12-31"
]

for td in test_dates:
    t_m = pd.Timestamp(td)
    scores = []
    
    # Compute universe average 20d return as bench proxy
    bench_r20_list = []
    for t in tickers:
        df = data_daily[t].loc[:t_m]
        if len(df) >= 21:
            P = df["close"].iloc[-1]
            r20 = P / df["close"].iloc[-21] - 1.0
            bench_r20_list.append(r20)
    bench_r20 = np.mean(bench_r20_list) if bench_r20_list else 0.0
    
    for t in tickers:
        df = data_daily[t].loc[:t_m]
        if len(df) < 70:
            continue
        P = df["close"].iloc[-1]
        if not (10.0 <= P <= 450.0):
            continue
        
        # ADV20
        adv20 = (df["close"] * df["volume"]).tail(20).median()
        # Daily ATR14
        h, l, c_prev = df["high"], df["low"], df["close"].shift(1)
        tr = np.maximum(h - l, np.maximum((h - c_prev).abs(), (l - c_prev).abs()))
        atr14 = tr.ewm(alpha=1/14).mean().iloc[-1]
        atr14_pct = (atr14 / P) * 100.0
        
        r20 = P / df["close"].iloc[-21] - 1.0
        r60 = P / df["close"].iloc[-61] - 1.0
        ema20 = df["close"].ewm(span=20).mean().iloc[-1]
        e = P / ema20 - 1.0
        x20 = r20 - bench_r20
        
        scores.append({
            "ticker": t.replace(".NS", ""),
            "P": round(P, 2),
            "adv20_cr": round(adv20 / 1e7, 2),
            "atr_pct": round(atr14_pct, 2),
            "r20": r20,
            "r60": r60,
            "e": e,
            "x20": x20
        })
        
    N = len(scores)
    if N == 0:
        continue
        
    # Percentile ranks
    for feat in ["r20", "r60", "e", "x20"]:
        vals = sorted([s[feat] for s in scores])
        for s in scores:
            # mid-rank percentile
            rank = (vals.index(s[feat]) + len(vals) - 1 - list(reversed(vals)).index(s[feat])) / 2.0 + 1.0
            s[f"p_{feat}"] = (rank - 0.5) / N
            
    for s in scores:
        s["RS"] = round(0.35 * s["p_r20"] + 0.30 * s["p_r60"] + 0.15 * s["p_e"] + 0.20 * s["p_x20"], 3)
        
    scores.sort(key=lambda x: x["RS"], reverse=True)
    print(f"\n=== Month starting after {td} (Pool N={N}) ===")
    top = [f"{s['ticker']}({s['RS']})" for s in scores[:8]]
    bottom = [f"{s['ticker']}({s['RS']})" for s in scores[8:]]
    print("TOP Leaders:", ", ".join(top))
    print("BOTTOM Laggards:", ", ".join(bottom))
