import pandas as pd

v34 = pd.read_csv(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V34.3\fold_2026\trades_intraday.csv")
v36 = pd.read_csv(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V36\fold_2026\trades_intraday.csv")

print("--- MTNL around 2026-10-01 in V34.3 ---")
print(v34[v34['symbol'] == 'MTNL'])

print("\n--- MTNL around 2026-10-01 in V36 ---")
print(v36[v36['symbol'] == 'MTNL'])

print("\n--- What was V36 doing on 2026-10-01? ---")
print(v36[v36['entry_time'].str.startswith('2026-10-01')])
