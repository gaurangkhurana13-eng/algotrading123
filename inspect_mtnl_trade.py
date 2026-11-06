import pandas as pd

v34 = pd.read_csv(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V34.3\fold_2026\trades_intraday.csv")
v36 = pd.read_csv(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V36\fold_2026\trades_intraday.csv")

row_34 = v34.loc[94]
row_36 = v36.loc[137]

print("=== V34.3 MTNL Trade 94 ===")
for k, v in row_34.items():
    print(f"  {k}: {v}")

print("\n=== V36 MTNL Trade 137 ===")
for k, v in row_36.items():
    print(f"  {k}: {v}")
