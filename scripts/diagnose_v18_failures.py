import pandas as pd
from test_v18_calibrated import df_v18

print("=== V18 EXIT REASON BREAKDOWN ===")
for r in df_v18["exit_reason"].unique():
    dfr = df_v18[df_v18["exit_reason"] == r]
    print(f"{r:<30}: Trades={len(dfr):<3} | Wins={(dfr['net_pnl']>0).sum():<3} | Losses={(dfr['net_pnl']<0).sum():<3} | Net=Rs {dfr['net_pnl'].sum():>7.2f}")

print("\n=== V18 SYMBOL BREAKDOWN ===")
for s in df_v18["symbol"].unique():
    dfs = df_v18[df_v18["symbol"] == s]
    print(f"{s:<15}: Trades={len(dfs):<3} | Wins={(dfs['net_pnl']>0).sum():<3} | Losses={(dfs['net_pnl']<0).sum():<3} | Net=Rs {dfs['net_pnl'].sum():>7.2f}")
