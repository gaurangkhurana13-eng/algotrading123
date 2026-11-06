import pandas as pd

v34 = pd.read_csv(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V34.3\fold_2026\trades_intraday.csv")
v36 = pd.read_csv(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V36\fold_2026\trades_intraday.csv")

print("--- V34.3 vs V36 (Fold 3 Bull) ---")
print(f"V34.3: Trades={len(v34)}, Gross PnL={v34['gross_pnl'].sum():.2f}, Charges={v34['charges'].sum():.2f}, Net={v34['net_pnl'].sum():.2f}")
print(f"V36:   Trades={len(v36)}, Gross PnL={v36['gross_pnl'].sum():.2f}, Charges={v36['charges'].sum():.2f}, Net={v36['net_pnl'].sum():.2f}")

diff_gross = v36['gross_pnl'].sum() - v34['gross_pnl'].sum()
diff_chg = v36['charges'].sum() - v34['charges'].sum()
diff_net = v36['net_pnl'].sum() - v34['net_pnl'].sum()
print(f"Difference: Gross={diff_gross:+.2f}, Charges={diff_chg:+.2f}, Net={diff_net:+.2f}")

v34_wins = v34[v34['net_pnl'] > 0]
v34_losses = v34[v34['net_pnl'] < 0]
v36_wins = v36[v36['net_pnl'] > 0]
v36_losses = v36[v36['net_pnl'] < 0]

print(f"\nV34.3 Wins: count={len(v34_wins)}, sum={v34_wins['net_pnl'].sum():.2f}, avg={v34_wins['net_pnl'].mean():.2f}, max={v34_wins['net_pnl'].max():.2f}")
print(f"V34.3 Losses: count={len(v34_losses)}, sum={v34_losses['net_pnl'].sum():.2f}, avg={v34_losses['net_pnl'].mean():.2f}, min={v34_losses['net_pnl'].min():.2f}")

print(f"\nV36 Wins: count={len(v36_wins)}, sum={v36_wins['net_pnl'].sum():.2f}, avg={v36_wins['net_pnl'].mean():.2f}, max={v36_wins['net_pnl'].max():.2f}")
print(f"V36 Losses: count={len(v36_losses)}, sum={v36_losses['net_pnl'].sum():.2f}, avg={v36_losses['net_pnl'].mean():.2f}, min={v36_losses['net_pnl'].min():.2f}")

print("\nTop 5 Winners V34.3:")
print(v34.nlargest(5, 'net_pnl')[['symbol', 'entry_time', 'shares', 'gross_pnl', 'charges', 'net_pnl', 'exit_reason']])

print("\nTop 5 Winners V36:")
print(v36.nlargest(5, 'net_pnl')[['symbol', 'entry_time', 'shares', 'gross_pnl', 'charges', 'net_pnl', 'exit_reason']])
