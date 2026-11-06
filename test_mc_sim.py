import numpy as np

def simulate_trade_mc(entry_p, atr, tp_mult=2.4, sl_mult=1.0, be_mult=1.0, drift=0.001, hold_bars=12, qty=100, n_sims=10000):
    # Daily/15m volatility per bar
    sigma = atr / np.sqrt(hold_bars)
    
    # 10,000 paths of 12 steps
    dt = 1.0
    # Shock matrix: (10000, hold_bars)
    shocks = np.random.normal(loc=drift, scale=sigma, size=(n_sims, hold_bars))
    # Cumulative return paths
    cum_ret = np.cumsum(shocks, axis=1)
    paths = entry_p * (1.0 + cum_ret)
    
    tp_price = entry_p * (1.0 + atr * tp_mult)
    sl_price = entry_p * (1.0 - atr * sl_mult)
    be_price = entry_p * (1.0 + atr * be_mult)
    
    wins = 0
    losses = 0
    breakevens = 0
    net_pnls = []
    
    # Check outcomes across paths
    for i in range(n_sims):
        path = paths[i]
        be_hit = False
        curr_sl = sl_price
        hit = False
        
        for p in path:
            if not be_hit and p >= be_price:
                be_hit = True
                curr_sl = entry_p * 1.0005 # Move SL to breakeven
            
            if p >= tp_price:
                wins += 1
                gross = (tp_price - entry_p) * qty
                net_pnls.append(gross)
                hit = True
                break
            elif p <= curr_sl:
                if be_hit:
                    breakevens += 1
                    net_pnls.append(0.0)
                else:
                    losses += 1
                    gross = (sl_price - entry_p) * qty
                    net_pnls.append(gross)
                hit = True
                break
        
        if not hit:
            # Time exit
            final_p = path[-1]
            gross = (final_p - entry_p) * qty
            if gross > 0: wins += 1
            else: losses += 1
            net_pnls.append(gross)
            
    net_arr = np.array(net_pnls)
    ev = np.mean(net_arr)
    win_pct = wins / n_sims
    print(f"MC 10,000 Sims: EV=Rs {ev:.2f} | Win Rate={win_pct*100:.1f}% | Losses={losses/n_sims*100:.1f}% | BE={breakevens/n_sims*100:.1f}%")
    return ev > 0 and win_pct >= 0.50

simulate_trade_mc(entry_p=100.0, atr=0.015, tp_mult=2.4, sl_mult=1.0, be_mult=1.0, drift=0.002, qty=100)
