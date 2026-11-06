import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, 'C:/Users/admin/.gemini/antigravity/scratch/new_trade/strategies')
sys.path.insert(0, 'C:/Users/admin/.gemini/antigravity/scratch/new_trade/strategies/factory')
from data_manager import load_clean_ticker_data, BASKET
from framework import calculate_atr, calc_dhan_charges, compute_metrics

def true_supertrend(df, period=10, mult=2.5):
    h = df['high'].to_numpy(float)
    l = df['low'].to_numpy(float)
    c = df['close'].to_numpy(float)
    atr = calculate_atr(df, period).to_numpy(float)
    n = len(df)
    hl2 = (h + l) / 2.0
    lower_band = hl2 - mult * atr
    upper_band = hl2 + mult * atr
    final_lower = np.zeros(n)
    final_upper = np.zeros(n)
    direction = np.ones(n, dtype=int)
    start_idx = period
    while start_idx < n and np.isnan(atr[start_idx]):
        start_idx += 1
    if start_idx >= n:
        return pd.Series(1, index=df.index)
    final_lower[start_idx] = lower_band[start_idx]
    final_upper[start_idx] = upper_band[start_idx]
    for i in range(start_idx + 1, n):
        if lower_band[i] > final_lower[i-1] or c[i-1] < final_lower[i-1]:
            final_lower[i] = lower_band[i]
        else:
            final_lower[i] = final_lower[i-1]
        if upper_band[i] < final_upper[i-1] or c[i-1] > final_upper[i-1]:
            final_upper[i] = upper_band[i]
        else:
            final_upper[i] = final_upper[i-1]
        if direction[i-1] == -1 and c[i] > final_upper[i]:
            direction[i] = 1
        elif direction[i-1] == 1 and c[i] < final_lower[i]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
    return pd.Series(direction, index=df.index)

def get_shifted_vwap(df: pd.DataFrame) -> pd.Series:
    df_temp = df.copy()
    df_temp['_date'] = df_temp.index.date
    tp = (df_temp['high'] + df_temp['low'] + df_temp['close']) / 3.0
    df_temp['_tpv'] = tp * df_temp['volume']
    cum_tpv = df_temp.groupby('_date')['_tpv'].cumsum()
    cum_vol = df_temp.groupby('_date')['volume'].cumsum()
    return (cum_tpv / (cum_vol + 1e-9)).shift(1)

def run_bidir_sim(data_map, basket, params):
    all_ts = sorted(list(set().union(*[d['df'].index for d in data_map.values()])))
    capital = 500.0
    max_portfolio_notional = 2450.0
    slippage = 0.0005
    sl_mult = params.get('sl_mult', 1.00)
    p1_trig = params.get('p1_trig', 0.85)
    p1_lock = params.get('p1_lock', 0.40)
    p2_trig = params.get('p2_trig', 1.70)
    trail_dist = params.get('trail_dist', 0.65)
    
    positions = {}
    trades = []
    
    for ts in all_ts:
        t_time = ts.time()
        live_tickers = list(positions.keys())
        for ticker in live_tickers:
            p = positions[ticker]
            t_data = data_map[ticker]
            if ts not in t_data['df'].index:
                continue
            row = t_data['df'].loc[ts]
            cur_c = row['close']
            cur_h = row['high']
            cur_l = row['low']
            direction = p['direction']
            
            # 15:10 IST square-off
            if t_time >= pd.Timestamp('15:10').time():
                if direction == 'long':
                    raw_exit = cur_c * (1.0 - slippage)
                    exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                    gross = round((exit_p - p['entry_p']) * p['qty'], 2)
                    chg = round(calc_dhan_charges(p['entry_p'], exit_p, p['qty'], is_intraday=True), 2)
                else:
                    raw_exit = cur_c * (1.0 + slippage)
                    exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                    gross = round((p['entry_p'] - exit_p) * p['qty'], 2)
                    chg = round(calc_dhan_charges(exit_p, p['entry_p'], p['qty'], is_intraday=True), 2)
                net = round(gross - chg, 2)
                clean_sym = ticker.replace('.NS', '').replace('NSE:', '')
                trades.append({
                    'symbol': clean_sym, 'direction': direction,
                    'entry_time': str(p['entry_t']), 'exit_time': str(ts),
                    'entry_price': p['entry_p'], 'exit_price': exit_p,
                    'shares': p['qty'], 'gross_pnl': gross, 'charges': chg,
                    'net_pnl': net, 'exit_reason': '15:10_INTRADAY_SQUARE_OFF'
                })
                del positions[ticker]
                continue
                
            if direction == 'long':
                p['favorable_seen'] = max(p['favorable_seen'], cur_h)
                mfe_atr = (p['favorable_seen'] - p['entry_p']) / p['entry_atr'] if p['entry_atr'] > 0 else 0.0
                if mfe_atr >= p2_trig:
                    p['phase'] = 2
                    p['active_sl'] = max(p['active_sl'], p['favorable_seen'] - (trail_dist * p['entry_atr']))
                elif mfe_atr >= p1_trig and p['phase'] < 1:
                    p['phase'] = 1
                    p['active_sl'] = max(p['active_sl'], p['entry_p'] + (p1_lock * p['entry_atr']))
                if cur_l <= p['active_sl']:
                    raw_exit = p['active_sl'] * (1.0 - slippage)
                    exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                    gross = round((exit_p - p['entry_p']) * p['qty'], 2)
                    chg = round(calc_dhan_charges(p['entry_p'], exit_p, p['qty'], is_intraday=True), 2)
                    net = round(gross - chg, 2)
                    reason = 'PHASE_2_MOONSHOT_TRAIL_HIT' if p['phase'] == 2 else ('PHASE_1_RATCHET_FLOOR_HIT' if p['phase'] == 1 else 'INITIAL_SL_HIT')
                    clean_sym = ticker.replace('.NS', '').replace('NSE:', '')
                    trades.append({
                        'symbol': clean_sym, 'direction': direction,
                        'entry_time': str(p['entry_t']), 'exit_time': str(ts),
                        'entry_price': p['entry_p'], 'exit_price': exit_p,
                        'shares': p['qty'], 'gross_pnl': gross, 'charges': chg,
                        'net_pnl': net, 'exit_reason': reason
                    })
                    del positions[ticker]
            else: # short
                p['favorable_seen'] = min(p['favorable_seen'], cur_l)
                mfe_atr = (p['entry_p'] - p['favorable_seen']) / p['entry_atr'] if p['entry_atr'] > 0 else 0.0
                if mfe_atr >= p2_trig:
                    p['phase'] = 2
                    p['active_sl'] = min(p['active_sl'], p['favorable_seen'] + (trail_dist * p['entry_atr']))
                elif mfe_atr >= p1_trig and p['phase'] < 1:
                    p['phase'] = 1
                    p['active_sl'] = min(p['active_sl'], p['entry_p'] - (p1_lock * p['entry_atr']))
                if cur_h >= p['active_sl']:
                    raw_exit = p['active_sl'] * (1.0 + slippage)
                    exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                    gross = round((p['entry_p'] - exit_p) * p['qty'], 2)
                    chg = round(calc_dhan_charges(exit_p, p['entry_p'], p['qty'], is_intraday=True), 2)
                    net = round(gross - chg, 2)
                    reason = 'PHASE_2_MOONSHOT_TRAIL_HIT' if p['phase'] == 2 else ('PHASE_1_RATCHET_FLOOR_HIT' if p['phase'] == 1 else 'INITIAL_SL_HIT')
                    clean_sym = ticker.replace('.NS', '').replace('NSE:', '')
                    trades.append({
                        'symbol': clean_sym, 'direction': direction,
                        'entry_time': str(p['entry_t']), 'exit_time': str(ts),
                        'entry_price': p['entry_p'], 'exit_price': exit_p,
                        'shares': p['qty'], 'gross_pnl': gross, 'charges': chg,
                        'net_pnl': net, 'exit_reason': reason
                    })
                    del positions[ticker]
                    
        # Entries
        current_live_notional = sum(p['entry_p'] * p['qty'] for p in positions.values())
        avail_notional = max_portfolio_notional - current_live_notional
        if avail_notional < 15.0:
            continue
        if not (pd.Timestamp('09:20').time() <= t_time <= pd.Timestamp('13:30').time()):
            continue
            
        for ticker in basket:
            if ticker in positions:
                continue
            t_data = data_map[ticker]
            if ts not in t_data['df'].index:
                continue
            idx = t_data['df'].index.get_loc(ts)
            if idx < 100:
                continue
            is_l = t_data['long_cond'].iloc[idx]
            is_s = t_data['short_cond'].iloc[idx]
            if not is_l and not is_s:
                continue
            cur_c = t_data['df']['close'].iloc[idx]
            cur_h = t_data['df']['high'].iloc[idx]
            cur_l = t_data['df']['low'].iloc[idx]
            entry_atr = max(t_data['atr'].iloc[idx], 0.05)
            
            if is_l:
                direction = 'long'
                raw_entry = cur_c * (1.0 + slippage)
                entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
                stop_dist = sl_mult * entry_atr
                active_sl = entry_p - stop_dist
            else:
                direction = 'short'
                raw_entry = cur_c * (1.0 - slippage)
                entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
                stop_dist = sl_mult * entry_atr
                active_sl = entry_p + stop_dist
                
            friction_share = 0.002067 * entry_p
            denom = stop_dist + friction_share
            qty_risk = int(14.00 / denom) if denom > 0 else 1
            qty_margin = int(avail_notional / entry_p)
            qty = min(qty_risk, qty_margin)
            
            if qty >= 1:
                notional_used = entry_p * qty
                if current_live_notional + notional_used <= max_portfolio_notional:
                    positions[ticker] = {
                        'direction': direction, 'entry_p': entry_p, 'entry_t': ts,
                        'qty': qty, 'entry_atr': entry_atr, 'active_sl': active_sl,
                        'favorable_seen': entry_p, 'phase': 0
                    }
                    current_live_notional += notional_used
                    avail_notional = max_portfolio_notional - current_live_notional
                    if avail_notional < 15.0:
                        break
    return pd.DataFrame(trades)

def main():
    params = {'sl_mult': 1.00, 'p1_trig': 0.85, 'p1_lock': 0.40, 'p2_trig': 1.70, 'trail_dist': 0.65}
    is_map, oos_map = {}, {}
    cutoff = pd.Timestamp('2026-09-20')

    for t in BASKET:
        df = load_clean_ticker_data(t)
        c, h, l, v = df['close'], df['high'], df['low'], df['volume']
        macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
        sig = macd.ewm(span=9).mean()
        cross_dn = (macd < sig) & (macd.shift(1) >= sig.shift(1))
        cross_up = (macd > sig) & (macd.shift(1) <= sig.shift(1))
        st = true_supertrend(df, 10, 2.5)
        vwap = get_shifted_vwap(df)
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(7).mean()
        loss = (-delta.clip(upper=0)).rolling(7).mean()
        rsi7 = 100 - (100 / (1 + gain / (loss + 1e-9)))
        
        tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
        atr10 = tr.rolling(10).mean()
        atr14 = calculate_atr(df, 14)
        up_move = h - h.shift(1)
        down_move = l.shift(1) - l
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(10).mean() / (atr10 + 1e-9))
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(10).mean() / (atr10 + 1e-9))
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
        adx = dx.rolling(10).mean()
        
        ema50 = c.ewm(span=50).mean()
        ema200 = c.ewm(span=200).mean()
        
        # Relative Strength Regime Filter:
        # Long only when stock is above 200 EMA (Structural Bull)
        # Short only when stock is below 200 EMA (Structural Bear)
        long_sig = cross_up & (macd > 0) & (c > ema50) & (c > ema200) & (rsi7 > 50) & (adx > 18) & (plus_di > minus_di) & (c > vwap) & (st == 1)
        short_sig = cross_dn & (macd < 0) & (c < ema50) & (c < ema200) & (rsi7 < 50) & (adx > 18) & (minus_di > plus_di) & (c < vwap) & (st == -1)
        
        is_mask = df.index <= cutoff
        oos_mask = df.index > cutoff
        is_map[t] = {'df': df[is_mask], 'atr': atr14[is_mask], 'long_cond': long_sig[is_mask], 'short_cond': short_sig[is_mask]}
        oos_map[t] = {'df': df[oos_mask], 'atr': atr14[oos_mask], 'long_cond': long_sig[oos_mask], 'short_cond': short_sig[oos_mask]}

    df_is = run_bidir_sim(is_map, BASKET, params)
    df_oos = run_bidir_sim(oos_map, BASKET, params)

    m_is = compute_metrics(df_is, 500.0)
    m_oos = compute_metrics(df_oos, 500.0)

    print('=== IN-SAMPLE (Up to Sept 20, 2026) ===')
    print('Trades:', m_is['total_trades'], 'Wins:', m_is['wins'], 'Losses:', m_is['losses'])
    print('Win Rate:', m_is['win_rate'], '% | PF:', m_is['profit_factor'], '| Net PnL: Rs', m_is['net_pnl'])
    print('Avg Win: Rs', m_is['avg_win'], '| Avg Loss: Rs', m_is['avg_loss'])
    if len(df_is):
        print('Directions:', df_is['direction'].value_counts().to_dict())

    print('\n=== OUT-OF-SAMPLE (Sept 21 - Oct 28, 2026 Crash) ===')
    print('Trades:', m_oos['total_trades'], 'Wins:', m_oos['wins'], 'Losses:', m_oos['losses'])
    print('Win Rate:', m_oos['win_rate'], '% | PF:', m_oos['profit_factor'], '| Net PnL: Rs', m_oos['net_pnl'])
    print('Avg Win: Rs', m_oos['avg_win'], '| Avg Loss: Rs', m_oos['avg_loss'])
    if len(df_oos):
        print('Directions:', df_oos['direction'].value_counts().to_dict())
        for d in ['long', 'short']:
            sub = df_oos[df_oos['direction'] == d]
            if len(sub):
                m_sub = compute_metrics(sub, 500.0)
                wr_s = m_sub['win_rate']
                pf_s = m_sub['profit_factor']
                pnl_s = m_sub['net_pnl']
                print(f'OOS {d.upper()}: {len(sub)} trades, WR {wr_s}%, PF {pf_s}, Net PnL: Rs {pnl_s}')

if __name__ == '__main__':
    main()
