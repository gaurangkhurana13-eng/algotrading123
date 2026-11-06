import os, sys, pickle
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))
from framework import calc_dhan_charges, calculate_atr

V29_30_UNIVERSE = {
    "J&KBANK", "UTKARSHBNK", "MANALIPETC", "IDEA", "PCJEWELLER",
    "NIACL", "MTNL", "IFCI", "TEXRAIL", "SWSOLAR",
    "TTML", "RENUKA", "RCF", "NBCC", "JYOTHYLAB",
    "RBLBANK", "ELECTCAST", "PNB", "GSFC", "TATASTEEL",
    "SUZLON", "CENTRALBK", "IOB", "IRCON", "NMDC",
    "RITES", "NCC", "ZEEL", "DELTACORP", "RPOWER"
}

SECTOR_MAP = {
    "FIN": {"PNB", "CENTRALBK", "IOB", "J&KBANK", "UTKARSHBNK", "RBLBANK", "IFCI", "NIACL"},
    "INFRA": {"IRCON", "RITES", "TEXRAIL", "NBCC", "NCC"},
    "COMMODITIES": {"TATASTEEL", "NMDC", "ELECTCAST", "RCF", "GSFC", "MANALIPETC", "RENUKA"},
    "ENERGY": {"SUZLON", "RPOWER", "SWSOLAR", "IDEA", "MTNL", "TTML"},
    "CONSUMER": {"PCJEWELLER", "JYOTHYLAB", "ZEEL", "DELTACORP"}
}
SYM_TO_SECTOR = {s: sec for sec, syms in SECTOR_MAP.items() for s in syms}

core_27 = {
    'CANBK', 'DELTACORP', 'FEDERALBNK', 'GMRINFRA', 'GSFC', 'IDFCFIRSTB', 'IOC',
    'JYOTHYLAB', 'MANAPPURAM', 'MOTHERSON', 'NBCC', 'NCC', 'NATIONALUM', 'NMDC',
    'POONAWALLA', 'PNB', 'RBLBANK', 'RCF', 'SAIL', 'SUZLON', 'TATACHEM',
    'TATASTEEL', 'TATAPOWER', 'TRIDENT', 'UNIONBANK', 'YESBANK', 'ZEEL'
}

def compute_vwap_shifted(df: pd.DataFrame) -> pd.Series:
    df_temp = df.copy()
    df_temp["_date"] = df_temp.index.date
    tp = (df_temp["high"] + df_temp["low"] + df_temp["close"]) / 3.0
    df_temp["_tpv"] = tp * df_temp["volume"]
    cum_tpv = df_temp.groupby("_date")["_tpv"].cumsum()
    cum_vol = df_temp.groupby("_date")["volume"].cumsum()
    return (cum_tpv / (cum_vol + 1e-9)).shift(1)

def compute_er10(series_close: pd.Series) -> pd.Series:
    change = (series_close - series_close.shift(10)).abs()
    volatility = series_close.diff().abs().rolling(10).sum()
    return change / (volatility + 1e-9)

def compute_adx_di(df: pd.DataFrame, n: int = 10):
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff()
    dn = -l.diff()
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
    atr_s = tr.rolling(n).mean()
    pdi = 100 * pd.Series(pdm, index=df.index).rolling(n).mean() / (atr_s + 1e-9)
    mdi = 100 * pd.Series(mdm, index=df.index).rolling(n).mean() / (atr_s + 1e-9)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    adx = dx.rolling(n).mean()
    return adx, pdi, mdi

def run_simulation(
    raw_pool,
    start,
    end,
    fold_label,
    universe_filter=None,
    phi_max=0.45,
    symbol_lockout_days=2,
    alpha_streak=0.085,
    beta_streak=0.038
):
    ticker_map = {}
    for k in raw_pool.keys():
        clean = k.replace('.NS', '')
        if universe_filter is not None and clean not in universe_filter:
            continue
        if clean not in ticker_map or not k.endswith('.NS'):
            ticker_map[clean] = k

    data_map = {}
    for clean, raw_k in ticker_map.items():
        df_full = raw_pool[raw_k].loc[start:end].copy()
        if len(df_full) < 150:
            continue
        c, h, l, v, o = df_full['close'], df_full['high'], df_full['low'], df_full['volume'], df_full['open']
        atr14 = calculate_atr(df_full, 14)
        atr_pct = (atr14 / c) * 100.0
        ema9 = c.ewm(span=9).mean()
        ema20 = c.ewm(span=20).mean()
        ema50 = c.ewm(span=50).mean()
        ema110 = c.ewm(span=110).mean()

        delta = c.diff()
        gain = delta.clip(lower=0).rolling(7).mean()
        loss = (-delta.clip(upper=0)).rolling(7).mean()
        rsi7 = 100.0 - (100.0 / (1.0 + gain / (loss + 1e-9)))

        macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
        sig = macd.ewm(span=9).mean()
        macd_cross_up = (macd > sig) & (macd.shift(1) <= sig.shift(1))

        adx, pdi, mdi = compute_adx_di(df_full, 10)
        vwap_lag = compute_vwap_shifted(df_full)
        v_sma20 = v.rolling(20).mean()
        rvol = v / (v_sma20 + 1e-9)
        er10 = compute_er10(c)

        high_5 = h.rolling(5).max().shift(1)
        lookback = min(300, len(c) - 1)
        c_300 = c.shift(lookback)
        mom_cond = (c > c_300)

        df_full['_date'] = df_full.index.date
        day_open = df_full.groupby('_date')['open'].transform('first')
        daily_close = df_full.groupby('_date')['close'].last()
        prev_close_map = daily_close.shift(1).to_dict()
        df_full['prev_close'] = df_full['_date'].map(prev_close_map)
        df_full['open_gap_pct'] = (day_open - df_full['prev_close']) / (df_full['prev_close'] + 1e-9) * 100.0
        df_full['day_ret'] = (df_full['close'] / day_open) - 1.0
        df_full['1h_ret'] = (df_full['close'] / df_full['close'].shift(4)) - 1.0

        univ_cond = (
            (c > ema50) & (c > ema110) &
            (c > vwap_lag) &
            (atr_pct >= 0.50) &
            mom_cond &
            (c <= 450.0) & (c >= 10.0)
        )
        if clean not in V29_30_UNIVERSE:
            phi_friction = (0.05 + 0.00206 * c) / (atr14 + 1e-9)
            unseen_ok = (c >= 30.0) & (atr14 >= 0.25) & (phi_friction <= phi_max)
            univ_cond = univ_cond & unseen_ok

        t1_base = (
            macd_cross_up & (macd > 0) &
            (c > ema20) &
            (rsi7 > 50.0) & (adx >= 18.0) & (pdi > mdi) &
            (er10 >= 0.35) &
            (rvol >= 1.20)
        )
        t1_tqs = (t1_base & (rvol >= 1.25) & (c > high_5))

        sup_prev = np.maximum(ema9.shift(1), vwap_lag.shift(1)) + (0.15 * atr14)
        t2_pullback = (
            (macd > sig) & (macd > 0) & (pdi > mdi) &
            (ema9 > ema20) & (ema20 > ema50) &
            (c > ema9) &
            (l.shift(1) <= sup_prev) &
            (v.shift(1) < v.shift(2)) & (v >= 1.15 * v.shift(1)) & (rvol >= 1.20) &
            (c > h.shift(1)) & (c.shift(1) <= h.shift(2)) & (c > o) &
            (rsi7 >= 52.0) & (rsi7 <= 74.0) &
            (((c - ema20) / (atr14 + 1e-9)) <= 2.10)
        )

        data_map[clean] = {
            'df': df_full, 'c': c, 'h': h, 'l': l, 'v': v, 'o': o, 'atr14': atr14,
            'ema9': ema9, 'ema20': ema20, 'rsi7': rsi7, 'adx': adx, 'vwap_lag': vwap_lag,
            'rvol': rvol, 'er10': er10,
            'univ_cond': univ_cond, 't1_base': t1_base, 't1_tqs': t1_tqs, 't2_pullback': t2_pullback,
            'day_ret': df_full['day_ret'], '1h_ret': df_full['1h_ret'],
            'c_300': c_300, 'day_open': day_open, 'open_gap_pct': df_full['open_gap_pct']
        }

    bench_day = pd.concat([d['day_ret'] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
    bench_1h  = pd.concat([d['1h_ret'] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
    c300_breadth = pd.concat([(d['c'] > d['c_300']).astype(float) for d in data_map.values()], axis=1, sort=False).mean(axis=1)

    for clean, d in data_map.items():
        b_d = bench_day.reindex(d['day_ret'].index)
        b_1 = bench_1h.reindex(d['1h_ret'].index)
        d['g1_cond'] = (d['day_ret'] > b_d)
        d['g2_cond'] = (d['1h_ret'] > b_1)
        d['excess_1h'] = d['1h_ret'] - b_1

    all_ts = sorted(list(set().union(*[d['df'].index for d in data_map.values()])))
    ticker_list = sorted(list(data_map.keys()))
    ticker_to_idx = {sym: i for i, sym in enumerate(ticker_list)}
    N_stocks = len(ticker_list)

    bullish_matrix = np.zeros((len(all_ts), N_stocks), dtype=bool)
    for i, s in enumerate(ticker_list):
        s_c = data_map[s]['c'].reindex(all_ts).ffill().to_numpy()
        s_vwap = data_map[s]['vwap_lag'].reindex(all_ts).ffill().to_numpy()
        s_ex = data_map[s]['excess_1h'].reindex(all_ts).fillna(0.0).to_numpy()
        bullish_matrix[:, i] = (s_c > s_vwap) & (s_ex > 0)

    sec_p_indices = {}
    for sec, syms in SECTOR_MAP.items():
        sec_p_indices[sec] = {}
        for s in syms:
            if s in ticker_to_idx:
                sec_p_indices[sec][s] = [ticker_to_idx[p] for p in syms if p != s and p in ticker_to_idx]

    initial_equity = 500.0
    ek = initial_equity
    hwm = ek
    positions = {}
    trades = []
    daily_traded = {}
    symbol_lockouts = {}
    last_exit_times = set()
    slippage = 0.0005
    hard_sl_mult = 0.90

    t2_start_t = pd.Timestamp('09:45').time()
    t2_end_t = pd.Timestamp('13:15').time()

    for ts_idx, ts in enumerate(all_ts):
        if ts.weekday() >= 5:
            continue
        cur_date = ts.date()
        if cur_date not in daily_traded:
            daily_traded[cur_date] = set()
        t_time = ts.time()

        b_c300 = c300_breadth.get(ts, 0.50)
        b_day_now = bench_day.get(ts, 0.0)
        b_1h_now = bench_1h.get(ts, 0.0)

        is_bear_env = (b_c300 < 0.40) or (b_day_now < -0.0060)
        is_severe_bleed_day = (b_day_now < -0.0120)

        # 1. Manage Active Positions
        for ticker in list(positions.keys()):
            p = positions[ticker]
            t_data = data_map[ticker]
            if ts not in t_data['df'].index:
                continue
            row = t_data['df'].loc[ts]
            cur_c, cur_h, cur_l = row['close'], row['high'], row['low']
            cur_rsi = t_data['rsi7'].loc[ts]
            cur_vwap = t_data['vwap_lag'].loc[ts]
            cur_ema9 = t_data['ema9'].loc[ts]
            cur_ema20 = t_data['ema20'].loc[ts]
            cur_adx = t_data['adx'].loc[ts]
            cur_rvol = t_data['rvol'].loc[ts]

            p['bars_held'] += 1
            p['high_seen'] = max(p['high_seen'], cur_h)
            mfe_atr = (p['high_seen'] - p['entry_p']) / p['entry_atr'] if p['entry_atr'] > 0 else 0.0

            shooting_star = (cur_rvol >= 2.60) and (((cur_h - cur_c) / (cur_h - cur_l + 1e-6)) >= 0.45)
            extreme_rsi = (cur_rsi >= 86.0)
            break_ema9 = (cur_c < p.get('prev_l', cur_l)) and (cur_c < cur_ema9)
            circuit_prox = (mfe_atr >= 8.00)
            psi_climax = shooting_star or extreme_rsi or break_ema9 or circuit_prox

            theta_expand = (cur_c > cur_ema9 > cur_ema20) and (cur_adx >= 24.0 or cur_rvol >= 1.80) and (cur_rsi >= 56.0) and (not psi_climax)
            prev_l = p.get('prev_l', cur_l)

            # Master V41 Titan Dual-Engine Ratchet:
            if mfe_atr >= 3.50:
                if psi_climax:
                    stop_ladder = p['high_seen'] - (0.10 * p['entry_atr'])
                elif theta_expand:
                    # Unleash massive multi-ATR runners
                    stop_ladder = max(p['entry_p'] + (2.00 * p['entry_atr']), prev_l - (0.08 * p['entry_atr']), p['high_seen'] - (0.35 * p['entry_atr']))
                else:
                    stop_ladder = p['high_seen'] - (0.18 * p['entry_atr'])
            elif mfe_atr >= 2.25:
                if psi_climax:
                    stop_ladder = p['high_seen'] - (0.10 * p['entry_atr'])
                elif theta_expand:
                    stop_ladder = max(p['entry_p'] + (1.20 * p['entry_atr']), prev_l - (0.10 * p['entry_atr']), p['high_seen'] - (0.35 * p['entry_atr']))
                else:
                    stop_ladder = p['high_seen'] - (0.20 * p['entry_atr'])
            elif mfe_atr >= 1.25:
                if theta_expand:
                    stop_ladder = max(p['entry_p'] + (0.55 * p['entry_atr']), prev_l - (0.12 * p['entry_atr']), p['high_seen'] - (0.38 * p['entry_atr']))
                else:
                    stop_ladder = p['high_seen'] - (0.28 * p['entry_atr'])
            elif mfe_atr >= 0.85:
                # STRICT PROFIT LOCK (+0.40 ATR)
                stop_ladder = p['entry_p'] + (0.40 * p['entry_atr'])
            else:
                stop_ladder = p['entry_p'] - (0.90 * p['entry_atr'])

            if mfe_atr >= 0.85 and cur_c < cur_ema9 and cur_rsi < 55.0:
                stop_ladder += 0.05 * p['entry_atr']

            p['active_sl'] = max(p['active_sl'], round(stop_ladder, 2))
            p['prev_l'] = cur_l

            def finalize_trade(exit_fill, exit_reason):
                tot_qty = p['qty']
                gross = round((exit_fill - p['entry_p']) * tot_qty, 2)
                chg = round(calc_dhan_charges(p['entry_p'], exit_fill, tot_qty, is_intraday=True), 2)
                net = round(gross - chg, 2)
                trades.append({
                    'symbol': ticker,
                    'direction': 'LONG',
                    'entry_time': p['entry_t'].strftime('%Y-%m-%d %H:%M:%S'),
                    'exit_time': ts.strftime('%Y-%m-%d %H:%M:%S'),
                    'entry_price': p['entry_p'],
                    'exit_price': exit_fill,
                    'shares': tot_qty,
                    'sl': p['initial_sl'],
                    'gross_pnl': gross,
                    'charges': chg,
                    'net_pnl': net,
                    'exit_reason': exit_reason
                })
                last_exit_times.add(ts)
                if exit_reason == 'STOP' and exit_fill <= p['entry_p']:
                    symbol_lockouts[ticker] = cur_date + pd.Timedelta(days=symbol_lockout_days)
                del positions[ticker]

            if cur_l <= p['active_sl']:
                raw_fill = min(cur_h, p['active_sl']) * (1.0 - slippage)
                exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
                finalize_trade(exit_fill, 'STOP')
                continue
            if p['bars_held'] >= 4 and mfe_atr < 0.40 and cur_c < p['entry_p'] and (cur_rsi < 46.0 or cur_c < cur_vwap):
                raw_fill = cur_c * (1.0 - slippage)
                exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
                finalize_trade(exit_fill, 'SCRATCH')
                continue
            if t_time >= pd.Timestamp('14:45').time():
                raw_fill = cur_c * (1.0 - slippage)
                exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
                finalize_trade(exit_fill, 'EOD')
                continue

        # 2. Dynamic Equity & Risk Budget
        ek = initial_equity + sum(t['net_pnl'] for t in trades)
        hwm = max(hwm, ek)
        dd_pct = (hwm - ek) / max(hwm, initial_equity)

        # Single Slot Discipline
        if len(positions) >= 1 or ts in last_exit_times or t_time < pd.Timestamp('09:20').time() or t_time > pd.Timestamp('14:15').time():
            continue

        # Structural Session Armor: Skip midday deadzone (11:30 to 12:30) and 10:00 post-open trap
        if pd.Timestamp('11:30').time() <= t_time <= pd.Timestamp('12:30').time():
            continue
        if pd.Timestamp('10:00').time() <= t_time <= pd.Timestamp('10:05').time():
            continue

        avail_margin = (0.95 * ek) - sum((p['entry_p'] * p['qty']) / 5.0 for p in positions.values())
        if avail_margin < 10.0:
            continue
        avail_notional = avail_margin * 5.0

        consec_losses = 0
        for t in reversed(trades):
            if t['net_pnl'] < 0:
                consec_losses += 1
            else:
                break

        if consec_losses >= 2 or (ek > 2500.0 and dd_pct > 0.16):
            alpha, beta = 0.015, 0.010
        elif consec_losses == 1:
            alpha, beta = 0.035, 0.020
        else:
            alpha, beta = alpha_streak, beta_streak

        if ek <= 2500.0:
            raw_r = 14.0 + (alpha * max(0.0, ek - 500.0))
        else:
            raw_r = 14.0 + (alpha * 2000.0) + (beta * (ek - 2500.0))

        gate_v2_cap = min(0.090 * ek, 258.0)
        if dd_pct > 0.15:
            gate_v2_cap *= 0.50
        target_risk = min(raw_r, gate_v2_cap)

        t_bullish = bullish_matrix[ts_idx, :]
        sum_bullish = int(np.sum(t_bullish))
        k_cluster = {}
        for clean in data_map.keys():
            sec = SYM_TO_SECTOR.get(clean)
            tick_i = ticker_to_idx[clean]
            if sec and clean in sec_p_indices.get(sec, {}):
                p_idxs = sec_p_indices[sec][clean]
                if p_idxs:
                    k_cluster[clean] = float(np.mean(t_bullish[p_idxs]))
                else:
                    k_cluster[clean] = 0.0
            else:
                k_cluster[clean] = float(sum_bullish - t_bullish[tick_i]) / max(1, N_stocks - 1)

        candidates = []
        for ticker in data_map.keys():
            if ticker in positions or ticker in daily_traded[cur_date]:
                continue
            if ticker in symbol_lockouts and cur_date <= symbol_lockouts[ticker]:
                continue
            t_data = data_map[ticker]
            if ts not in t_data['df'].index:
                continue
            idx = t_data['df'].index.get_loc(ts)
            if idx < 100:
                continue

            if not (t_data['univ_cond'].iloc[idx] and t_data['g1_cond'].iloc[idx] and t_data['g2_cond'].iloc[idx]):
                continue

            # Strict Opening Gap Filter
            gap_pct = t_data['open_gap_pct'].iloc[idx]
            if not np.isnan(gap_pct) and gap_pct > 1.50:
                continue

            if is_bear_env:
                k_cl = k_cluster.get(ticker, 0.0)
                decoupling = ((t_data['c'].iloc[idx] - t_data['day_open'].iloc[idx]) / (t_data['day_open'].iloc[idx] + 1e-9)) - b_day_now
                auth_pass = (k_cl >= 0.25) or (decoupling >= 0.0120 and t_data['rvol'].iloc[idx] >= 1.50)
                if not auth_pass:
                    continue

            if is_severe_bleed_day:
                pass_trigger = t_data['t1_tqs'].iloc[idx]
            else:
                pass_t1 = t_data['t1_base'].iloc[idx]
                pass_t2 = (t_data['t2_pullback'].iloc[idx] and (b_day_now >= -0.0040) and (t2_start_t <= t_time <= t2_end_t))
                pass_trigger = pass_t1 or pass_t2

            if not pass_trigger:
                continue

            z_mom = t_data['excess_1h'].iloc[idx]
            z_vol = min(3.0, t_data['rvol'].iloc[idx]) / 3.0
            z_er = min(1.0, t_data['er10'].iloc[idx])
            bar_h, bar_l, bar_c = t_data['h'].iloc[idx], t_data['l'].iloc[idx], t_data['c'].iloc[idx]
            z_spread = (bar_c - bar_l) / (bar_h - bar_l + 1e-6)
            base_score = (0.45 * z_mom) + (0.25 * z_vol) + (0.15 * z_er) + (0.15 * z_spread)
            final_alpha = base_score * (1.0 + 0.35 * k_cluster.get(ticker, 0.0))

            candidates.append((
                ticker,
                bar_c,
                bar_h,
                bar_l,
                t_data['v'].iloc[idx],
                max(t_data['atr14'].iloc[idx], 0.05),
                final_alpha
            ))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[6], reverse=True)
        best = candidates[0]
        ticker, cur_c, cur_h, cur_l, cur_v, entry_atr, _ = best

        raw_entry = cur_c * (1.0 + slippage)
        entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
        active_sl = round(entry_p - (hard_sl_mult * entry_atr), 2)
        sl_dist = max(entry_p - active_sl, 0.05)
        friction = 0.002067 * entry_p
        denom = sl_dist + friction

        qty_r = int(target_risk / denom)
        qty_m = int((avail_notional * 0.90) / entry_p)
        qty_v = int(cur_v * 0.05)
        qty = min(qty_r, qty_m, qty_v)

        if qty < 1:
            continue

        positions[ticker] = {
            'entry_t': ts,
            'entry_p': entry_p,
            'initial_sl': active_sl,
            'active_sl': active_sl,
            'entry_atr': entry_atr,
            'qty': qty,
            'high_seen': cur_h,
            'prev_l': cur_l,
            'bars_held': 0
        }
        daily_traded[cur_date].add(ticker)

    final_cap = initial_equity + sum(t['net_pnl'] for t in trades)
    wr = (sum(1 for t in trades if t['net_pnl'] > 0) / len(trades) * 100.0) if trades else 0.0
    net = sum(t['net_pnl'] for t in trades)
    wins = [t['net_pnl'] for t in trades if t['net_pnl'] > 0]
    losses = [t['net_pnl'] for t in trades if t['net_pnl'] < 0]
    gp = sum(wins)
    gl = abs(sum(losses))
    pf = round(gp / gl, 2) if gl > 0 else 999.0
    return final_cap, trades

def load_pool_by_spec(spec):
    if spec == 'COMPOSITE_1':
        u1 = pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))
        u2 = pickle.load(open(ROOT / 'data/unseen_high_beta_10_15m_1y.pkl', 'rb'))
        u3 = pickle.load(open(ROOT / 'data/unseen_10_15m_1y.pkl', 'rb'))
        return {**u1, **u2, **u3}
    elif spec == 'COMPOSITE_12':
        b1 = pickle.load(open(ROOT / 'data/unseen_10_15m_1y.pkl', 'rb'))
        b3 = pickle.load(open(ROOT / 'data/unseen_batch3_10_15m_1y.pkl', 'rb'))
        return {**b1, **b3}
    elif spec == 'COMPOSITE_13':
        return pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))
    else:
        return pickle.load(open(spec, 'rb'))

def eval_trades(df_t, initial_cap=500.0):
    n = len(df_t)
    if n > 0:
        win_trades = df_t[df_t['net_pnl'] > 0]
        loss_trades = df_t[df_t['net_pnl'] < 0]
        wr = (len(win_trades) / n) * 100.0
        net = df_t['net_pnl'].sum()
        gw = win_trades['net_pnl'].sum()
        gl = abs(loss_trades['net_pnl'].sum())
        pf = (gw / gl) if gl > 0 else 999.0
        
        eq = [initial_cap]
        cur = initial_cap
        for p in df_t['net_pnl']:
            cur += p
            eq.append(cur)
        pk = eq[0]
        mdd = 0.0
        for e in eq:
            if e > pk: pk = e
            dd = (pk - e) / pk * 100.0
            if dd > mdd: mdd = dd
    else:
        wr = net = pf = mdd = 0.0
    return n, round(wr, 1), round(net, 2), round(pf, 2), round(mdd, 1)

if __name__ == '__main__':
    from scripts.test_v41_titan_prototype import folds_def

    print("="*105)
    print("RUNNING COMPLETE 15-FOLD TEST OF ENHANCED V41 TITAN")
    print("="*105)

    rows = []
    all_trades = []

    for fid, fname, pdesc, pool_spec, s_dt, e_dt, u_filt in folds_def:
        pool = load_pool_by_spec(pool_spec)
        if u_filt == 'COMPOSITE_13_UNIV':
            clean_78 = {k.replace('.NS', '') for k in pool.keys()}
            filt = clean_78 - V29_30_UNIVERSE
        elif u_filt is None:
            filt = set(k.replace('.NS', '') for k in pool.keys())
        else:
            filt = u_filt

        cap, trades = run_simulation(pool, s_dt, e_dt, fname, universe_filter=filt)
        df_t = pd.DataFrame(trades)
        n, wr, net, pf, dd = eval_trades(df_t)
        if not df_t.empty:
            all_trades.extend(trades)

        status = "GREEN 🟢" if net > 0 else "RED 🔴"
        print(f"[{fid:2d}] {fname:30s} | Trades: {n:3d} | WR: {wr:5.1f}% | Net: Rs {net:+8.2f} | PF: {pf:5.2f} | DD: {dd:4.1f}% | {status}")
        rows.append({
            'fid': fid, 'name': fname, 'trades': n, 'wr': wr, 'net': net, 'pf': pf, 'dd': dd
        })

    df_res = pd.DataFrame(rows)
    total_net = df_res['net'].sum()
    green_count = (df_res['net'] > 0).sum()
    print("\n" + "="*105)
    print(f"V41 TITAN MASTER SUMMARY: Total Net PnL = Rs {total_net:+,.2f} | Green Folds = {green_count}/15 ({green_count/15*100:.1f}%)")
    print("="*105)
