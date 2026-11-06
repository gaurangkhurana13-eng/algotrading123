import os
import sys
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure stdout uses UTF-8
sys.stdout.reconfigure(encoding='utf-8')

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

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

SECTOR_MAP = {
    "FIN": {"PNB", "CENTRALBK", "IOB", "J&KBANK", "UTKARSHBNK", "RBLBANK", "IFCI", "NIACL"},
    "INFRA": {"IRCON", "RITES", "TEXRAIL", "NBCC", "NCC"},
    "COMMODITIES": {"TATASTEEL", "NMDC", "ELECTCAST", "RCF", "GSFC", "MANALIPETC", "RENUKA"},
    "ENERGY": {"SUZLON", "RPOWER", "SWSOLAR", "IDEA", "MTNL", "TTML"},
    "CONSUMER": {"PCJEWELLER", "JYOTHYLAB", "ZEEL", "DELTACORP"}
}

SYM_TO_SECTOR = {}
for s_name, syms in SECTOR_MAP.items():
    for s in syms:
        SYM_TO_SECTOR[s] = s_name

V29_30_UNIVERSE = {
    "J&KBANK", "UTKARSHBNK", "MANALIPETC", "IDEA", "PCJEWELLER",
    "NIACL", "MTNL", "IFCI", "TEXRAIL", "SWSOLAR",
    "TTML", "RENUKA", "RCF", "NBCC", "JYOTHYLAB",
    "RBLBANK", "ELECTCAST", "PNB", "GSFC", "TATASTEEL",
    "SUZLON", "CENTRALBK", "IOB", "IRCON", "NMDC",
    "RITES", "NCC", "ZEEL", "DELTACORP", "RPOWER"
}

core_27 = {
    'CANBK', 'DELTACORP', 'FEDERALBNK', 'GMRINFRA', 'GSFC', 'IDFCFIRSTB', 'IOC',
    'JYOTHYLAB', 'MANAPPURAM', 'MOTHERSON', 'NBCC', 'NCC', 'NATIONALUM', 'NMDC',
    'POONAWALLA', 'PNB', 'RBLBANK', 'RCF', 'SAIL', 'SUZLON', 'TATACHEM',
    'TATASTEEL', 'TATAPOWER', 'TRIDENT', 'UNIONBANK', 'YESBANK', 'ZEEL'
}

folds = [
    ("2026_BULL", "COMPOSITE_1", "2026-01-01", "2026-10-31", V29_30_UNIVERSE),
    ("2023_FULL", ROOT_DIR / "data/2023-24/universe_30_15m.pkl", "2023-01-02", "2023-12-29", V29_30_UNIVERSE),
    ("2022_BEAR", ROOT_DIR / "data/2022-23/universe_30_15m.pkl", "2022-01-01", "2022-12-31", V29_30_UNIVERSE),
    ("2021_FULL", ROOT_DIR / "data/2021-22/universe_30_15m.pkl", "2021-01-01", "2021-12-31", {'CENTRALBK', 'DELTACORP', 'GSFC', 'IDEA', 'IOB', 'JYOTHYLAB', 'NBCC', 'NCC', 'NIACL', 'NMDC', 'PNB', 'RBLBANK', 'RCF', 'RENUKA', 'RITES', 'SUZLON', 'SWSOLAR', 'TATASTEEL', 'TTML', 'ZEEL'}),
    ("2020_FULL", ROOT_DIR / "data/2020_FULL/universe_30_15m.pkl", "2020-01-01", "2020-12-31", core_27),
]

def load_pool(spec):
    if spec == "COMPOSITE_1":
        u1 = pickle.load(open(ROOT_DIR / "data/universe_78_15m_1y.pkl", "rb"))
        u2 = pickle.load(open(ROOT_DIR / "data/unseen_high_beta_10_15m_1y.pkl", "rb"))
        u3 = pickle.load(open(ROOT_DIR / "data/unseen_10_15m_1y.pkl", "rb"))
        return {**u1, **u2, **u3}
    return pickle.load(open(spec, "rb"))

pools = {}
for label, path, st, ed, u_filt in folds:
    pools[label] = load_pool(path)

print("Pools preloaded.")
def run_scanner_experiment(
    setup_name: str,
    allow_macd: bool,
    allow_pullback: bool,
    allow_reclaim_ma20: bool,
    allow_breakout_20d: bool,
    allow_bull_flag: bool,
    rvol_thresh: float = 1.30,
    rec_rvol: float = 1.30,
    bo_rvol: float = 1.35,
    flag_rvol: float = 1.30,
    bo_er_min: float = 0.35,
    rec_ext_max: float = 1.60,
    bo_ext_max: float = 1.80
):
    total_net = 0.0
    total_trades = 0
    total_wins = 0
    total_charges = 0.0
    per_regime = {}

    for label, path, st, ed, u_filt in folds:
        raw_pool = pools[label]
        ticker_map = {}
        for k in raw_pool.keys():
            clean = k.replace(".NS", "")
            if u_filt is not None and clean not in u_filt:
                continue
            if clean not in ticker_map or not k.endswith(".NS"):
                ticker_map[clean] = k

        data_map = {}
        for clean, raw_k in ticker_map.items():
            df_full = raw_pool[raw_k].loc[st:ed].copy()
            if len(df_full) < 150:
                continue
            c = df_full["close"]
            h = df_full["high"]
            l = df_full["low"]
            v = df_full["volume"]
            o = df_full["open"]

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

            df_full["_date"] = df_full.index.date
            day_open = df_full.groupby("_date")["open"].transform("first")
            daily_close = df_full.groupby("_date")["close"].last()
            daily_high = df_full.groupby("_date")["high"].max()
            daily_low = df_full.groupby("_date")["low"].min()

            prev_close_map = daily_close.shift(1).to_dict()
            df_full["prev_close"] = df_full["_date"].map(prev_close_map)
            df_full["open_gap_pct"] = (day_open - df_full["prev_close"]) / (df_full["prev_close"] + 1e-9) * 100.0
            df_full["day_ret"] = (df_full["close"] / day_open) - 1.0
            df_full["1h_ret"] = (df_full["close"] / df_full["close"].shift(4)) - 1.0

            # Daily 20-day high: highest close or high of previous 20 sessions
            d_hi20_dict = daily_high.rolling(20).max().shift(1).to_dict()
            df_full["d_hi20"] = df_full["_date"].map(d_hi20_dict)

            # Daily Bull Flag (10-session run > 10%, 5-session consolidation < 5%)
            d_run10 = (daily_close.shift(5) / daily_close.shift(15) - 1.0)
            d_rng5 = (daily_high.shift(1).rolling(5).max() - daily_low.shift(1).rolling(5).min()) / daily_close.shift(1)
            d_flag_high = daily_high.shift(1).rolling(5).max().to_dict()
            d_flag_setup = (d_run10 > 0.08) & (d_rng5 < 0.05)
            df_full["d_flag_active"] = df_full["_date"].map(d_flag_setup.to_dict()).fillna(False)
            df_full["d_flag_high"] = df_full["_date"].map(d_flag_high)

            # Base Universe Filter
            univ_cond = (
                (c > ema50) & (c > ema110) &
                (c > vwap_lag) &
                (atr_pct >= 0.50) &
                mom_cond &
                (c <= 450.0) & (c >= 10.0)
            )

            if clean not in V29_30_UNIVERSE:
                phi_friction = (0.05 + 0.00206 * c) / (atr14 + 1e-9)
                unseen_ok = (c >= 30.0) & (atr14 >= 0.25) & (phi_friction <= 0.45)
                univ_cond = univ_cond & unseen_ok

            # 1. V42 MACD Cross
            t1_base = (
                macd_cross_up & (macd > 0) &
                (c > ema20) &
                (rsi7 > 50.0) & (adx >= 18.0) & (pdi > mdi) &
                (er10 >= 0.35) &
                (rvol >= rvol_thresh)
            )
            t1_tqs = (t1_base & (rvol >= 1.35) & (c > high_5))

            # 2. V42 Pullback
            sup_prev = np.maximum(ema9.shift(1), vwap_lag.shift(1)) + (0.15 * atr14)
            t2_pullback = (
                (macd > sig) & (macd > 0) & (pdi > mdi) &
                (ema9 > ema20) & (ema20 > ema50) &
                (c > ema9) &
                (l.shift(1) <= sup_prev) &
                (v.shift(1) < v.shift(2)) & (v >= 1.15 * v.shift(1)) & (rvol >= rvol_thresh) &
                (c > h.shift(1)) & (c.shift(1) <= h.shift(2)) & (c > o) &
                (rsi7 >= 52.0) & (rsi7 <= 74.0) &
                (((c - ema20) / (atr14 + 1e-9)) <= 2.10)
            )

            # 3. Scanner Reclaim MA20
            # Volume-confirmed reclaim of 20 MA:
            # Price was below EMA20 (or dipped under it), now closes cleanly above EMA20
            ext_ma20 = (c - ema20) / (atr14 + 1e-9)
            trig_reclaim_ma20 = (
                (l.shift(1) <= ema20.shift(1)) &
                (c > ema20) &
                (macd > sig) & (macd > 0) &
                (rvol >= rec_rvol) &
                (rsi7 >= 50.0) & (rsi7 <= 72.0) &
                (adx >= 18.0) & (pdi > mdi) &
                (c > vwap_lag) &
                (ext_ma20 <= rec_ext_max) &
                (c > o)
            )

            # 4. Scanner Breakout 20D
            # First close above the 20-day high with volume confirmation
            # Check: was the previous bar or yesterday below 20d high?
            # To catch the FIRST close above 20d high on the current day:
            is_above_20d = (c > df_full["d_hi20"])
            # Within today's session, is this the first bar to close above?
            first_bar_above_20d = is_above_20d & (~is_above_20d.groupby(df_full["_date"]).shift(1).fillna(False))
            trig_breakout_20d = (
                first_bar_above_20d &
                (rvol >= bo_rvol) &
                (er10 >= bo_er_min) & (adx >= 20.0) & (pdi > mdi) &
                (rsi7 >= 52.0) & (rsi7 <= 78.0) &
                (ext_ma20 <= bo_ext_max) &
                (c > o)
            )

            # 5. Scanner Bull Flag
            # Flagpole: 10 bars expansion > 2.5% + 5 bars tight consolidation < 1.5%
            flag_run = (c.shift(5) / (c.shift(15) + 1e-9) - 1.0)
            flag_rng = (h.shift(1).rolling(5).max() - l.shift(1).rolling(5).min()) / (c.shift(1) + 1e-9)
            flag_high = h.shift(1).rolling(5).max()
            first_flag_break = (c > flag_high) & (c.shift(1) <= flag_high.shift(1))
            trig_bull_flag = (
                (flag_run >= 0.025) &
                (flag_rng <= 0.015) &
                first_flag_break &
                (rvol >= flag_rvol) &
                (adx >= 20.0) & (pdi > mdi) &
                (rsi7 >= 52.0) & (rsi7 <= 75.0) &
                (c > o)
            )

            data_map[clean] = {
                "df": df_full, "c": c, "h": h, "l": l, "v": v, "o": o, "atr14": atr14,
                "ema9": ema9, "ema20": ema20, "rsi7": rsi7, "adx": adx, "vwap_lag": vwap_lag,
                "rvol": rvol, "er10": er10,
                "univ_cond": univ_cond, "t1_base": t1_base, "t1_tqs": t1_tqs, "t2_pullback": t2_pullback,
                "trig_reclaim_ma20": trig_reclaim_ma20,
                "trig_breakout_20d": trig_breakout_20d,
                "trig_bull_flag": trig_bull_flag,
                "day_ret": df_full["day_ret"], "1h_ret": df_full["1h_ret"],
                "c_300": c_300, "day_open": day_open, "open_gap_pct": df_full["open_gap_pct"]
            }

        bench_day = pd.concat([d["day_ret"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
        bench_1h  = pd.concat([d["1h_ret"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
        c300_breadth = pd.concat([(d["c"] > d["c_300"]).astype(float) for d in data_map.values()], axis=1, sort=False).mean(axis=1)

        for clean, d in data_map.items():
            b_d = bench_day.reindex(d["day_ret"].index)
            b_1 = bench_1h.reindex(d["1h_ret"].index)
            d["g1_cond"] = (d["day_ret"] > b_d)
            d["g2_cond"] = (d["1h_ret"] > b_1)
            d["excess_1h"] = d["1h_ret"] - b_1

        all_ts = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))
        ticker_list = sorted(list(data_map.keys()))
        ticker_to_idx = {sym: i for i, sym in enumerate(ticker_list)}
        N_stocks = len(ticker_list)

        bullish_matrix = np.zeros((len(all_ts), N_stocks), dtype=bool)
        for i, s in enumerate(ticker_list):
            s_c = data_map[s]["c"].reindex(all_ts).ffill().to_numpy()
            s_vwap = data_map[s]["vwap_lag"].reindex(all_ts).ffill().to_numpy()
            s_ex = data_map[s]["excess_1h"].reindex(all_ts).fillna(0.0).to_numpy()
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
        daily_losses = {}
        symbol_lockouts = {}
        last_exit_times = set()
        slippage = 0.0005
        hard_sl_mult = 0.90

        t2_start_t = pd.Timestamp("09:45").time()
        t2_end_t = pd.Timestamp("13:15").time()

        for ts_idx, ts in enumerate(all_ts):
            if ts.weekday() >= 5:
                continue
            cur_date = ts.date()
            if cur_date not in daily_traded:
                daily_traded[cur_date] = set()
            if cur_date not in daily_losses:
                daily_losses[cur_date] = 0
            t_time = ts.time()

            b_c300 = c300_breadth.get(ts, 0.50)
            b_day_now = bench_day.get(ts, 0.0)

            is_bear_env = (b_c300 < 0.40) or (b_day_now < -0.0060)
            is_severe_bleed_day = (b_day_now < -0.0120)

            # Active position management
            for ticker in list(positions.keys()):
                p = positions[ticker]
                t_data = data_map[ticker]
                if ts not in t_data["df"].index:
                    continue
                row = t_data["df"].loc[ts]
                cur_c = row["close"]
                cur_h = row["high"]
                cur_l = row["low"]
                cur_rsi = t_data["rsi7"].loc[ts]
                cur_vwap = t_data["vwap_lag"].loc[ts]
                cur_ema9 = t_data["ema9"].loc[ts]
                cur_ema20 = t_data["ema20"].loc[ts]
                cur_adx = t_data["adx"].loc[ts]
                cur_rvol = t_data["rvol"].loc[ts]

                p["bars_held"] += 1
                p["high_seen"] = max(p["high_seen"], cur_h)
                mfe_atr = (p["high_seen"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0

                shooting_star = (cur_rvol >= 2.60) and (((cur_h - cur_c) / (cur_h - cur_l + 1e-6)) >= 0.45)
                extreme_rsi = (cur_rsi >= 86.0)
                break_ema9 = (cur_c < p.get("prev_l", cur_l)) and (cur_c < cur_ema9)
                circuit_prox = (mfe_atr >= 8.00)
                psi_climax = shooting_star or extreme_rsi or break_ema9 or circuit_prox

                theta_expand = (cur_c > cur_ema9 > cur_ema20) and (cur_adx >= 24.0 or cur_rvol >= 1.80) and (cur_rsi >= 56.0) and (not psi_climax)
                prev_l = p.get("prev_l", cur_l)

                if mfe_atr >= 3.50:
                    if psi_climax:
                        stop_ladder = p["high_seen"] - (0.10 * p["entry_atr"])
                    elif theta_expand:
                        stop_ladder = max(p["entry_p"] + (2.00 * p["entry_atr"]), prev_l - (0.08 * p["entry_atr"]), p["high_seen"] - (0.35 * p["entry_atr"]))
                    else:
                        stop_ladder = p["high_seen"] - (0.18 * p["entry_atr"])
                elif mfe_atr >= 2.25:
                    if psi_climax:
                        stop_ladder = p["high_seen"] - (0.10 * p["entry_atr"])
                    elif theta_expand:
                        stop_ladder = max(p["entry_p"] + (1.20 * p["entry_atr"]), prev_l - (0.10 * p["entry_atr"]), p["high_seen"] - (0.35 * p["entry_atr"]))
                    else:
                        stop_ladder = p["high_seen"] - (0.20 * p["entry_atr"])
                elif mfe_atr >= 1.25:
                    if theta_expand:
                        stop_ladder = max(p["entry_p"] + (0.55 * p["entry_atr"]), prev_l - (0.12 * p["entry_atr"]), p["high_seen"] - (0.38 * p["entry_atr"]))
                    else:
                        stop_ladder = p["high_seen"] - (0.28 * p["entry_atr"])
                elif mfe_atr >= 0.85:
                    stop_ladder = p["entry_p"] + (0.40 * p["entry_atr"])
                elif mfe_atr >= 0.45:
                    stop_ladder = p["entry_p"] + (0.05 * p["entry_atr"])
                else:
                    stop_ladder = p["entry_p"] - (0.90 * p["entry_atr"])

                if mfe_atr >= 0.85 and cur_c < cur_ema9 and cur_rsi < 55.0:
                    stop_ladder += 0.05 * p["entry_atr"]

                p["active_sl"] = max(p["active_sl"], round(stop_ladder, 2))
                p["prev_l"] = cur_l

                def finalize_trade(exit_fill, exit_reason):
                    tot_qty = p["qty"]
                    gross = round((exit_fill - p["entry_p"]) * tot_qty, 2)
                    chg = round(calc_dhan_charges(p["entry_p"], exit_fill, tot_qty, is_intraday=True), 2)
                    net = round(gross - chg, 2)
                    trades.append({
                        "symbol": ticker,
                        "direction": "LONG",
                        "entry_time": p["entry_t"].strftime("%Y-%m-%d %H:%M:%S"),
                        "exit_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "entry_price": p["entry_p"],
                        "exit_price": exit_fill,
                        "shares": tot_qty,
                        "sl": p["initial_sl"],
                        "gross_pnl": gross,
                        "charges": chg,
                        "net_pnl": net,
                        "exit_reason": exit_reason,
                        "pattern": p.get("pattern", "UNKNOWN")
                    })
                    last_exit_times.add(ts)
                    if net < 0:
                        daily_losses[cur_date] += 1
                    if exit_reason == "STOP" and exit_fill <= p["entry_p"]:
                        symbol_lockouts[ticker] = cur_date + pd.Timedelta(days=2)
                    del positions[ticker]

                if cur_l <= p["active_sl"]:
                    raw_fill = min(cur_h, p["active_sl"]) * (1.0 - slippage)
                    exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
                    finalize_trade(exit_fill, "STOP")
                    continue
                if p["bars_held"] >= 4 and mfe_atr < 0.40 and cur_c < p["entry_p"] and (cur_rsi < 46.0 or cur_c < cur_vwap):
                    raw_fill = cur_c * (1.0 - slippage)
                    exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
                    finalize_trade(exit_fill, "SCRATCH")
                    continue
                if t_time >= pd.Timestamp("14:45").time():
                    raw_fill = cur_c * (1.0 - slippage)
                    exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
                    finalize_trade(exit_fill, "EOD")
                    continue

            # Budget
            ek = initial_equity + sum(t["net_pnl"] for t in trades)
            hwm = max(hwm, ek)
            dd_pct = (hwm - ek) / max(hwm, initial_equity)

            if len(positions) >= 1 or ts in last_exit_times or t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:15").time():
                continue
            if pd.Timestamp("11:30").time() <= t_time <= pd.Timestamp("12:30").time():
                continue
            if pd.Timestamp("10:00").time() <= t_time <= pd.Timestamp("10:05").time():
                continue
            if daily_losses.get(cur_date, 0) >= 2:
                continue

            avail_margin = (0.95 * ek) - sum((p["entry_p"] * p["qty"]) / 5.0 for p in positions.values())
            if avail_margin < 10.0:
                continue
            avail_notional = avail_margin * 5.0

            consec_losses = 0
            for t in reversed(trades):
                if t["net_pnl"] < 0:
                    consec_losses += 1
                else:
                    break

            if consec_losses >= 2 or (ek > 2500.0 and dd_pct > 0.16):
                alpha, beta = 0.015, 0.010
            elif consec_losses == 1:
                alpha, beta = 0.035, 0.020
            else:
                alpha, beta = 0.085, 0.038

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
                if ts not in t_data["df"].index:
                    continue
                idx = t_data["df"].index.get_loc(ts)
                if idx < 100:
                    continue

                if not (t_data["univ_cond"].iloc[idx] and t_data["g1_cond"].iloc[idx] and t_data["g2_cond"].iloc[idx]):
                    continue

                gap_pct = t_data["open_gap_pct"].iloc[idx]
                if not np.isnan(gap_pct) and gap_pct > 1.50:
                    continue

                if is_bear_env:
                    k_cl = k_cluster.get(ticker, 0.0)
                    decoupling = ((t_data["c"].iloc[idx] - t_data["day_open"].iloc[idx]) / (t_data["day_open"].iloc[idx] + 1e-9)) - b_day_now
                    auth_pass = (k_cl >= 0.25) or (decoupling >= 0.0120 and t_data["rvol"].iloc[idx] >= 1.50)
                    if not auth_pass:
                        continue

                pass_t1 = t_data["t1_base"].iloc[idx] and allow_macd
                pass_t2 = t_data["t2_pullback"].iloc[idx] and allow_pullback and (b_day_now >= -0.0040) and (t2_start_t <= t_time <= t2_end_t)
                pass_rec = t_data["trig_reclaim_ma20"].iloc[idx] and allow_reclaim_ma20
                pass_bo = t_data["trig_breakout_20d"].iloc[idx] and allow_breakout_20d
                pass_flag = t_data["trig_bull_flag"].iloc[idx] and allow_bull_flag

                active_pattern = None
                if is_severe_bleed_day:
                    if t_data["t1_tqs"].iloc[idx] and allow_macd:
                        active_pattern = "V42_MACD_CROSS"
                else:
                    if pass_t1:
                        active_pattern = "V42_MACD_CROSS"
                    elif pass_rec:
                        active_pattern = "RECLAIM_MA20"
                    elif pass_bo:
                        active_pattern = "BREAKOUT_20D"
                    elif pass_flag:
                        active_pattern = "BULL_FLAG"
                    elif pass_t2:
                        active_pattern = "V42_PULLBACK"

                if not active_pattern:
                    continue

                z_mom = t_data["excess_1h"].iloc[idx]
                z_vol = min(3.0, t_data["rvol"].iloc[idx]) / 3.0
                z_er = min(1.0, t_data["er10"].iloc[idx])
                bar_h = t_data["h"].iloc[idx]
                bar_l = t_data["l"].iloc[idx]
                bar_c = t_data["c"].iloc[idx]
                z_spread = (bar_c - bar_l) / (bar_h - bar_l + 1e-6)
                base_score = (0.45 * z_mom) + (0.25 * z_vol) + (0.15 * z_er) + (0.15 * z_spread)
                
                conv_mult = 1.0
                if active_pattern == "V42_MACD_CROSS":
                    conv_mult = 1.20
                elif active_pattern == "RECLAIM_MA20":
                    conv_mult = 1.10
                elif active_pattern == "BREAKOUT_20D":
                    conv_mult = 1.00
                elif active_pattern == "BULL_FLAG":
                    conv_mult = 0.95

                final_alpha = base_score * (1.0 + 0.35 * k_cluster.get(ticker, 0.0)) * conv_mult

                candidates.append((
                    ticker,
                    bar_c,
                    bar_h,
                    bar_l,
                    t_data["v"].iloc[idx],
                    max(t_data["atr14"].iloc[idx], 0.05),
                    final_alpha,
                    active_pattern
                ))

            if not candidates:
                continue

            candidates.sort(key=lambda x: x[6], reverse=True)
            best = candidates[0]
            ticker, cur_c, cur_h, cur_l, cur_v, entry_atr, _, best_pattern = best

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
                "entry_t": ts,
                "entry_p": entry_p,
                "initial_sl": active_sl,
                "active_sl": active_sl,
                "entry_atr": entry_atr,
                "qty": qty,
                "high_seen": cur_h,
                "prev_l": cur_l,
                "bars_held": 0,
                "pattern": best_pattern
            }
            daily_traded[cur_date].add(ticker)

        net = sum(t["net_pnl"] for t in trades)
        chg = sum(t["charges"] for t in trades)
        wr = (sum(1 for t in trades if t["net_pnl"] > 0) / len(trades) * 100.0) if trades else 0.0
        wins = [t["net_pnl"] for t in trades if t["net_pnl"] > 0]
        losses = [t["net_pnl"] for t in trades if t["net_pnl"] < 0]
        gp = sum(wins)
        gl = abs(sum(losses))
        pf = round(gp / gl, 2) if gl > 0 else 999.0

        equity_curve = [500.0]
        for t in trades:
            equity_curve.append(equity_curve[-1] + t["net_pnl"])
        eq_series = pd.Series(equity_curve)
        peak = eq_series.cummax()
        dd_s = (peak - eq_series) / peak
        max_dd = dd_s.max() * 100.0

        per_regime[label] = {
            "net": net, "trades": len(trades), "wr": wr, "pf": pf, "dd": max_dd, "charges": chg, "wallet": 500.0 + net
        }
        total_net += net
        total_trades += len(trades)
        total_wins += sum(1 for t in trades if t["net_pnl"] > 0)
        total_charges += chg

    overall_wr = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0
    print(f"\n==================== EXPERIMENT: {setup_name} ====================")
    for label, r in per_regime.items():
        print(f"  {label:10s} | Net: Rs {r['net']:+8.2f} | Tr: {r['trades']:3d} | WR: {r['wr']:5.1f}% | PF: {r['pf']:5.2f} | DD: {r['dd']:5.1f}% | Wallet: Rs {r['wallet']:8.2f}")
    print(f"  --> TOTAL OVERALL | Net: Rs {total_net:+8.2f} | Trades: {total_trades:4d} | Win Rate: {overall_wr:5.1f}% | Charges: Rs {total_charges:7.2f}")

if __name__ == "__main__":
    # Test Individual Scanner Setups Standalone
    run_scanner_experiment(
        setup_name="STANDALONE RECLAIM_MA20 (Only Reclaim MA20, No MACD)",
        allow_macd=False, allow_pullback=False, allow_reclaim_ma20=True, allow_breakout_20d=False, allow_bull_flag=False,
        rec_rvol=1.30
    )
    run_scanner_experiment(
        setup_name="STANDALONE BREAKOUT_20D (Only First Close > 20d High)",
        allow_macd=False, allow_pullback=False, allow_reclaim_ma20=False, allow_breakout_20d=True, allow_bull_flag=False,
        bo_rvol=1.35, bo_er_min=0.35
    )
    run_scanner_experiment(
        setup_name="STANDALONE BULL_FLAG (Only Flagpole + Tight Flag)",
        allow_macd=False, allow_pullback=False, allow_reclaim_ma20=False, allow_breakout_20d=False, allow_bull_flag=True,
        flag_rvol=1.30
    )
    # Test Combined Master Architect Model
    run_scanner_experiment(
        setup_name="MASTER BHAVISHYA V42 + HIGH-CONVICTION RECLAIM_MA20",
        allow_macd=True, allow_pullback=True, allow_reclaim_ma20=True, allow_breakout_20d=False, allow_bull_flag=False,
        rec_rvol=1.30
    )
    run_scanner_experiment(
        setup_name="MASTER BHAVISHYA V42 + RECLAIM_MA20 + FILTERED BREAKOUT_20D",
        allow_macd=True, allow_pullback=True, allow_reclaim_ma20=True, allow_breakout_20d=True, allow_bull_flag=False,
        rec_rvol=1.30, bo_rvol=1.40, bo_er_min=0.40
    )
    run_scanner_experiment(
        setup_name="MASTER BHAVISHYA V42 + ALL 3 SCANNER PATTERNS (SYNTHESIS)",
        allow_macd=True, allow_pullback=True, allow_reclaim_ma20=True, allow_breakout_20d=True, allow_bull_flag=True,
        rec_rvol=1.30, bo_rvol=1.40, bo_er_min=0.40, flag_rvol=1.35
    )
