import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT / "strategies"))
from framework import calc_dhan_charges, calculate_atr

data_path = ROOT / "data" / "2024-25" / "universe_30_15m.pkl"
with open(data_path, "rb") as f:
    pool = pickle.load(f)

# Load helper functions
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

def test_config(
    use_lockout=True,
    skip_midday_deadzone=True,
    time_machine_odds=True,
    min_odds=0.50
):
    start = "2024-01-01"
    end = "2024-12-31"

    ticker_map = {}
    for k in pool.keys():
        clean = k.replace(".NS", "")
        if clean in V29_30_UNIVERSE and (clean not in ticker_map or not k.endswith(".NS")):
            ticker_map[clean] = k

    data_map = {}
    for clean, raw_k in ticker_map.items():
        df_full = pool[raw_k].loc[start:end].copy()
        if len(df_full) < 150:
            continue
        c, h, l, v, o = df_full["close"], df_full["high"], df_full["low"], df_full["volume"], df_full["open"]
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
        df_full["day_ret"] = (df_full["close"] / day_open) - 1.0
        df_full["1h_ret"] = (df_full["close"] / df_full["close"].shift(4)) - 1.0

        univ_cond = (
            (c > ema50) & (c > ema110) &
            (c > vwap_lag) &
            (atr_pct >= 0.50) &
            mom_cond &
            (c <= 450.0) & (c >= 10.0)
        )

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

        # Precompute rolling historical session slot odds (Day Trader Skill Time Machine)
        # For each date D and slot, look back 20 sessions: what fraction of times did high exceed close + 1.0*atr before low touched close - 0.9*atr?
        # Vectorized approximation for speed:
        # Fwd max high in next 8 bars vs fwd min low in next 8 bars
        fwd_h8 = h.iloc[::-1].rolling(8).max().iloc[::-1].shift(-1)
        fwd_l8 = l.iloc[::-1].rolling(8).min().iloc[::-1].shift(-1)
        target_hit = (fwd_h8 - c) >= (1.0 * atr14)
        stop_hit = (c - fwd_l8) >= (0.9 * atr14)
        # Slot win indicator: target hit and (target hit earlier or stop not hit)
        slot_win = target_hit & (~stop_hit | ((fwd_h8 - c) / (atr14 + 1e-6) > (c - fwd_l8) / (atr14 + 1e-6)))
        # Rolling 100-bar (approx 20 sessions for same slot) win rate, shifted by 8 bars to prevent lookahead
        roll_odds = slot_win.astype(float).shift(8).rolling(150, min_periods=30).mean()

        data_map[clean] = {
            "df": df_full, "c": c, "h": h, "l": l, "v": v, "o": o, "atr14": atr14,
            "ema9": ema9, "rsi7": rsi7, "vwap_lag": vwap_lag, "rvol": rvol, "er10": er10,
            "univ_cond": univ_cond, "t1_base": t1_base, "t1_tqs": t1_tqs, "t2_pullback": t2_pullback,
            "day_ret": df_full["day_ret"], "excess_1h": df_full["1h_ret"],
            "roll_odds": roll_odds
        }

    all_ts = sorted(list(set.union(*[set(d["df"].index) for d in data_map.values()])))
    tickers = sorted(list(data_map.keys()))
    ticker_to_idx = {t: i for i, t in enumerate(tickers)}

    initial_equity = 500.0
    ek = initial_equity
    hwm = ek
    positions = {}
    trades = []
    daily_traded = {}
    last_exit_times = set()
    slippage = 0.0005
    hard_sl_mult = 0.90

    alpha_streak = 0.082
    beta_streak = 0.035

    t2_start_t = pd.Timestamp("09:45").time()
    t2_end_t = pd.Timestamp("13:15").time()

    # Lockout tracker: ticker -> date until which locked out
    symbol_lockouts = {}

    for ts_idx, ts in enumerate(all_ts):
        if ts.weekday() >= 5:
            continue
        cur_date = ts.date()
        if cur_date not in daily_traded:
            daily_traded[cur_date] = set()
        t_time = ts.time()

        # 1. Manage Active Positions
        for ticker in list(positions.keys()):
            p = positions[ticker]
            t_data = data_map[ticker]
            if ts not in t_data["df"].index:
                continue
            row = t_data["df"].loc[ts]
            cur_c, cur_h, cur_l = row["close"], row["high"], row["low"]
            cur_rsi = t_data["rsi7"].loc[ts]
            cur_vwap = t_data["vwap_lag"].loc[ts]
            cur_ema9 = t_data["ema9"].loc[ts]
            cur_rvol = t_data["rvol"].loc[ts]

            p["bars_held"] += 1
            p["high_seen"] = max(p["high_seen"], cur_h)
            mfe_atr = (p["high_seen"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0

            shooting_star = (cur_rvol >= 2.60) and (((cur_h - cur_c) / (cur_h - cur_l + 1e-6)) >= 0.45)
            extreme_rsi = (cur_rsi >= 86.0)
            break_ema9 = (cur_c < p.get("prev_l", cur_l)) and (cur_c < cur_ema9)
            circuit_prox = (mfe_atr >= 8.00)
            psi_climax = shooting_star or extreme_rsi or break_ema9 or circuit_prox

            if mfe_atr >= 3.00:
                if psi_climax:
                    stop_ladder = p["high_seen"] - (0.08 * p["entry_atr"])
                else:
                    stop_ladder = p["high_seen"] - (0.28 * p["entry_atr"])
                p["active_sl"] = max(p["active_sl"], stop_ladder)
            elif mfe_atr >= 1.20:
                p["active_sl"] = max(p["active_sl"], p["entry_p"] + (0.05 * p["entry_atr"]))

            def finalize_trade(exit_p: float, reason: str):
                gross = (exit_p - p["entry_p"]) * p["qty"]
                charges = calc_dhan_charges(p["entry_p"], exit_p, p["qty"])
                net = gross - charges
                trades.append({
                    "symbol": ticker, "entry_time": p["entry_time"], "exit_time": ts,
                    "entry_p": p["entry_p"], "exit_p": exit_p, "qty": p["qty"],
                    "gross_pnl": round(gross, 2), "charges": round(charges, 2), "net_pnl": round(net, 2),
                    "exit_reason": reason, "bars_held": p["bars_held"], "direction": "BUY"
                })
                last_exit_times.add(ts)
                if use_lockout and reason == "STOP":
                    # Lock out ticker for next 2 trading days
                    symbol_lockouts[ticker] = cur_date + pd.Timedelta(days=3)
                del positions[ticker]

            if cur_l <= p["active_sl"]:
                raw_fill = min(cur_c, p["active_sl"]) * (1.0 - slippage)
                exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
                finalize_trade(exit_fill, "STOP")
                continue

            if t_time >= pd.Timestamp("14:45").time():
                raw_fill = cur_c * (1.0 - slippage)
                exit_fill = round(max(cur_l, min(cur_h, raw_fill)), 2)
                finalize_trade(exit_fill, "EOD")
                continue

        # 2. Dynamic Equity & Risk Budget
        ek = initial_equity + sum(t["net_pnl"] for t in trades)
        hwm = max(hwm, ek)
        dd_pct = (hwm - ek) / max(hwm, initial_equity)

        # Single slot discipline
        if len(positions) >= 1 or ts in last_exit_times or t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:15").time():
            continue

        # Day Trader Skill: Avoid mid-day dead zone (11:30 to 12:30) where institutional volume dries up and chop bleeds capital
        if skip_midday_deadzone and (pd.Timestamp("11:30").time() <= t_time <= pd.Timestamp("12:30").time()):
            continue

        # Day Trader Skill: Avoid 10:00 AM post-open fakeout trap
        if skip_midday_deadzone and (pd.Timestamp("10:00").time() <= t_time <= pd.Timestamp("10:05").time()):
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
            alpha, beta = alpha_streak, beta_streak

        if ek <= 2500.0:
            raw_r = 14.0 + (alpha * max(0.0, ek - 500.0))
        else:
            raw_r = 14.0 + (alpha * 2000.0) + (beta * (ek - 2500.0))

        gate_v2_cap = min(0.090 * ek, 258.0)
        if dd_pct > 0.15:
            gate_v2_cap *= 0.50
        target_risk = min(raw_r, gate_v2_cap)

        # 3. Scan Candidates
        candidates = []
        for ticker in tickers:
            if ticker in positions or ticker in daily_traded[cur_date]:
                continue
            if use_lockout and ticker in symbol_lockouts and cur_date <= symbol_lockouts[ticker]:
                continue

            t_data = data_map[ticker]
            if ts not in t_data["df"].index:
                continue
            idx = t_data["df"].index.get_loc(ts)

            if not t_data["univ_cond"].iloc[idx]:
                continue

            # Day Trader Skill: Time Machine Pre-Trade Odds Filter
            if time_machine_odds:
                odds = t_data["roll_odds"].iloc[idx]
                if not np.isnan(odds) and odds < min_odds:
                    continue

            pass_t1 = t_data["t1_base"].iloc[idx]
            pass_t2 = (t_data["t2_pullback"].iloc[idx] and (t2_start_t <= t_time <= t2_end_t))
            pass_trigger = pass_t1 or pass_t2

            if not pass_trigger:
                continue

            z_mom = t_data["excess_1h"].iloc[idx]
            z_vol = min(3.0, t_data["rvol"].iloc[idx]) / 3.0
            z_er = min(1.0, t_data["er10"].iloc[idx])
            bar_h, bar_l, bar_c = t_data["h"].iloc[idx], t_data["l"].iloc[idx], t_data["c"].iloc[idx]
            z_spread = (bar_c - bar_l) / (bar_h - bar_l + 1e-6)
            base_score = (0.45 * z_mom) + (0.25 * z_vol) + (0.15 * z_er) + (0.15 * z_spread)

            candidates.append((
                ticker, bar_c, bar_h, bar_l, t_data["v"].iloc[idx],
                max(t_data["atr14"].iloc[idx], 0.05), base_score
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

        if qty <= 0:
            continue

        actual_alloc = (qty * entry_p) / 5.0
        if actual_alloc > (avail_margin + 1e-4):
            continue

        positions[ticker] = {
            "entry_p": entry_p, "qty": qty, "entry_atr": entry_atr,
            "active_sl": active_sl, "bars_held": 0, "high_seen": cur_h,
            "entry_time": ts, "prev_l": cur_l
        }
        daily_traded[cur_date].add(ticker)

    tdf = pd.DataFrame(trades)
    if not tdf.empty:
        wins = tdf[tdf["net_pnl"] > 0]
        losses = tdf[tdf["net_pnl"] < 0]
        wr = len(wins) / len(tdf) * 100.0
        net_pnl = tdf["net_pnl"].sum()
        cum = 500.0 + tdf["net_pnl"].cumsum()
        hwm = cum.cummax()
        dd = (hwm - cum) / hwm * 100.0
        max_dd = dd.max()
        pf = (wins["net_pnl"].sum() / abs(losses["net_pnl"].sum())) if not losses.empty and abs(losses["net_pnl"].sum()) > 0 else 999.0
        return len(tdf), wr, net_pnl, pf, max_dd, ek
    return 0, 0, 0, 0, 0, ek

print("--- TESTING EXPERIMENTAL HYBRID CONFIGURATIONS ON FULL YEAR 2024 (30 STOCKS) ---")
configs = [
    ("Baseline (No Filter)", False, False, False, 0.0),
    ("With Symbol Lockout (V38 Defense)", True, False, False, 0.0),
    ("With Midday Deadzone Avoidance (Day Trader Skill)", False, True, False, 0.0),
    ("With Lockout + Deadzone Avoidance", True, True, False, 0.0),
    ("With Lockout + Deadzone + Time Machine Odds (min 0.48)", True, True, True, 0.48),
    ("With Lockout + Deadzone + Time Machine Odds (min 0.50)", True, True, True, 0.50),
]

for name, lock, dead, odds, m_odds in configs:
    tr, wr, net, pf, mdd, cap = test_config(lock, dead, odds, m_odds)
    print(f"{name:55s} | Trd: {tr:3d} | WR: {wr:5.1f}% | Net: Rs {net:+7.2f} | PF: {pf:4.2f} | MaxDD: {mdd:4.1f}% | End: Rs {cap:6.2f}")
