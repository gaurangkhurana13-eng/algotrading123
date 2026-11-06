import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

# Load 2025-26 multi-year data
data_path = ROOT_DIR / "data" / "2025-26" / "universe_14_15m.pkl"
with open(data_path, "rb") as f:
    univ_data = pickle.load(f)

tickers_combined = [
    "PCJEWELLER.NS", "RPOWER.NS", "DELTACORP.NS", "RITES.NS",
    "IDEA.NS", "MTNL.NS", "ZEEL.NS", "UTKARSHBNK.NS", "IFCI.NS", "NCC.NS",
    "TEXRAIL.NS", "JPPOWER.NS", "PTC.NS", "NIACL.NS"
]

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

def run_simulation(start_date, end_date, config_name, er_thresh=0.40, rvol_thresh=1.4, adx_thresh=20.0, atr_pct_thresh=1.1, use_20d_mom=True):
    print(f"\n=======================================================")
    print(f"RUNNING: {config_name} [{start_date} to {end_date}]")
    print(f"Filters: ER10>={er_thresh}, RVOL>={rvol_thresh}, ADX>={adx_thresh}, ATR%>={atr_pct_thresh}%, 20d_Mom={use_20d_mom}")
    print(f"=======================================================")
    
    data_map = {}
    for t in tickers_combined:
        raw_sym = t.replace(".NS", "")
        df = univ_data.get(t, univ_data.get(raw_sym))
        if df is None:
            continue
        df = df.copy()
        c = df["close"]
        h = df["high"]
        l = df["low"]
        v = df["volume"]
        atr14 = calculate_atr(df, 14)
        atr_pct = (atr14 / c) * 100.0
        ema50 = c.ewm(span=50).mean()
        ema110 = c.ewm(span=110).mean()
        vwap_lag = compute_vwap_shifted(df)
        v_sma20 = v.rolling(20).mean()
        rvol = v / (v_sma20 + 1e-9)
        er10 = compute_er10(c)
        
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(7).mean()
        loss = (-delta.clip(upper=0)).rolling(7).mean()
        rsi7 = 100 - (100 / (1 + gain / (loss + 1e-9)))
        
        macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
        sig = macd.ewm(span=9).mean()
        macd_cross_up = (macd > sig) & (macd.shift(1) <= sig.shift(1))
        
        tr = np.maximum(h - l, np.maximum((h - c.shift(1)).abs(), (l - c.shift(1)).abs()))
        atr10 = tr.rolling(10).mean()
        up_move = h - h.shift(1)
        down_move = l.shift(1) - l
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(10).mean() / (atr10 + 1e-9))
        minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(10).mean() / (atr10 + 1e-9))
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
        adx = dx.rolling(10).mean()
        
        # 20-day momentum (20 days * 25 bars = 500 bars)
        mom20d = (c > c.shift(500)) if use_20d_mom else pd.Series(True, index=df.index)
        
        # V23 Enhanced Entry Filter
        entry_cond = (
            macd_cross_up & (macd > 0) &
            (c > ema50) & (c > ema110) &
            (c > vwap_lag) &
            (rsi7 > 50.0) & (adx >= adx_thresh) & (plus_di > minus_di) &
            (er10 >= er_thresh) &
            (rvol >= rvol_thresh) &
            (atr_pct >= atr_pct_thresh) &
            mom20d &
            (c <= 450.0) & (c >= 10.0)
        )
        
        data_map[t] = {
            "df": df, "c": c, "h": h, "l": l, "v": v, "atr14": atr14,
            "rsi7": rsi7, "vwap_lag": vwap_lag, "rvol": rvol, "er10": er10,
            "entry_cond": entry_cond
        }

    all_ts = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))
    slippage = 0.0005
    hard_sl_mult = 0.90
    alpha_streak = 0.082
    beta_streak = 0.035

    positions = {}
    trades = []
    daily_traded = {}
    last_exit_times = set()

    T_START = pd.Timestamp(start_date)
    T_END = pd.Timestamp(end_date)

    for ts in all_ts:
        if ts.weekday() >= 5:
            continue
        cur_date = ts.date()
        if cur_date not in daily_traded:
            daily_traded[cur_date] = set()
        t_time = ts.time()
        t_time_int = t_time.hour * 100 + t_time.minute
        
        # Manage Active Positions
        for ticker in list(positions.keys()):
            p = positions[ticker]
            t_data = data_map[ticker]
            if ts not in t_data["df"].index:
                continue
            row = t_data["df"].loc[ts]
            cur_c, cur_h, cur_l = row["close"], row["high"], row["low"]
            cur_rsi = t_data["rsi7"].loc[ts]
            cur_vwap = t_data["vwap_lag"].loc[ts]
            cur_v = row["volume"]
            
            p["bars_held"] += 1
            p["high_seen"] = max(p["high_seen"], cur_h)
            mfe_atr = (p["high_seen"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0
            
            # V22 Asymmetric Accelerated Ratchet
            if mfe_atr >= 2.50:
                p["active_sl"] = max(p["active_sl"], round(p["high_seen"] - (0.12 * p["entry_atr"]), 2))
            elif mfe_atr >= 1.50 or (mfe_atr >= 1.15 and t_time_int >= 1415):
                p["active_sl"] = max(p["active_sl"], round(p["high_seen"] - (0.18 * p["entry_atr"]), 2))
            elif mfe_atr >= 1.15:
                p["active_sl"] = max(p["active_sl"], round(p["high_seen"] - (0.28 * p["entry_atr"]), 2))
            elif mfe_atr >= 0.85:
                p["active_sl"] = max(p["active_sl"], round(p["entry_p"] + (0.40 * p["entry_atr"]), 2))

            def finalize_trade(exit_price_raw, exit_reason):
                tot_qty = p["qty"]
                if cur_v > 0 and tot_qty > int(0.045 * cur_v):
                    tot_qty = max(1, int(0.045 * cur_v))
                exit_fill = round(max(cur_l, min(cur_h, exit_price_raw)), 2)
                gross = round(tot_qty * (exit_fill - p["entry_p"]), 2)
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
                    "bars": p["bars_held"],
                    "mfe": mfe_atr
                })
                last_exit_times.add(ts)
                del positions[ticker]

            if cur_l <= p["active_sl"]:
                finalize_trade(p["active_sl"] * (1.0 - slippage), "STOP")
                continue
            if p["bars_held"] >= 3 and mfe_atr < 0.40 and cur_c < p["entry_p"] and (cur_rsi < 46.0 or cur_c < cur_vwap):
                finalize_trade(cur_c * (1.0 - slippage), "SCRATCH")
                continue
            if t_time >= pd.Timestamp("14:45").time():
                finalize_trade(cur_c * (1.0 - slippage), "EOD")
                continue

        # Strictly filter entries
        if ts < T_START or ts > T_END:
            continue

        # Sizing
        ek = 500.0 + sum(t["net_pnl"] for t in trades)
        hwm = 500.0
        cum = 500.0
        for t in trades:
            cum += t["net_pnl"]
            if cum > hwm:
                hwm = cum
        dd_pct = (hwm - ek) / max(hwm, 500.0)
        
        if len(positions) >= 1 or ts in last_exit_times or t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:15").time():
            continue
            
        current_notional = sum(p["entry_p"] * p["qty"] for p in positions.values())
        avail_margin = (0.95 * ek) - (current_notional / 5.0)
        if avail_margin < 10.0:
            continue
        avail_notional = avail_margin * 5.0
        
        gate_v2_cap = min(0.09 * ek, 258.0)
        if dd_pct > 0.15:
            gate_v2_cap *= 0.5
            
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
            
        candidates = []
        for ticker in tickers_combined:
            if ticker in positions or ticker in daily_traded[cur_date]:
                continue
            t_data = data_map[ticker]
            if ts not in t_data["df"].index:
                continue
            idx = t_data["df"].index.get_loc(ts)
            if idx < 500 or not t_data["entry_cond"].iloc[idx]:
                continue
            cur_c = t_data["c"].iloc[idx]
            cur_h = t_data["h"].iloc[idx]
            cur_l = t_data["l"].iloc[idx]
            cur_v = t_data["v"].iloc[idx]
            entry_atr = max(t_data["atr14"].iloc[idx], 0.05)
            conviction_score = t_data["rvol"].iloc[idx] * t_data["er10"].iloc[idx]
            candidates.append({
                "ticker": ticker, "c": cur_c, "h": cur_h, "l": cur_l, "v": cur_v,
                "atr": entry_atr, "score": conviction_score, "er": t_data["er10"].iloc[idx], "rvol": t_data["rvol"].iloc[idx]
            })
            
        if not candidates:
            continue
            
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]
        ticker = best["ticker"]
        cur_c = best["c"]
        cur_h = best["h"]
        cur_l = best["l"]
        cur_v = best["v"]
        entry_atr = best["atr"]
        cur_er = best["er"]
        cur_rvol = best["rvol"]
        
        er_bonus = 0.25 * max(0.0, cur_er - 0.35)
        rvol_bonus = 0.05 * min(max(0.0, cur_rvol - 1.5), 3.0)
        weight = min(1.20, max(0.85, 0.85 + er_bonus + rvol_bonus))
        
        target_risk = min(raw_r * weight, gate_v2_cap)
        raw_entry = cur_c * (1.0 + slippage)
        entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
        active_sl = round(entry_p - (hard_sl_mult * entry_atr), 2)
        sl_dist = max(entry_p - active_sl, 0.05)
        friction = 0.002067 * entry_p
        denom = sl_dist + friction
        
        qty_r = int(target_risk / denom)
        qty_m = int((avail_notional * 0.90) / entry_p)
        qty = min(qty_r, qty_m)
        if cur_v > 0:
            qty = min(qty, int(0.045 * cur_v))
            
        if qty >= 1:
            positions[ticker] = {
                "entry_p": entry_p, "qty": qty, "entry_atr": entry_atr,
                "active_sl": active_sl, "initial_sl": active_sl,
                "high_seen": entry_p, "bars_held": 0, "entry_t": ts
            }
            daily_traded[cur_date].add(ticker)

    tdf = pd.DataFrame(trades)
    if len(tdf) > 0:
        wins = tdf[tdf["net_pnl"] > 0]
        losses = tdf[tdf["net_pnl"] <= 0]
        wr = len(wins) / len(tdf) * 100.0
        net_pnl = tdf["net_pnl"].sum()
        fin_wallet = 500.0 + net_pnl
        gross_p = tdf[tdf["gross_pnl"] > 0]["gross_pnl"].sum()
        gross_l = abs(tdf[tdf["gross_pnl"] < 0]["gross_pnl"].sum())
        pf = wins["net_pnl"].sum() / max(abs(losses["net_pnl"].sum()), 1e-6)
        avg_win = wins["net_pnl"].mean()
        avg_loss = losses["net_pnl"].mean()
        payoff = avg_win / max(abs(avg_loss), 1e-6)
        
        cum_arr = np.cumsum([500.0] + list(tdf["net_pnl"]))
        peaks = np.maximum.accumulate(cum_arr)
        dds = (peaks - cum_arr) / peaks * 100.0
        max_dd = np.max(dds)
        
        print(f"Total Trades:      {len(tdf)}")
        print(f"Win Rate:          {wr:.2f}% ({len(wins)} Wins, {len(losses)} Losses)")
        print(f"Gross PnL:         Rs {gross_p - gross_l:.2f}")
        print(f"Total Charges:     Rs {tdf['charges'].sum():.2f}")
        print(f"Net Realized PnL:  Rs {net_pnl:.2f}")
        print(f"Final Wallet:      Rs {fin_wallet:.2f} ({fin_wallet / 500.0:.2f}x)")
        print(f"Profit Factor:     {pf:.2f}")
        print(f"Avg Win / Loss:    Rs {avg_win:.2f} / Rs {avg_loss:.2f} (Payoff: {payoff:.2f})")
        print(f"Max Drawdown:      {max_dd:.2f}%")
        
        # Monthly breakdown
        tdf["month"] = pd.to_datetime(tdf["entry_time"]).dt.strftime("%Y-%m")
        m_summary = tdf.groupby("month").agg(
            trades=("net_pnl", "count"),
            net_pnl=("net_pnl", "sum"),
            win_rate=("net_pnl", lambda x: (x > 0).mean() * 100.0)
        )
        print("\nMonthly Breakdown:")
        print(m_summary.to_string())
        return tdf, net_pnl, wr, pf, max_dd
    else:
        print("Zero trades executed.")
        return pd.DataFrame(), 0, 0, 0, 0

if __name__ == "__main__":
    print("=== BHAVISHYA V23 QUANTITATIVE EVALUATION ===")
    
    # 1. Unseen Period: June 1, 2025 to Dec 31, 2025
    t1, net1, wr1, pf1, dd1 = run_simulation(
        "2025-06-01 09:15:00", "2025-12-31 15:30:00",
        config_name="V23_OOS_2025_H2",
        er_thresh=0.40, rvol_thresh=1.4, adx_thresh=20.0, atr_pct_thresh=1.1, use_20d_mom=True
    )
    
    # 2. Bull Period: Jan 1, 2026 to Oct 30, 2026
    t2, net2, wr2, pf2, dd2 = run_simulation(
        "2026-01-01 09:15:00", "2026-10-30 15:30:00",
        config_name="V23_BULL_2026",
        er_thresh=0.40, rvol_thresh=1.4, adx_thresh=20.0, atr_pct_thresh=1.1, use_20d_mom=True
    )
