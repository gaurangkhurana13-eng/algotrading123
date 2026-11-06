import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import yfinance as yf

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

# 1. Download & Prep Nifty Daily for Regime Filter (Strictly Lagged 1-Day)
print("1. Fetching Nifty Daily Data for Macro Regime Filter...")
nifty_df = yf.download("^NSEI", start="2021-08-01", end="2023-02-01", progress=False)
if isinstance(nifty_df.columns, pd.MultiIndex):
    nifty_df.columns = [c[0].lower() for c in nifty_df.columns]
else:
    nifty_df.columns = [c.lower() for c in nifty_df.columns]

nifty_close = nifty_df["close"]
nifty_ema50 = nifty_close.ewm(span=50).mean()
nifty_bullish_daily = (nifty_close > nifty_ema50).shift(1)
nifty_regime_map = {d.date(): bool(nifty_bullish_daily.loc[d]) for d in nifty_bullish_daily.index}

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

def run_30_stock_v29(use_regime_filter: bool = False):
    data_file = ROOT_DIR / "data" / "2022-23" / "universe_30_15m.pkl"
    with open(data_file, "rb") as f:
        raw_pool = pickle.load(f)
        
    pool = {k: v for k, v in raw_pool.items() if not k.endswith(".NS")}
    print(f"Loaded {len(pool)} unique tickers for 2022 test.")
    
    data_map = {}
    for t, df in pool.items():
        df = df.copy().sort_index()
        c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
        
        atr14 = calculate_atr(df, 14)
        atr_pct = (atr14 / c) * 100.0
        ema9 = c.ewm(span=9).mean()
        ema50 = c.ewm(span=50).mean()
        ema110 = c.ewm(span=110).mean()
        
        delta = c.diff()
        gain = delta.clip(lower=0).rolling(7).mean()
        loss = (-delta.clip(upper=0)).rolling(7).mean()
        rsi7 = 100 - (100 / (1 + gain / (loss + 1e-9)))
        
        macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
        sig = macd.ewm(span=9).mean()
        macd_cross_up = (macd > sig) & (macd.shift(1) <= sig.shift(1))
        
        adx, pdi, mdi = compute_adx_di(df, 10)
        vwap_lag = compute_vwap_shifted(df)
        v_sma20 = v.rolling(20).mean()
        rvol = v / (v_sma20 + 1e-9)
        er10 = compute_er10(c)
        
        lookback = min(300, len(c) - 1)
        mom_cond = (c > c.shift(lookback))
        
        g4_v29 = (
            macd_cross_up & (macd > 0) &
            (c > ema50) & (c > ema110) &
            (c > vwap_lag) &
            (rsi7 > 50.0) & (adx >= 18.0) & (pdi > mdi) &
            (er10 >= 0.35) &
            (rvol >= 1.2) &
            (atr_pct >= 0.50) &
            mom_cond &
            (c <= 450.0) & (c >= 10.0)
        )
        
        df["_date"] = df.index.date
        day_open = df.groupby("_date")["open"].transform("first")
        df["day_ret"] = (df["close"] / day_open) - 1.0
        df["1h_ret"] = (df["close"] / df["close"].shift(4)) - 1.0
        
        data_map[t] = {
            "df": df, "c": c, "h": h, "l": l, "v": v, "atr14": atr14, "atr_pct": atr_pct,
            "ema9": ema9, "rsi7": rsi7, "vwap_lag": vwap_lag, "rvol": rvol, "er10": er10,
            "g4_v29": g4_v29, "day_ret": df["day_ret"], "1h_ret": df["1h_ret"]
        }
        
    bench_day_ret = pd.concat([d["day_ret"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
    bench_1h_ret = pd.concat([d["1h_ret"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)
    
    for t, d in data_map.items():
        b_day = bench_day_ret.reindex(d["day_ret"].index)
        b_1h = bench_1h_ret.reindex(d["1h_ret"].index)
        d["excess_1h"] = d["1h_ret"] - b_1h
        
    all_ts = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))
    
    slippage = 0.0005
    initial_equity = 500.0
    capital = initial_equity
    positions = {}
    trades = []
    daily_traded = {}
    last_exit_times = set()
    
    for ts in all_ts:
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
            cur_ema9 = t_data["ema9"].loc[ts]
            
            p["bars_held"] += 1
            p["high_seen"] = max(p["high_seen"], cur_h)
            mfe_atr = (p["high_seen"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0
            
            if mfe_atr >= 4.00:
                stop_ladder = p["high_seen"] - (0.10 * p["entry_atr"])
            elif mfe_atr >= 3.00:
                stop_ladder = p["high_seen"] - (0.14 * p["entry_atr"])
            elif mfe_atr >= 2.00:
                stop_ladder = p["high_seen"] - (0.20 * p["entry_atr"])
            elif mfe_atr >= 1.15:
                stop_ladder = p["high_seen"] - (0.28 * p["entry_atr"])
            elif mfe_atr >= 0.85:
                stop_ladder = p["entry_p"] + (0.40 * p["entry_atr"])
            else:
                stop_ladder = p["entry_p"] - (0.90 * p["entry_atr"])
                
            if mfe_atr >= 0.85 and cur_c < cur_ema9 and cur_rsi < 55.0:
                stop_ladder += 0.05 * p["entry_atr"]
                
            p["active_sl"] = max(p["active_sl"], round(stop_ladder, 2))
            
            def finalize_trade(exit_fill, exit_reason):
                tot_qty = p["qty"]
                gross = round((exit_fill - p["entry_p"]) * tot_qty, 2)
                chg = round(calc_dhan_charges(p["entry_p"], exit_fill, tot_qty, is_intraday=True), 2)
                net = round(gross - chg, 2)
                trades.append({
                    "symbol": ticker,
                    "entry_time": p["entry_t"].strftime("%Y-%m-%d %H:%M:%S"),
                    "exit_time": ts.strftime("%Y-%m-%d %H:%M:%S"),
                    "entry_price": p["entry_p"],
                    "exit_price": exit_fill,
                    "qty": tot_qty,
                    "gross_pnl": gross,
                    "charges": chg,
                    "net_pnl": net,
                    "bars_held": p["bars_held"],
                    "exit_reason": exit_reason
                })
                nonlocal capital
                capital += net
                last_exit_times.add(ts)
                del positions[ticker]
                
            if cur_l <= p["active_sl"]:
                fill = min(cur_c, p["active_sl"]) * (1.0 - slippage)
                finalize_trade(fill, "TRAIL_SL")
            elif t_time >= pd.Timestamp("15:10").time():
                fill = cur_c * (1.0 - slippage)
                finalize_trade(fill, "EOD_SQUAREOFF")
                
        # 2. Check Entries (Single Slot compounding)
        if len(positions) == 0:
            if t_time < pd.Timestamp("09:45").time() or t_time > pd.Timestamp("14:30").time():
                continue
            if ts in last_exit_times:
                continue
                
            if use_regime_filter:
                is_nifty_bullish = nifty_regime_map.get(cur_date, True)
                if not is_nifty_bullish:
                    continue
                cur_bench_day = bench_day_ret.loc[ts] if ts in bench_day_ret.index else 0.0
                if cur_bench_day < -0.002:
                    continue
                    
            candidates = []
            for ticker, t_data in data_map.items():
                if ts not in t_data["df"].index:
                    continue
                if ticker in daily_traded[cur_date]:
                    continue
                    
                row = t_data["df"].loc[ts]
                if t_data["g4_v29"].loc[ts]:
                    ex_1h = t_data["excess_1h"].loc[ts]
                    r_1h = t_data["1h_ret"].loc[ts]
                    if ex_1h > 0 and r_1h > 0:
                        candidates.append((ticker, ex_1h, row["close"], t_data["atr14"].loc[ts]))
                        
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                best_ticker, best_ex, best_c, best_atr = candidates[0]
                
                entry_fill = round(best_c * (1.0 + slippage), 2)
                pos_val = max(50.0, capital * 0.98)
                qty = int(pos_val / entry_fill)
                
                if qty >= 1:
                    initial_sl = round(entry_fill - (0.90 * best_atr), 2)
                    positions[best_ticker] = {
                        "entry_p": entry_fill,
                        "entry_t": ts,
                        "qty": qty,
                        "entry_atr": best_atr,
                        "active_sl": initial_sl,
                        "high_seen": entry_fill,
                        "bars_held": 0
                    }
                    daily_traded[cur_date].add(best_ticker)
                    
    tdf = pd.DataFrame(trades)
    return capital, tdf

if __name__ == "__main__":
    print("\n" + "="*70)
    print("RUNNING 2022 FULL YEAR ON COMPLETE 30-STOCK UNIVERSE")
    print("="*70)
    
    cap_raw, tdf_raw = run_30_stock_v29(use_regime_filter=False)
    cap_reg, tdf_reg = run_30_stock_v29(use_regime_filter=True)
    
    print("\n" + "-"*50)
    print(f"2022 30-STOCKS BASELINE: Final Capital: Rs {cap_raw:.2f}, Net PnL: Rs {cap_raw - 500:.2f}")
    if len(tdf_raw) > 0:
        wr_raw = (tdf_raw["net_pnl"] > 0).mean() * 100
        print(f"  Total Trades: {len(tdf_raw)}, Win Rate: {wr_raw:.1f}%, Charges: Rs {tdf_raw['charges'].sum():.2f}")
        
    print("\n" + "-"*50)
    print(f"2022 30-STOCKS WITH REGIME FILTER: Final Capital: Rs {cap_reg:.2f}, Net PnL: Rs {cap_reg - 500:.2f}")
    if len(tdf_reg) > 0:
        wr_reg = (tdf_reg["net_pnl"] > 0).mean() * 100
        print(f"  Total Trades: {len(tdf_reg)}, Win Rate: {wr_reg:.1f}%, Charges: Rs {tdf_reg['charges'].sum():.2f}")
    print("="*70)
