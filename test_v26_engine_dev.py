import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

# 1. Load 78 Stocks Universe
with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    universe = pickle.load(f)

def compute_vwap_shifted(df: pd.DataFrame) -> pd.Series:
    df_temp = df.copy()
    df_temp["_date"] = df_temp.index.date
    tp = (df_temp["high"] + df_temp["low"] + df_temp["close"]) / 3.0
    df_temp["_tpv"] = tp * df_temp["volume"]
    cum_tpv = df_temp.groupby("_date")["_tpv"].cumsum()
    cum_vol = df_temp.groupby("_date")["volume"].cumsum()
    return (cum_tpv / (cum_vol + 1e-9)).shift(1)

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

# Simple industry mapping for universe stocks
SECTOR_MAP = {
    "BANK": ["BANKBARODA", "BANKINDIA", "CANBK", "CENTRALBK", "EQUITASBNK", "IDBI", "IDFCFIRSTB", "J&KBANK", "PNB", "RBLBANK", "SBIN", "SOUTHBANK", "UCOBANK", "UJJIVANSFB", "UNIONBANK", "UTKARSHBNK", "YESBANK", "BANDHANBNK"],
    "POWER": ["ADANIPOWER", "CESC", "JPPOWER", "NHPC", "NTPC", "POWERGRID", "PTC", "RPOWER", "RTNPOWER", "SJVN", "TATAPOWER", "SWSOLAR", "SUZLON"],
    "METALS": ["COALINDIA", "GMDCLTD", "HINDALCO", "JINDALSTEL", "JSWSTEEL", "NATIONALUM", "NMDC", "SAIL", "TATASTEEL", "VEDL"],
    "RAIL_INFRA": ["HCC", "HUDCO", "IRB", "IRCON", "IRCTC", "IRFC", "NBCC", "NCC", "RAILTEL", "RITES", "RVNL", "TEXRAIL", "GMRAIRPORT"],
    "TELECOM_MEDIA": ["IDEA", "MTNL", "DISHTV", "HATHWAY", "DEN", "ZEEL", "TTML", "FCS_SOFT", "FSL", "SUBEXLTD"],
    "OIL_GAS_CHEM": ["BPCL", "GAIL", "IOC", "OIL", "ONGC", "PETRONET", "BODALCHEM", "CASTROLIND", "GSFC", "RCF", "SPIC"],
    "CONSUMER_JEWEL": ["PCJEWELLER", "DELTACORP", "RENUKA", "BCLIND", "TRIDENT", "ALOKINDS", "ELECTCAST", "SEPC", "VIKASLIFE"],
    "FINANCIALS": ["IFCI", "RECLTD", "PFC", "NIACL", "GICRE", "L&TFH", "MANAPPURAM", "MUTHOOTFIN"]
}

TICKER_TO_SECTOR = {}
for sec, tickers in SECTOR_MAP.items():
    for t in tickers:
        TICKER_TO_SECTOR[t.upper()] = sec

data_map = {}
for sym, df in universe.items():
    if len(df) < 500:
        continue
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]
    
    recent_p = c.iloc[-1]
    if recent_p < 10.0 or recent_p > 450.0:
        continue
        
    atr14 = calculate_atr(df, 14)
    atr_pct = (atr14 / c) * 100.0
    mean_atr_pct = atr_pct.dropna().mean()
    if mean_atr_pct < 0.40 or mean_atr_pct > 3.5:
        continue
        
    ema20 = c.ewm(span=20).mean()
    ema60 = c.ewm(span=60).mean()
    ema9 = c.ewm(span=9).mean()
    
    vwap_lag = compute_vwap_shifted(df)
    vwap_slope = vwap_lag - vwap_lag.shift(2)
    
    v_sma20 = v.rolling(20).mean()
    rvol = v / (v_sma20 + 1e-9)
    
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(7).mean()
    loss = (-delta.clip(upper=0)).rolling(7).mean()
    rsi7 = 100 - (100 / (1 + gain / (loss + 1e-9)))
    
    macd = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    sig = macd.ewm(span=9).mean()
    
    adx, pdi, mdi = compute_adx_di(df, 10)
    
    nr = (h - l) / c
    nr4 = (nr <= nr.rolling(4).min())
    hh4 = h.rolling(4).max().shift(1)
    
    vel = c.pct_change(4)
    acc = vel - vel.shift(1)
    
    df["_date"] = df.index.date
    day_open = df.groupby("_date")["open"].transform("first")
    day_ret = (c / day_open) - 1.0
    
    clean_sym = sym.replace(".NS", "")
    sec = TICKER_TO_SECTOR.get(clean_sym.upper(), "OTHER")
    
    data_map[clean_sym] = {
        "df": df, "c": c, "h": h, "l": l, "v": v, "atr14": atr14, "atr_pct": atr_pct,
        "ema20": ema20, "ema60": ema60, "ema9": ema9, "vwap_lag": vwap_lag, "vwap_slope": vwap_slope,
        "rvol": rvol, "rsi7": rsi7, "macd": macd, "sig": sig, "adx": adx, "pdi": pdi, "mdi": mdi,
        "nr4": nr4, "hh4": hh4, "vel": vel, "acc": acc, "day_ret": day_ret, "sector": sec
    }

bench_1h_series = pd.concat([d["vel"] for d in data_map.values()], axis=1, sort=False).mean(axis=1)

for sym, d in data_map.items():
    b_1h = bench_1h_series.reindex(d["df"].index)
    g1 = (d["c"] > d["ema20"]) & (d["ema20"] > d["ema60"])
    g2 = (d["macd"] > d["sig"]) & (d["sig"] > 0)
    g3 = (d["rsi7"] > 50.0) & (d["adx"] >= 18.0) & (d["pdi"] > d["mdi"])
    g4 = (d["rvol"] >= 1.35)
    g5 = (d["c"] > d["hh4"]) & d["nr4"].shift(1).fillna(False)
    g6 = (d["vel"] >= 0.0035) & (d["acc"] >= 0)
    g7 = (d["vel"] - b_1h >= 0.0010)
    
    vwap_ok = (d["c"] > d["vwap_lag"]) & (d["vwap_slope"] > 0)
    price_ok = (d["c"] >= 10.0) & (d["c"] <= 450.0)
    
    d["signal"] = g1 & g2 & g3 & g4 & g5 & g6 & g7 & vwap_ok & price_ok
    d["excess_1h"] = d["vel"] - b_1h
    d["score"] = d["rvol"] * (d["vel"].clip(lower=0) ** 1.5) * (1.0 + (d["atr_pct"] / 100.0))

print("Feature and Gates computation complete across all stocks!")

# 2. Multi-Slot Portfolio Simulation Engine
all_ts = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))

slippage = 0.0005
hard_sl_mult = 0.90
initial_equity = 500.0
equity = initial_equity

positions = {} # symbol -> pos_dict
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
    t_time_int = t_time.hour * 100 + t_time.minute
    
    # ── 1. Process Open Positions ──
    for sym in list(positions.keys()):
        p = positions[sym]
        t_data = data_map[sym]
        if ts not in t_data["df"].index:
            continue
        row = t_data["df"].loc[ts]
        cur_c, cur_h, cur_l = row["close"], row["high"], row["low"]
        cur_rsi = t_data["rsi7"].loc[ts]
        cur_vwap = t_data["vwap_lag"].loc[ts]
        cur_v = row["volume"]
        cur_ema9 = t_data["ema9"].loc[ts]
        
        p["bars_held"] += 1
        p["hh"] = max(p["hh"], cur_h)
        mfe_atr = (p["hh"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0
        
        # DeepSeek V26 Exit Ladder
        if mfe_atr >= 3.00:
            stop_ladder = p["hh"] - (0.14 * p["entry_atr"])
        elif mfe_atr >= 2.00:
            stop_ladder = p["hh"] - (0.20 * p["entry_atr"])
        elif mfe_atr >= 1.15:
            stop_ladder = p["hh"] - (0.28 * p["entry_atr"])
        elif mfe_atr >= 0.85:
            stop_ladder = p["entry_p"] + (0.40 * p["entry_atr"])
        else:
            stop_ladder = p["entry_p"] - (0.90 * p["entry_atr"])
            
        # Velocity decay tightening: mfe >= 0.85 and close < ema9 and rsi7 < 55
        if mfe_atr >= 0.85 and cur_c < cur_ema9 and cur_rsi < 55.0:
            stop_ladder += 0.05 * p["entry_atr"]
            
        p["active_sl"] = max(p["active_sl"], round(stop_ladder, 2))
        
        def finalize_trade(exit_price_raw, exit_reason):
            tot_qty = p["qty"]
            if cur_v > 0 and tot_qty > int(0.045 * cur_v):
                tot_qty = max(1, int(0.045 * cur_v))
            exit_fill = round(max(cur_l, min(cur_h, exit_price_raw)), 2)
            gross = round(tot_qty * (exit_fill - p["entry_p"]), 2)
            chg = round(calc_dhan_charges(p["entry_p"], exit_fill, tot_qty, is_intraday=True), 2)
            net = round(gross - chg, 2)
            trades.append({
                "symbol": sym,
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
                "mfe": round(mfe_atr, 2)
            })
            last_exit_times.add(ts)
            del positions[sym]
            
        if cur_l <= p["active_sl"]:
            finalize_trade(min(p["active_sl"], row["open"]) * (1.0 - slippage), "STOP")
            continue
        # V26 Stagnation Scratch: bars >= 4, mfe < 0.40, close < entry, (rsi < 48 or close < vwap)
        if p["bars_held"] >= 4 and mfe_atr < 0.40 and cur_c < p["entry_p"] and (cur_rsi < 48.0 or cur_c < cur_vwap):
            finalize_trade(cur_c * (1.0 - slippage), "STAG")
            continue
        # Force flat at 14:45
        if t_time >= pd.Timestamp("14:45").time():
            finalize_trade(cur_c * (1.0 - slippage), "EOD")
            continue
            
    # ── 2. Equity & Margin Accounting ──
    ek = initial_equity + sum(t["net_pnl"] for t in trades)
    hwm = initial_equity
    cum = initial_equity
    for t in trades:
        cum += t["net_pnl"]
        if cum > hwm:
            hwm = cum
    dd_pct = (hwm - ek) / max(hwm, initial_equity)
    
    # Max slots determination
    max_slots = 2 if ek < 5000.0 else 3
    avail_slots = max_slots - len(positions)
    
    if avail_slots <= 0 or ts in last_exit_times or t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:15").time():
        continue
        
    current_notional = sum(p["entry_p"] * p["qty"] for p in positions.values())
    used_margin = current_notional / 5.0
    avail_margin = (0.95 * ek) - used_margin
    if avail_margin < 10.0:
        continue
        
    avail_notional_per_slot = (avail_margin * 5.0) / avail_slots
    
    # Risk budget per slot: B_agg = min(0.10 * E, 320.0), per slot = B_agg / max_slots
    # Gate V2 cap: min(0.09 * ek, 258.0), halved if DD > 15%
    b_agg = min(0.10 * ek, 320.0)
    slot_risk = b_agg / max_slots
    gate_cap = min(0.09 * ek, 258.0)
    if dd_pct > 0.15:
        gate_cap *= 0.50
    target_risk = min(slot_risk, gate_cap)
    
    # ── 3. Candidate Selection with Sector Isolation ──
    active_sectors = {p["sector"] for p in positions.values()}
    
    candidates = []
    for sym, t_data in data_map.items():
        if sym in positions or sym in daily_traded[cur_date]:
            continue
        sec = t_data["sector"]
        if sec in active_sectors:
            continue
        if ts not in t_data["df"].index:
            continue
        idx = t_data["df"].index.get_loc(ts)
        if idx < 100 or not t_data["signal"].iloc[idx]:
            continue
            
        cur_c = t_data["c"].iloc[idx]
        cur_h = t_data["h"].iloc[idx]
        cur_l = t_data["l"].iloc[idx]
        cur_v = t_data["v"].iloc[idx]
        entry_atr = max(t_data["atr14"].iloc[idx], 0.05)
        score = t_data["score"].iloc[idx]
        
        candidates.append({
            "symbol": sym, "sector": sec, "c": cur_c, "h": cur_h, "l": cur_l, "v": cur_v,
            "atr": entry_atr, "score": score
        })
        
    if not candidates:
        continue
        
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    for cand in candidates:
        if avail_slots <= 0:
            break
        sym = cand["symbol"]
        sec = cand["sector"]
        if sec in active_sectors:
            continue
            
        cur_c = cand["c"]
        cur_h = cand["h"]
        cur_l = cand["l"]
        cur_v = cand["v"]
        entry_atr = cand["atr"]
        
        raw_entry = cur_c * (1.0 + slippage)
        entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
        active_sl = round(entry_p - (hard_sl_mult * entry_atr), 2)
        sl_dist = max(entry_p - active_sl, 0.05)
        friction = 0.002067 * entry_p
        denom = sl_dist + friction
        
        qty_r = int(target_risk / denom)
        qty_m = int((avail_notional_per_slot * 0.90) / entry_p)
        qty = min(qty_r, qty_m)
        if cur_v > 0:
            qty = min(qty, int(0.045 * cur_v))
            
        if qty >= 1:
            positions[sym] = {
                "entry_p": entry_p, "qty": qty, "entry_atr": entry_atr,
                "active_sl": active_sl, "initial_sl": active_sl,
                "hh": entry_p, "bars_held": 0, "entry_t": ts, "sector": sec
            }
            daily_traded[cur_date].add(sym)
            active_sectors.add(sec)
            avail_slots -= 1

tdf = pd.DataFrame(trades)
print("\n" + "="*50)
print(f"BHAVISHYA V26 SIMULATION RESULTS (2026)")
print("="*50)
print(f"Total Trades: {len(tdf)}")
if len(tdf) > 0:
    wins = tdf[tdf["net_pnl"] > 0]
    losses = tdf[tdf["net_pnl"] < 0]
    wr = len(wins) / len(tdf) * 100.0
    net = tdf["net_pnl"].sum()
    gross = tdf["gross_pnl"].sum()
    chg = tdf["charges"].sum()
    cum = initial_equity + tdf["net_pnl"].cumsum()
    hwm = cum.cummax()
    dd = (hwm - cum) / hwm * 100.0
    max_dd = dd.max()
    
    net_win = wins["net_pnl"].sum()
    net_loss = abs(losses["net_pnl"].sum())
    pf = round(net_win / net_loss, 2) if net_loss > 0 else 999.0
    
    print(f"Wins: {len(wins)} | Losses: {len(losses)} | Win Rate: {wr:.2f}%")
    print(f"Net Profit: Rs {net:.2f} | Final Wallet: Rs {cum.iloc[-1]:.2f} ({(cum.iloc[-1]/initial_equity):.2f}x)")
    print(f"Gross Profit: Rs {gross:.2f} | Charges: Rs {chg:.2f}")
    print(f"Profit Factor: {pf} | Max Drawdown: {max_dd:.2f}%")
    print(f"Avg Win: Rs {wins['net_pnl'].mean():.2f} | Avg Loss: Rs {losses['net_pnl'].mean():.2f}")
    print(f"Exit Reasons: {tdf['exit_reason'].value_counts().to_dict()}")
