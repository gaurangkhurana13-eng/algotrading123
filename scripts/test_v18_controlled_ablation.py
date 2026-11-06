import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr
import strategy_evaluate_gate_v2 as gate

# Load Top-10 Verified Data
with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    univ = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_high_beta_10_15m_1y.pkl", "rb") as f:
    d_beta = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_10_15m_1y.pkl", "rb") as f:
    d_unseen = pickle.load(f)

tickers_10 = [
    "PCJEWELLER.NS", "RPOWER.NS", "DELTACORP.NS", "RITES.NS",
    "IDEA.NS", "MTNL.NS", "ZEEL.NS",
    "UTKARSHBNK.NS", "IFCI.NS", "NCC.NS"
]

all_dfs = {}
for t in ["PCJEWELLER.NS", "RPOWER.NS", "IDEA.NS", "MTNL.NS", "UTKARSHBNK.NS", "IFCI.NS"]:
    all_dfs[t] = univ[t].copy()
all_dfs["DELTACORP.NS"] = d_beta["DELTACORP.NS"].copy()
all_dfs["RITES.NS"] = d_beta["RITES.NS"].copy()
all_dfs["NCC.NS"] = d_beta["NCC.NS"].copy()
all_dfs["ZEEL.NS"] = d_unseen["ZEEL.NS"].copy()

def compute_vwap_shifted(df: pd.DataFrame) -> pd.Series:
    df_temp = df.copy()
    df_temp["_date"] = df_temp.index.date
    tp = (df_temp["high"] + df_temp["low"] + df_temp["close"]) / 3.0
    df_temp["_tpv"] = tp * df_temp["volume"]
    cum_tpv = df_temp.groupby("_date")["_tpv"].cumsum()
    cum_vol = df_temp.groupby("_date")["volume"].cumsum()
    return (cum_tpv / (cum_vol + 1e-9)).shift(1)

data_map = {}
for t in tickers_10:
    df = all_dfs[t]
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]
    atr14 = calculate_atr(df, 14)
    ema50 = c.ewm(span=50).mean()
    ema110 = c.ewm(span=110).mean()
    vwap_lag = compute_vwap_shifted(df)
    v_sma20 = v.rolling(20).mean()
    rvol = v / (v_sma20 + 1e-9)
    
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
    
    entry_cond = (
        macd_cross_up & (macd > 0) &
        (c > ema50) & (c > ema110) &
        (c > vwap_lag) &
        (rsi7 > 50.0) & (adx > 18.0) & (plus_di > minus_di) &
        (c <= 450.0)
    )
    data_map[t] = {
        "df": df, "c": c, "h": h, "l": l, "v": v, "atr14": atr14,
        "rsi7": rsi7, "vwap_lag": vwap_lag, "rvol": rvol, "entry_cond": entry_cond
    }

all_ts = sorted(list(set().union(*[d["df"].index for d in data_map.values()])))
slippage = 0.0005

def run_simulation(
    hard_sl_mult: float = 1.00,
    sizing_denom_mult: float = 1.00,
    single_risk_cap: float = 260.0,
    use_3gear_trail: bool = False,
    use_bar2_scratch: bool = False
):
    positions = {}
    trades = []
    daily_traded = {}
    last_exit_times = set()

    for ts in all_ts:
        cur_date = ts.date()
        if cur_date not in daily_traded:
            daily_traded[cur_date] = set()
        t_time = ts.time()
        t_time_int = t_time.hour * 100 + t_time.minute
        
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
            cur_atr = t_data["atr14"].loc[ts]
            
            p["bars_held"] += 1
            p["high_seen"] = max(p["high_seen"], cur_h)
            mfe_atr = (p["high_seen"] - p["entry_p"]) / p["entry_atr"] if p["entry_atr"] > 0 else 0.0
            
            # Fast Scratch Option
            if use_bar2_scratch:
                if p["bars_held"] == 2 and mfe_atr < 0.35 and cur_c < cur_vwap and cur_rsi < 48.0:
                    raw_exit = cur_c * (1.0 - slippage)
                    exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                    gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                    chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
                    net = round(gross - chg, 2)
                    trades.append({"net_pnl": net, "exit_reason": "FAST_SCRATCH"})
                    last_exit_times.add(ts)
                    del positions[ticker]
                    continue
                    
            if not use_3gear_trail:
                # V17 Standard Harvester
                if mfe_atr >= 1.35:
                    p["active_sl"] = max(p["active_sl"], round(p["high_seen"] - (0.45 * p["entry_atr"]), 2))
                elif mfe_atr >= 1.70:
                    p["active_sl"] = max(p["active_sl"], round(p["high_seen"] - (0.65 * p["entry_atr"]), 2))
                elif mfe_atr >= 0.85:
                    p["active_sl"] = max(p["active_sl"], round(p["entry_p"] + (0.40 * p["entry_atr"]), 2))
            else:
                # 3-Gear Chandelier Trail
                idx_now = t_data["df"].index.get_loc(ts)
                low_t1 = t_data["l"].iloc[idx_now - 1] if idx_now >= 1 else cur_l
                low_t2 = t_data["l"].iloc[idx_now - 2] if idx_now >= 2 else low_t1
                
                if mfe_atr >= 0.75:
                    p["active_sl"] = max(p["active_sl"], round(p["entry_p"] + (0.30 * p["entry_atr"]), 2))
                if mfe_atr >= 1.35:
                    profit_floor = p["entry_p"] + (0.85 * p["entry_atr"])
                    structural_low = min(low_t1, low_t2) - (0.08 * cur_atr)
                    wide_ch = p["high_seen"] - (0.80 * cur_atr)
                    p["active_sl"] = max(p["active_sl"], round(max(profit_floor, structural_low, wide_ch), 2))
                if mfe_atr >= 2.40 or (mfe_atr >= 1.35 and t_time_int >= 1415):
                    climax_ch = p["high_seen"] - (0.38 * cur_atr)
                    p["active_sl"] = max(p["active_sl"], round(climax_ch, 2))

            # Stop hit check
            if cur_l <= p["active_sl"]:
                raw_exit = p["active_sl"] * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
                net = round(gross - chg, 2)
                trades.append({"net_pnl": net, "exit_reason": "STOP_HIT"})
                last_exit_times.add(ts)
                del positions[ticker]
                continue
                
            # Stagnation scratch
            if p["bars_held"] >= 3 and mfe_atr < 0.40 and cur_c < p["entry_p"] and (cur_rsi < 46.0 or cur_c < cur_vwap):
                raw_exit = cur_c * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
                net = round(gross - chg, 2)
                trades.append({"net_pnl": net, "exit_reason": "STAGNATION_SCRATCH"})
                last_exit_times.add(ts)
                del positions[ticker]
                continue
                
            # 14:45 Square-off
            if t_time >= pd.Timestamp("14:45").time():
                raw_exit = cur_c * (1.0 - slippage)
                exit_p = round(max(cur_l, min(cur_h, raw_exit)), 2)
                gross = round((exit_p - p["entry_p"]) * p["qty"], 2)
                chg = round(calc_dhan_charges(p["entry_p"], exit_p, p["qty"], is_intraday=True), 2)
                net = round(gross - chg, 2)
                trades.append({"net_pnl": net, "exit_reason": "1445_FLAT"})
                last_exit_times.add(ts)
                del positions[ticker]
                continue
                
        closed_before = trades
        ek = 500.0 + sum(t["net_pnl"] for t in closed_before)
        hwm = 500.0
        cum = 500.0
        for t in closed_before:
            cum += t["net_pnl"]
            if cum > hwm:
                hwm = cum
                
        consec_losses = 0
        for t in reversed(closed_before):
            if t["net_pnl"] < 0:
                consec_losses += 1
            else:
                break
                
        if len(positions) >= 2 or ts in last_exit_times or t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:15").time():
            continue
            
        current_notional = sum(p["entry_p"] * p["qty"] for p in positions.values())
        avail_margin = (0.95 * ek) - (current_notional / 5.0)
        if avail_margin < 10.0:
            continue
        avail_notional = avail_margin * 5.0
        
        # Sizing Spline
        if consec_losses >= 2 or ((hwm - ek)/max(hwm, 500.0) > 0.16):
            alpha, beta = 0.015, 0.010
        elif consec_losses == 1:
            alpha, beta = 0.035, 0.020
        else:
            alpha, beta = 0.082, 0.035
            
        if ek <= 2500.0:
            raw_r = 14.0 + alpha * max(0.0, ek - 500.0)
        else:
            raw_r = 14.0 + alpha * 2000.0 + beta * (ek - 2500.0)
            
        target_risk = min(raw_r * 1.12, single_risk_cap)
        
        for ticker in tickers_10:
            if ticker in positions or ticker in daily_traded[cur_date]:
                continue
            t_data = data_map[ticker]
            if ts not in t_data["df"].index:
                continue
            idx = t_data["df"].index.get_loc(ts)
            if idx < 100 or not t_data["entry_cond"].iloc[idx]:
                continue
                
            cur_c = t_data["c"].iloc[idx]
            cur_h = t_data["h"].iloc[idx]
            cur_l = t_data["l"].iloc[idx]
            bar_vol = t_data["v"].iloc[idx]
            entry_atr = max(t_data["atr14"].iloc[idx], 0.05)
            raw_entry = cur_c * (1.0 + slippage)
            entry_p = round(max(cur_l, min(cur_h, raw_entry)), 2)
            
            active_sl = round(entry_p - (hard_sl_mult * entry_atr), 2)
            friction = 0.002067 * entry_p
            denom = (sizing_denom_mult * entry_atr) + friction
            
            qty_r = int(target_risk / denom)
            qty_m = int((avail_notional * 0.90) / entry_p)
            qty = min(qty_r, qty_m)
            if bar_vol > 0:
                qty = min(qty, int(0.045 * bar_vol))
                
            if qty >= 1:
                positions[ticker] = {
                    "entry_p": entry_p, "qty": qty, "entry_atr": entry_atr,
                    "active_sl": active_sl, "high_seen": entry_p, "bars_held": 0
                }
                daily_traded[cur_date].add(ticker)
                break

    df_res = pd.DataFrame(trades)
    n = len(df_res)
    wins = (df_res["net_pnl"] > 0).sum()
    wr = (wins / n * 100.0) if n else 0.0
    net = df_res["net_pnl"].sum()
    gp = df_res[df_res["net_pnl"] > 0]["net_pnl"].sum()
    gl = abs(df_res[df_res["net_pnl"] < 0]["net_pnl"].sum())
    pf = (gp / gl) if gl > 0 else 0.0
    avg_win = gp / wins if wins else 0.0
    avg_loss = -gl / (n - wins) if (n - wins) else 0.0
    worst = df_res["net_pnl"].min() if n else 0.0
    path = 500.0 + np.cumsum(df_res["net_pnl"].to_numpy(float))
    peak = np.maximum.accumulate(np.maximum(path, 500.0))
    mdd = float(((peak - path) / peak).max()) * 100.0 if len(path) else 0.0
    return {
        "n": n, "wr": wr, "net": net, "pf": pf,
        "final": path[-1] if len(path) else 500.0,
        "avg_win": avg_win, "avg_loss": avg_loss,
        "worst": worst, "mdd": mdd
    }

print("=== STEP-BY-STEP ABLATION ON TOP-10 VERIFIED UNIVERSE ===")
res_v17 = run_simulation(hard_sl_mult=1.00, sizing_denom_mult=1.00, single_risk_cap=260.0)
print(f"V17 Baseline       : Net=Rs {res_v17['net']:>7.2f} | Final=Rs {res_v17['final']:>7.2f} | PF={res_v17['pf']:.2f} | WR={res_v17['wr']:.1f}% | AvgWin=Rs {res_v17['avg_win']:>6.2f} | AvgLoss=Rs {res_v17['avg_loss']:>6.2f} | Worst=Rs {res_v17['worst']:>7.2f} | MDD={res_v17['mdd']:.1f}%")

res_cap135 = run_simulation(hard_sl_mult=1.00, sizing_denom_mult=1.00, single_risk_cap=135.0)
print(f"Risk Cap Rs 135    : Net=Rs {res_cap135['net']:>7.2f} | Final=Rs {res_cap135['final']:>7.2f} | PF={res_cap135['pf']:.2f} | WR={res_cap135['wr']:.1f}% | AvgWin=Rs {res_cap135['avg_win']:>6.2f} | AvgLoss=Rs {res_cap135['avg_loss']:>6.2f} | Worst=Rs {res_cap135['worst']:>7.2f} | MDD={res_cap135['mdd']:.1f}%")

res_decoupled = run_simulation(hard_sl_mult=0.85, sizing_denom_mult=1.00, single_risk_cap=135.0)
print(f"Decoupled 0.85x SL : Net=Rs {res_decoupled['net']:>7.2f} | Final=Rs {res_decoupled['final']:>7.2f} | PF={res_decoupled['pf']:.2f} | WR={res_decoupled['wr']:.1f}% | AvgWin=Rs {res_decoupled['avg_win']:>6.2f} | AvgLoss=Rs {res_decoupled['avg_loss']:>6.2f} | Worst=Rs {res_decoupled['worst']:>7.2f} | MDD={res_decoupled['mdd']:.1f}%")

res_3gear = run_simulation(hard_sl_mult=0.85, sizing_denom_mult=1.00, single_risk_cap=135.0, use_3gear_trail=True)
print(f"With 3-Gear Trail  : Net=Rs {res_3gear['net']:>7.2f} | Final=Rs {res_3gear['final']:>7.2f} | PF={res_3gear['pf']:.2f} | WR={res_3gear['wr']:.1f}% | AvgWin=Rs {res_3gear['avg_win']:>6.2f} | AvgLoss=Rs {res_3gear['avg_loss']:>6.2f} | Worst=Rs {res_3gear['worst']:>7.2f} | MDD={res_3gear['mdd']:.1f}%")
