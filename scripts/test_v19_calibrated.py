import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "scripts"))
from test_v19_direct import raw_dfs, DhanFrictionModel, ActivePosition, BhavishyaV19Engine

def prepare_v19_calibrated_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy().sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(d["timestamp"])
    d["date"] = ts.dt.date
    d["time_hm"] = ts.dt.hour * 100 + ts.dt.minute

    o, h, l, c, v = d["open"], d["high"], d["low"], d["close"], d["volume"]

    # 1. Wilder's ATR(14) & NATR(14)
    prev_c = c.shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - prev_c).abs(),
        (l - prev_c).abs()
    ], axis=1).max(axis=1)
    d["atr_14"] = tr.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    d["natr_14"] = d["atr_14"] / c.replace(0, np.nan)
    d["tick_to_atr"] = 0.05 / d["atr_14"].replace(0, np.nan)

    # 2. Shifted Intraday Cumulative VWAP
    typ_price = (h + l + c) / 3.0
    tp_vol = typ_price * v
    cum_tp_vol = tp_vol.groupby(d["date"]).cumsum()
    cum_vol = v.groupby(d["date"]).cumsum().replace(0, np.nan)
    raw_vwap = cum_tp_vol / cum_vol
    d["vwap_shifted"] = raw_vwap.groupby(d["date"]).shift(1).fillna(typ_price)

    # 3. Structural Regime EMAs (EMA 50 & EMA 110)
    d["ema_50"] = c.ewm(span=50, adjust=False).mean()
    d["ema_110"] = c.ewm(span=110, adjust=False).mean()

    # 4. MACD (12, 26, 9)
    ema_12 = c.ewm(span=12, adjust=False).mean()
    ema_26 = c.ewm(span=26, adjust=False).mean()
    d["macd"] = ema_12 - ema_26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]

    # 5. RSI(7)
    delta = c.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / 7.0, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / 7.0, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-9)
    d["rsi_7"] = 100.0 - (100.0 / (1.0 + rs))

    # 6. ADX(10), +DI(10), -DI(10)
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr_10 = tr.ewm(alpha=1.0 / 10.0, adjust=False).mean().replace(0, 1e-9)
    d["plus_di"] = 100.0 * pd.Series(plus_dm, index=d.index).ewm(alpha=1.0 / 10.0, adjust=False).mean() / tr_10
    d["minus_di"] = 100.0 * pd.Series(minus_dm, index=d.index).ewm(alpha=1.0 / 10.0, adjust=False).mean() / tr_10
    dx = 100.0 * (d["plus_di"] - d["minus_di"]).abs() / (d["plus_di"] + d["minus_di"]).replace(0, 1e-9)
    d["adx_10"] = dx.ewm(alpha=1.0 / 10.0, adjust=False).mean()

    # 7. Kaufman Efficiency Ratio (ER_10) & RVOL(20)
    net_disp_10 = (c - c.shift(10)).abs()
    sum_path_10 = c.diff().abs().rolling(10).sum().replace(0, 1e-9)
    d["er_10"] = (net_disp_10 / sum_path_10).clip(0.0, 1.0)
    vol_ma_20 = v.shift(1).rolling(20).mean().replace(0, np.nan)
    d["rvol_20"] = (v / vol_ma_20).fillna(1.0)

    # 8. Microstructure Filters: CLV, Bar Range Ratio, WBR_20
    bar_range = (h - l).replace(0, 1e-9)
    body_size = (c - o).abs()
    wick_size = (bar_range - body_size).clip(lower=0.0)

    d["clv"] = (c - l) / bar_range
    d["bar_range_atr"] = (h - l) / d["atr_14"].replace(0, 1e-9)
    d["wbr_20"] = (wick_size.shift(1).rolling(20).sum() / bar_range.shift(1).rolling(20).sum()).fillna(0.50)

    # 9. Structural Prior Lows & Highs
    d["low_prev_1"] = l.shift(1)
    d["low_prev_2"] = l.shift(2)
    d["high_prev_1"] = h.shift(1)

    # 10. Calibrated Entry Signal
    macd_bull = (d["macd"] > 0.0) & (d["macd"] > d["macd_signal"]) & (d["macd_hist"] > d["macd_hist"].shift(1))
    mom_bull = (d["rsi_7"] > 50.0) & (d["rsi_7"] < 80.0) & (d["adx_10"] > 18.0) & (d["plus_di"] > d["minus_di"])
    regime_bull = (c > d["ema_50"]) & (d["ema_50"] >= d["ema_110"] * 0.995)
    vwap_band = ((c - d["vwap_shifted"]) >= 0.05 * d["atr_14"]) & ((c - d["vwap_shifted"]) <= 2.80 * d["atr_14"])

    microstructure_ok = (
        (d["tick_to_atr"] <= 0.15) &     # Valid for liquid midcaps
        (d["natr_14"] >= 0.0045) &       # NATR >= 0.45%
        (d["wbr_20"] <= 0.60) &          # Rejects heavy PSU chop
        (d["er_10"] >= 0.25) &           # Directional persistence
        (d["rvol_20"] >= 1.05) &         # Above-average volume
        (d["clv"] >= 0.55) &             # Upper 45% close (no shooting star)
        (d["bar_range_atr"] <= 2.20)     # No extreme exhaustion blowoff
    )

    time_ok = (d["time_hm"] >= 920) & (d["time_hm"] <= 1430)
    d["entry_signal"] = macd_bull & mom_bull & regime_bull & vwap_band & microstructure_ok & time_ok

    d["conviction_mult"] = (
        0.85
        + 0.20 * ((d["er_10"] - 0.25) / 0.45).clip(0.0, 1.0)
        + 0.15 * ((d["rvol_20"] - 1.05) / 2.0).clip(0.0, 1.0)
    ).clip(0.85, 1.20)

    return d

# Re-run engine with calibrated features
import test_v19_direct
test_v19_direct.prepare_v19_features = prepare_v19_calibrated_features

engine = BhavishyaV19Engine(
    initial_capital=500.0,
    hard_unit_risk_cap=95.0,
    physical_stop_atr=0.85,
    max_concurrent_slots=2
)

# Overwrite run_portfolio_backtest to use prepare_v19_calibrated_features
def run_calibrated(self, symbol_dfs):
    prepared = {sym: prepare_v19_calibrated_features(df) for sym, df in symbol_dfs.items()}
    all_frames = []
    for sym, df in prepared.items():
        df_sym = df.copy()
        df_sym["symbol"] = sym
        all_frames.append(df_sym)
    panel = pd.concat(all_frames, ignore_index=True).sort_values(["timestamp", "symbol"])

    active_positions = {}
    closed_trades = []
    peak_equity = self.wallet_equity
    max_dd_pct = 0.0

    for ts, bar_slice in panel.groupby("timestamp", sort=True):
        for _, row in bar_slice.iterrows():
            sym = row["symbol"]
            if sym not in active_positions:
                continue

            pos = active_positions[sym]
            should_exit, raw_exit, reason = self.update_position_bar(pos, row)

            if should_exit:
                exit_fill = self.friction.apply_exit_slippage(raw_exit)
                buy_notional = pos.total_buy_notional
                sell_notional = pos.total_qty * exit_fill
                num_entries = 2 if pos.qty_pyramid > 0 else 1

                taxes = self.friction.compute_taxes(buy_notional, sell_notional, num_entries)
                gross_pnl = sell_notional - buy_notional
                net_pnl = round(gross_pnl - taxes, 2)

                self.wallet_equity = round(self.wallet_equity + net_pnl, 2)
                peak_equity = max(peak_equity, self.wallet_equity)
                dd = (peak_equity - self.wallet_equity) / peak_equity * 100.0
                max_dd_pct = max(max_dd_pct, dd)

                if net_pnl < 0:
                    self.consecutive_losses += 1
                else:
                    self.consecutive_losses = 0

                avg_entry = round(buy_notional / pos.total_qty, 2)
                closed_trades.append({
                    "symbol": sym,
                    "direction": "long",
                    "entry_time": str(pos.entry_time),
                    "exit_time": str(ts),
                    "entry_price": avg_entry,
                    "exit_price": exit_fill,
                    "shares": pos.total_qty,
                    "sl": pos.stop_price,
                    "gross_pnl": round(gross_pnl, 2),
                    "charges": taxes,
                    "net_pnl": net_pnl,
                    "exit_reason": reason,
                    "qty_base": pos.qty_base,
                    "qty_pyramid": pos.qty_pyramid,
                    "wallet_after": self.wallet_equity
                })
                del active_positions[sym]
            else:
                self.try_house_money_pyramid(pos, row, active_positions)

        max_slots = self._allowed_total_slots()
        if len(active_positions) >= max_slots:
            continue

        any_slot_at_risk = any(not p.is_risk_free for p in active_positions.values())
        if any_slot_at_risk:
            continue

        candidates = bar_slice[
            (bar_slice["entry_signal"]) & (~bar_slice["symbol"].isin(active_positions.keys()))
        ].sort_values("conviction_mult", ascending=False)

        if candidates.empty:
            continue

        top = candidates.iloc[0]
        sym = top["symbol"]
        entry_fill = self.friction.apply_entry_slippage(top["close"])
        atr = float(top["atr_14"])
        natr = float(top["natr_14"])

        unit_risk_rs = self._compute_base_unit_risk(natr, float(top["conviction_mult"]))
        qty_risk = int(np.floor(unit_risk_rs / atr))
        free_bp = self._free_buying_power(active_positions)
        qty_margin = int(np.floor(free_bp / entry_fill))
        qty = min(qty_risk, qty_margin)

        if qty >= 1:
            init_stop = round(entry_fill - self.physical_stop_atr * atr, 2)
            active_positions[sym] = ActivePosition(
                symbol=sym,
                entry_time=ts,
                entry_price=entry_fill,
                raw_entry_close=float(top["close"]),
                entry_mid=0.5 * (float(top["open"]) + float(top["close"])),
                entry_vol=float(top["volume"]),
                atr_at_entry=atr,
                natr_at_entry=natr,
                qty_base=qty,
                stop_price=init_stop,
                highest_high=float(top["high"])
            )

    trades_df = pd.DataFrame(closed_trades)
    summary = self._compile_telemetry(trades_df, max_dd_pct)
    return trades_df, summary

BhavishyaV19Engine.run_portfolio_backtest = run_calibrated

print("Running Calibrated Bhavishya V19 Engine Simulation...")
df_trades, summary = engine.run_portfolio_backtest(raw_dfs)

print("\n=== CALIBRATED BHAVISHYA V19 TELEMETRY ===")
for k, v in summary.items():
    print(f"  {k}: {v}")

if not df_trades.empty:
    print("\n=== EXIT REASON BREAKDOWN ===")
    print(df_trades["exit_reason"].value_counts())
    print("\n=== TOP 5 WINNERS ===")
    print(df_trades.sort_values("net_pnl", ascending=False)[["symbol", "net_pnl", "gross_pnl", "shares", "qty_pyramid", "exit_reason"]].head(5).to_string(index=False))
    print("\n=== WORST 5 LOSERS ===")
    print(df_trades.sort_values("net_pnl").head(5)[["symbol", "net_pnl", "gross_pnl", "shares", "exit_reason"]].to_string(index=False))
