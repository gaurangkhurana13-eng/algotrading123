import pickle
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT_DIR / "strategies"))
from framework import calc_dhan_charges, calculate_atr

# Load Data Pools
with open(ROOT_DIR / "data" / "universe_78_15m_1y.pkl", "rb") as f:
    univ = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_high_beta_10_15m_1y.pkl", "rb") as f:
    d_beta = pickle.load(f)
with open(ROOT_DIR / "data" / "unseen_10_15m_1y.pkl", "rb") as f:
    d_unseen = pickle.load(f)

all_pools = {}
all_pools.update(univ)
all_pools.update(d_beta)
all_pools.update(d_unseen)

tickers_12 = [
    "PCJEWELLER.NS", "RPOWER.NS", "DELTACORP.NS", "RITES.NS",
    "IDEA.NS", "MTNL.NS", "ZEEL.NS", "UTKARSHBNK.NS", "IFCI.NS", "NCC.NS",
    "PTC.NS", "NIACL.NS"
]

# Build dictionary of DataFrames formatted with timestamp column
raw_dfs = {}
for t in tickers_12:
    if t in all_pools:
        df = all_pools[t].copy()
        if "timestamp" not in df.columns:
            df["timestamp"] = df.index
        raw_dfs[t.replace(".NS", "")] = df

print("Loaded symbols for V19:", list(raw_dfs.keys()))

# Import Argon's V19 classes and functions
@dataclass(frozen=True)
class DhanFrictionModel:
    slippage_per_leg: float = 0.0005       # 0.05% per leg
    brokerage_rate: float = 0.0003         # 0.03% per executed order
    brokerage_cap_per_order: float = 20.0  # Rs 20 max per order
    stt_sell_rate: float = 0.00025         # 0.025% on sell notional
    nse_txn_rate: float = 0.0000297        # 0.00297% per leg
    sebi_rate: float = 0.000001            # 0.0001% per leg
    stamp_buy_rate: float = 0.00003        # 0.003% on buy notional
    gst_rate: float = 0.18                 # 18% on (Brokerage + NSE Txn + SEBI)

    def apply_entry_slippage(self, raw_price: float) -> float:
        return round(raw_price * (1.0 + self.slippage_per_leg), 2)

    def apply_exit_slippage(self, raw_price: float) -> float:
        return round(raw_price * (1.0 - self.slippage_per_leg), 2)

    def compute_taxes(self, buy_notional: float, sell_notional: float, num_entry_orders: int = 1) -> float:
        if num_entry_orders == 1:
            brok_buy = min(self.brokerage_cap_per_order, buy_notional * self.brokerage_rate)
        else:
            b1 = min(self.brokerage_cap_per_order, (buy_notional * 0.606) * self.brokerage_rate)
            b2 = min(self.brokerage_cap_per_order, (buy_notional * 0.394) * self.brokerage_rate)
            brok_buy = b1 + b2

        brok_sell = min(self.brokerage_cap_per_order, sell_notional * self.brokerage_rate)
        total_brok = brok_buy + brok_sell

        stt = sell_notional * self.stt_sell_rate
        turnover = buy_notional + sell_notional
        exch_txn = turnover * self.nse_txn_rate
        sebi_fee = turnover * self.sebi_rate
        stamp = buy_notional * self.stamp_buy_rate
        gst = self.gst_rate * (total_brok + exch_txn + sebi_fee)

        return round(total_brok + stt + exch_txn + sebi_fee + stamp + gst, 2)

def prepare_v19_features(df: pd.DataFrame) -> pd.DataFrame:
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

    # 2. Shifted Intraday Cumulative VWAP (.shift(1) eliminates intrabar look-ahead)
    typ_price = (h + l + c) / 3.0
    tp_vol = typ_price * v
    cum_tp_vol = tp_vol.groupby(d["date"]).cumsum()
    cum_vol = v.groupby(d["date"]).cumsum().replace(0, np.nan)
    raw_vwap = cum_tp_vol / cum_vol
    d["vwap_shifted"] = raw_vwap.groupby(d["date"]).shift(1).fillna(typ_price)

    # 3. Structural Regime EMAs (EMA 50 & EMA 110) + Fast Trail EMA 9
    d["ema_9"] = c.ewm(span=9, adjust=False).mean()
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

    # 8. V19 Microstructure Filters: CLV, Bar Range Ratio, and 20-Bar Wick-to-Body Ratio (WBR_20)
    bar_range = (h - l).replace(0, 1e-9)
    body_size = (c - o).abs()
    wick_size = (bar_range - body_size).clip(lower=0.0)

    d["clv"] = (c - l) / bar_range
    d["bar_range_atr"] = (h - l) / d["atr_14"].replace(0, 1e-9)
    d["wbr_20"] = (wick_size.shift(1).rolling(20).sum() / bar_range.shift(1).rolling(20).sum()).fillna(0.50)

    # 9. Structural Prior Lows & Prior Highs
    d["low_prev_1"] = l.shift(1)
    d["low_prev_2"] = l.shift(2)
    d["high_prev_1"] = h.shift(1)

    # 10. V19 Entry Signal Boolean
    macd_bull = (d["macd"] > 0.0) & (d["macd"] > d["macd_signal"]) & (d["macd_hist"] > d["macd_hist"].shift(1))
    mom_bull = (d["rsi_7"] > 52.0) & (d["rsi_7"] < 78.0) & (d["adx_10"] > 18.0) & (d["plus_di"] > d["minus_di"])
    regime_bull = (c > d["ema_50"]) & (d["ema_50"] >= d["ema_110"] * 0.997)
    vwap_band = ((c - d["vwap_shifted"]) >= 0.10 * d["atr_14"]) & ((c - d["vwap_shifted"]) <= 2.35 * d["atr_14"])

    microstructure_ok = (
        (d["tick_to_atr"] <= 0.14) &
        (d["natr_14"] >= 0.0058) &
        (d["wbr_20"] <= 0.56) &
        (d["er_10"] >= 0.28) &
        (d["rvol_20"] >= 1.20) &
        (d["clv"] >= 0.68) &
        (d["bar_range_atr"] <= 1.75)
    )

    time_ok = (d["time_hm"] >= 930) & (d["time_hm"] <= 1430)
    d["entry_signal"] = macd_bull & mom_bull & regime_bull & vwap_band & microstructure_ok & time_ok

    d["conviction_mult"] = (
        0.85
        + 0.20 * ((d["er_10"] - 0.28) / 0.42).clip(0.0, 1.0)
        + 0.15 * ((d["rvol_20"] - 1.20) / 2.30).clip(0.0, 1.0)
    ).clip(0.85, 1.20)

    return d

@dataclass
class ActivePosition:
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    raw_entry_close: float
    entry_mid: float
    entry_vol: float
    atr_at_entry: float
    natr_at_entry: float
    qty_base: int
    qty_pyramid: int = 0
    pyramid_entry_price: float = 0.0
    stop_price: float = 0.0
    highest_high: float = 0.0
    bars_held: int = 0
    is_risk_free: bool = False

    @property
    def total_qty(self) -> int:
        return self.qty_base + self.qty_pyramid

    @property
    def total_buy_notional(self) -> float:
        return (self.qty_base * self.entry_price) + (self.qty_pyramid * self.pyramid_entry_price)

class BhavishyaV19Engine:
    def __init__(
        self,
        initial_capital: float = 500.0,
        leverage: float = 5.0,
        hard_unit_risk_cap: float = 92.0,
        physical_stop_atr: float = 0.82,
        max_concurrent_slots: int = 3
    ):
        self.initial_capital = initial_capital
        self.wallet_equity = initial_capital
        self.leverage = leverage
        self.hard_unit_risk_cap = hard_unit_risk_cap
        self.physical_stop_atr = physical_stop_atr
        self.max_concurrent_slots = max_concurrent_slots
        self.friction = DhanFrictionModel()
        self.consecutive_losses = 0

    def _compute_base_unit_risk(self, natr: float, conviction_mult: float) -> float:
        eq = max(self.wallet_equity, 250.0)
        if eq <= 2200.0:
            r_raw = 14.50 + 0.033 * (eq - 500.0)
        elif eq <= 5500.0:
            r_raw = 70.60 + 0.0065 * (eq - 2200.0)
        else:
            r_raw = self.hard_unit_risk_cap

        f_friction = float(np.clip(natr / 0.0075, 0.72, 1.05))
        streak_mult = 0.82 if self.consecutive_losses >= 2 else 1.00
        target_risk = r_raw * conviction_mult * f_friction * streak_mult
        return float(min(target_risk, self.hard_unit_risk_cap))

    def _allowed_total_slots(self) -> int:
        if self.wallet_equity < 1800.0:
            return 1
        elif self.wallet_equity < 4500.0:
            return 2
        return self.max_concurrent_slots

    def _used_margin(self, positions: Dict[str, ActivePosition]) -> float:
        total_notional = sum(p.total_buy_notional for p in positions.values())
        return total_notional / self.leverage

    def _free_buying_power(self, positions: Dict[str, ActivePosition]) -> float:
        free_margin = max(0.0, self.wallet_equity * 0.95 - self._used_margin(positions))
        return free_margin * self.leverage

    def update_position_bar(self, pos: ActivePosition, row: pd.Series) -> Tuple[bool, float, str]:
        pos.bars_held += 1
        o, h, l, c, v = row["open"], row["high"], row["low"], row["close"], row["volume"]
        atr = pos.atr_at_entry

        if l <= pos.stop_price:
            fill_raw = min(o, pos.stop_price)
            reason = "TRAIL_STOP" if pos.is_risk_free else "HARD_STOP_0.82ATR"
            return True, fill_raw, reason

        if h > pos.highest_high:
            pos.highest_high = h
        mfe = pos.highest_high - pos.entry_price
        mfe_atr = mfe / atr

        if row["time_hm"] >= 1510:
            return True, c, "EOD_1510_FLAT"

        if not pos.is_risk_free:
            if pos.bars_held in (1, 2) and mfe_atr < 0.25:
                vwap_break = c < row["vwap_shifted"]
                heavy_red_dump = (c < o) and (v >= 0.85 * pos.entry_vol)
                if (c < pos.entry_mid) and (vwap_break or heavy_red_dump):
                    return True, c, "FAST_SCRATCH_T1_ORDERFLOW"

            if pos.bars_held == 2 and mfe_atr < 0.18:
                if (c < pos.entry_price - 0.15 * atr) and (row["rsi_7"] < 48.0):
                    return True, c, "FAST_SCRATCH_T2_STALL"

            if pos.bars_held >= 3 and mfe_atr < 0.40:
                if (c <= pos.entry_price + 0.05 * atr) or (row["rsi_7"] < 47.0):
                    return True, c, "FAST_SCRATCH_T3_TIMESTOP"

        new_stop = pos.stop_price

        if mfe_atr >= 0.80:
            p1_stop = pos.entry_price + 0.18 * atr
            new_stop = max(new_stop, p1_stop)
            pos.is_risk_free = True

        if mfe_atr >= 1.40:
            swing_floor = min(row["low_prev_1"], row["low_prev_2"]) - 0.15 * atr
            p2_stop = max(
                pos.entry_price + 0.65 * atr,
                swing_floor,
                pos.highest_high - 0.92 * atr
            )
            new_stop = max(new_stop, p2_stop)

        if mfe_atr >= 2.80:
            p3_stop = max(
                row["low_prev_1"] - 0.10 * atr,
                pos.highest_high - 0.60 * atr
            )
            new_stop = max(new_stop, p3_stop)

        climax_wick = (mfe_atr >= 2.50) and (row["rsi_7"] >= 83.0) and (row["clv"] <= 0.45)
        if (mfe_atr >= 4.00) or climax_wick:
            p4_stop = pos.highest_high - 0.35 * atr
            new_stop = max(new_stop, p4_stop)

        if new_stop < c:
            pos.stop_price = round(max(pos.stop_price, new_stop), 2)

        return False, 0.0, ""

    def try_house_money_pyramid(
        self, pos: ActivePosition, row: pd.Series, positions: Dict[str, ActivePosition]
    ) -> None:
        if pos.qty_pyramid > 0 or row["time_hm"] > 1415:
            return

        c, h = row["close"], row["high"]
        atr = pos.atr_at_entry
        current_gain_atr = (c - pos.entry_price) / atr

        if (
            current_gain_atr >= 1.25
            and c > row["high_prev_1"]
            and row["rvol_20"] >= 1.35
            and row["rsi_7"] < 79.0
        ):
            desired_q2 = int(np.floor(pos.qty_base * 0.65))
            if desired_q2 < 1:
                return

            pyr_fill = self.friction.apply_entry_slippage(c)
            free_bp = self._free_buying_power(positions)
            max_afford_q2 = int(np.floor(free_bp / pyr_fill))
            q2 = min(desired_q2, max_afford_q2)

            if q2 >= 1:
                pos.qty_pyramid = q2
                pos.pyramid_entry_price = pyr_fill
                unified_stop = round(pos.entry_price + 0.45 * atr, 2)
                if unified_stop < c:
                    pos.stop_price = max(pos.stop_price, unified_stop)
                    pos.is_risk_free = True

    def run_portfolio_backtest(self, symbol_dfs: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict]:
        prepared = {sym: prepare_v19_features(df) for sym, df in symbol_dfs.items()}

        all_frames = []
        for sym, df in prepared.items():
            df_sym = df.copy()
            df_sym["symbol"] = sym
            all_frames.append(df_sym)
        panel = pd.concat(all_frames, ignore_index=True).sort_values(["timestamp", "symbol"])

        active_positions: Dict[str, ActivePosition] = {}
        closed_trades: List[Dict] = []
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

    def _compile_telemetry(self, trades_df: pd.DataFrame, max_dd_pct: float) -> Dict:
        if trades_df.empty:
            return {"status": "NO_TRADES"}
        wins = trades_df[trades_df["net_pnl"] > 0]
        losses = trades_df[trades_df["net_pnl"] <= 0]
        gross_wins = trades_df[trades_df["gross_pnl"] > 0]["gross_pnl"].sum()
        total_taxes = trades_df["charges"].sum()
        net_win_sum = wins["net_pnl"].sum()
        net_loss_sum = abs(losses["net_pnl"].sum())

        return {
            "total_trades": len(trades_df),
            "win_rate_pct": round(len(wins) / len(trades_df) * 100.0, 2),
            "net_profit_rs": round(trades_df["net_pnl"].sum(), 2),
            "final_wallet_rs": round(self.wallet_equity, 2),
            "net_profit_factor": round(net_win_sum / max(net_loss_sum, 1e-6), 2),
            "avg_win_net_rs": round(wins["net_pnl"].mean(), 2) if not wins.empty else 0.0,
            "avg_loss_net_rs": round(losses["net_pnl"].mean(), 2) if not losses.empty else 0.0,
            "worst_single_loss_rs": round(trades_df["net_pnl"].min(), 2),
            "dhan_taxes_paid_rs": round(total_taxes, 2),
            "tax_to_gross_win_pct": round(total_taxes / max(gross_wins, 1e-6) * 100.0, 2),
            "max_drawdown_pct": round(max_dd_pct, 2)
        }

print("Running Bhavishya V19 Engine Simulation...")
engine = BhavishyaV19Engine(initial_capital=500.0)
df_trades, summary = engine.run_portfolio_backtest(raw_dfs)

print("\n=== BHAVISHYA V19 TELEMETRY ===")
for k, v in summary.items():
    print(f"  {k}: {v}")

if not df_trades.empty:
    print("\n=== EXIT REASON BREAKDOWN ===")
    print(df_trades["exit_reason"].value_counts())
    print("\n=== TOP 5 WINNERS ===")
    print(df_trades.sort_values("net_pnl", ascending=False)[["symbol", "net_pnl", "gross_pnl", "shares", "exit_reason"]].head(5).to_string(index=False))
    print("\n=== WORST 5 LOSERS ===")
    print(df_trades.sort_values("net_pnl").head(5)[["symbol", "net_pnl", "gross_pnl", "shares", "exit_reason"]].to_string(index=False))
