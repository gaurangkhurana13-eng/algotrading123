"""
Bhavishya Strategy V21: Order-Flow Physics & Continuous Compounding Engine
Asset Class: Indian Equities (NSE Cash 15m, Dhan 5x MIS Intraday)
Audit: Gate V2 Institutional Compliance (Zero Look-Ahead, Causal Order-Flow Scratches)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple


# ==============================================================================
# 1. PENNY-EXACT DHAN 5x MIS STATUTORY TAX & SLIPPAGE ENGINE
# ==============================================================================
@dataclass(frozen=True)
class DhanFrictionEngine:
    slippage_per_leg: float = 0.0005       # 0.05% adverse slippage per leg
    brokerage_rate: float = 0.0003         # 0.03% per executed order
    brokerage_cap_per_order: float = 20.0  # Rs 20 max per order
    stt_sell_rate: float = 0.00025         # 0.025% on sell turnover
    nse_txn_rate: float = 0.0000297        # 0.00297% per leg
    sebi_rate: float = 0.000001            # 0.0001% per leg
    stamp_buy_rate: float = 0.00003        # 0.003% on buy turnover
    gst_rate: float = 0.18                 # 18% on (Brokerage + NSE Txn + SEBI)

    def buy_fill(self, raw_price: float) -> float:
        return round(float(raw_price) * (1.0 + self.slippage_per_leg), 2)

    def sell_fill(self, raw_price: float) -> float:
        return round(float(raw_price) * (1.0 - self.slippage_per_leg), 2)

    def compute_statutory_charges(
        self,
        buy_notional_base: float,
        buy_notional_pyr: float,
        sell_notional_total: float
    ) -> float:
        """Calculates exact Dhan brokerage + STT + NSE Txn + Stamp + SEBI + 18% GST."""
        brok_buy_1 = min(self.brokerage_cap_per_order, buy_notional_base * self.brokerage_rate)
        brok_buy_2 = (
            min(self.brokerage_cap_per_order, buy_notional_pyr * self.brokerage_rate)
            if buy_notional_pyr > 0.0 else 0.0
        )
        brok_sell = min(self.brokerage_cap_per_order, sell_notional_total * self.brokerage_rate)
        total_brok = brok_buy_1 + brok_buy_2 + brok_sell

        total_buy = buy_notional_base + buy_notional_pyr
        turnover = total_buy + sell_notional_total

        stt = sell_notional_total * self.stt_sell_rate
        exch_txn = turnover * self.nse_txn_rate
        sebi_fee = turnover * self.sebi_rate
        stamp_duty = total_buy * self.stamp_buy_rate
        gst = self.gst_rate * (total_brok + exch_txn + sebi_fee)

        return round(total_brok + stt + exch_txn + sebi_fee + stamp_duty + gst, 2)


# ==============================================================================
# 2. CAUSAL (ZERO-LOOKAHEAD) MICROSTRUCTURE & SIGNAL PIPELINE
# ==============================================================================
def compute_v21_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Input: Single-symbol 15m DataFrame with ['timestamp', 'open', 'high', 'low', 'close', 'volume'].
    All cross-bar references use .shift(1) to guarantee 100% Gate V2 zero-lookahead compliance.
    """
    d = df.copy()
    if "timestamp" not in d.columns:
        d["timestamp"] = d.index
    d = d.sort_values("timestamp").reset_index(drop=True)
    ts = pd.to_datetime(d["timestamp"])
    d["date"] = ts.dt.date
    d["time_hm"] = ts.dt.hour * 100 + ts.dt.minute

    o = d["open"].astype(float)
    h = d["high"].astype(float)
    l = d["low"].astype(float)
    c = d["close"].astype(float)
    v = d["volume"].astype(float)

    # 1. Wilder's ATR(14) & NATR(14)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l).abs(), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    d["atr_14"] = tr.ewm(alpha=1.0 / 14.0, adjust=False).mean()
    d["natr_14"] = d["atr_14"] / c.replace(0.0, np.nan)
    d["tick_to_atr"] = 0.05 / d["atr_14"].replace(0.0, np.nan)

    # 2. Strictly Causal Shifted Intraday VWAP (.shift(1) within each trading session)
    typ = (h + l + c) / 3.0
    cum_pv = (typ * v).groupby(d["date"]).cumsum()
    cum_v = v.groupby(d["date"]).cumsum().replace(0.0, np.nan)
    vwap_curr = cum_pv / cum_v
    d["vwap_shifted"] = vwap_curr.groupby(d["date"]).shift(1).fillna(typ)

    # 3. Dual Structural Regime EMAs (50 & 110)
    d["ema_50"] = c.ewm(span=50, adjust=False).mean()
    d["ema_110"] = c.ewm(span=110, adjust=False).mean()

    # 4. MACD (12, 26, 9)
    ema_12 = c.ewm(span=12, adjust=False).mean()
    ema_26 = c.ewm(span=26, adjust=False).mean()
    d["macd"] = ema_12 - ema_26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]

    # 5. Wilder's RSI(7)
    delta = c.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / 7.0, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / 7.0, adjust=False).mean()
    d["rsi_7"] = 100.0 - (100.0 / (1.0 + (gain / loss.replace(0.0, 1e-9))))

    # 6. Wilder's ADX(10), +DI(10), -DI(10)
    up_m = h.diff()
    dn_m = -l.diff()
    p_dm = np.where((up_m > dn_m) & (up_m > 0.0), up_m, 0.0)
    m_dm = np.where((dn_m > up_m) & (dn_m > 0.0), dn_m, 0.0)
    tr_10 = tr.ewm(alpha=1.0 / 10.0, adjust=False).mean().replace(0.0, 1e-9)
    d["plus_di"] = 100.0 * pd.Series(p_dm, index=d.index).ewm(alpha=1.0 / 10.0, adjust=False).mean() / tr_10
    d["minus_di"] = 100.0 * pd.Series(m_dm, index=d.index).ewm(alpha=1.0 / 10.0, adjust=False).mean() / tr_10
    dx = 100.0 * (d["plus_di"] - d["minus_di"]).abs() / (d["plus_di"] + d["minus_di"]).replace(0.0, 1e-9)
    d["adx_10"] = dx.ewm(alpha=1.0 / 10.0, adjust=False).mean()

    # 7. Kaufman Efficiency Ratio ER(10) & Causal Relative Volume RVOL(20)
    disp_10 = (c - c.shift(10)).abs()
    path_10 = c.diff().abs().rolling(10).sum().replace(0.0, 1e-9)
    d["er_10"] = (disp_10 / path_10).clip(0.0, 1.0)
    vol_baseline = v.shift(1).rolling(20).mean().replace(0.0, np.nan)
    d["rvol_20"] = (v / vol_baseline).fillna(1.0)

    # 8. Order-Book Microstructure Metrics: CLV, Bar Range Ratio, and Shifted WBR(20)
    rng = (h - l).replace(0.0, 1e-9)
    body = (c - o).abs()
    wick = (rng - body).clip(lower=0.0)

    d["clv"] = (c - l) / rng
    d["bar_range_atr"] = (h - l) / d["atr_14"].replace(0.0, 1e-9)
    d["wbr_20"] = (
        wick.shift(1).rolling(20).sum() / rng.shift(1).rolling(20).sum().replace(0.0, 1e-9)
    ).fillna(0.50)

    # 9. Causal Prior Structural Swings for Phase 2/3 Trailing & Pyramiding
    d["low_prev_1"] = l.shift(1)
    d["low_prev_2"] = l.shift(2)
    d["low_prev_3"] = l.shift(3)
    d["high_prev_1"] = h.shift(1)

    # 10. V21 Syllabus Entry Filter (Restores V19 Microstructure Physics)
    macd_ok = (d["macd"] > 0.0) & (d["macd"] > d["macd_signal"]) & (d["macd_hist"] > d["macd_hist"].shift(1))
    mom_ok = (d["rsi_7"] >= 52.0) & (d["rsi_7"] <= 78.5) & (d["adx_10"] >= 18.0) & (d["plus_di"] > d["minus_di"])
    regime_ok = (c > d["ema_50"]) & (d["ema_50"] >= d["ema_110"] * 0.997)
    vwap_ok = ((c - d["vwap_shifted"]) >= 0.08 * d["atr_14"]) & ((c - d["vwap_shifted"]) <= 2.40 * d["atr_14"])

    microstructure_ok = (
        (d["tick_to_atr"] <= 0.14) &     # Valid tick-to-volatility bound for midcaps
        (d["natr_14"] >= 0.0056) &       # Caps Dhan statutory tax drag <= 18.9% of 1R
        (d["wbr_20"] <= 0.57) &          # Blocks PSU/cyclical pin-bar chop automatically
        (d["er_10"] >= 0.26) &           # Requires directional fractal efficiency
        (d["rvol_20"] >= 1.18) &         # Requires institutional volume expansion
        (d["clv"] >= 0.66) &             # Closes in top 34% of bar (no supply upper wick)
        (d["bar_range_atr"] >= 0.45) &
        (d["bar_range_atr"] <= 1.80)     # Blocks exhausted climax blow-off bars
    )

    session_ok = (d["time_hm"] >= 930) & (d["time_hm"] <= 1430)
    d["entry_signal"] = macd_ok & mom_ok & regime_ok & vwap_ok & microstructure_ok & session_ok

    # Ticker-Agnostic Conviction Multiplier (0.88x to 1.18x)
    d["conviction_mult"] = (
        0.88
        + 0.18 * ((d["er_10"] - 0.26) / 0.44).clip(0.0, 1.0)
        + 0.12 * ((d["rvol_20"] - 1.18) / 2.32).clip(0.0, 1.0)
    ).clip(0.88, 1.18)

    return d


# ==============================================================================
# 3. BHAVISHYA V21 EXECUTION STATE MACHINE
# ==============================================================================
@dataclass
class V21Position:
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float           # Slippage-adjusted base entry fill
    entry_open: float            # O_0 of breakout bar
    entry_mid: float             # M_0 = 0.5 * (O_0 + C_0) of breakout bar
    entry_vol: float             # V_0 of breakout bar
    atr_at_entry: float          # A_0
    natr_at_entry: float
    qty_base: int
    qty_pyramid: int = 0
    pyramid_entry_price: float = 0.0
    stop_price: float = 0.0
    highest_high: float = 0.0
    bars_held: int = 0
    is_risk_free: bool = False   # True once Stop >= Entry_1 + 0.20 * A_0 (Unlocks next slot)

    @property
    def total_qty(self) -> int:
        return self.qty_base + self.qty_pyramid

    @property
    def buy_notional_base(self) -> float:
        return self.qty_base * self.entry_price

    @property
    def buy_notional_pyramid(self) -> float:
        return self.qty_pyramid * self.pyramid_entry_price

    @property
    def total_buy_notional(self) -> float:
        return self.buy_notional_base + self.buy_notional_pyramid


class BhavishyaV21Engine:
    def __init__(
        self,
        initial_capital: float = 500.0,
        leverage: float = 5.0,
        hard_unit_risk_cap: float = 102.0,    # 1.00x ATR denom -> -Rs 99.78 worst net loss at 0.84x stop
        physical_stop_atr: float = 0.84,      # Decoupled physical hard stop multiplier
        pyramid_scale_ratio: float = 0.70,    # House-money tranche ratio Q_2 = 0.70 * Q_1
        max_slots_cap: int = 3
    ):
        self.initial_capital = initial_capital
        self.wallet_equity = initial_capital
        self.leverage = leverage
        self.hard_unit_risk_cap = hard_unit_risk_cap
        self.physical_stop_atr = physical_stop_atr
        self.pyramid_scale_ratio = pyramid_scale_ratio
        self.max_slots_cap = max_slots_cap
        self.friction = DhanFrictionEngine()
        self.consecutive_losses = 0

    def compute_monotonic_unit_risk(self, natr: float, conviction_mult: float) -> float:
        """
        C^0-Continuous 3-Stage Monotonic Risk Spline (dR/dE >= 0 everywhere).
        Eliminates the Rs 2,000 equity cliff completely.
        """
        e = max(self.wallet_equity, 250.0)
        if e <= 1500.0:
            r_base = 16.00 + 0.0420 * max(0.0, e - 500.0)
        elif e <= 3200.0:
            r_base = 58.00 + 0.0160 * (e - 1500.0)
        elif e <= 5500.0:
            r_base = 85.20 + 0.005565 * (e - 3200.0)
        else:
            r_base = 98.00

        f_fric = float(np.clip(natr / 0.0072, 0.78, 1.05))
        m_streak = 0.85 if self.consecutive_losses >= 3 else 1.00

        r_unit = r_base * conviction_mult * f_fric * m_streak
        return float(min(r_unit, self.hard_unit_risk_cap))

    def _allowed_slots(self) -> int:
        if self.wallet_equity < 1600.0:
            return 1
        elif self.wallet_equity < 4200.0:
            return 2
        return self.max_slots_cap

    def _free_buying_power(self, active_positions: Dict[str, V21Position]) -> float:
        used_notional = sum(p.total_buy_notional for p in active_positions.values())
        used_margin = used_notional / self.leverage
        free_margin = max(0.0, (self.wallet_equity * 0.95) - used_margin)
        return free_margin * self.leverage

    def evaluate_bar_and_corridor(
        self, pos: V21Position, row: pd.Series
    ) -> Tuple[bool, float, str]:
        """
        Executes intrabar stop check, V21 Order-Flow Scratch (preserving Healthy Pauses),
        and 4-Phase Asymmetric Structural Trail Corridor.
        """
        pos.bars_held += 1
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
        v = float(row["volume"])
        atr_0 = pos.atr_at_entry
        atr_cur = float(row["atr_14"])

        # 1. Conservative Intrabar Hard/Trailing Stop Priority
        if l <= pos.stop_price:
            raw_exit = min(o, pos.stop_price)
            reason = "TRAIL_STOP" if pos.is_risk_free else "HARD_STOP_0.84ATR"
            return True, raw_exit, reason

        # Update Peak MFE
        if h > pos.highest_high:
            pos.highest_high = h
        mfe_atr = (pos.highest_high - pos.entry_price) / atr_0

        # 2. Strict 15:10 IST Intraday Flat
        if int(row["time_hm"]) >= 1510:
            return True, c, "EOD_1510_FLAT"

        # 3. V21 Order-Flow Scratch (Distinguishes Healthy Low-Vol Pauses from Distribution Traps)
        if not pos.is_risk_free:
            below_midpoint = c < pos.entry_mid
            below_vwap = c < float(row["vwap_shifted"])

            # State 2: Tier-1 Active Distribution Dump on Bar 1 or Bar 2
            if pos.bars_held in (1, 2) and mfe_atr < 0.25 and below_midpoint and below_vwap:
                heavy_red_dump = (c < o) and (v >= 0.85 * pos.entry_vol)
                bottom_close_slam = float(row["clv"]) <= 0.25
                if heavy_red_dump or bottom_close_slam:
                    return True, c, "SCRATCH_T1_DISTRIBUTION"

            # State 3: Tier-2 Full Breakout Body Breakdown on Bar 2
            if pos.bars_held == 2 and mfe_atr < 0.20:
                if (c < pos.entry_open) and below_vwap and (float(row["rsi_7"]) < 46.5):
                    return True, c, "SCRATCH_T2_BODY_FAILURE"

            # State 4: Tier-3 Structural Time-Decay Scratch on Bar >= 3
            if pos.bars_held >= 3 and mfe_atr < 0.38:
                if below_midpoint and (float(row["rsi_7"]) < 46.0):
                    return True, c, "SCRATCH_T3_STRUCTURAL_DECAY"

        # 4. V21 4-Phase Asymmetric Structural Trail Corridor
        new_stop = pos.stop_price

        # Phase 1: Friction-Cleared Breakeven Ratchet (0.82x to 1.45x ATR)
        if mfe_atr >= 0.82:
            new_stop = max(new_stop, pos.entry_price + 0.20 * atr_0)
            pos.is_risk_free = True  # Unlocks Zero-Risk Slot 2!

        # Phase 2: 2-Bar Structural Swing Corridor (1.45x to 2.40x ATR)
        if 1.45 <= mfe_atr < 2.40:
            swing_2bar = min(float(row["low_prev_1"]), float(row["low_prev_2"])) - 0.16 * atr_cur
            p2_stop = max(
                pos.entry_price + 0.65 * atr_0,
                swing_2bar,
                pos.highest_high - 0.88 * atr_cur
            )
            new_stop = max(new_stop, p2_stop)

        # Phase 3: Super-Runner 3-Bar Structural Corridor (MFE >= 2.40x ATR)
        if mfe_atr >= 2.40:
            swing_3bar = min(
                float(row["low_prev_1"]),
                float(row["low_prev_2"]),
                float(row["low_prev_3"])
            ) - 0.12 * atr_cur
            p3_stop = max(
                pos.entry_price + 1.55 * atr_0,
                swing_3bar,
                pos.highest_high - 0.98 * atr_cur
            )
            new_stop = max(new_stop, p3_stop)

        # Phase 4: Parabolic Climax Harvester (MFE >= 3.80x ATR or High-RSI Rejection Wick)
        climax_wick = (mfe_atr >= 2.60) and (float(row["rsi_7"]) >= 82.5) and (float(row["clv"]) <= 0.42)
        if mfe_atr >= 3.80 or climax_wick:
            p4_stop = pos.highest_high - 0.36 * atr_cur
            new_stop = max(new_stop, p4_stop)

        if new_stop < c:
            pos.stop_price = round(max(pos.stop_price, new_stop), 2)

        return False, 0.0, ""

    def try_house_money_pyramid(
        self, pos: V21Position, row: pd.Series, active_positions: Dict[str, V21Position]
    ) -> None:
        """
        Injects Q_2 = 0.70 * Q_1 pyramid tranche when gain >= 1.20x ATR_0, Close > High[t-1],
        and RVOL >= 1.30, locking unified stop at Entry_1 + 0.45 * ATR_0 (Net Risk = -0.075R).
        """
        if pos.qty_pyramid > 0 or int(row["time_hm"]) > 1415:
            return

        c = float(row["close"])
        atr_0 = pos.atr_at_entry
        gain_atr = (c - pos.entry_price) / atr_0

        if (
            gain_atr >= 1.20
            and c > float(row["high_prev_1"])
            and float(row["rvol_20"]) >= 1.30
            and float(row["rsi_7"]) < 79.5
        ):
            desired_q2 = int(np.floor(pos.qty_base * self.pyramid_scale_ratio))
            if desired_q2 < 1:
                return

            pyr_fill = self.friction.buy_fill(c)
            free_bp = self._free_buying_power(active_positions)
            max_afford_q2 = int(np.floor(free_bp / pyr_fill))
            q2 = min(desired_q2, max_afford_q2)

            if q2 >= 1:
                pos.qty_pyramid = q2
                pos.pyramid_entry_price = pyr_fill
                unified_stop = round(pos.entry_price + 0.45 * atr_0, 2)
                if unified_stop < c:
                    pos.stop_price = max(pos.stop_price, unified_stop)
                    pos.is_risk_free = True

    def run_backtest(self, symbol_dfs: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict]:
        """
        Synchronized chronological portfolio backtest across the stock basket.
        Enforces Zero-Risk Locked Concurrency, House-Money Pyramiding, and Dhan statutory charges.
        """
        prepared = {sym: compute_v21_features(df) for sym, df in symbol_dfs.items()}
        frames = []
        for sym, df in prepared.items():
            d_sym = df.copy()
            d_sym["symbol"] = sym
            frames.append(d_sym)
        panel = pd.concat(frames, ignore_index=True).sort_values(["timestamp", "symbol"])

        active_positions: Dict[str, V21Position] = {}
        closed_trades: List[Dict] = []
        peak_equity = self.wallet_equity
        max_dd_pct = 0.0

        for ts, bar_slice in panel.groupby("timestamp", sort=True):
            # 1. Manage Open Positions
            for _, row in bar_slice.iterrows():
                sym = str(row["symbol"])
                if sym not in active_positions:
                    continue

                pos = active_positions[sym]
                should_exit, raw_exit, reason = self.evaluate_bar_and_corridor(pos, row)

                if should_exit:
                    exit_fill = self.friction.sell_fill(raw_exit)
                    sell_notional = pos.total_qty * exit_fill
                    taxes = self.friction.compute_statutory_charges(
                        pos.buy_notional_base, pos.buy_notional_pyramid, sell_notional
                    )
                    gross_pnl = sell_notional - pos.total_buy_notional
                    net_pnl = round(gross_pnl - taxes, 2)

                    self.wallet_equity = round(self.wallet_equity + net_pnl, 2)
                    peak_equity = max(peak_equity, self.wallet_equity)
                    dd = (peak_equity - self.wallet_equity) / peak_equity * 100.0
                    max_dd_pct = max(max_dd_pct, dd)

                    if net_pnl < 0.0:
                        self.consecutive_losses += 1
                    else:
                        self.consecutive_losses = 0

                    closed_trades.append({
                        "symbol": sym,
                        "entry_time": pos.entry_time,
                        "exit_time": ts,
                        "qty_base": pos.qty_base,
                        "qty_pyramid": pos.qty_pyramid,
                        "entry_price": pos.entry_price,
                        "exit_price": exit_fill,
                        "gross_pnl": round(gross_pnl, 2),
                        "dhan_taxes": taxes,
                        "net_pnl": net_pnl,
                        "bars_held": pos.bars_held,
                        "exit_reason": reason,
                        "wallet_after": self.wallet_equity
                    })
                    del active_positions[sym]
                else:
                    self.try_house_money_pyramid(pos, row, active_positions)

            # 2. Check Zero-Risk Slot Gate for New Entries
            if len(active_positions) >= self._allowed_slots():
                continue

            # NEVER open a secondary slot while any active slot still carries open initial risk
            if any(not p.is_risk_free for p in active_positions.values()):
                continue

            candidates = bar_slice[
                (bar_slice["entry_signal"]) & (~bar_slice["symbol"].isin(active_positions.keys()))
            ].sort_values("conviction_mult", ascending=False)

            if candidates.empty:
                continue

            top = candidates.iloc[0]
            sym = str(top["symbol"])
            entry_fill = self.friction.buy_fill(float(top["close"]))
            atr_0 = float(top["atr_14"])
            natr_0 = float(top["natr_14"])

            unit_risk_rs = self.compute_monotonic_unit_risk(natr_0, float(top["conviction_mult"]))
            qty_risk = int(np.floor(unit_risk_rs / atr_0))
            free_bp = self._free_buying_power(active_positions)
            qty_margin = int(np.floor(free_bp / entry_fill))
            qty = min(qty_risk, qty_margin)

            if qty >= 1:
                init_stop = round(entry_fill - self.physical_stop_atr * atr_0, 2)
                active_positions[sym] = V21Position(
                    symbol=sym,
                    entry_time=ts,
                    entry_price=entry_fill,
                    entry_open=float(top["open"]),
                    entry_mid=0.5 * (float(top["open"]) + float(top["close"])),
                    entry_vol=float(top["volume"]),
                    atr_at_entry=atr_0,
                    natr_at_entry=natr_0,
                    qty_base=qty,
                    stop_price=init_stop,
                    highest_high=float(top["high"])
                )

        trades_df = pd.DataFrame(closed_trades)
        return trades_df, self._summarize(trades_df, max_dd_pct)

    def _summarize(self, trades_df: pd.DataFrame, max_dd_pct: float) -> Dict:
        if trades_df.empty:
            return {"status": "NO_TRADES"}
        wins = trades_df[trades_df["net_pnl"] > 0.0]
        losses = trades_df[trades_df["net_pnl"] <= 0.0]
        win_sum = wins["net_pnl"].sum()
        loss_sum = abs(losses["net_pnl"].sum())

        return {
            "total_trades": len(trades_df),
            "win_rate_pct": round(len(wins) / len(trades_df) * 100.0, 2),
            "net_profit_rs": round(trades_df["net_pnl"].sum(), 2),
            "final_wallet_rs": round(self.wallet_equity, 2),
            "net_profit_factor": round(win_sum / max(loss_sum, 1e-6), 2),
            "avg_win_rs": round(wins["net_pnl"].mean(), 2) if not wins.empty else 0.0,
            "avg_loss_rs": round(losses["net_pnl"].mean(), 2) if not losses.empty else 0.0,
            "worst_single_loss_rs": round(trades_df["net_pnl"].min(), 2),
            "total_taxes_rs": round(trades_df["dhan_taxes"].sum(), 2),
            "max_drawdown_pct": round(max_dd_pct, 2)
        }
