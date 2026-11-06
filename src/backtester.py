"""
Precision Backtesting Engine
----------------------------
Simulates bar-by-bar execution identical to TradingView Strategy Tester:
- Initial Capital: ₹100,000
- Position Sizing: 20% equity per trade
- Realistic Commission: 0.03% per fill
- Dynamic Stop Loss & Take Profit (intra-bar high/low evaluation)
- Tracks complete trade log and mark-to-market equity curve
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class Trade:
    trade_id: int
    direction: str          # "LONG" or "SHORT"
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    pnl_pct: float
    commission: float
    exit_reason: str        # "TAKE_PROFIT", "STOP_LOSS", "REVERSAL", "END_OF_DATA"


class BacktestEngine:
    def __init__(self, initial_capital: float = 100000.0, position_pct: float = 20.0, commission_pct: float = 0.03):
        self.initial_capital = initial_capital
        self.position_pct = position_pct
        self.commission_pct = commission_pct

    def run(self, df: pd.DataFrame, use_tp_sl: bool = True, trade_direction: str = "Both") -> Dict[str, Any]:
        """
        Executes backtest over signals DataFrame.
        Expected columns: open, high, low, close, buy_signal, sell_signal, long_sl, long_tp, short_sl, short_tp
        """
        capital = self.initial_capital
        cash = capital
        position = None  # None, or dict with details
        trades: List[Trade] = []
        equity_curve = []
        timestamps = []

        allow_long = trade_direction in ["Both", "Long Only"]
        allow_short = trade_direction in ["Both", "Short Only"]

        for i in range(len(df)):
            ts = df.index[i]
            o = df["open"].iloc[i]
            h = df["high"].iloc[i]
            l = df["low"].iloc[i]
            c = df["close"].iloc[i]

            buy_sig = df["buy_signal"].iloc[i]
            sell_sig = df["sell_signal"].iloc[i]
            long_sl = df["long_sl"].iloc[i] if "long_sl" in df else c * 0.98
            long_tp = df["long_tp"].iloc[i] if "long_tp" in df else c * 1.04
            short_sl = df["short_sl"].iloc[i] if "short_sl" in df else c * 1.02
            short_tp = df["short_tp"].iloc[i] if "short_tp" in df else c * 0.96

            # 1. Manage Active Position (Check intra-bar SL/TP)
            if position is not None:
                p_type = position["type"]
                entry_px = position["entry_price"]
                shares = position["shares"]
                pos_sl = position["sl"]
                pos_tp = position["tp"]

                exit_triggered = False
                exit_price = None
                exit_reason = None

                if p_type == "LONG":
                    # Conservative check: if SL hit
                    if use_tp_sl and l <= pos_sl:
                        exit_triggered = True
                        exit_price = min(o, pos_sl) if o < pos_sl else pos_sl  # Gap handling
                        exit_reason = "STOP_LOSS"
                    elif use_tp_sl and h >= pos_tp:
                        exit_triggered = True
                        exit_price = max(o, pos_tp) if o > pos_tp else pos_tp  # Gap handling
                        exit_reason = "TAKE_PROFIT"
                    elif sell_sig and allow_short:
                        # Reversal exit
                        exit_triggered = True
                        exit_price = c
                        exit_reason = "REVERSAL"

                elif p_type == "SHORT":
                    if use_tp_sl and h >= pos_sl:
                        exit_triggered = True
                        exit_price = max(o, pos_sl) if o > pos_sl else pos_sl
                        exit_reason = "STOP_LOSS"
                    elif use_tp_sl and l <= pos_tp:
                        exit_triggered = True
                        exit_price = min(o, pos_tp) if o < pos_tp else pos_tp
                        exit_reason = "TAKE_PROFIT"
                    elif buy_sig and allow_long:
                        exit_triggered = True
                        exit_price = c
                        exit_reason = "REVERSAL"

                if exit_triggered:
                    exit_value = shares * exit_price
                    comm_exit = exit_value * (self.commission_pct / 100.0)
                    total_comm = position["commission"] + comm_exit

                    if p_type == "LONG":
                        pnl = (exit_price - entry_px) * shares - total_comm
                    else:
                        pnl = (entry_px - exit_price) * shares - total_comm

                    pnl_pct = (pnl / (shares * entry_px)) * 100.0
                    cash += (position["cost_basis"] + pnl)

                    trades.append(Trade(
                        trade_id=len(trades) + 1,
                        direction=p_type,
                        entry_time=position["entry_time"],
                        exit_time=ts,
                        entry_price=entry_px,
                        exit_price=exit_price,
                        shares=shares,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        commission=total_comm,
                        exit_reason=exit_reason
                    ))
                    position = None

            # 2. Check for New Entries (Execution on confirmed close)
            current_total_equity = cash + (0 if position is None else self._calc_unrealized(position, c))
            if position is None:
                if buy_sig and allow_long:
                    allocation = current_total_equity * (self.position_pct / 100.0)
                    if allocation > 0 and c > 0:
                        shares = allocation / c
                        comm = allocation * (self.commission_pct / 100.0)
                        cash -= (allocation + comm)
                        position = {
                            "type": "LONG",
                            "entry_time": ts,
                            "entry_price": c,
                            "shares": shares,
                            "cost_basis": allocation,
                            "commission": comm,
                            "sl": long_sl,
                            "tp": long_tp
                        }
                elif sell_sig and allow_short:
                    allocation = current_total_equity * (self.position_pct / 100.0)
                    if allocation > 0 and c > 0:
                        shares = allocation / c
                        comm = allocation * (self.commission_pct / 100.0)
                        cash -= (allocation + comm)
                        position = {
                            "type": "SHORT",
                            "entry_time": ts,
                            "entry_price": c,
                            "shares": shares,
                            "cost_basis": allocation,
                            "commission": comm,
                            "sl": short_sl,
                            "tp": short_tp
                        }

            # 3. Mark to Market Equity
            bar_equity = cash + (0 if position is None else self._calc_unrealized(position, c))
            equity_curve.append(bar_equity)
            timestamps.append(ts)

        # Close any lingering open position at last bar
        if position is not None:
            c = df["close"].iloc[-1]
            p_type = position["type"]
            shares = position["shares"]
            entry_px = position["entry_price"]
            exit_val = shares * c
            comm_exit = exit_val * (self.commission_pct / 100.0)
            total_comm = position["commission"] + comm_exit
            if p_type == "LONG":
                pnl = (c - entry_px) * shares - total_comm
            else:
                pnl = (entry_px - c) * shares - total_comm
            pnl_pct = (pnl / (shares * entry_px)) * 100.0
            cash += (position["cost_basis"] + pnl)
            trades.append(Trade(
                trade_id=len(trades) + 1,
                direction=p_type,
                entry_time=position["entry_time"],
                exit_time=df.index[-1],
                entry_price=entry_px,
                exit_price=c,
                shares=shares,
                pnl=pnl,
                pnl_pct=pnl_pct,
                commission=total_comm,
                exit_reason="END_OF_DATA"
            ))
            equity_curve[-1] = cash

        # Create Equity Series
        eq_series = pd.Series(equity_curve, index=pd.DatetimeIndex(timestamps), name="equity")

        # Performance Metrics
        metrics = self._calculate_metrics(eq_series, trades)

        return {
            "metrics": metrics,
            "trades": trades,
            "trades_df": pd.DataFrame([t.__dict__ for t in trades]) if trades else pd.DataFrame(),
            "equity_curve": eq_series
        }

    @staticmethod
    def _calc_unrealized(position: dict, current_price: float) -> float:
        cost = position["cost_basis"]
        shares = position["shares"]
        entry = position["entry_price"]
        if position["type"] == "LONG":
            return cost + (current_price - entry) * shares
        else:
            return cost + (entry - current_price) * shares

    def _calculate_metrics(self, equity_curve: pd.Series, trades: List[Trade]) -> Dict[str, Any]:
        initial = self.initial_capital
        final_equity = equity_curve.iloc[-1] if not equity_curve.empty else initial
        net_profit = final_equity - initial
        net_profit_pct = (net_profit / initial) * 100.0

        # Drawdown
        running_max = equity_curve.cummax()
        drawdowns = (equity_curve - running_max) / running_max * 100.0
        max_drawdown_pct = abs(drawdowns.min()) if not drawdowns.empty else 0.0
        max_drawdown_val = abs((equity_curve - running_max).min()) if not equity_curve.empty else 0.0

        total_trades = len(trades)
        if total_trades > 0:
            profits = [t.pnl for t in trades if t.pnl > 0]
            losses = [t.pnl for t in trades if t.pnl <= 0]
            winning_trades = len(profits)
            losing_trades = len(losses)
            win_rate = (winning_trades / total_trades) * 100.0

            gross_profit = sum(profits) if profits else 0.0
            gross_loss = abs(sum(losses)) if losses else 0.0
            profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

            avg_trade = sum([t.pnl for t in trades]) / total_trades
            avg_trade_pct = sum([t.pnl_pct for t in trades]) / total_trades
            total_commissions = sum([t.commission for t in trades])
        else:
            winning_trades = 0
            losing_trades = 0
            win_rate = 0.0
            profit_factor = 0.0
            gross_profit = 0.0
            gross_loss = 0.0
            avg_trade = 0.0
            avg_trade_pct = 0.0
            total_commissions = 0.0

        return {
            "initial_capital": round(initial, 2),
            "final_equity": round(final_equity, 2),
            "net_profit": round(net_profit, 2),
            "net_profit_pct": round(net_profit_pct, 2),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": round(win_rate, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_drawdown_pct, 2),
            "max_drawdown_val": round(max_drawdown_val, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "avg_trade_pnl": round(avg_trade, 2),
            "avg_trade_pnl_pct": round(avg_trade_pct, 2),
            "total_commissions": round(total_commissions, 2)
        }
