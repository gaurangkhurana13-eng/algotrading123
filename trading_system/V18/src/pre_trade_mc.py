"""
Pre-Trade Monte Carlo Decision Engine for Bhavishya V12
Simulates 10,000 future price paths before submitting an order.
V12 Innovation: Uses Student's t distribution (df=5) for realistic fat-tailed return modeling.
"""

import numpy as np
from .dhan_engine import calc_dhan_charges

class PreTradeMonteCarloEngine:
    def __init__(self, config: dict):
        self.config = config
        self.mc_cfg = config.get("pre_trade_mc", {})
        self.n_sims = self.mc_cfg.get("iterations", 10000)
        self.min_win_prob = self.mc_cfg.get("min_win_prob", 0.50)
        self.min_expected_ev = self.mc_cfg.get("min_expected_ev", 0.0)
        self.max_cvar_loss = self.mc_cfg.get("max_cvar_loss", -40.0)
        self.df = self.mc_cfg.get("degrees_of_freedom", 5)
        self.seed = self.mc_cfg.get("seed", 42)
        self.rng = np.random.default_rng(self.seed)

    def evaluate_candidate(self, snapshot: dict) -> dict:
        required_keys = ["timestamp", "ticker", "close", "atr_norm", "prob", "qty", "tp_pct", "sl_pct", "hold_bars"]
        for k in required_keys:
            if k not in snapshot or snapshot[k] is None:
                return {
                    "decision": "INSUFFICIENT EVIDENCE",
                    "reason": f"Missing value for {k}",
                    "stats": {},
                    "snapshot": snapshot
                }
            if isinstance(snapshot[k], (int, float, np.number)) and np.isnan(snapshot[k]):
                return {
                    "decision": "INSUFFICIENT EVIDENCE",
                    "reason": f"NaN value for {k}",
                    "stats": {},
                    "snapshot": snapshot
                }

        entry_p   = float(snapshot["close"])
        atr       = float(snapshot["atr_norm"])
        prob      = float(snapshot["prob"])
        qty       = int(snapshot["qty"])
        tp_pct    = float(snapshot["tp_pct"])
        sl_pct    = float(snapshot["sl_pct"])
        hold_bars = int(snapshot["hold_bars"])

        if entry_p <= 0 or atr <= 0 or qty <= 0:
            return {
                "decision": "INSUFFICIENT EVIDENCE",
                "reason": "Invalid entry price, ATR, or quantity",
                "stats": {},
                "snapshot": snapshot
            }

        # Model drift per 15m bar
        step_drift = ((prob - 0.50) * atr) / np.sqrt(hold_bars)
        step_vol   = atr / np.sqrt(hold_bars)

        # Generate 10,000 paths using Student's t distribution with df=5
        # Standardized Student's t has variance df / (df - 2)
        scale = np.sqrt((self.df - 2) / self.df)
        t_draws = self.rng.standard_t(df=self.df, size=(self.n_sims, hold_bars)) * scale
        path_innovations = step_drift + step_vol * t_draws

        # Cumulative log return paths
        log_paths = np.cumsum(path_innovations, axis=1)
        sim_prices = entry_p * np.exp(log_paths)

        tp_barrier = entry_p * (1.0 + tp_pct)
        sl_barrier = entry_p * (1.0 - sl_pct)

        exit_prices = np.zeros(self.n_sims)
        for i in range(self.n_sims):
            path = sim_prices[i]
            hit_tp = np.where(path >= tp_barrier)[0]
            hit_sl = np.where(path <= sl_barrier)[0]

            idx_tp = hit_tp[0] if len(hit_tp) > 0 else 999999
            idx_sl = hit_sl[0] if len(hit_sl) > 0 else 999999

            if idx_tp < idx_sl and idx_tp < hold_bars:
                exit_prices[i] = tp_barrier
            elif idx_sl < idx_tp and idx_sl < hold_bars:
                exit_prices[i] = sl_barrier
            else:
                exit_prices[i] = path[-1]

        gross_pnls = (exit_prices - entry_p) * qty
        mean_exit = float(np.mean(exit_prices))
        dhan_est_charges = calc_dhan_charges(entry_p, mean_exit, qty)
        net_pnls = gross_pnls - dhan_est_charges

        wins = np.sum(net_pnls > 0)
        win_prob = wins / self.n_sims
        expected_ev = float(np.mean(net_pnls))
        p5_var = float(np.percentile(net_pnls, 5))
        cvar_loss = float(np.mean(net_pnls[net_pnls <= p5_var]))

        stats = {
            "win_prob": round(win_prob, 4),
            "expected_net_ev": round(expected_ev, 2),
            "var_95_loss": round(p5_var, 2),
            "cvar_95_loss": round(cvar_loss, 2),
            "dhan_charges_est": round(dhan_est_charges, 2)
        }

        # Decision Logic
        if win_prob < self.min_win_prob:
            return {
                "decision": "REJECT ENTRY",
                "reason": f"Simulated Win Probability {win_prob*100:.1f}% below minimum {self.min_win_prob*100:.1f}%",
                "stats": stats,
                "snapshot": snapshot
            }

        if expected_ev <= self.min_expected_ev:
            return {
                "decision": "REJECT ENTRY",
                "reason": f"Simulated Net EV ₹{expected_ev:.2f} <= ₹{self.min_expected_ev:.2f}",
                "stats": stats,
                "snapshot": snapshot
            }

        if cvar_loss < self.max_cvar_loss:
            return {
                "decision": "REJECT ENTRY",
                "reason": f"Fat-Tailed CVaR 95% Loss ₹{cvar_loss:.2f} exceeds tolerance ₹{self.max_cvar_loss:.2f}",
                "stats": stats,
                "snapshot": snapshot
            }

        return {
            "decision": "ALLOW ENTRY",
            "reason": f"Passed all Pre-Trade Monte Carlo criteria (EV: ₹{expected_ev:.2f}, WinProb: {win_prob*100:.1f}%)",
            "stats": stats,
            "snapshot": snapshot
        }
