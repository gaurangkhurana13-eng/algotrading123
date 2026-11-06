"""
Pre-Trade Monte Carlo Decision Engine for Bhavishya V23
V23 Innovation: Pre-Trade Monte Carlo Bayesian Drift Shrinkage & Microstructure Friction Penalty
- Shrinks empirical directional drift toward the uninformative zero prior using James-Stein Bayesian shrinkage (lambda = 0.65).
- Anchors excess probability to the empirical event base rate P0 (~0.27) rather than arbitrary 0.50.
- Deducts an explicit microstructure half-spread crossing drag (2 bps / bar) in the 10,000-path simulation.
- Retains Student's t distribution (df=5) for fat-tailed kurtosis modeling.
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
        self.drift_shrinkage = self.mc_cfg.get("drift_shrinkage_lambda", 0.65)
        self.spread_drag_per_bar = self.mc_cfg.get("spread_drag_per_bar", 0.0002)
        self.prior_base_rate = self.mc_cfg.get("prior_base_rate", 0.274)
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

        # Bayesian James-Stein Drift Shrinkage anchored to empirical base rate
        prob_excess = prob - self.prior_base_rate
        raw_step_drift = (prob_excess * atr) / np.sqrt(hold_bars)
        shrunk_step_drift = self.drift_shrinkage * raw_step_drift

        # Net drift per 15m bar including microstructure spread crossing penalty
        net_step_drift = shrunk_step_drift - self.spread_drag_per_bar
        step_vol   = atr / np.sqrt(hold_bars)

        # Standardized Student's t with df=5 has variance df / (df - 2)
        scale = np.sqrt((self.df - 2) / self.df)
        t_draws = self.rng.standard_t(df=self.df, size=(self.n_sims, hold_bars)) * scale
        path_innovations = net_step_drift + step_vol * t_draws

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
            "win_prob": round(float(win_prob), 4),
            "expected_ev": round(float(expected_ev), 2),
            "p5_var": round(float(p5_var), 2),
            "cvar_loss": round(float(cvar_loss), 2),
            "raw_drift": round(float(raw_step_drift), 6),
            "shrunk_drift": round(float(shrunk_step_drift), 6),
            "spread_drag": round(float(self.spread_drag_per_bar), 6)
        }

        # Multi-gate Pre-Trade Decision
        if win_prob < self.min_win_prob:
            decision = "REJECT"
            reason = f"Win probability {win_prob*100:.1f}% below minimum {self.min_win_prob*100:.1f}%"
        elif expected_ev < self.min_expected_ev:
            decision = "REJECT"
            reason = f"Expected EV Rs {expected_ev:.2f} is non-positive"
        elif cvar_loss < self.max_cvar_loss:
            decision = "REJECT"
            reason = f"Fat-tailed CVaR Rs {cvar_loss:.2f} exceeds risk ceiling Rs {self.max_cvar_loss:.2f}"
        else:
            decision = "ALLOW ENTRY"
            reason = f"Passed all gates: Win Prob {win_prob*100:.1f}%, EV Rs {expected_ev:.2f}, CVaR Rs {cvar_loss:.2f}"

        return {
            "decision": decision,
            "reason": reason,
            "stats": stats,
            "snapshot": snapshot
        }
