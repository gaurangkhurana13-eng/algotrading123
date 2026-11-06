"""
Pre-Trade Monte Carlo Module for Bhavishya V30: Terminal-Aware Monte Carlo Engine (MC_term).
Evaluates fat-tailed Student's t (df=5) paths over dynamic horizon H_rem(t) to 15:15 IST.
A path is counted as a win if:
  1. It hits the TP barrier before SL, OR
  2. It never hits SL and terminates above +0.65x ATR at session close.
"""

import numpy as np
from .dhan_engine import calc_dhan_charges

class PreTradeMonteCarloEngine:
    def __init__(self, config: dict):
        mc_cfg = config.get("pre_trade_mc", {})
        self.n_sims        = int(mc_cfg.get("iterations", 2000))
        self.min_win_prob  = float(mc_cfg.get("min_win_prob", 0.49))
        self.min_ev        = float(mc_cfg.get("min_expected_ev", 0.0))
        self.max_cvar      = float(mc_cfg.get("max_cvar_loss", -45.0))
        self.df            = int(mc_cfg.get("degrees_of_freedom", 5))
        self.seed          = int(mc_cfg.get("seed", 42))
        self.terminal_gain_atr = float(mc_cfg.get("terminal_gain_atr", 0.65))
        self.ratchet_trigger_atr = float(config.get("ratchet_trigger_atr", 1.70))
        self.ratchet_floor_atr   = float(config.get("ratchet_floor_atr", 1.45))
        self.rng           = np.random.default_rng(self.seed)

    def evaluate_candidate(self, snapshot: dict) -> dict:
        entry_p   = float(snapshot["close"])
        atr_norm  = float(snapshot["atr_norm"])
        prob      = float(snapshot["prob"])
        qty       = int(snapshot["qty"])
        tp_pct    = float(snapshot["tp_pct"])
        sl_pct    = float(snapshot["sl_pct"])
        hold_bars = int(snapshot["hold_bars"])

        if entry_p <= 0 or atr_norm <= 0 or qty <= 0:
            return {
                "decision": "INSUFFICIENT EVIDENCE",
                "reason": "Invalid entry price, ATR, or quantity",
                "stats": {},
                "snapshot": snapshot
            }

        if hold_bars < 3:
            return {
                "decision": "REJECT",
                "reason": f"Horizon too short ({hold_bars} bars < 3 bars)",
                "stats": {"win_prob": 0.0, "expected_ev": -100.0, "cvar_loss": -100.0},
                "snapshot": snapshot
            }

        # Model drift per 15m bar
        step_drift = ((prob - 0.50) * atr_norm) / np.sqrt(hold_bars)
        step_vol   = atr_norm / np.sqrt(hold_bars)

        scale = np.sqrt((self.df - 2) / self.df)
        t_draws = self.rng.standard_t(df=self.df, size=(self.n_sims, hold_bars)) * scale
        path_innovations = step_drift + step_vol * t_draws

        log_paths = np.cumsum(path_innovations, axis=1)
        sim_prices = entry_p * np.exp(log_paths)

        p_trigger = entry_p * (1.0 + self.ratchet_trigger_atr * atr_norm)
        p_floor   = entry_p * (1.0 + self.ratchet_floor_atr * atr_norm)
        p_ceiling = entry_p * (1.0 + tp_pct)
        p_sl      = entry_p * (1.0 - sl_pct)

        exit_prices = np.zeros(self.n_sims)

        for i in range(self.n_sims):
            path = sim_prices[i]
            hit_trig = np.where(path >= p_trigger)[0]
            hit_sl   = np.where(path <= p_sl)[0]

            idx_trig = hit_trig[0] if len(hit_trig) > 0 else 999999
            idx_sl   = hit_sl[0] if len(hit_sl) > 0 else 999999

            if idx_trig < idx_sl and idx_trig < hold_bars:
                post_path = path[idx_trig:]
                hit_ceil = np.where(post_path >= p_ceiling)[0]
                if len(hit_ceil) > 0:
                    exit_prices[i] = p_ceiling
                else:
                    exit_prices[i] = max(p_floor, post_path[-1])
            elif idx_sl < idx_trig and idx_sl < hold_bars:
                exit_prices[i] = p_sl
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
            "cvar_loss": round(float(cvar_loss), 2)
        }

        # Multi-gate Pre-Trade Decision
        if win_prob < self.min_win_prob:
            decision = "REJECT"
            reason = f"Terminal-Aware win probability {win_prob*100:.1f}% below minimum {self.min_win_prob*100:.1f}%"
        elif expected_ev <= self.min_ev:
            decision = "REJECT"
            reason = f"Expected Net EV ₹{expected_ev:.2f} is non-positive"
        elif cvar_loss < self.max_cvar:
            decision = "REJECT"
            reason = f"5% CVaR tail loss ₹{cvar_loss:.2f} breaches safety limit ₹{self.max_cvar:.2f}"
        else:
            decision = "ALLOW ENTRY"
            reason = "Approved by Terminal-Aware Fat-Tail MC Engine"

        return {
            "decision": decision,
            "reason": reason,
            "stats": stats,
            "snapshot": snapshot
        }
