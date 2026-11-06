"""
Pre-Trade Monte Carlo Module for Bhavishya V32: Terminal-Aware Loss-Shield Monte Carlo Engine (MC_term).
Evaluates fat-tailed Student's t (df=5) paths over dynamic horizon H_rem(t) to 15:15 IST.
Simulates:
  - Phase 0: -1.50x ATR SL
  - Shield 1: at +0.95x ATR, SL moves to -0.50x ATR
  - Shield 2: at +1.35x ATR, SL moves to Breakeven (+0.15x ATR)
  - Ratchet: at +1.75x ATR, locks +1.45x ATR floor
  - Ceiling: +2.95x ATR (or +3.45x ATR)
"""

import numpy as np
from .dhan_engine import calc_dhan_charges

class PreTradeMonteCarloEngine:
    def __init__(self, config: dict):
        mc_cfg = config.get("pre_trade_mc", {})
        self.n_sims        = int(mc_cfg.get("iterations", 1500))
        self.df            = int(mc_cfg.get("degrees_of_freedom", 5))
        self.seed          = int(mc_cfg.get("seed", 42))

        self.s1_trig       = float(config.get("shield1_trigger_atr", 0.95))
        self.s1_sl         = float(config.get("shield1_sl_atr", 0.50))
        self.s2_trig       = float(config.get("shield2_trigger_atr", 1.35))
        self.ratchet_trig  = float(config.get("ratchet_trigger_atr", 1.75))
        self.ratchet_floor = float(config.get("ratchet_floor_atr", 1.45))
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
            return {"win_prob": 0.0, "expected_ev": 0.0}

        if hold_bars < 3:
            return {"win_prob": 0.0, "expected_ev": -100.0}

        # Model drift per 15m bar
        step_drift = ((prob - 0.50) * atr_norm) / np.sqrt(hold_bars)
        step_vol   = atr_norm / np.sqrt(hold_bars)

        scale = np.sqrt((self.df - 2) / self.df)
        t_draws = self.rng.standard_t(df=self.df, size=(self.n_sims, hold_bars)) * scale
        path_innovations = step_drift + step_vol * t_draws

        log_paths = np.cumsum(path_innovations, axis=1)
        sim_prices = entry_p * np.exp(log_paths)

        p_s1_trig = entry_p * (1.0 + self.s1_trig * atr_norm)
        p_s1_sl   = entry_p * (1.0 - self.s1_sl * atr_norm)
        p_s2_trig = entry_p * (1.0 + self.s2_trig * atr_norm)
        p_s2_sl   = entry_p * (1.0 + 0.15 * atr_norm)
        p_r_trig  = entry_p * (1.0 + self.ratchet_trig * atr_norm)
        p_r_flr   = entry_p * (1.0 + self.ratchet_floor * atr_norm)
        p_ceiling = entry_p * (1.0 + tp_pct)
        p_sl_init = entry_p * (1.0 - sl_pct)

        exit_prices = np.zeros(self.n_sims)

        for i in range(self.n_sims):
            path = sim_prices[i]
            active_sl = p_sl_init
            exited = False

            for k in range(hold_bars):
                p_k = path[k]
                
                # Update SL levels
                if p_k >= p_r_trig:
                    active_sl = max(active_sl, p_r_flr)
                elif p_k >= p_s2_trig:
                    active_sl = max(active_sl, p_s2_sl)
                elif p_k >= p_s1_trig:
                    active_sl = max(active_sl, p_s1_sl)

                if p_k >= p_ceiling:
                    exit_prices[i] = p_ceiling
                    exited = True
                    break
                elif p_k <= active_sl:
                    exit_prices[i] = active_sl
                    exited = True
                    break

            if not exited:
                exit_prices[i] = path[-1]

        gross_pnls = (exit_prices - entry_p) * qty
        mean_exit = float(np.mean(exit_prices))
        dhan_est_charges = calc_dhan_charges(entry_p, mean_exit, qty)
        net_pnls = gross_pnls - dhan_est_charges

        wins = np.sum(net_pnls > 0)
        win_prob = wins / self.n_sims
        expected_ev = float(np.mean(net_pnls))

        return {
            "win_prob": round(float(win_prob), 4),
            "expected_ev": round(float(expected_ev), 2)
        }
