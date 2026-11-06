"""
Pre-Trade Monte Carlo Module for Bhavishya V33: Rank-Centered Hybrid Drift Monte Carlo Engine (MC_term).
Evaluates fat-tailed Student's t (df=5) paths over dynamic horizon H_rem(t) to 15:15 IST.
Decouples drift from raw Platt probability using Cross-Sectional RankScore & Volume Impulse.
Simulates:
  - Phase 0: Invariant -1.50x ATR SL
  - Phase 1: at +1.60x ATR, locks hard floor at +1.35x ATR
  - Phase 2: at +2.30x ATR, locks floor at +1.95x ATR
  - TP Ceiling: +2.85x ATR
"""

import numpy as np
from .dhan_engine import calc_dhan_charges

class PreTradeMonteCarloEngine:
    def __init__(self, config: dict):
        mc_cfg = config.get("pre_trade_mc", {})
        self.n_sims        = int(mc_cfg.get("iterations", 2500))
        self.df            = int(mc_cfg.get("degrees_of_freedom", 5))
        self.seed          = int(mc_cfg.get("seed", 42))

        self.p1_trig       = float(config.get("ratchet_phase1_trigger", 1.60))
        self.p1_floor      = float(config.get("ratchet_phase1_floor", 1.35))
        self.p2_trig       = float(config.get("ratchet_phase2_trigger", 2.30))
        self.p2_floor      = float(config.get("ratchet_phase2_floor", 1.95))
        self.tp_ceiling    = float(config.get("tp_ceiling", 2.85))
        self.sl_atr_mult   = float(config.get("sl_atr_mult", 1.5))
        self.rng           = np.random.default_rng(self.seed)

    def evaluate_candidate(self, snapshot: dict) -> dict:
        entry_p    = float(snapshot["close"])
        atr_norm   = float(snapshot["atr_norm"])
        rank_score = float(snapshot.get("rank_score", 0.50))
        cs_rs      = float(snapshot.get("cs_rs", 0.0))
        rvol       = float(snapshot.get("rvol", 1.0))
        qty        = int(snapshot.get("qty", 10))
        hold_bars  = int(snapshot["hold_bars"])

        if entry_p <= 0 or atr_norm <= 0:
            return {"win_prob": 0.0, "expected_ev": 0.0}

        if hold_bars < 3:
            return {"win_prob": 0.0, "expected_ev": -100.0}

        # DeepSeek-R1 Section 1.3 & 1.4: Rank-Centered Hybrid Drift
        vol_confirm = min(rvol / 2.0, 1.0)
        sgn_cs = np.sign(cs_rs)
        
        # Combined normalized drift per bar
        drift_factor = 0.65 * (rank_score - 0.50) + 0.35 * sgn_cs * vol_confirm
        step_drift   = (drift_factor * atr_norm) / np.sqrt(hold_bars)
        
        # df=5 variance is 5/3, so scale standard Student's t by 1/sqrt(3) to match ATR variance
        step_vol = (atr_norm / np.sqrt(3.0)) / np.sqrt(hold_bars)

        scale = np.sqrt((self.df - 2) / self.df)
        t_draws = self.rng.standard_t(df=self.df, size=(self.n_sims, hold_bars)) * scale
        path_innovations = step_drift + step_vol * t_draws

        log_paths = np.cumsum(path_innovations, axis=1)
        sim_prices = entry_p * np.exp(log_paths)

        p_p1_trig = entry_p * (1.0 + self.p1_trig * atr_norm)
        p_p1_flr  = entry_p * (1.0 + self.p1_floor * atr_norm)
        p_p2_trig = entry_p * (1.0 + self.p2_trig * atr_norm)
        p_p2_flr  = entry_p * (1.0 + self.p2_floor * atr_norm)
        p_ceiling = entry_p * (1.0 + self.tp_ceiling * atr_norm)
        p_sl_init = entry_p * (1.0 - self.sl_atr_mult * atr_norm)

        exit_prices = np.zeros(self.n_sims)

        for i in range(self.n_sims):
            path = sim_prices[i]
            active_sl = p_sl_init
            phase = 0
            exited = False

            for k in range(hold_bars):
                p_k = path[k]

                # Update Ratchet Levels (Never premature, Phase 0 invariant until +1.60x)
                if p_k >= p_p2_trig:
                    phase = 2
                    active_sl = max(active_sl, p_p2_flr)
                elif p_k >= p_p1_trig:
                    phase = max(phase, 1)
                    active_sl = max(active_sl, p_p1_flr)

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
