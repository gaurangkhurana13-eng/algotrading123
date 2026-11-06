"""
Pre-Trade Monte Carlo Module for Bhavishya V34: Conviction-Centered Fat-Tail Engine (MC_term).
Evaluates fat-tailed Student's t (df=5) paths over dynamic horizon H_rem(t) to 15:05 IST.
Simulates:
  - Phase 0: Invariant -1.50x ATR SL
  - Phase 1: at +1.60x ATR, locks hard floor at +1.35x ATR
  - Phase 2: at +2.30x ATR, locks floor at +1.95x ATR
  - Phase 3: at +3.20x ATR, locks floor at +2.80x ATR
  - Final TP Ceiling: +4.20x ATR
  - Trade-Relative Stagnation Exits (H >= 6, 8, 10)
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
        self.p3_trig       = float(config.get("ratchet_phase3_trigger", 3.20))
        self.p3_floor      = float(config.get("ratchet_phase3_floor", 2.80))
        self.tp_ceiling    = float(config.get("tp_ceiling", 4.20))
        self.sl_atr_mult   = float(config.get("sl_atr_mult", 1.5))
        self.rng           = np.random.default_rng(self.seed)

    def evaluate_candidate(self, snapshot: dict) -> dict:
        entry_p    = float(snapshot["close"])
        atr_norm   = float(snapshot["atr_norm"])
        z_score    = float(snapshot.get("z_score", 1.28))
        rank_score = float(snapshot.get("rank_score", 0.95))
        cs_rs      = float(snapshot.get("cs_rs", 0.0))
        rvol       = float(snapshot.get("rvol", 1.0))
        qty        = int(snapshot.get("qty", 10))
        hold_bars  = int(snapshot["hold_bars"])

        if entry_p <= 0 or atr_norm <= 0:
            return {"win_prob": 0.0, "expected_ev": 0.0}

        if hold_bars < 3:
            return {"win_prob": 0.0, "expected_ev": -100.0}

        vol_confirm = min(rvol / 2.0, 1.0)
        sgn_cs = np.sign(cs_rs)
        
        # Conviction & Momentum normalized drift
        z_norm = (z_score - 1.28) / 2.0
        drift_factor = 0.65 * max(-0.20, min(1.0, z_norm)) + 0.35 * sgn_cs * vol_confirm
        step_drift   = (drift_factor * atr_norm) / np.sqrt(hold_bars)
        
        # Scale standard Student's t by 1/sqrt(3) to match ATR variance (df=5 variance is 5/3)
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
        p_p3_trig = entry_p * (1.0 + self.p3_trig * atr_norm)
        p_p3_flr  = entry_p * (1.0 + self.p3_floor * atr_norm)
        p_ceiling = entry_p * (1.0 + self.tp_ceiling * atr_norm)
        p_sl_init = entry_p * (1.0 - self.sl_atr_mult * atr_norm)

        exit_prices = np.zeros(self.n_sims)

        for i in range(self.n_sims):
            path = sim_prices[i]
            active_sl = p_sl_init
            max_p = entry_p
            exited = False

            for k in range(hold_bars):
                p_k = path[k]
                if p_k > max_p:
                    max_p = p_k

                # Update Ratchet Levels (Never premature, Phase 0 invariant until +1.60x)
                if max_p >= p_p3_trig:
                    active_sl = max(active_sl, p_p3_flr)
                elif max_p >= p_p2_trig:
                    active_sl = max(active_sl, p_p2_flr)
                elif max_p >= p_p1_trig:
                    active_sl = max(active_sl, p_p1_flr)

                # Barrier checks
                if p_k >= p_ceiling:
                    exit_prices[i] = p_ceiling
                    exited = True
                    break
                elif p_k <= active_sl:
                    exit_prices[i] = active_sl
                    exited = True
                    break

                # Trade-Relative Stagnation Exits
                h_bars = k + 1
                mfe_atr = (max_p - entry_p) / (entry_p * atr_norm + 1e-9)
                if h_bars >= 6 and mfe_atr < 0.50 and p_k <= entry_p + 0.10 * entry_p * atr_norm:
                    exit_prices[i] = p_k
                    exited = True
                    break
                elif h_bars >= 8 and mfe_atr < 1.00:
                    exit_prices[i] = p_k
                    exited = True
                    break
                elif h_bars >= 10 and mfe_atr < 1.50:
                    exit_prices[i] = p_k
                    exited = True
                    break

            if not exited:
                exit_prices[i] = path[-1]

        gross_pnls = (exit_prices - entry_p) * qty
        sample_exit = float(np.median(exit_prices))
        charges = calc_dhan_charges(entry_p, sample_exit, qty)
        net_pnls = gross_pnls - charges

        win_prob = float(np.mean(net_pnls > 0))
        ev = float(np.mean(net_pnls))

        return {
            "win_prob": round(win_prob, 4),
            "expected_ev": round(ev, 2)
        }
