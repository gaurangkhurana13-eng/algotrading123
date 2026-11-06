"""
Pre-Trade Monte Carlo Module for Bhavishya V31: Terminal-Aware Three-Gear Monte Carlo Engine.
Evaluates fat-tailed Student's t (df=5) paths over dynamic horizon H_rem(t) to 15:15 IST.
Simulates:
  - Gear 1: +1.65x ATR trigger -> locks +1.35x ATR floor
  - Gear 2: +2.30x ATR trigger -> locks +1.95x ATR floor
  - Regime Ceiling: +2.65x ATR (standard) or +3.60x ATR (runner)
  - Structural SL: Invariant -1.50x ATR
"""

import numpy as np
from .dhan_engine import calc_dhan_charges

class PreTradeMonteCarloEngine:
    def __init__(self, config: dict):
        mc_cfg = config.get("pre_trade_mc", {})
        self.n_sims        = int(mc_cfg.get("iterations", 2000))
        self.min_win_prob  = float(mc_cfg.get("min_win_prob", 0.50))
        self.min_ev        = float(mc_cfg.get("min_expected_ev", 0.0))
        self.max_cvar      = float(mc_cfg.get("max_cvar_loss", -45.0))
        self.df            = int(mc_cfg.get("degrees_of_freedom", 5))
        self.seed          = int(mc_cfg.get("seed", 42))

        self.gear1_trig    = float(config.get("ratchet_gear1_trigger", 1.65))
        self.gear1_floor   = float(config.get("ratchet_gear1_floor", 1.35))
        self.gear2_trig    = float(config.get("ratchet_gear2_trigger", 2.30))
        self.gear2_floor   = float(config.get("ratchet_gear2_floor", 1.95))
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

        p_g1_trig = entry_p * (1.0 + self.gear1_trig * atr_norm)
        p_g1_flr  = entry_p * (1.0 + self.gear1_floor * atr_norm)
        p_g2_trig = entry_p * (1.0 + self.gear2_trig * atr_norm)
        p_g2_flr  = entry_p * (1.0 + self.gear2_floor * atr_norm)
        p_ceiling = entry_p * (1.0 + tp_pct)
        p_sl      = entry_p * (1.0 - sl_pct)

        exit_prices = np.zeros(self.n_sims)

        for i in range(self.n_sims):
            path = sim_prices[i]
            hit_g1 = np.where(path >= p_g1_trig)[0]
            hit_sl = np.where(path <= p_sl)[0]

            idx_g1 = hit_g1[0] if len(hit_g1) > 0 else 999999
            idx_sl = hit_sl[0] if len(hit_sl) > 0 else 999999

            if idx_g1 < idx_sl and idx_g1 < hold_bars:
                # Gear 1 activated before SL
                post_path = path[idx_g1:]
                hit_ceil = np.where(post_path >= p_ceiling)[0]
                hit_g2   = np.where(post_path >= p_g2_trig)[0]
                
                if len(hit_ceil) > 0:
                    exit_prices[i] = p_ceiling
                elif len(hit_g2) > 0:
                    # Gear 2 activated
                    idx_g2 = hit_g2[0]
                    g2_post = post_path[idx_g2:]
                    hit_ceil_g2 = np.where(g2_post >= p_ceiling)[0]
                    if len(hit_ceil_g2) > 0:
                        exit_prices[i] = p_ceiling
                    else:
                        exit_prices[i] = max(p_g2_flr, g2_post[-1])
                else:
                    exit_prices[i] = max(p_g1_flr, post_path[-1])
            elif idx_sl < idx_g1 and idx_sl < hold_bars:
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
            "cvar_loss": round(float(cvar_loss), 2),
            "p5_var": round(float(p5_var), 2)
        }

        # Decision Hurdle
        if win_prob >= self.min_win_prob and expected_ev > self.min_ev and cvar_loss > self.max_cvar:
            return {
                "decision": "ALLOW ENTRY",
                "reason": f"MC passed: WinProb={win_prob:.2f} >= {self.min_win_prob:.2f}, EV=Rs {expected_ev:.2f} > 0",
                "stats": stats,
                "snapshot": snapshot
            }
        else:
            reasons = []
            if win_prob < self.min_win_prob:
                reasons.append(f"WinProb={win_prob:.2f} < {self.min_win_prob:.2f}")
            if expected_ev <= self.min_ev:
                reasons.append(f"EV=Rs {expected_ev:.2f} <= {self.min_ev:.2f}")
            if cvar_loss <= self.max_cvar:
                reasons.append(f"CVaR=Rs {cvar_loss:.2f} <= {self.max_cvar:.2f}")
            return {
                "decision": "REJECT",
                "reason": f"MC rejected: {', '.join(reasons)}",
                "stats": stats,
                "snapshot": snapshot
            }
