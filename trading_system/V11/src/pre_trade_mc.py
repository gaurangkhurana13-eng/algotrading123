"""
Pre-Entry Monte Carlo Decision Engine
Generates 10,000 forward path simulations before an order is placed.
Outputs:
  - ALLOW ENTRY
  - REJECT ENTRY
  - INSUFFICIENT EVIDENCE
Logs timestamped snapshot with complete decision parameters to JSONL.
"""

import numpy as np
from .dhan_engine import calc_dhan_charges

class PreTradeMonteCarloEngine:
    def __init__(self, config: dict):
        self.mc_cfg = config.get("pre_trade_mc", {})
        self.n_sims = self.mc_cfg.get("iterations", 10000)
        self.min_win_prob = self.mc_cfg.get("min_win_prob", 0.50)
        self.min_ev = self.mc_cfg.get("min_expected_ev", 0.0)
        self.max_cvar_loss = self.mc_cfg.get("max_cvar_loss", -45.0)
        self.seed = self.mc_cfg.get("seed", 42)
        self.rng = np.random.default_rng(self.seed)

    def evaluate_candidate(self, snapshot: dict) -> dict:
        """
        Takes a candidate market snapshot available at decision timestamp.
        Simulates 10,000 paths and returns decision and distribution statistics.
        """
        # 1. Validate inputs
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
                "reason": "Non-positive price, atr, or quantity",
                "stats": {},
                "snapshot": snapshot
            }

        # 2. Model drift and volatility
        # Conservative drift calibrated by probability edge over 50%
        drift = (prob - 0.50) * atr * 0.12
        sigma = atr / np.sqrt(hold_bars)

        # 3. Simulate 10,000 paths (vectorized)
        shocks = self.rng.normal(loc=drift, scale=sigma, size=(self.n_sims, hold_bars))
        cum_ret = np.cumsum(shocks, axis=1)
        paths = entry_p * (1.0 + cum_ret)

        tp_p = entry_p * (1.0 + tp_pct)
        sl_p = entry_p * (1.0 - sl_pct)
        charges = calc_dhan_charges(entry_p, entry_p, qty)

        max_p = np.max(paths, axis=1)
        min_p = np.min(paths, axis=1)

        tp_hits = max_p >= tp_p
        sl_hits = min_p <= sl_p

        net_pnls = np.zeros(self.n_sims)
        outcomes = [] # "TP", "SL", "TIME"

        for i in range(self.n_sims):
            if tp_hits[i] and not sl_hits[i]:
                net = (tp_p - entry_p) * qty - charges
                net_pnls[i] = net
                outcomes.append("TP")
            elif sl_hits[i] and not tp_hits[i]:
                net = (sl_p - entry_p) * qty - charges
                net_pnls[i] = net
                outcomes.append("SL")
            else:
                # Path crossed both or neither; inspect sequential arrival
                path = paths[i]
                hit = False
                for p in path:
                    if p >= tp_p:
                        net_pnls[i] = (tp_p - entry_p) * qty - charges
                        outcomes.append("TP")
                        hit = True
                        break
                    elif p <= sl_p:
                        net_pnls[i] = (sl_p - entry_p) * qty - charges
                        outcomes.append("SL")
                        hit = True
                        break
                if not hit:
                    net_pnls[i] = (path[-1] - entry_p) * qty - charges
                    outcomes.append("TIME")

        # 4. Distribution Statistics
        win_prob = float(np.mean(net_pnls > 0))
        tp_prob  = float(np.mean([o == "TP" for o in outcomes]))
        sl_prob  = float(np.mean([o == "SL" for o in outcomes]))
        ev_pnl   = float(np.mean(net_pnls))
        median_pnl = float(np.median(net_pnls))
        p5_var   = float(np.percentile(net_pnls, 5))
        cvar_95  = float(np.mean(net_pnls[net_pnls <= p5_var]))
        prob_exceed_risk = float(np.mean(net_pnls < -35.0))

        stats = {
            "n_sims": self.n_sims,
            "win_prob": round(win_prob, 4),
            "tp_hit_prob": round(tp_prob, 4),
            "sl_hit_prob": round(sl_prob, 4),
            "expected_net_ev": round(ev_pnl, 2),
            "median_pnl": round(median_pnl, 2),
            "var_95": round(p5_var, 2),
            "cvar_95": round(cvar_95, 2),
            "prob_exceed_risk_limit": round(prob_exceed_risk, 4),
            "charges_est": charges
        }

        # 5. Predefined Entry Decision Criteria
        # Must have:
        # a) Win probability >= min_win_prob
        # b) Expected EV > min_expected_ev
        # c) Tail risk (CVaR 95%) must not breach max_cvar_loss
        if (win_prob >= self.min_win_prob) and (ev_pnl > self.min_ev) and (cvar_95 >= self.max_cvar_loss):
            decision = "ALLOW ENTRY"
            reason = f"Positive EV (₹{ev_pnl:.2f}), WinProb {win_prob*100:.1f}%, CVaR95 ₹{cvar_95:.2f} within risk limits"
        else:
            decision = "REJECT ENTRY"
            fail_reasons = []
            if win_prob < self.min_win_prob: fail_reasons.append(f"WinProb {win_prob*100:.1f}% < {self.min_win_prob*100:.0f}%")
            if ev_pnl <= self.min_ev: fail_reasons.append(f"EV ₹{ev_pnl:.2f} <= ₹{self.min_ev:.2f}")
            if cvar_95 < self.max_cvar_loss: fail_reasons.append(f"CVaR95 ₹{cvar_95:.2f} breaches limit ₹{self.max_cvar_loss:.2f}")
            reason = "; ".join(fail_reasons)

        return {
            "decision": decision,
            "reason": reason,
            "stats": stats,
            "snapshot": snapshot
        }
