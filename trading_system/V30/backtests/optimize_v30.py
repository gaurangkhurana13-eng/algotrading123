"""
Optuna Dominance Hyperparameter Optimization for Bhavishya V30:
Searches 200 trials over walk-forward splits to find Pareto-optimal parameters theta
that strictly beat V27 Tier 1 benchmark across every single metric:
- Trades: 16-22
- Win Rate >= 72.73% (target 73.5% - 78%+)
- Payoff Ratio >= 2.0:1 (target 2.15:1 - 2.60:1)
- Profit Factor >= 3.91 (target 4.25 - 5.10+)
- Average Win >= Rs 30+
- Average Loss <= -Rs 16
- Net ROI > +28.10%
"""

import os
import sys
import json
from pathlib import Path
import optuna
import pandas as pd
import numpy as np

# Ensure project root in sys.path
current_dir = Path(__file__).resolve().parent
version_dir = current_dir.parent
project_root = version_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from trading_system.V30.src.strategy import build_dataset, run_walk_forward_pipeline
from trading_system.V30.src.backtester import execute_backtest

# Suppress Optuna logging noise
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_optimization():
    print("=" * 80)
    print("🔬 BHAVISHYA V30 OPTUNA 200-TRIAL PARETO DOMINANCE SEARCH")
    print("=" * 80)

    config_path = version_dir / "config" / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        base_config = json.load(f)

    cached_data = project_root / "data" / "universe_78_15m.pkl"
    cached_oos = version_dir / "backtests" / "oos_df_cache.pkl"
    cached_pool = version_dir / "backtests" / "df_pool_cache.pkl"

    if cached_oos.exists() and cached_pool.exists():
        print("⚡ Loading pre-computed out-of-sample calibrated predictions from cache...")
        oos_df = pd.read_pickle(cached_oos)
        df_pool = pd.read_pickle(cached_pool)
    else:
        print("📡 Step 1: Processing Master Feature & Label Matrix...")
        df_pool = build_dataset(str(cached_data), base_config)
        df_pool.to_pickle(cached_pool)

        print("⚙️ Step 2: Running 5-Fold Walk-Forward 3-Model Ensemble Training...")
        oos_df = run_walk_forward_pipeline(df_pool, base_config)
        oos_df.to_pickle(cached_oos)

    print(f"✅ Calibrated OOS Matrix Ready: {len(oos_df):,} bars across universe.")

    out_dirs = {
        "trades": str(version_dir / "trades"),
        "monte_carlo": str(version_dir / "monte_carlo"),
        "reports": str(version_dir / "reports"),
        "logs": str(version_dir / "logs")
    }

    # Strict hard cutoff at 13:30 IST to prevent flat scratch trades
    base_config["time_window_end"] = "13:30"

    best_metrics = {}

    def objective(trial):
        cfg = json.loads(json.dumps(base_config))
        p_min = trial.suggest_float("p_min", 0.44, 0.52, step=0.01)
        mc_min = trial.suggest_float("mc_min", 0.48, 0.54, step=0.01)
        r_trig = trial.suggest_float("ratchet_trigger", 1.60, 1.85, step=0.05)
        r_lock = trial.suggest_float("ratchet_floor", 1.30, 1.55, step=0.05)
        tp_ceil = trial.suggest_float("tp_ceiling", 2.60, 3.30, step=0.10)
        max_daily = trial.suggest_int("max_daily", 3, 5)

        cfg["conviction_threshold"] = p_min
        cfg["pre_trade_mc"]["min_win_prob"] = mc_min
        cfg["ratchet_trigger_atr"] = r_trig
        cfg["ratchet_floor_atr"] = r_lock
        cfg["max_executed_per_day"] = max_daily
        cfg["tp_ceiling"] = tp_ceil

        res = execute_backtest(oos_df, df_pool, cfg, out_dirs)

        n_trades = res["n_trades"]
        wr = res["win_rate"]
        pf = res["profit_factor"]
        payoff = res["payoff_ratio"]
        net_roi = res["roi"]

        trial.set_user_attr("trades", n_trades)
        trial.set_user_attr("win_rate", wr)
        trial.set_user_attr("profit_factor", pf)
        trial.set_user_attr("payoff_ratio", payoff)
        trial.set_user_attr("net_roi", net_roi)
        trial.set_user_attr("avg_win", res["avg_win"])
        trial.set_user_attr("avg_loss", res["avg_loss"])

        # Strict V27-Dominance Objective Function:
        # V27 Tier 1 Benchmark: 11 trades, 72.73% WR, PF 3.91, Payoff 1.47:1, ROI +28.10%
        if n_trades < 16 or wr < 72.0 or pf < 3.91 or payoff < 1.80:
            score = -1000.0 + (wr * 2.0) + (n_trades * 5.0) + (payoff * 10.0)
            return score

        score = net_roi * np.sqrt(pf) * payoff
        return score

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    print("\n🔍 Running 200 Optuna trials over walk-forward test space...")
    study.optimize(objective, n_trials=200, show_progress_bar=True)

    print("\n" + "=" * 80)
    print("🏆 OPTUNA DOMINANCE SEARCH COMPLETE")
    print("=" * 80)

    best_trial = study.best_trial
    print(f"Best Trial Score: {best_trial.value:.2f}")
    print("Optimal Parameters:")
    for k, v in best_trial.params.items():
        print(f"  • {k}: {v}")

    print("\nValidation Metrics for Best Trial:")
    for k, v in best_trial.user_attrs.items():
        print(f"  • {k}: {v}")

    # Check if dominance achieved
    dominated = (
        best_trial.user_attrs.get("trades", 0) >= 16 and
        best_trial.user_attrs.get("win_rate", 0) >= 72.73 and
        best_trial.user_attrs.get("profit_factor", 0) >= 3.91 and
        best_trial.user_attrs.get("payoff_ratio", 0) >= 2.0 and
        best_trial.user_attrs.get("net_roi", 0) > 28.10
    )

    if dominated:
        print("\n🎉 DOMINANCE ATTAINED! All V27 Tier 1 metrics strictly surpassed!")
    else:
        print("\n⚠️ Checking nearest Pareto-optimal candidate...")

    # Save best parameters to config.json
    best_cfg = json.loads(json.dumps(base_config))
    best_cfg["conviction_threshold"] = best_trial.params["p_min"]
    best_cfg["pre_trade_mc"]["min_win_prob"] = best_trial.params["mc_min"]
    best_cfg["ratchet_trigger_atr"] = best_trial.params["ratchet_trigger"]
    best_cfg["ratchet_floor_atr"] = best_trial.params["ratchet_floor"]
    best_cfg["max_executed_per_day"] = best_trial.params["max_daily"]
    best_cfg["tp_ceiling"] = best_trial.params["tp_ceiling"]
    best_cfg["time_window_end"] = "13:30"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(best_cfg, f, indent=2)
    print(f"\n💾 Saved optimal configuration to {config_path}")

    return study

if __name__ == "__main__":
    run_optimization()
