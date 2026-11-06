import sys, pickle, time
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

from scripts.multi_slot_sim_engine import (
    run_multi_slot_sim,
    stocks_30_legacy,
    core_27,
    u_2021,
    V29_30_UNIVERSE
)

# Load 12 Timeline Pools
print("Loading 10-year pools...")
t0 = time.time()
p_2016_h1 = pickle.load(open(ROOT / 'data/2016_H1_UNSEEN/universe_30_15m.pkl', 'rb'))
p_2016_h2 = pickle.load(open(ROOT / 'data/2016_H2_UNSEEN/universe_30_15m.pkl', 'rb'))
p_2017 = pickle.load(open(ROOT / 'data/2017_UNSEEN/universe_30_15m.pkl', 'rb'))
p_2018 = pickle.load(open(ROOT / 'data/2018_UNSEEN/universe_30_15m.pkl', 'rb'))
p_2019 = pickle.load(open(ROOT / 'data/2019_UNSEEN/universe_30_15m.pkl', 'rb'))
p_2020 = pickle.load(open(ROOT / 'data/2020_FULL/universe_30_15m.pkl', 'rb'))
p_2021 = pickle.load(open(ROOT / 'data/2021-22/universe_30_15m.pkl', 'rb'))
p_2022 = pickle.load(open(ROOT / 'data/2022-23/universe_30_15m.pkl', 'rb'))
p_2023 = pickle.load(open(ROOT / 'data/2023-24/universe_30_15m.pkl', 'rb'))
p_2024 = pickle.load(open(ROOT / 'data/2024-25/universe_14_15m.pkl', 'rb'))
p_2025 = pickle.load(open(ROOT / 'data/2025-26/universe_14_15m.pkl', 'rb'))
p_2026 = pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))
print(f"Loaded in {time.time()-t0:.2f}s")

timeline = [
    ("2016_H1", p_2016_h1, "2016-01-01", "2016-06-30", stocks_30_legacy),
    ("2016_H2", p_2016_h2, "2016-07-01", "2016-12-31", stocks_30_legacy),
    ("2017", p_2017, "2017-01-01", "2017-12-31", stocks_30_legacy),
    ("2018", p_2018, "2018-01-01", "2018-12-31", stocks_30_legacy),
    ("2019", p_2019, "2019-01-01", "2019-12-31", None),
    ("2020", p_2020, "2020-01-01", "2020-12-31", core_27),
    ("2021", p_2021, "2021-01-01", "2021-12-31", u_2021),
    ("2022", p_2022, "2022-01-01", "2022-12-31", V29_30_UNIVERSE),
    ("2023", p_2023, "2023-01-02", "2023-12-29", V29_30_UNIVERSE),
    ("2024", p_2024, "2024-01-01", "2024-12-31", V29_30_UNIVERSE),
    ("2025", p_2025, "2025-01-01", "2025-12-31", V29_30_UNIVERSE),
    ("2026", p_2026, "2026-01-01", "2026-10-29", V29_30_UNIVERSE),
]

def run_all_eras(max_slots=1, slot_unlock=1000.0, formula="formula_b", sector_div=True, init_cap=500.0):
    results = []
    for era_name, pool, s_dt, e_dt, u_filt in timeline:
        res = run_multi_slot_sim(
            pool, s_dt, e_dt,
            universe_filter=u_filt,
            base_risk=25.0,
            initial_equity=init_cap,
            max_slots=max_slots,
            slot_unlock_equity=slot_unlock,
            formula_mode=formula,
            enforce_sector_diversification=sector_div
        )
        res['era'] = era_name
        results.append(res)
    return results

def main():
    print("=" * 135)
    print("BHAVISHYA V42 MULTI-SLOT ARCHITECTURE MASTER BENCHMARK (10-YEAR HISTORICAL DATASET 2016-2026)")
    print("Starting Capital per Era: Rs 500.00 | Slot 2 Unlock Threshold: Rs 1,000.00 | Actual Dhan Brokerage + Slippage")
    print("=" * 135)

    # 1. Benchmark 1-Slot Baseline
    print("\n>>> RUNNING EXPERIMENT 1: 1-SLOT BASELINE (V42 ORIGINAL)...")
    t0 = time.time()
    res_1slot = run_all_eras(max_slots=1)
    print(f"Completed in {time.time()-t0:.1f}s")

    # 2. Benchmark 2-Slot Formula B (4.5% Cap per Slot, 9% Combined)
    print("\n>>> RUNNING EXPERIMENT 2: 2-SLOT CONCURRENCY (FORMULA B: 4.5% Cap per Slot + Sector Div)...")
    t0 = time.time()
    res_2slot_b = run_all_eras(max_slots=2, slot_unlock=1000.0, formula="formula_b", sector_div=True)
    print(f"Completed in {time.time()-t0:.1f}s")

    # 3. Benchmark 2-Slot Formula C (Dynamic Slot 2 Allocation)
    print("\n>>> RUNNING EXPERIMENT 3: 2-SLOT CONCURRENCY (FORMULA C: Dynamic Slot 2 + Sector Div)...")
    t0 = time.time()
    res_2slot_c = run_all_eras(max_slots=2, slot_unlock=1000.0, formula="formula_c", sector_div=True)
    print(f"Completed in {time.time()-t0:.1f}s")

    # 4. Benchmark 2-Slot Formula B WITHOUT Sector Diversification
    print("\n>>> RUNNING EXPERIMENT 4: 2-SLOT CONCURRENCY (NO SECTOR ISOLATION ABLATION)...")
    t0 = time.time()
    res_2slot_nosec = run_all_eras(max_slots=2, slot_unlock=1000.0, formula="formula_b", sector_div=False)
    print(f"Completed in {time.time()-t0:.1f}s")

    # Print Comparative Table
    print("\n" + "=" * 145)
    print(f"{'Era':<8} | {'1-Slot Net':<11} {'1-Slot DD':<10} | {'2-Slot B Net':<13} {'2-Slot B DD':<12} {'WR %':<7} | {'2-Slot C Net':<13} {'2-Slot C DD':<12} {'WR %':<7} | {'NoSec Net':<11} {'NoSec DD':<9}")
    print("-" * 145)

    tot_1_net, tot_b_net, tot_c_net, tot_nosec_net = 0.0, 0.0, 0.0, 0.0
    tot_1_trades, tot_b_trades, tot_c_trades, tot_nosec_trades = 0, 0, 0, 0
    tot_1_time_util, tot_b_time_util, tot_b_2slot_time = 0.0, 0.0, 0.0
    tot_1_margin_util, tot_b_margin_util = 0.0, 0.0

    for i in range(len(timeline)):
        era = timeline[i][0]
        r1 = res_1slot[i]
        rb = res_2slot_b[i]
        rc = res_2slot_c[i]
        rns = res_2slot_nosec[i]

        tot_1_net += r1['net_pnl']
        tot_b_net += rb['net_pnl']
        tot_c_net += rc['net_pnl']
        tot_nosec_net += rns['net_pnl']

        tot_1_trades += r1['trades_count']
        tot_b_trades += rb['trades_count']
        tot_c_trades += rc['trades_count']
        tot_nosec_trades += rns['trades_count']

        tot_1_time_util += r1['time_util_pct']
        tot_b_time_util += rb['time_util_pct']
        tot_b_2slot_time += rb['two_slot_time_pct']
        tot_1_margin_util += r1['avg_margin_util_pct']
        tot_b_margin_util += rb['avg_margin_util_pct']

        print(f"{era:<8} | Rs {r1['net_pnl']:+8.2f} {r1['max_dd']:7.1f}%  | Rs {rb['net_pnl']:+10.2f} {rb['max_dd']:9.1f}% {rb['win_rate']:6.1f}% | Rs {rc['net_pnl']:+10.2f} {rc['max_dd']:9.1f}% {rc['win_rate']:6.1f}% | Rs {rns['net_pnl']:+8.2f} {rns['max_dd']:7.1f}%")

    print("-" * 145)
    print(f"{'TOTAL':<8} | Rs {tot_1_net:+8.2f}          | Rs {tot_b_net:+10.2f}              | Rs {tot_c_net:+10.2f}              | Rs {tot_nosec_net:+8.2f}")
    print(f"{'TRADES':<8} | {tot_1_trades:<20} | {tot_b_trades:<26} | {tot_c_trades:<26} | {tot_nosec_trades}")
    print(f"{'AVG TIME UTIL':<15}: 1-Slot = {tot_1_time_util/12:.2f}% | 2-Slot B = {tot_b_time_util/12:.2f}% (Concurrent 2-Slot active = {tot_b_2slot_time/12:.3f}%)")
    print(f"{'AVG MARGIN UTIL'}: 1-Slot = {tot_1_margin_util/12:.2f}% | 2-Slot B = {tot_b_margin_util/12:.2f}%")

    # 5. Continuous 10-Year Compounding Test
    print("\n" + "=" * 135)
    print("CONTINUOUS 10-YEAR COMPOUNDING SIMULATION (2016 -> 2026 UNBROKEN CHAIN)")
    print("Starting Capital: Rs 500.00 at Jan 2016 -> Equity rolls over across all 12 eras")
    print("=" * 135)

    def run_continuous_chain(max_slots=1, formula="formula_b"):
        ek = 500.0
        all_trades = []
        era_summaries = []
        for era_name, pool, s_dt, e_dt, u_filt in timeline:
            res = run_multi_slot_sim(
                pool, s_dt, e_dt,
                universe_filter=u_filt,
                base_risk=25.0,
                initial_equity=ek,
                max_slots=max_slots,
                slot_unlock_equity=1000.0,
                formula_mode=formula,
                enforce_sector_diversification=True
            )
            era_net = res['net_pnl']
            start_ek = ek
            ek = res['final_cap']
            all_trades.extend(res['trades'])
            era_summaries.append({
                "era": era_name, "start_ek": start_ek, "final_ek": ek, "net": era_net, "trades": res['trades_count']
            })
        
        # Calculate full curve DD
        eq_curve = [500.0]
        for t in all_trades:
            eq_curve.append(eq_curve[-1] + t["net_pnl"])
        eq_s = pd.Series(eq_curve)
        pk = eq_s.cummax()
        max_dd = ((pk - eq_s) / pk).max() * 100.0
        wr = (sum(1 for t in all_trades if t['net_pnl'] > 0) / len(all_trades) * 100.0) if all_trades else 0.0

        return ek, ek - 500.0, max_dd, wr, len(all_trades), era_summaries

    chain_1_final, chain_1_net, chain_1_dd, chain_1_wr, chain_1_tr, _ = run_continuous_chain(max_slots=1)
    print(f"\n1-Slot Continuous Compounding:")
    print(f"Final Capital: Rs {chain_1_final:.2f} | Net PnL: Rs {chain_1_net:.2f} | Max DD: {chain_1_dd:.2f}% | Win Rate: {chain_1_wr:.1f}% | Trades: {chain_1_tr}")

    chain_2_final, chain_2_net, chain_2_dd, chain_2_wr, chain_2_tr, summaries = run_continuous_chain(max_slots=2, formula="formula_b")
    print(f"\n2-Slot Continuous Compounding (Formula B: 4.5% Cap per Slot):")
    print(f"Final Capital: Rs {chain_2_final:.2f} | Net PnL: Rs {chain_2_net:.2f} | Max DD: {chain_2_dd:.2f}% | Win Rate: {chain_2_wr:.1f}% | Trades: {chain_2_tr}")

    print("\nEra-by-Era Compounding Progression (2-Slot Concurrency):")
    for s in summaries:
        print(f"  {s['era']:<8}: Start Rs {s['start_ek']:9.2f} -> End Rs {s['final_ek']:9.2f} (Net: Rs {s['net']:+9.2f}, {s['trades']:2d} trades)")

    # Save all results to pickle for permanent record
    output_data = {
        "res_1slot": res_1slot,
        "res_2slot_b": res_2slot_b,
        "res_2slot_c": res_2slot_c,
        "res_2slot_nosec": res_2slot_nosec,
        "continuous_1slot": {"final": chain_1_final, "net": chain_1_net, "dd": chain_1_dd, "wr": chain_1_wr, "trades": chain_1_tr},
        "continuous_2slot": {"final": chain_2_final, "net": chain_2_net, "dd": chain_2_dd, "wr": chain_2_wr, "trades": chain_2_tr, "summaries": summaries}
    }
    with open(ROOT / 'data/multi_slot_10yr_benchmark_results.pkl', 'wb') as f:
        pickle.dump(output_data, f)
    print(f"\nSaved benchmark results to {ROOT / 'data/multi_slot_10yr_benchmark_results.pkl'}")

if __name__ == "__main__":
    main()
